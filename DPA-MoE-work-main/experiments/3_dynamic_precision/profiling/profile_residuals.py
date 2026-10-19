"""Per-expert chunk-residual measurement + greedy budget solver.

For each (layer, expert) measures

    R_e^{(K)}  =  E_{x routed to e} [ ‖ E_e^orig(x) − E_e^{(K)}(x) ‖_2 ]

across K = 2 .. 8, then runs a greedy budget solver per layer:

    K_e ∈ {2, …, 8},   Σ_e K_e ≤ B,   B ∈ {2k, 3k, …, 8k}     (k = E)

and emits one layout CSV per B.

Inputs:
  --hf-dir         HF BF16 model dir
  --codebook-dir   ss_anybcq_layer/ — per-layer packed-bytes safetensors
  --corpus         train/calibration text file(s), never eval/test data
  --max-tokens     cap on calibration tokens [16384]
  --max-per-expert reservoir cap per (layer, expert) [256]
  --seed           fixed calibration seed [0]
  --out            output dir [cache/experiment/Qwen3-30B-A3B/layout_solver]

Outputs:
  expert_residuals.npz   R[L, E, 7]  A[L, E]  N[L, E]
  layouts_B{2..8}k.csv   layer,expert,kept_chunks
  layouts_summary.csv    budget_label, B, predicted_error_total
"""
from __future__ import annotations

import argparse
import csv
import gc
import heapq
import pathlib
import random
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent.parent.parent   # profiling → 3_dynamic_precision → experiments → DPA-MoE

sys.path.insert(0, str(_HERE))
from dpa_moe.format import HEADER_STRUCT            # noqa: E402
from _collector import Collector, make_gate_hook   # noqa: E402
from _compute import compute_layer_residuals       # noqa: E402


# ── Solver ─────────────────────────────────────────────────────────────────────

def greedy_layer(R_layer: np.ndarray, a_layer: np.ndarray, B: int,
                 K_min: int = 2, K_max: int = 8) -> np.ndarray:
    """Greedy budget allocation per layer.

    R_layer: (E, n_K) residuals, K_min..K_max.
    a_layer: (E,) activity scores.
    Returns K[e] for e in 0..E-1.  Σ K ≤ B.
    """
    E, nK = R_layer.shape
    assert nK == K_max - K_min + 1
    K = np.full(E, K_min, dtype=np.int32)
    used = K_min * E
    if used > B:
        return K
    h = []
    for e in range(E):
        if K_max <= K_min:
            continue
        if np.isnan(R_layer[e, 0]) or np.isnan(R_layer[e, 1]):
            continue
        gain = float(a_layer[e]) * float(R_layer[e, 0] - R_layer[e, 1])
        if gain > 0:
            heapq.heappush(h, (-gain, e, K_min))
    remaining = B - used
    while remaining > 0 and h:
        neg_gain, e, k = heapq.heappop(h)
        if K[e] != k:
            continue
        K[e] = k + 1
        remaining -= 1
        if k + 1 < K_max:
            i_next = (k + 1) - K_min
            if np.isnan(R_layer[e, i_next]) or np.isnan(R_layer[e, i_next + 1]):
                continue
            gain = float(a_layer[e]) * float(R_layer[e, i_next] - R_layer[e, i_next + 1])
            if gain > 0:
                heapq.heappush(h, (-gain, e, k + 1))
    return K


def predicted_error(R_layer: np.ndarray, a_layer: np.ndarray,
                    K_assign: np.ndarray, K_min: int = 2) -> float:
    """Σ_e a_e · R_e^(K_e), valid (non-nan) entries only."""
    total = 0.0
    for e in range(R_layer.shape[0]):
        k_idx = int(K_assign[e]) - K_min
        v = R_layer[e, k_idx]
        if not np.isnan(v):
            total += float(a_layer[e]) * float(v)
    return total


# ── Driver helpers ─────────────────────────────────────────────────────────────

def _shard_stem(tok_offset: int, layer_start: int, layer_end: int,
                n_layers: int, shard_flag: bool) -> str:
    """Return output file stem based on shard parameters."""
    is_partial = (layer_start > 0 or layer_end < n_layers
                  or tok_offset > 0 or shard_flag)
    if not is_partial:
        return "expert_residuals"
    if shard_flag or tok_offset > 0:
        return f"expert_residuals_tok{tok_offset}"
    return f"expert_residuals_L{layer_start}_{layer_end}"


def _load_model(hf_dir: pathlib.Path, device: str):
    """Load HF BF16 model; patch rope_scaling quirks. Returns (model, tok, cfg)."""
    import transformers.utils.import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    if isinstance(getattr(cfg, "rope_scaling", None), dict) and "type" not in cfg.rope_scaling:
        cfg.rope_scaling = None
    _is_custom = bool(getattr(cfg, "auto_map", None))
    if not _is_custom and getattr(cfg, "rope_parameters", None) is None:
        cfg.rope_parameters = {"rope_type": "default",
                               "rope_theta": float(getattr(cfg, "rope_theta", 10000.0))}
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True, attn_implementation="eager")
    model.config.use_cache = False
    model.eval()
    return model, tok, cfg


def _tokenize_corpus(corpus_paths: list[str], tokenizer,
                     offset: int, max_tokens: int) -> torch.Tensor:
    """Concatenate corpus files, tokenize, slice [offset:offset+max_tokens]."""
    text = "\n".join(pathlib.Path(p).read_text() for p in corpus_paths)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return ids[offset : offset + max_tokens]


def _run_forward(model, token_ids: torch.Tensor, layer_start: int, layer_end: int,
                 seq_len: int, collector: Collector, device: str, log) -> None:
    """Forward pass in seq_len-token chunks; fills collector buffers."""
    handles = []
    for L in range(layer_start, layer_end):
        mlp = model.model.layers[L].mlp
        if not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
            continue
        if not hasattr(mlp.gate, "top_k"):
            log(f"  layer {L}: mlp.gate has no top_k "
                f"(got {type(mlp.gate).__name__}); skip")
            continue
        h = mlp.gate.register_forward_hook(make_gate_hook(L, mlp, collector))
        handles.append(h)
    log(f"hooks: {len(handles)} MoE layers")
    log("forward pass...")
    for start in range(0, token_ids.numel(), seq_len):
        chunk = token_ids[start:start + seq_len].unsqueeze(0).to(device)
        with torch.no_grad():
            model(chunk)
        if (start // seq_len) % 4 == 0:
            log(f"  forward: {start + chunk.shape[1]} / {token_ids.numel()}")
    for h in handles:
        h.remove()
    log("forward done")


def _save_shard(R: np.ndarray, A_mean: np.ndarray, N: np.ndarray,
                N_eff: np.ndarray, K_min: int, K_max: int,
                out_dir: pathlib.Path, stem: str, seed: int,
                corpus: list[str]) -> None:
    """Write expert_residuals .npz + .safetensors for one shard."""
    np.savez_compressed(out_dir / f"{stem}.npz",
                        R=R, A=A_mean.astype(np.float32),
                        N=N.astype(np.int64),
                        N_effective=N_eff,
                        K_min=K_min, K_max=K_max,
                        seed=seed,
                        corpus=np.array(corpus))
    save_safetensors(
        {"R": torch.from_numpy(R),
         "activity_mean": torch.from_numpy(A_mean.astype(np.float32)),
         "routed_token_count": torch.from_numpy(N.astype(np.int64))},
        str(out_dir / f"{stem}.safetensors"),
        metadata={"description": "Per-(layer, expert) expert-output residual "
                                  "R_e^(K) = E[‖E_e^orig(x) − E_e^(K)(x)‖_2] / n_x",
                  "K_min": str(K_min), "K_max": str(K_max),
                  "n_layers": str(R.shape[0]), "n_experts": str(R.shape[1]),
                  "seed": str(seed),
                  "corpus": ",".join(corpus)})


def _run_solver(R: np.ndarray, A_mean: np.ndarray, n_layers: int, n_experts: int,
                K_min: int, K_max: int, out_dir: pathlib.Path, log) -> None:
    """Greedy budget solver → layout_B{K}k.csv + layouts_summary.csv."""
    k = n_experts
    summary_rows = []
    for B_mult in range(K_min, K_max + 1):
        B_per_layer = B_mult * k
        layout_path = out_dir / f"layout_B{B_mult}k.csv"
        total_err = 0.0
        with open(layout_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["layer", "expert", "kept_chunks", "activity", "predicted_err_at_K"])
            for L in range(n_layers):
                Rl = R[L]
                al = A_mean[L]
                if np.all(np.isnan(Rl)):
                    continue
                K_assign = greedy_layer(Rl, al, B_per_layer, K_min, K_max)
                err = predicted_error(Rl, al, K_assign, K_min)
                total_err += err
                for e in range(n_experts):
                    k_e = int(K_assign[e])
                    Rerr = Rl[e, k_e - K_min]
                    w.writerow([L, e, k_e, f"{float(al[e]):.6e}",
                                "" if np.isnan(Rerr) else f"{float(Rerr):.6e}"])
        log(f"  B={B_mult}k → {layout_path.name} predicted_err_total={total_err:.4f}")
        summary_rows.append((f"B{B_mult}k", B_mult * k, total_err))
    with open(out_dir / "layouts_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["budget_label", "B_per_layer", "predicted_error_total"])
        w.writerows(summary_rows)


def _diagnostics(R: np.ndarray, layer_start: int, layer_end: int,
                 n_experts: int, log) -> None:
    """Print monotonicity check + NaN report."""
    n_non_mono = 0
    valid_curves = 0
    for L in range(layer_start, layer_end):
        for e in range(n_experts):
            r = R[L, e]
            if np.any(np.isnan(r)):
                continue
            valid_curves += 1
            if np.any(np.diff(r) > 0):
                n_non_mono += 1
    log(f"monotonicity: {n_non_mono}/{valid_curves} expert R-curves non-monotone "
        f"({100*n_non_mono/max(1,valid_curves):.1f}%)")
    n_fully_nan = int(np.sum(np.all(np.isnan(R[layer_start:layer_end]), axis=2)))
    n_total_le  = (layer_end - layer_start) * n_experts
    n_nan_cells = int(np.isnan(R[layer_start:layer_end]).sum())
    log(f"NaN: {n_fully_nan}/{n_total_le} expert slots fully unsampled "
        f"({100*n_fully_nan/n_total_le:.1f}%), "
        f"{n_nan_cells} total NaN cells in R")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, type=pathlib.Path)
    ap.add_argument("--codebook-dir", required=True, type=pathlib.Path)
    # IMPORTANT: default to the CALIBRATION corpus, not the test corpus.
    # The residuals + activity arrays feed the offline layout solver, whose
    # output is evaluated against wikitext-2-test in 3_dynamic_precision.
    # Profiling on the test set would let the solver overfit to the eval.
    ap.add_argument("--corpus", nargs="+", default=[
        str(REPO / "cache" / "datasets" / "mbpp_train_text.txt"),
        str(REPO / "cache" / "datasets" / "gsm8k_train_text.txt"),
        str(REPO / "cache" / "datasets" / "hellaswag_train_text.txt"),
        str(REPO / "cache" / "datasets" / "calibration_v3.txt"),
        str(REPO / "cache" / "datasets" / "calibration_v5.txt"),
        str(REPO / "cache" / "datasets" / "wikitext-2-raw-train.txt"),
    ], help="one or more text files; concatenated in order before tokenising")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-per-expert", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed for deterministic calibration profiling")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "cache" / "experiment" / "ss_anybcq" / "layout_solver")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=-1,
                    help="last layer (exclusive); -1 = all layers")
    ap.add_argument("--token-offset", type=int, default=0,
                    help="skip this many tokens before slicing max-tokens")
    ap.add_argument("--shard", action="store_true",
                    help="data-parallel mode: use _tok{offset} suffix even at offset=0")
    ap.add_argument("--force", action="store_true",
                    help="skip idempotency check and recompute even if output exists")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    def log(*a):
        print(f"[{time.time()-t0:.1f}s]", *a, flush=True)

    log("loading HF model (bf16)...")
    model, tok, cfg = _load_model(args.hf_dir, args.device)

    n_layers = cfg.num_hidden_layers
    n_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None)
    hidden = cfg.hidden_size
    top_k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_experts_active_per_tok", 8)
    log(f"model: {n_layers} layers, {n_experts} experts, hidden={hidden}, top_k={top_k}")

    layer_start = args.layer_start
    layer_end   = n_layers if args.layer_end == -1 else args.layer_end
    tok_offset  = args.token_offset
    log(f"layer shard: [{layer_start}, {layer_end}) of {n_layers}"
        + (f"  token offset: {tok_offset}" if tok_offset > 0 else ""))

    log("tokenizing corpus...")
    log(f"  corpus files: {args.corpus}")
    log(f"  calibration seed: {args.seed}")
    ids = _tokenize_corpus(args.corpus, tok, tok_offset, args.max_tokens)
    log(f"corpus tokens: {ids.numel()}")

    collector = Collector(n_layers, n_experts, hidden, top_k,
                          args.max_per_expert, args.device)
    _run_forward(model, ids, layer_start, layer_end,
                 args.seq_len, collector, args.device, log)

    W_sum_cpu = collector.W_sum.cpu().numpy().astype(np.float64)
    N_cpu     = collector.N.cpu().numpy().astype(np.int64)
    A         = W_sum_cpu / np.maximum(1, N_cpu)
    A_mean    = W_sum_cpu / max(1, ids.numel())
    N         = N_cpu
    log(f"  routing: tokens={ids.numel()}, hot experts (a_mean>0.5%): "
        f"{int((A_mean > 0.005).sum())} / {n_layers * n_experts}")
    np.savez_compressed(args.out / "routing_activity.npz",
                        A=A.astype(np.float32),
                        A_mean=A_mean.astype(np.float32),
                        N=N.astype(np.int64))
    save_safetensors(
        {"activity_mean": torch.from_numpy(A_mean.astype(np.float32)),
         "activity_routed_mean": torch.from_numpy(A.astype(np.float32)),
         "routed_token_count": torch.from_numpy(N.astype(np.int64))},
        str(args.out / "routing_activity.safetensors"),
        metadata={"description": "Per-(layer, expert) routing activity from calibration corpus",
                  "n_layers": str(n_layers), "n_experts": str(n_experts),
                  "n_tokens": str(int(ids.numel())), "top_k": str(int(top_k))})

    sample_paths = sorted(args.codebook_dir.glob("layer_*.safetensors"))
    if not sample_paths:
        raise FileNotFoundError(f"no layer_*.safetensors in {args.codebook_dir}")
    sample_path = sample_paths[0]
    with safe_open(sample_path, framework="pt") as fp:
        sample_b = bytes(fp.get_tensor("e000.gate.bytes").numpy())
    sample_hdr = HEADER_STRUCT.unpack(sample_b[:HEADER_STRUCT.size])
    base_p   = int(sample_hdr[3])
    target_p = int(sample_hdr[4])
    K_min, K_max = base_p, target_p
    log(f"codebook: base_p={base_p}  target_p={target_p}  →  K ∈ [{K_min}, {K_max}]")

    R = np.full((n_layers, n_experts, K_max - K_min + 1), np.nan, dtype=np.float32)
    stem = _shard_stem(tok_offset, layer_start, layer_end, n_layers, args.shard)
    for L in range(layer_start, layer_end):
        layer_path = args.codebook_dir / f"layer_{L:02d}.safetensors"
        if not layer_path.exists():
            log(f"  layer {L}: codebook missing, skip")
            continue
        hf_mlp = model.model.layers[L].mlp
        if not hasattr(hf_mlp, "experts"):
            continue
        layer_per_expert = collector.gather_per_expert(L)
        Rl = compute_layer_residuals(
            L, layer_path, hf_mlp, layer_per_expert,
            K_min, K_max, base_p, args.device, log)
        R[L] = Rl
        N_eff = np.minimum(N_cpu, args.max_per_expert).astype(np.int64)
        _save_shard(R, A_mean, N, N_eff, K_min, K_max, args.out, stem,
                    args.seed, [str(p) for p in args.corpus])
        del layer_per_expert
        collector.free_layer(L)
        torch.cuda.empty_cache()
        gc.collect()

    _diagnostics(R, layer_start, layer_end, n_experts, log)
    log("residuals done")

    _run_solver(R, A_mean, n_layers, n_experts, K_min, K_max, args.out, log)
    log("solver done")


if __name__ == "__main__":
    main()
