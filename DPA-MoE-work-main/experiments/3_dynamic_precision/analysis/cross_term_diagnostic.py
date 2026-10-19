#!/usr/bin/env python3
"""Measure routed MoE reconstruction-error approximation error.

The runtime allocator uses a diagonal approximation:

    ||sum_e g_e delta_e||^2 ~= sum_e g_e^2 ||delta_e||^2

This diagnostic reruns a small calibration slice, computes the exact routed
error with cross terms, and reports the ratio against both:

  1. sample_diag: diagonal approximation on the same sampled activations
  2. profile_diag: runtime-style per-expert mean R[L,e,K] weighted by sum(g^2)
"""
from __future__ import annotations

import argparse
import csv
import gc
import pathlib
import random
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent.parent.parent

sys.path.insert(0, str(_HERE.parent / "profiling"))
sys.path.insert(0, str(_HERE))
from _collector import Collector, make_gate_hook, expert_forward  # noqa: E402
from _compute import _plane_deltas, unpack_chunk_bytes            # noqa: E402
from profile_residuals import _load_model, _tokenize_corpus                 # noqa: E402


def _fill_profile_nan(R: np.ndarray) -> np.ndarray:
    out = R.astype(np.float32, copy=True)
    for L in range(out.shape[0]):
        for k in range(out.shape[2]):
            col = out[L, :, k]
            mask = np.isnan(col)
            if mask.any() and not mask.all():
                col[mask] = np.nanmean(col)
                out[L, :, k] = col
    return np.nan_to_num(out, nan=0.0)


def _compute_layer(
    layer_idx: int,
    layer_path: pathlib.Path,
    hf_mlp,
    collector: Collector,
    profile_R: np.ndarray | None,
    K_min: int,
    K_max: int,
    base_p: int,
    device: str,
    log,
) -> list[dict]:
    X = torch.cat(collector.X_chunks[layer_idx], dim=0)
    Idx = torch.cat(collector.Idx_chunks[layer_idx], dim=0)
    Sc = torch.cat(collector.Score_chunks[layer_idx], dim=0).float()
    T = int(X.shape[0])
    hidden = int(X.shape[1])
    n_K = K_max - K_min + 1
    n_experts = collector.n_experts

    exact_acc = torch.zeros((n_K, T, hidden), dtype=torch.float32, device=device)
    sample_diag = torch.zeros((n_K,), dtype=torch.float64, device=device)
    gate_sq_sum = torch.zeros((n_experts,), dtype=torch.float64, device=device)

    flat_e = Idx.reshape(-1)
    token_idx = torch.arange(T, device=Idx.device).unsqueeze(1).expand_as(Idx).reshape(-1)
    flat_w = Sc.reshape(-1)
    gate_sq_sum.scatter_add_(0, flat_e.long(), (flat_w.double() * flat_w.double()))

    sorted_e, perm = torch.sort(flat_e)
    sorted_token = token_idx[perm]
    sorted_w = flat_w[perm]
    unique_e, counts = torch.unique_consecutive(sorted_e, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts

    experts = hf_mlp.experts
    fused_experts = hasattr(experts, "gate_up_proj")
    with safe_open(layer_path, framework="pt") as fp, torch.no_grad():
        keys = set(fp.keys())
        for ue, st, ct in zip(unique_e.tolist(), starts.tolist(), counts.tolist()):
            e = int(ue)
            rows = sorted_token[st : st + ct]
            ws = sorted_w[st : st + ct].to(device=device, dtype=torch.float32)
            Xg = X[rows].to(device=device, dtype=torch.bfloat16)

            if fused_experts:
                gate_up = experts.gate_up_proj
                down_p = experts.down_proj
                _, two_I, _ = gate_up.shape
                I = two_I // 2
                Wg_orig = gate_up[e, :I, :].to(device=device, dtype=torch.bfloat16)
                Wu_orig = gate_up[e, I:, :].to(device=device, dtype=torch.bfloat16)
                Wd_orig = down_p[e].to(device=device, dtype=torch.bfloat16)
            else:
                expert = experts[e]
                Wg_orig = expert.gate_proj.weight.to(device=device, dtype=torch.bfloat16)
                Wu_orig = expert.up_proj.weight.to(device=device, dtype=torch.bfloat16)
                Wd_orig = expert.down_proj.weight.to(device=device, dtype=torch.bfloat16)

            y_orig = expert_forward(Xg, Wg_orig, Wu_orig, Wd_orig).to(torch.float32)
            cbs = {}
            for kind in ("gate", "up", "down"):
                key = f"e{e:03d}.{kind}.bytes"
                if key not in keys:
                    raise KeyError(key)
                cbs[kind] = unpack_chunk_bytes(fp.get_tensor(key).numpy().tobytes())

            base_g, dlt_g, n_g, ng_g, gs_g, m_g, _ = _plane_deltas(cbs["gate"], device)
            base_u, dlt_u, n_u, ng_u, gs_u, m_u, _ = _plane_deltas(cbs["up"], device)
            base_d, dlt_d, n_d, ng_d, gs_d, m_d, _ = _plane_deltas(cbs["down"], device)
            W_g = base_g.clone()
            W_u = base_u.clone()
            W_d = base_d.clone()
            for K in range(K_min, K_max + 1):
                k_idx = K - K_min
                k_extras = K - base_p
                if k_extras >= 1:
                    W_g = W_g + dlt_g[:, :, k_extras - 1]
                    W_u = W_u + dlt_u[:, :, k_extras - 1]
                    W_d = W_d + dlt_d[:, :, k_extras - 1]
                Wg = W_g.view(n_g, ng_g * gs_g)[:, :m_g].to(torch.bfloat16)
                Wu = W_u.view(n_u, ng_u * gs_u)[:, :m_u].to(torch.bfloat16)
                Wd = W_d.view(n_d, ng_d * gs_d)[:, :m_d].to(torch.bfloat16)
                y_K = expert_forward(Xg, Wg, Wu, Wd).to(torch.float32)
                delta = y_orig.reshape(Xg.shape[0], -1) - y_K.reshape(Xg.shape[0], -1)
                d2 = (delta * delta).sum(dim=-1).double()
                sample_diag[k_idx] += ((ws.double() * ws.double()) * d2).sum()
                exact_acc[k_idx, rows, :] += ws[:, None] * delta

            del W_g, W_u, W_d, dlt_g, dlt_u, dlt_d, base_g, base_u, base_d
            del y_orig, Xg
            torch.cuda.empty_cache()

    exact = (exact_acc.double() * exact_acc.double()).sum(dim=(1, 2)).cpu().numpy()
    sample = sample_diag.cpu().numpy()
    gate_sq = gate_sq_sum.cpu().numpy()
    rows_out = []
    for K in range(K_min, K_max + 1):
        k_idx = K - K_min
        profile = float(np.dot(gate_sq, profile_R[layer_idx, :, k_idx])) if profile_R is not None else np.nan
        ex = float(exact[k_idx])
        sd = float(sample[k_idx])
        rows_out.append({
            "layer": layer_idx,
            "K": K,
            "tokens": T,
            "exact": ex,
            "sample_diag": sd,
            "profile_diag": profile,
            "cross_term": ex - sd,
            "exact_over_sample_diag": ex / sd if sd > 0 else np.nan,
            "profile_diag_over_exact": profile / ex if ex > 0 else np.nan,
        })
    log(f"layer {layer_idx}: done")
    del exact_acc
    collector.free_layer(layer_idx)
    torch.cuda.empty_cache()
    gc.collect()
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, type=pathlib.Path)
    ap.add_argument("--codebook-dir", required=True, type=pathlib.Path)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--profile-npz", type=pathlib.Path, default=None)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=-1)
    ap.add_argument("--token-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()

    def log(*parts):
        print(f"[{time.time() - t0:.1f}s]", *parts, flush=True)

    log("loading model")
    model, tok, cfg = _load_model(args.hf_dir, args.device)
    n_layers = cfg.num_hidden_layers
    n_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None)
    hidden = cfg.hidden_size
    top_k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_experts_active_per_tok", 8)
    layer_end = n_layers if args.layer_end == -1 else args.layer_end

    log("tokenizing")
    ids = _tokenize_corpus(args.corpus, tok, args.token_offset, args.max_tokens)
    log(f"tokens={ids.numel()} layers=[{args.layer_start},{layer_end})")

    collector = Collector(n_layers, n_experts, hidden, top_k, args.max_tokens, args.device)
    handles = []
    for L in range(args.layer_start, layer_end):
        mlp = model.model.layers[L].mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts") and hasattr(mlp.gate, "top_k"):
            handles.append(mlp.gate.register_forward_hook(make_gate_hook(L, mlp, collector)))
    log(f"hooks={len(handles)}")
    for start in range(0, ids.numel(), args.seq_len):
        chunk = ids[start:start + args.seq_len].unsqueeze(0).to(args.device)
        with torch.no_grad():
            model(chunk)
        log(f"forward {start + chunk.shape[1]} / {ids.numel()}")
    for h in handles:
        h.remove()

    sample_path = sorted(args.codebook_dir.glob("layer_*.safetensors"))[0]
    with safe_open(sample_path, framework="pt") as fp:
        from format import HEADER_STRUCT
        sample_b = bytes(fp.get_tensor("e000.gate.bytes").numpy())
    sample_hdr = HEADER_STRUCT.unpack(sample_b[:HEADER_STRUCT.size])
    base_p = int(sample_hdr[3])
    K_min = base_p
    K_max = int(sample_hdr[4])

    profile_R = None
    if args.profile_npz is not None:
        profile_R = _fill_profile_nan(np.load(args.profile_npz, allow_pickle=True)["R"])

    rows = []
    for L in range(args.layer_start, layer_end):
        layer_path = args.codebook_dir / f"layer_{L:02d}.safetensors"
        if not layer_path.exists() or not collector.X_chunks[L]:
            continue
        rows.extend(_compute_layer(
            L, layer_path, model.model.layers[L].mlp, collector,
            profile_R, K_min, K_max, base_p, args.device, log))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fieldnames = [
            "layer", "K", "tokens", "exact", "sample_diag", "profile_diag",
            "cross_term", "exact_over_sample_diag", "profile_diag_over_exact",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
