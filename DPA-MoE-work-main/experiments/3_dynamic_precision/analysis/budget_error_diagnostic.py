#!/usr/bin/env python3
"""Measure same-budget dynamic mixed-precision residual error.

For a small calibration slice, this evaluates the allocator-facing question:

    given a Kbar budget and profile-DP precision assignment K[e],
    what is the actual routed residual ||sum_e g_e Delta_e(K[e])||^2?

It also reports the diagonal estimate used by the allocator on the same
sampled activations. Integer Kbar budgets additionally include uniform-K
exact/diagonal residuals for direct same-budget comparison.
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
from _collector import Collector, expert_forward, make_gate_hook  # noqa: E402
from _compute import _plane_deltas, unpack_chunk_bytes            # noqa: E402
from profile_residuals import _load_model, _tokenize_corpus                 # noqa: E402
from check_allocator import allocate                              # noqa: E402


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


def _gate_sq_by_expert(idx: torch.Tensor, score: torch.Tensor, n_experts: int) -> np.ndarray:
    out = torch.zeros((n_experts,), dtype=torch.float64, device=score.device)
    flat_e = idx.reshape(-1).long()
    flat_w = score.reshape(-1).double()
    out.scatter_add_(0, flat_e, flat_w * flat_w)
    return out.cpu().numpy()


def _assignments_for_layer(
    R_layer: np.ndarray,
    gate_sq: np.ndarray,
    budgets: list[float],
    K_min: int,
    K_max: int,
) -> dict[float, list[int]]:
    experts = np.flatnonzero(gate_sq > 0).astype(int).tolist()
    gates = np.sqrt(gate_sq[experts]).astype(float).tolist()
    return {
        kbar: allocate(R_layer, experts, gates, kbar, K_min, K_max)
        for kbar in budgets
    }, experts


def _compute_layer(
    layer_idx: int,
    layer_path: pathlib.Path,
    hf_mlp,
    collector: Collector,
    profile_R: np.ndarray,
    budgets: list[float],
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
    n_experts = collector.n_experts

    gate_sq = _gate_sq_by_expert(Idx, Sc, n_experts)
    dyn_by_budget, active_experts = _assignments_for_layer(
        profile_R[layer_idx], gate_sq, budgets, K_min, K_max)
    active_set = set(active_experts)
    dyn_k_by_budget = {
        kbar: {e: K for e, K in zip(active_experts, ks)}
        for kbar, ks in dyn_by_budget.items()
    }

    exact_dyn = {
        kbar: torch.zeros((T, hidden), dtype=torch.float32, device=device)
        for kbar in budgets
    }
    diag_dyn = {kbar: torch.zeros((), dtype=torch.float64, device=device) for kbar in budgets}
    exact_uni = {
        kbar: torch.zeros((T, hidden), dtype=torch.float32, device=device)
        for kbar in budgets if abs(kbar - round(kbar)) < 1e-9
    }
    diag_uni = {kbar: torch.zeros((), dtype=torch.float64, device=device) for kbar in exact_uni}

    flat_e = Idx.reshape(-1)
    token_idx = torch.arange(T, device=Idx.device).unsqueeze(1).expand_as(Idx).reshape(-1)
    flat_w = Sc.reshape(-1)
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
            if e not in active_set:
                continue
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
            delta_by_k: dict[int, torch.Tensor] = {}
            for K in range(K_min, K_max + 1):
                k_extras = K - base_p
                if k_extras >= 1:
                    W_g = W_g + dlt_g[:, :, k_extras - 1]
                    W_u = W_u + dlt_u[:, :, k_extras - 1]
                    W_d = W_d + dlt_d[:, :, k_extras - 1]
                Wg = W_g.view(n_g, ng_g * gs_g)[:, :m_g].to(torch.bfloat16)
                Wu = W_u.view(n_u, ng_u * gs_u)[:, :m_u].to(torch.bfloat16)
                Wd = W_d.view(n_d, ng_d * gs_d)[:, :m_d].to(torch.bfloat16)
                y_K = expert_forward(Xg, Wg, Wu, Wd).to(torch.float32)
                delta_by_k[K] = y_orig.reshape(Xg.shape[0], -1) - y_K.reshape(Xg.shape[0], -1)

            for kbar in budgets:
                K = dyn_k_by_budget[kbar][e]
                delta = delta_by_k[K]
                exact_dyn[kbar][rows] += ws[:, None] * delta
                diag_dyn[kbar] += ((ws.double() * ws.double()) * (delta.double() * delta.double()).sum(dim=-1)).sum()

            for kbar in exact_uni:
                K = int(round(kbar))
                delta = delta_by_k[K]
                exact_uni[kbar][rows] += ws[:, None] * delta
                diag_uni[kbar] += ((ws.double() * ws.double()) * (delta.double() * delta.double()).sum(dim=-1)).sum()

            del W_g, W_u, W_d, dlt_g, dlt_u, dlt_d, base_g, base_u, base_d
            del y_orig, Xg, delta_by_k
            torch.cuda.empty_cache()

    rows_out = []
    for kbar in budgets:
        dyn_exact = float((exact_dyn[kbar].double() * exact_dyn[kbar].double()).sum().cpu())
        dyn_diag = float(diag_dyn[kbar].cpu())
        ks = dyn_by_budget[kbar]
        row = {
            "layer": layer_idx,
            "tokens": T,
            "kbar": kbar,
            "active_experts": len(active_experts),
            "target_sum_k": int(round(len(active_experts) * kbar)),
            "actual_sum_k": int(sum(ks)),
            "dynamic_exact": dyn_exact,
            "dynamic_diag": dyn_diag,
            "dynamic_exact_over_diag": dyn_exact / dyn_diag if dyn_diag > 0 else np.nan,
            "uniform_k": "",
            "uniform_exact": np.nan,
            "uniform_diag": np.nan,
            "dynamic_exact_over_uniform_exact": np.nan,
        }
        if kbar in exact_uni:
            uni_exact = float((exact_uni[kbar].double() * exact_uni[kbar].double()).sum().cpu())
            uni_diag = float(diag_uni[kbar].cpu())
            row.update({
                "uniform_k": int(round(kbar)),
                "uniform_exact": uni_exact,
                "uniform_diag": uni_diag,
                "dynamic_exact_over_uniform_exact": dyn_exact / uni_exact if uni_exact > 0 else np.nan,
            })
        rows_out.append(row)

    log(f"layer {layer_idx}: done")
    collector.free_layer(layer_idx)
    torch.cuda.empty_cache()
    gc.collect()
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, type=pathlib.Path)
    ap.add_argument("--codebook-dir", required=True, type=pathlib.Path)
    ap.add_argument("--profile-npz", required=True, type=pathlib.Path)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budgets", nargs="+", type=float, default=[3, 4, 5, 6, 7])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=4)
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
    log(f"tokens={ids.numel()} layers=[{args.layer_start},{layer_end}) budgets={args.budgets}")

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
    profile_R = _fill_profile_nan(np.load(args.profile_npz, allow_pickle=True)["R"])

    rows = []
    for L in range(args.layer_start, layer_end):
        layer_path = args.codebook_dir / f"layer_{L:02d}.safetensors"
        if not layer_path.exists() or not collector.X_chunks[L]:
            continue
        rows.extend(_compute_layer(
            L, layer_path, model.model.layers[L].mlp, collector,
            profile_R, args.budgets, K_min, K_max, base_p, args.device, log))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fieldnames = [
            "layer", "tokens", "kbar", "active_experts", "target_sum_k",
            "actual_sum_k", "dynamic_exact", "dynamic_diag",
            "dynamic_exact_over_diag", "uniform_k", "uniform_exact",
            "uniform_diag", "dynamic_exact_over_uniform_exact",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
