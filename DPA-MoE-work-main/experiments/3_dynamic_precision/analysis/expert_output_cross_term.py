#!/usr/bin/env python3
"""Measure diagonal approximation error for BF16 routed expert outputs.

For each MoE layer this computes:

    exact = sum_t ||sum_u g[t,u] E_{e[t,u]}(x_t)||^2
    diag  = sum_t sum_u g[t,u]^2 ||E_{e[t,u]}(x_t)||^2

This is the scale of the cross-expert terms in the real BF16 expert mixture,
not the quantization residual profile.
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

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent.parent.parent

sys.path.insert(0, str(_HERE.parent / "profiling"))
sys.path.insert(0, str(_HERE))
from _collector import Collector, expert_forward, make_gate_hook  # noqa: E402
from profile_residuals import _load_model, _tokenize_corpus                 # noqa: E402


def compute_layer(layer_idx: int, hf_mlp, collector: Collector, device: str, log) -> dict:
    X = torch.cat(collector.X_chunks[layer_idx], dim=0)
    Idx = torch.cat(collector.Idx_chunks[layer_idx], dim=0)
    Sc = torch.cat(collector.Score_chunks[layer_idx], dim=0).float()
    T = int(X.shape[0])
    hidden = int(X.shape[1])

    exact_acc = torch.zeros((T, hidden), dtype=torch.float32, device=device)
    diag = torch.zeros((), dtype=torch.float64, device=device)

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
    with torch.no_grad():
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
                Wg = gate_up[e, :I, :].to(device=device, dtype=torch.bfloat16)
                Wu = gate_up[e, I:, :].to(device=device, dtype=torch.bfloat16)
                Wd = down_p[e].to(device=device, dtype=torch.bfloat16)
            else:
                expert = experts[e]
                Wg = expert.gate_proj.weight.to(device=device, dtype=torch.bfloat16)
                Wu = expert.up_proj.weight.to(device=device, dtype=torch.bfloat16)
                Wd = expert.down_proj.weight.to(device=device, dtype=torch.bfloat16)
            y = expert_forward(Xg, Wg, Wu, Wd).to(torch.float32).reshape(Xg.shape[0], -1)
            diag += ((ws.double() * ws.double()) * (y.double() * y.double()).sum(dim=-1)).sum()
            exact_acc[rows, :] += ws[:, None] * y
            del Xg, y

    exact = float((exact_acc.double() * exact_acc.double()).sum().item())
    diag_f = float(diag.item())
    collector.free_layer(layer_idx)
    del exact_acc
    torch.cuda.empty_cache()
    gc.collect()
    log(f"layer {layer_idx}: exact/diag={exact / diag_f if diag_f else float('nan'):.6f}")
    return {
        "layer": layer_idx,
        "tokens": T,
        "exact": exact,
        "diag": diag_f,
        "cross_term": exact - diag_f,
        "exact_over_diag": exact / diag_f if diag_f > 0 else np.nan,
        "cross_over_exact": (exact - diag_f) / exact if exact > 0 else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, type=pathlib.Path)
    ap.add_argument("--corpus", nargs="+", required=True)
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

    model, tok, cfg = _load_model(args.hf_dir, args.device)
    n_layers = cfg.num_hidden_layers
    n_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None)
    hidden = cfg.hidden_size
    top_k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_experts_active_per_tok", 8)
    layer_end = n_layers if args.layer_end == -1 else args.layer_end
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

    rows = []
    for L in range(args.layer_start, layer_end):
        if collector.X_chunks[L]:
            rows.append(compute_layer(L, model.model.layers[L].mlp, collector, args.device, log))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fieldnames = ["layer", "tokens", "exact", "diag", "cross_term", "exact_over_diag", "cross_over_exact"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
