"""Merge per-shard expert_residuals_tok*.npz (data-parallel) or
expert_residuals_L*_*.npz (layer-parallel) into one expert_residuals.npz.

Data-parallel merge: each shard covers all layers but a different token
slice.  R[L,e,K] is a mean over N_effective[L,e] samples.  Combine with
weighted average:

    R_merged = sum_i(R_i * N_eff_i) / sum_i(N_eff_i)

Layer-parallel merge: each shard covers a non-overlapping layer range with
NaN for other layers.  Combine by taking the non-NaN slice per layer.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from safetensors.torch import save_file as save_safetensors
import torch


def enforce_monotonic(R: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Force R[..., K+1] < R[..., K] for finite adjacent entries.

    If a later precision has higher/equal residual, clamp it to the previous
    residual minus eps. Residuals are squared errors, so the clamp is floored
    at zero.
    """
    out = R.astype(np.float32, copy=True)
    n_fix = 0
    for k in range(1, out.shape[2]):
        prev = out[:, :, k - 1]
        cur = out[:, :, k]
        bad = np.isfinite(prev) & np.isfinite(cur) & (cur >= prev)
        if np.any(bad):
            cur[bad] = np.maximum(prev[bad] - eps, 0.0)
            out[:, :, k] = cur
            n_fix += int(np.sum(bad))
    return out, n_fix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--enforce-monotonic", action="store_true",
                    help="clamp R[..., K+1] >= R[..., K] to R[..., K] - eps")
    ap.add_argument("--monotonic-eps", type=float, default=1e-6,
                    help="epsilon used with --enforce-monotonic")
    args = ap.parse_args()

    d0 = np.load(args.shards[0], allow_pickle=True)
    K_min = int(d0["K_min"])
    K_max = int(d0["K_max"])
    n_layers, n_experts, n_K = d0["R"].shape

    # Detect merge mode from first shard: if it has N_effective and no full-NaN
    # layers, treat as data-parallel; otherwise layer-parallel.
    has_neff = "N_effective" in d0
    first_has_all_layers = not np.any(np.all(np.isnan(d0["R"]), axis=(1, 2)))
    data_parallel = has_neff and first_has_all_layers

    if data_parallel:
        # Weighted average: R_merged = sum(R_i * N_eff_i) / sum(N_eff_i)
        R_num   = np.zeros((n_layers, n_experts, n_K), dtype=np.float64)
        N_total = np.zeros((n_layers, n_experts), dtype=np.float64)
        A_num   = np.zeros((n_layers, n_experts), dtype=np.float64)
        N_raw   = np.zeros((n_layers, n_experts), dtype=np.int64)

        for shard in args.shards:
            d = np.load(shard, allow_pickle=True)
            R_i   = d["R"].astype(np.float64)        # (L, E, K)
            N_eff = d["N_effective"].astype(np.float64)  # (L, E)
            A_i   = d["A"].astype(np.float64)        # (L, E)
            N_i   = d["N"].astype(np.int64)
            valid = ~np.isnan(R_i)                   # (L, E, K)
            # Broadcast N_eff across K dimension for weighting.
            w = N_eff[:, :, np.newaxis] * valid
            R_num   += R_i * w
            N_total += N_eff * np.any(valid, axis=2)
            A_num   += A_i * N_eff
            N_raw   += N_i

        # Where total weight is 0 (expert never sampled), leave NaN.
        w_total = N_total[:, :, np.newaxis].clip(min=1e-9)
        R_merged = np.where(N_total[:, :, np.newaxis] > 0,
                            R_num / w_total, np.nan).astype(np.float32)
        A_merged = (A_num / np.maximum(N_total, 1e-9)).astype(np.float32)
        N_merged = N_raw
        print(f"data-parallel merge: {len(args.shards)} shards, "
              f"weighted by N_effective")
    else:
        # Layer-parallel: take non-NaN slice per layer.
        R_merged = np.full((n_layers, n_experts, n_K), np.nan, dtype=np.float32)
        A_merged = np.zeros((n_layers, n_experts), dtype=np.float32)
        N_merged = np.zeros((n_layers, n_experts), dtype=np.int64)
        for shard in args.shards:
            d = np.load(shard, allow_pickle=True)
            has_data = ~np.all(np.isnan(d["R"]), axis=(1, 2))
            R_merged[has_data] = d["R"][has_data].astype(np.float32)
            A_merged[has_data] = d["A"][has_data].astype(np.float32)
            N_merged[has_data] = d["N"][has_data].astype(np.int64)
        print(f"layer-parallel merge: {len(args.shards)} shards")

    n_nan = int(np.isnan(R_merged).sum())
    print(f"R({n_layers}, {n_experts}, K∈[{K_min},{K_max}])  NaN={n_nan}")
    if args.enforce_monotonic:
        R_merged, n_fixed = enforce_monotonic(R_merged, args.monotonic_eps)
        if n_fixed:
            print(f"monotonic clamp: fixed {n_fixed} adjacent violations "
                  f"with eps={args.monotonic_eps:g}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "expert_residuals.npz",
                        R=R_merged, A=A_merged, N=N_merged,
                        K_min=K_min, K_max=K_max)
    save_safetensors(
        {"R": torch.from_numpy(R_merged),
         "activity_mean": torch.from_numpy(A_merged),
         "routed_token_count": torch.from_numpy(N_merged)},
        str(args.out_dir / "expert_residuals.safetensors"),
        metadata={"K_min": str(K_min), "K_max": str(K_max),
                  "n_layers": str(n_layers), "n_experts": str(n_experts)})
    print(f"wrote {args.out_dir / 'expert_residuals.npz'}")


if __name__ == "__main__":
    main()
