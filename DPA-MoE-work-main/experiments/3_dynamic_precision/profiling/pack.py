"""Write the runtime-facing residuals binary from expert_residuals.npz.

Binary format (consumed by ``DPA_MOE_KBAR_PROFILE_FILE`` in
``dpa-moe-ext/moe/common/moe_scheduler.cpp``):

    uint32_t  n_layers
    uint32_t  n_experts
    uint32_t  K_min
    uint32_t  K_max
    float32[] R   shape (n_layers, n_experts, K_max-K_min+1), C-order

NaN entries (cold experts that were not sampled enough during
calibration) are filled with the per-layer per-K mean across the
*sampled* experts.  This gives the runtime allocator a "typical
expert" fallback instead of zero (which would silently produce
ΔR=0 for every K-step on that expert and starve the greedy walk).
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np


def enforce_monotonic(R: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Force finite residual rows to be strictly non-increasing with K.

    For any adjacent violation, set R[..., K+1] = max(R[..., K] - eps, 0).
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
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=pathlib.Path,
                     help="expert_residuals.npz from profile_residuals.py")
    ap.add_argument("--out", required=True, type=pathlib.Path,
                     help="output binary path")
    ap.add_argument("--enforce-monotonic", action="store_true",
                    help="clamp R[..., K+1] >= R[..., K] to R[..., K] - eps")
    ap.add_argument("--monotonic-eps", type=float, default=1e-6,
                    help="epsilon used with --enforce-monotonic")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    R = d["R"].astype(np.float32)
    K_min = int(d["K_min"])
    K_max = int(d["K_max"])
    n_layers, n_experts, n_K = R.shape
    if n_K != (K_max - K_min + 1):
        sys.exit(f"npz shape mismatch: K dim {n_K} != K_max−K_min+1 = {K_max-K_min+1}")
    # Replace NaN with the per-layer per-K mean across sampled experts.
    # If a whole layer is NaN (no calibration data at all), fall back to 0
    # so the runtime still gets a well-defined ΔR=0 across K (allocator
    # treats it as "no preference", spreads budget evenly).
    n_nan_before = int(np.isnan(R).sum())
    R_clean = R.copy()
    for L in range(R_clean.shape[0]):
        for k in range(R_clean.shape[2]):
            col = R_clean[L, :, k]
            mask = np.isnan(col)
            if mask.any() and not mask.all():
                col[mask] = np.nanmean(col)
                R_clean[L, :, k] = col
    # Any remaining NaN (whole layer dead) → 0
    R_clean = np.nan_to_num(R_clean, nan=0.0)
    if n_nan_before:
        print(f"  filled {n_nan_before} NaN entries with per-layer per-K mean")
    if args.enforce_monotonic:
        R_clean, n_fixed = enforce_monotonic(R_clean, args.monotonic_eps)
        if n_fixed:
            print(f"  monotonic clamp: fixed {n_fixed} adjacent violations "
                  f"with eps={args.monotonic_eps:g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<IIII", n_layers, n_experts, K_min, K_max))
        fh.write(R_clean.tobytes(order="C"))

    print(f"wrote {args.out}  (L={n_layers} E={n_experts} K∈[{K_min},{K_max}]; "
           f"{16 + R_clean.nbytes} bytes)")


if __name__ == "__main__":
    main()
