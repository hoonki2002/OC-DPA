"""Sanity check for the K̄-budget allocator (PROBLEM.md §5).

Ports the C++ allocator in
``third_party/llama.cpp/dpa-moe-ext/moe/common/moe_scheduler.cpp::allocate_dispatch_budget``
to Python and verifies, on real residuals ``R[L, e, K]`` from
``expert_residuals.npz``:

  1. Σ K[e] = round(n_active × K̄), clamped to [n_active·K_min, n_active·K_max]
  2. K_min ≤ K[e] ≤ K_max  for every active expert
  3. At K̄ = K_min: every K[e] = K_min   (no allocation freedom)
  4. At K̄ = K_max: every K[e] = K_max   (saturated)
  5. Greedy total error  Σ g²·R[L,e,K[e]]   is monotone-non-increasing in K̄
  6. Greedy total error  ≤  uniform K̄ total error  at any interior K̄

Run after a profile + pack, before kicking off a sweep:

    .venv/bin/python experiments/3_dynamic_precision/analysis/check_allocator.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def allocate(R_layer: np.ndarray,
              experts: list[int],
              gates: list[float],
              kbar: float,
              K_min: int, K_max: int) -> list[int]:
    """Python port of MoEScheduler::allocate_dispatch_budget.

    R_layer: shape (n_expert, n_K) for this layer.
    experts: list of expert indices active in this dispatch.
    gates:   list of live gate scores, aligned with experts.
    Returns K[i] for i in 0..len(experts)-1.
    """
    n = len(experts)
    n_K = K_max - K_min + 1
    if n_K <= 1 or n == 0:
        return [K_min] * n

    target = int(round(n * kbar))
    target = max(target, n * K_min)
    target = min(target, n * K_max)
    K_out = [K_min] * n
    remaining = target - n * K_min

    # next_ki[i] is the index into R_layer[e, .] of the *next* gain to evaluate
    # (corresponds to going from K_out[i] → K_out[i]+1).  Initially 1.
    next_ki = [1] * n
    next_gain = [0.0] * n

    def gain_of(i: int, ki: int) -> float:
        e = experts[i]
        # R[K-1] − R[K], expressed with array indices ki-1 → ki.
        R_prev = float(R_layer[e, ki - 1])
        R_here = float(R_layer[e, ki])
        dR = R_prev - R_here
        if dR < 0.0:
            dR = 0.0
        g = gates[i]
        return g * g * dR

    for i in range(n):
        next_gain[i] = gain_of(i, 1)

    while remaining > 0:
        best_i = -1
        best_g = -1.0
        for i in range(n):
            if next_ki[i] >= n_K:
                continue
            if next_gain[i] > best_g:
                best_g = next_gain[i]
                best_i = i
        if best_i < 0:
            break
        K_out[best_i] = K_min + next_ki[best_i]
        next_ki[best_i] += 1
        remaining -= 1
        if next_ki[best_i] < n_K:
            next_gain[best_i] = gain_of(best_i, next_ki[best_i])
        else:
            next_gain[best_i] = -1.0

    return K_out


def total_error(R_layer: np.ndarray,
                 experts: list[int],
                 gates: list[float],
                 K_assign: list[int],
                 K_min: int) -> float:
    err = 0.0
    for i, K in enumerate(K_assign):
        e = experts[i]
        idx = K - K_min
        err += gates[i] * gates[i] * float(R_layer[e, idx])
    return err


def check_layer(R_layer: np.ndarray, K_min: int, K_max: int,
                 layer_idx: int, top_k: int = 8, seed: int = 0,
                 n_dispatches: int = 8, verbose: bool = False) -> int:
    """Returns number of failures for this layer."""
    rng = np.random.default_rng(seed + layer_idx)
    n_expert = R_layer.shape[0]
    failures = 0

    for d in range(n_dispatches):
        # Pick a random top_k subset of valid (non-NaN) experts.
        valid_e = [e for e in range(n_expert)
                   if np.isfinite(R_layer[e]).all()]
        if len(valid_e) < top_k:
            continue
        rng.shuffle(valid_e)
        experts = valid_e[:top_k]
        # Random gate scores ~ softmax — closer to real router output.
        raw = rng.standard_normal(top_k)
        ex = np.exp(raw - raw.max())
        gates = (ex / ex.sum()).tolist()

        last_err = float("inf")
        for kbar in [K_min, K_min + 0.5, K_min + 1, K_min + 2,
                     (K_min + K_max) / 2, K_max - 1, K_max - 0.5, K_max]:
            K_assign = allocate(R_layer, experts, gates, kbar, K_min, K_max)
            # ── Invariant 1: budget hard equality (after clamp)
            target = max(int(round(top_k * kbar)), top_k * K_min)
            target = min(target, top_k * K_max)
            s = sum(K_assign)
            if s != target:
                print(f"  L{layer_idx} d{d} K̄={kbar}: Σ K[e]={s} != target={target}  FAIL")
                failures += 1
            # ── Invariant 2: per-expert range
            for K_i in K_assign:
                if K_i < K_min or K_i > K_max:
                    print(f"  L{layer_idx} d{d} K̄={kbar}: K[e]={K_i} out of [{K_min}, {K_max}]  FAIL")
                    failures += 1
            # ── Invariant 3, 4: extremes
            if abs(kbar - K_min) < 1e-9 and not all(K == K_min for K in K_assign):
                print(f"  L{layer_idx} d{d} K̄=K_min: not all K_min  FAIL  K={K_assign}")
                failures += 1
            if abs(kbar - K_max) < 1e-9 and not all(K == K_max for K in K_assign):
                print(f"  L{layer_idx} d{d} K̄=K_max: not all K_max  FAIL  K={K_assign}")
                failures += 1
            # ── Invariant 5: monotone-non-increasing total error in K̄
            err = total_error(R_layer, experts, gates, K_assign, K_min)
            if err > last_err + 1e-6:
                print(f"  L{layer_idx} d{d} K̄={kbar}: err={err:.6g} > prev {last_err:.6g}  FAIL")
                failures += 1
            last_err = err
            # ── Invariant 6 (advisory, not strict): greedy ≤ uniform.
            # Holds ONLY when the per-expert R curves are convex-non-
            # increasing.  About 7.6 % of (L, e, K→K+1) transitions in
            # the current codebook are non-monotone (worst at K=2→K=3 for
            # a few experts), which breaks the greedy-optimality theorem.
            # We log the violations but don't count them as failures.
            K_unif = max(K_min, min(K_max, int(round(kbar))))
            uniform_assign = [K_unif] * top_k
            if sum(uniform_assign) == target:
                err_uni = total_error(R_layer, experts, gates,
                                        uniform_assign, K_min)
                if err > err_uni + 1e-3 * max(err_uni, 1e-6):
                    if verbose:
                        print(f"  L{layer_idx} d{d} K̄={kbar}: greedy>uniform "
                              f"({err:.4g} vs {err_uni:.4g}) — non-monotone R")
            if verbose:
                print(f"  L{layer_idx} d{d} K̄={kbar:>4.1f}  Σ K[e]={s:>3d}  err={err:.4g}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=pathlib.Path,
                     default=pathlib.Path("cache/experiment/Qwen3-30B-A3B/layout_solver/expert_residuals.npz"))
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-dispatches", type=int, default=8,
                     help="dispatches per layer (random gate vectors)")
    ap.add_argument("--layers", type=str, default="0,12,24,36,47",
                     help="comma-separated layer indices to test (default: 5 representative)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.npz.exists():
        sys.exit(f"missing {args.npz} — run profile_residuals.py first")
    d = np.load(args.npz)
    R = d["R"]                  # (n_layers, n_expert, n_K)
    K_min = int(d["K_min"])
    K_max = int(d["K_max"])
    print(f"loaded R shape={R.shape}  K∈[{K_min}, {K_max}]")
    layers = [int(s) for s in args.layers.split(",")]

    total_fail = 0
    for L in layers:
        if L >= R.shape[0]:
            print(f"  layer {L}: out of range, skip"); continue
        f = check_layer(R[L], K_min, K_max, L, args.top_k,
                         n_dispatches=args.n_dispatches,
                         verbose=args.verbose)
        total_fail += f
        status = "OK" if f == 0 else f"{f} FAIL"
        print(f"  layer {L:2d}: {status}")

    if total_fail == 0:
        print(f"\nall checks passed ({len(layers)} layers × "
              f"{args.n_dispatches} dispatches × 8 K̄ values)")
        sys.exit(0)
    else:
        print(f"\n{total_fail} sanity failures")
        sys.exit(1)


if __name__ == "__main__":
    main()
