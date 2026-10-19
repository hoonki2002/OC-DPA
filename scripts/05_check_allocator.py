"""Does the on-chip allocator return the plane counts of DPA's dynamic program? (Sec. III-C)

Three allocators spend the same budget B = k-bar * N over the N routed experts of one dispatch:

  DP        DPA's knapsack on g^2 * R[l, e, k]                  (ocdpa.allocators.DPA)
  greedy    one plane at a time to the largest marginal gain     (DPAGreedy; exact when R is convex in k)
  hardware  the greedy rule on the allocator's fixed-point log datapath, with the router-logit
            term 2 (z_i - z_1) and the stored log gains         (DPAHardware)

The check counts the dispatches on which the plane counts agree with the DP, on random dispatches
(logits drawn per expert, top-N routed, gates from the softmax) and on the recorded routing. For the
fixed-point datapath, whose log gains are rounded, ties may break differently, so the objective
sum_e g_e^2 R[l, e, k_e] it reaches relative to the DP's is reported as well.

  python scripts/05_check_allocator.py --tag olmoe --topk 8        # reads data/trace_olmoe_long.npz, data/residual_olmoe.npy
"""
import argparse

import numpy as np, torch

from ocdpa.allocators import DPA, DPAGreedy, DPAHardware
from ocdpa.paths import data_file

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="olmoe", help="routing trace cache/trace_<tag><long>.npz")
ap.add_argument("--long", default="_long")
ap.add_argument("--residual", default=None, help="residual table (default: data/residual_<tag>.npy)")
ap.add_argument("--topk", type=int, default=8, help="routed experts per token (N)")
ap.add_argument("--kbar", type=int, default=4)
ap.add_argument("--dispatches", type=int, default=4096, help="random dispatches, spread over the layers")
ap.add_argument("--logit-std", type=float, default=2.0, help="spread of the random router logits")
ap.add_argument("--tokens", type=int, default=2048, help="recorded dispatches per layer to check")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

tr = np.load(data_file(f"trace_{args.tag}{args.long}.npz"))
idx, gate = torch.as_tensor(tr["idx"]).long(), torch.as_tensor(tr["gate"]).float()
L = idx.shape[0]
R = np.load(args.residual if args.residual else data_file(f"residual_{args.tag}.npy"))
R = R[R.shape[0] - L:]                                            # drop leading dense layers
E = R.shape[1]
dev = args.device
allocs = {"greedy": DPAGreedy(args.kbar, R, args.topk, dev), "hardware": DPAHardware(args.kbar, R, args.topk, dev)}
dp = DPA(args.kbar, R, args.topk, dev)


def objective(layer, ix, w, plan):
    cost = (w ** 2).unsqueeze(-1) * dp.res[layer][ix]              # (T, N, K+1)
    return cost.gather(-1, plan.unsqueeze(-1)).squeeze(-1).sum().item()


def agreement(layer, ix, w):
    """Per allocator: (dispatches with the DP's plane counts, objective reached); the DP's objective last."""
    ix, w = ix.to(dev), w.to(dev)
    dp.prepare(layer, ix, w)
    ref = dp.plan[layer]
    out = {"DP": (len(ix), objective(layer, ix, w, ref))}
    for name, a in allocs.items():
        a.prepare(layer, ix, w)
        out[name] = ((a.plan[layer] == ref).all(1).sum().item(), objective(layer, ix, w, a.plan[layer]))
    return out, len(ix)


def report(title, totals, n):
    base = totals["DP"][1]
    print(f"{title}: {n} dispatches")
    for k, (agree, obj) in totals.items():
        if k != "DP":
            print(f"  {k:9s} same plane counts as the DP on {100 * agree / n:6.2f}%   objective {100 * (obj / base - 1):+.3f}% vs the DP")


g = torch.Generator().manual_seed(0)
per_layer = max(1, args.dispatches // L)
def new_totals():
    return {k: (0, 0.0) for k in ["DP", *allocs]}


def add(tot, a):
    for k in tot:
        tot[k] = (tot[k][0] + a[k][0], tot[k][1] + a[k][1])


tot, n = new_totals(), 0
for layer in range(L):
    z = torch.randn(per_layer, E, generator=g) * args.logit_std
    w, ix = torch.softmax(z, -1).topk(args.topk, dim=-1)             # gates over all experts, N routed
    a, c = agreement(layer, ix, w)
    add(tot, a)
    n += c
report("random", tot, n)

tot, n = new_totals(), 0
for layer in range(L):
    a, c = agreement(layer, idx[layer, :args.tokens], gate[layer, :args.tokens])
    add(tot, a)
    n += c
report("recorded routing", tot, n)
