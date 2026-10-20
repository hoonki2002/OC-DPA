"""Area, energy and latency of the on-chip precision allocator (Sec. III-C).

Runs NeuroSim's `allocator` driver (neurosim_patch/allocator.cpp) for the
marginal-gain rule of OC-DPA and for the dynamic program it replaces, at the node of the logic die (22 nm by default, like the memory tiers),
for the routing width N and the table size (layers x experts) of each model.
The allocation must finish inside the two base-plane reads it hides behind.

  python scripts/11_allocator_ppa.py [--technode 22]
"""
import argparse, json

from ocdpa.hw.allocator_ppa import Allocator, run
from ocdpa.hw.stack import Stack

ap = argparse.ArgumentParser()
ap.add_argument("--technode", type=int, default=22)
ap.add_argument("--out", default="cache/allocator_ppa.json")
args = ap.parse_args()

MODELS = [("OLMoE-1B-7B", 16, 64, 8), ("DeepSeek-MoE-16B", 27, 64, 6), ("Qwen3-30B-A3B", 48, 128, 8)]
t_read = Stack(tiers=16, mux=8, share=16, write_share=16).t_read_ns()
shadow = 2 * t_read
print(f"lane read {t_read:.0f} ns; the allocator must finish within the two base-plane reads, "
      f"{shadow:.0f} ns (and they take 8x that for all input bits)\n")
out = {}
variants = [("marginal gain", dict(mode=0)), ("DP, 1 lane", dict(mode=1, dp_par=1)),
            ("DP, 16 lanes", dict(mode=1, dp_par=16)), ("DP, 32 lanes", dict(mode=1, dp_par=32))]
for name, L, E, n in MODELS:
    print(f"{name}: N = {n}, table {L * E} rows")
    print(f"  {'allocator':16s} {'cycles':>7} {'lat ns':>8} {'energy pJ':>10} {'area um2':>9}")
    for label, kw in variants:
        r = run(Allocator(n=n, lut_rows=L * E, banks=n, technode=args.technode, **kw))
        out.setdefault(name, {})[label] = r
        print(f"  {label:16s} {r['cycles']:7.0f} {r['latency_ns']:8.1f} {r['energy_pJ']:10.2f} {r['area_um2']:9.0f}")
    print()
import pathlib
pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print("wrote", args.out)
