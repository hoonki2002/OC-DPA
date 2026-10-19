"""The residual LUT of the on-chip allocator (Sec. III-C, Fig. 3), exported as the numbers the circuit stores.

The allocator compares, for every routed expert, the gate term 2 (z_i - z_1) plus the stored ln of the marginal gain
dR[layer, expert, k] = R[k] - R[k + 1] of its next plane. The LUT holds that logarithm, normalised per layer (a per-layer
constant shifts every expert of a dispatch alike and cannot change the order), as 9-bit signed fixed-point codes
(3 fractional + 6 integer bits), one row per (layer, expert) and one entry per plane step k = 2 ... 7 -> 8.

  python scripts/04_export_lut.py                              # OLMoE: data/residual_olmoe.npy -> results/lut_olmoe.hex
  python scripts/04_export_lut.py --residual PATH --out FILE   # another residual table

Writes FILE (one line per row: the six 9-bit codes as hex, step 2->3 first) and FILE.json (rows, entries, bits, bytes,
the range of the codes and how many saturate); the size of the table in the paper (6.75 KB for OLMoE) is `bytes`.
"""
import argparse, json
from pathlib import Path

import numpy as np, torch

from ocdpa.allocators import DPAHardware
from ocdpa.paths import data_file

ap = argparse.ArgumentParser()
ap.add_argument("--residual", default=None, help="residual table R[layer, expert, k] (default: data/residual_olmoe.npy)")
ap.add_argument("--out", default="results/lut_olmoe.hex")
ap.add_argument("--topk", type=int, default=8)
args = ap.parse_args()

R = np.load(args.residual if args.residual else data_file("residual_olmoe.npy"))
alloc = DPAHardware(4, R, args.topk, device="cpu")
lut = alloc.lut
L, E, S = lut.shape
bits, frac = alloc.lut_bits, 3
codes = torch.round(lut * 2 ** frac).long()
lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
assert int(codes.min()) >= lo and int(codes.max()) <= hi
words = (codes & (2 ** bits - 1)).reshape(L * E, S)
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
width = (bits + 3) // 4
out.write_text("\n".join(" ".join(f"{int(w):0{width}x}" for w in row) for row in words) + "\n")
summary = dict(layers=L, experts=E, rows=L * E, entries_per_row=S, bits=bits, bytes=L * E * S * bits // 8,
               code_min=int(codes.min()), code_max=int(codes.max()),
               saturated_low=int((codes == lo).sum()), saturated_high=int((codes == hi).sum()))
Path(str(out) + ".json").write_text(json.dumps(summary, indent=1))
print(f"{L} layers x {E} experts = {L * E} rows, {S} entries of {bits} bits: {summary['bytes']:,} B = {summary['bytes'] / 1024:.2f} KB")
print(f"codes {summary['code_min']} .. {summary['code_max']} (range {lo} .. {hi}); saturated: {summary['saturated_low']} low, {summary['saturated_high']} high")
print(f"wrote {out}")
