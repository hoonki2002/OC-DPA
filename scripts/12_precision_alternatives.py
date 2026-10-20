"""Fixed-precision alternatives to the any-precision stack: area and TTFT (OLMoE).

OC-DPA reads 4 of 8 planes in prefill and all 8 in decode from one copy of the
weights. The alternatives a fixed-precision AIMC could use instead:

  8-bit            the baseline: 8 planes, always read in full (load-sorted grouping, so its TTFT is the
                   second row of the ablation, not the in-order 3D-AIMC of the first)
  4-bit            native 4-bit weights, read in both phases (half the planes)
  4-bit + 8-bit    both a 4-bit copy (prefill) and an 8-bit copy (decode)
  OC-DPA           one 8-plane copy plus 8-bit scales (0.5625 bit per weight), prefill k-bar = 4

Memory per weight is 8, 4, 12 and 8.5625 bits; the area of each design is that of
its 3D stack (arrays, TSVs and ADC lanes) from the same model as scripts/10, at the point of the design
space (tiers per lane, columns per ADC) that is best for that design, as in scripts/10; the rows that
only change the allocation (uniform k-bar = 4, DPA) are priced at the point of OC-DPA. The 4-bit and
8-bit rows are plain bit-slicing: their own group scales are not counted (about 0.1 bit
per weight at 8 bits), whereas OC-DPA's 8-bit alpha/beta are.

Quality (KL divergence) of the fixed-precision rows comes from GPTQ copies:
  scripts/20_encode_native_gptq.py   -> cache/planes_olmoe_gptq4, cache/planes_olmoe_gptq8
  scripts/21_kl_sweep.py --planes cache/planes_olmoe_gptq8 --native-planes cache/planes_olmoe_gptq4 --native-only --home cpu
(rows "all 8" = native 8-bit, "native 4" = 4 bits throughout, "native 4 + 8" = 4-bit prefill
copy and 8-bit decode copy).

  python scripts/12_precision_alternatives.py --out cache/alternatives.json
"""
import argparse, json

import numpy as np

from ocdpa.compare import GRID_MUX, GRID_SHARE, NPU_LOGIC_MM2, SPEC, TIERS, evaluate, placement, scale_storage_mm2
from ocdpa.paths import data_file
from ocdpa.hw.npu import MODELS, NPU, NPU_CFG
from ocdpa.hw.stack import Tier, hetero_area
from ocdpa.planes import GS, KMAX, SCALE_BITS
from ocdpa.system import plans

ap = argparse.ArgumentParser()
ap.add_argument("--long", default="_long")
ap.add_argument("--kbar", type=int, default=4, help="prefill plane budget")
ap.add_argument("--static-at-ocdpa", action="store_true", help="price the static 4-bit and 4+8-bit rows at the point of OC-DPA instead of their own best point")
ap.add_argument("--out", default="cache/alternatives.json")
args = ap.parse_args()
npu = NPU(**NPU_CFG)


def footprint(m, planes_1t1r, planes_xbar, scale_bits=None, share=16, mux=8):
    """mm2 of a 16-tier stack holding every weight `planes_1t1r` planes in 1T1R and `planes_xbar` in crossbars."""
    params = m.n_moe * m.experts * 3 * m.hidden * m.moe_inter
    spec = ([Tier(1, 128, share, planes_1t1r)] if planes_1t1r else []) + [Tier(4, 256, share, planes_xbar)]
    a = hetero_area(params, spec, tiers=TIERS, mux=mux, input_share=8)
    mem = a["mem_mm2"] + scale_storage_mm2(params, scale_bits)
    logic = a["logic_mm2"] + NPU_LOGIC_MM2
    return dict(mem_mm2=mem, logic_mm2=logic, footprint_mm2=max(mem, logic))


m = next(x for x in MODELS if x.name == "OLMoE-1B-7B")
tag, resid = SPEC[m.name]
cal = np.load(data_file(f"trace_{tag}.npz"))["idx"].astype(np.int64)
tr = np.load(data_file(f"trace_{tag}{args.long}.npz"))
idx, gate = tr["idx"].astype(np.int64), tr["gate"]
R = np.load(data_file(resid)); R = R[R.shape[0] - idx.shape[0]:]
L = idx.shape[0]
loads = [np.bincount(cal[l].ravel(), minlength=m.experts) for l in range(L)]
slots_of = {sh: [placement(loads[l], sh, True) for l in range(L)] for sh in GRID_SHARE}
k8, k4 = plans(idx, gate, R, 8, m.topk, False), plans(idx, gate, R, 4, m.topk, False)
kk = plans(idx, gate, R, args.kbar, m.topk, False)
kd4 = plans(idx, gate, R, args.kbar, m.topk, True)


def run(kp, kd, share, mux, scale_bits=None):
    r = evaluate(m, idx, kp, kd, slots_of[share], share, npu, scale_bits=scale_bits, mux=mux)
    return dict(ttft_ms=r["prefill"]["ttft_ms"], energy_mJ=r["prefill"]["energy_mJ"], decode_tok_s=r["decode"]["tok_s"])


def row(bits, p1, p2, kp, kd, share, mux, scale_bits=None):
    r = dict(bits_per_weight=bits, share=share, mux=mux, **footprint(m, p1, p2, scale_bits, share, mux),
             **run(kp, kd, share, mux, scale_bits))
    r["prefill_tok_s_mm2"] = 1024e3 / r["ttft_ms"] / r["footprint_mm2"]
    return r


def tuned(*a, **kw):
    """The row at the (tiers per lane, columns per ADC) point with the highest prefill throughput per area."""
    return max((row(*a, share=sh, mux=mx, **kw) for sh in GRID_SHARE for mx in GRID_MUX),
               key=lambda r: r["prefill_tok_s_mm2"])


bits_dpa = 8 + (KMAX + 1) * SCALE_BITS / GS
ocdpa = tuned(bits_dpa, 2, 6, kd4, k8, scale_bits=SCALE_BITS)
at = dict(share=ocdpa["share"], mux=ocdpa["mux"])
rows = {
    "8-bit (best baseline)":      tuned(8.0, 2, 6, k8, k8),
    f"+ uniform k={args.kbar}":   row(bits_dpa, 2, 6, kk, k8, scale_bits=SCALE_BITS, **at),
    "+ DPA (OC-DPA)":             ocdpa,
    "Static 4-bit":               row(4.0, 2, 2, k4, k4, **at) if args.static_at_ocdpa else tuned(4.0, 2, 2, k4, k4),
    "Static 4-bit + 8-bit":       row(12.0, 4, 8, k4, k8, **at) if args.static_at_ocdpa else tuned(12.0, 4, 8, k4, k8),
}
print(m.name)
for name, r in rows.items():
    print(f"  {name:24s} {r['bits_per_weight']:6.2f} b/w  s={r['share']:<2d} mux {r['mux']}:1  {r['footprint_mm2']:6.0f} mm2  "
          f"TTFT {r['ttft_ms']:7.1f} ms  {r['prefill_tok_s_mm2']:5.1f} tok/s/mm2  E {r['energy_mJ']:6.1f} mJ  decode {r['decode_tok_s']:5.1f} tok/s")
mixed = footprint(m, 2, 6, share=ocdpa["share"], mux=ocdpa["mux"])
crossbar = footprint(m, 0, 8, share=ocdpa["share"], mux=ocdpa["mux"])
print(f"  base planes in 1T1R (2 + 6 planes) vs all eight planes in crossbars at the point of OC-DPA: memory die {mixed['mem_mm2']:.0f} vs "
      f"{crossbar['mem_mm2']:.0f} mm2 ({100 * (mixed['mem_mm2'] / crossbar['mem_mm2'] - 1):+.0f}%), footprint {mixed['footprint_mm2']:.0f} vs {crossbar['footprint_mm2']:.0f} mm2")
json.dump({m.name: rows, "memory_die_check": dict(point=at, mixed=mixed, all_crossbar=crossbar)}, open(args.out, "w"), indent=1, default=float)
