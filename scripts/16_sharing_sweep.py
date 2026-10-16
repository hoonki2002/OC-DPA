"""How many tiers should share an ADC lane, with and without DPA?

For sharing s = 1, 2, 4, 8, 16 tiers per lane (load-sorted grouping, as in G&Y; s = 16 with the experts in
order is 3D-AIMC) this prices two designs on a prefill of --prompt tokens (1024 by default):

  all planes    every expert read in full in prefill and decode                 (the baselines)
  + DPA         prefill budget k-bar on the on-chip allocator, decode at k = 8  (OC-DPA)

and reports the memory die, the logic die (the footprint is the larger one), TTFT, prefill energy and the
prefill throughput per area. The sharing at which the two dies are about equal is where area is spent
well: sharing more leaves the logic die idle, sharing less makes it the bigger die.

  python scripts/16_sharing_sweep.py --out cache/sharing_sweep.json
"""
import argparse, json

import numpy as np

from ocdpa.compare import SPEC, evaluate, placement
from ocdpa.hw.npu import MODELS, NPU, NPU_CFG
from ocdpa.planes import SCALE_BITS
from ocdpa.system import plans

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="OLMoE-1B-7B,DeepSeek-MoE-16B")
ap.add_argument("--prompt", type=int, default=1024, help="prompt tokens of the prefill (scored on trace tokens P..2P)")
ap.add_argument("--kbar", type=int, default=4, help="prefill plane budget with DPA")
ap.add_argument("--long", default="_long", help="suffix of the measurement trace, cache/trace_<tag><suffix>.npz")
ap.add_argument("--out", default="cache/sharing_sweep.json")
args = ap.parse_args()

npu = NPU(**NPU_CFG)
out = {}
for m in MODELS:
    if m.name not in args.models.split(","):
        continue
    tag, resid = SPEC[m.name]
    cal = np.load(f"cache/trace_{tag}.npz")["idx"].astype(np.int64)
    tr = np.load(f"cache/trace_{tag}{args.long}.npz")
    idx, gate = tr["idx"].astype(np.int64), tr["gate"]
    R = np.load(f"cache/{resid}"); R = R[R.shape[0] - idx.shape[0]:]
    L = idx.shape[0]
    loads = [np.bincount(cal[l].ravel(), minlength=m.experts) for l in range(L)]
    k8 = plans(idx, gate, R, 8, m.topk, False)
    k_dpa = plans(idx, gate, R, args.kbar, m.topk, True)

    def run(share, grouped, k_pre, scale_bits=None):
        slots = [placement(loads[l], share, grouped) for l in range(L)]
        r = evaluate(m, idx, k_pre, k8, slots, share, npu, P=args.prompt, scale_bits=scale_bits)
        t = r["prefill"]["ttft_ms"]
        return dict(share=share, mem_mm2=r["mem_mm2"], logic_mm2=r["logic_mm2"], footprint_mm2=r["footprint_mm2"],
                    ttft_ms=t, energy_mJ=r["prefill"]["energy_mJ"], tok_s_mm2=args.prompt * 1e3 / t / r["footprint_mm2"])

    rows = {"3D-AIMC (in order)": run(16, False, k8)}
    for s in (1, 2, 4, 8, 16):
        rows[f"s={s} all planes"] = run(s, True, k8)
        rows[f"s={s} + DPA"] = run(s, True, k_dpa, SCALE_BITS)
    out[m.name] = rows
    print(m.name)
    print(f"  {'design':22s} {'mem':>6} {'logic':>6} {'mm2':>6} {'TTFT ms':>8} {'mJ':>6} {'tok/s/mm2':>10}")
    for name, r in rows.items():
        print(f"  {name:22s} {r['mem_mm2']:6.0f} {r['logic_mm2']:6.0f} {r['footprint_mm2']:6.0f} "
              f"{r['ttft_ms']:8.1f} {r['energy_mJ']:6.1f} {r['tok_s_mm2']:10.1f}")
    for kind in ("all planes", "+ DPA"):
        best = max((r for n, r in rows.items() if n.endswith(kind)), key=lambda r: r["tok_s_mm2"])
        print(f"  best sharing {kind:10s}: s = {best['share']}  {best['tok_s_mm2']:.1f} tok/s/mm2")
json.dump(out, open(args.out, "w"), indent=1, default=float)
print("wrote", args.out)
