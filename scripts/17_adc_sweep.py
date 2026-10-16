"""How many ADCs, and with how much DPA on top? Both knobs that set the lane count are swept.

The lane count of the routed experts is set by two numbers: the tiers that share a lane (s) and the columns
that share an ADC inside a subarray (mux). For each pair this prices a prefill with all planes read (the
baselines) and with DPA at k-bar planes (OC-DPA), and reports the memory die, the logic die (the footprint is
the larger one), TTFT, and the prefill throughput per area. For models with always-on shared experts the
column mux of their dedicated lanes is swept as well, since it sets the time those experts take.

For each design the best point over the swept lane counts is printed, so the comparison is between the best
baseline and the best OC-DPA, not between two arbitrary configurations.

  python scripts/17_adc_sweep.py --out cache/adc_sweep.json
"""
import argparse, json

import numpy as np

from ocdpa.compare import SPEC, evaluate, placement
from ocdpa.hw.npu import MODELS, NPU, NPU_CFG
from ocdpa.planes import SCALE_BITS
from ocdpa.system import plans

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="OLMoE-1B-7B,DeepSeek-MoE-16B")
ap.add_argument("--share", default="1,2,4,8,16", help="tiers per lane")
ap.add_argument("--mux", default="8,4,2", help="columns per ADC lane of the routed experts")
ap.add_argument("--shared-mux", default="4,2,1", help="columns per ADC lane of the shared experts (if the model has any)")
ap.add_argument("--prompt", type=int, default=1024)
ap.add_argument("--kbar", type=int, default=4, help="prefill plane budget with DPA")
ap.add_argument("--long", default="_long")
ap.add_argument("--out", default="cache/adc_sweep.json")
args = ap.parse_args()
ints = lambda s: [int(v) for v in s.split(",")]

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
    shared = ints(args.shared_mux) if m.shared_inter else [None]
    rows = []
    for s in ints(args.share):
        slots = [placement(loads[l], s, True) for l in range(L)]
        for mux in ints(args.mux):
            for sm in shared:
                for design, kp, sb in (("all planes", k8, None), ("+ DPA", k_dpa, SCALE_BITS)):
                    r = evaluate(m, idx, kp, k8, slots, s, npu, P=args.prompt, scale_bits=sb, mux=mux, shared_mux=sm)
                    t = r["prefill"]["ttft_ms"]
                    rows.append(dict(design=design, share=s, mux=mux, shared_mux=sm, mem_mm2=r["mem_mm2"],
                                     logic_mm2=r["logic_mm2"], footprint_mm2=r["footprint_mm2"], ttft_ms=t,
                                     energy_mJ=r["prefill"]["energy_mJ"],
                                     tok_s_mm2=args.prompt * 1e3 / t / r["footprint_mm2"]))
        print(f"  {m.name}: sharing {s} done", flush=True)
    out[m.name] = rows
    print(m.name)
    for design in ("all planes", "+ DPA"):
        sel = sorted((r for r in rows if r["design"] == design), key=lambda r: -r["tok_s_mm2"])
        print(f"  best {design} (throughput per area):")
        for r in sel[:5]:
            print(f"    s={r['share']:2d} mux={r['mux']} shared-mux={r['shared_mux']}  mem {r['mem_mm2']:5.0f} logic {r['logic_mm2']:5.0f} mm2"
                  f"  TTFT {r['ttft_ms']:7.1f} ms  {r['tok_s_mm2']:5.1f} tok/s/mm2")
    best = {d: max((r for r in rows if r["design"] == d), key=lambda r: r["tok_s_mm2"]) for d in ("all planes", "+ DPA")}
    print(f"  OC-DPA over the best baseline: {best['+ DPA']['tok_s_mm2'] / best['all planes']['tok_s_mm2']:.2f}x throughput per area, "
          f"TTFT {best['all planes']['ttft_ms']:.0f} -> {best['+ DPA']['ttft_ms']:.0f} ms")
json.dump(out, open(args.out, "w"), indent=1, default=float)
print("wrote", args.out)
