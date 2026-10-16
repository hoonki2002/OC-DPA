"""Hardware results: the design comparison, the TTFT ablation and the sharing sweep.

For OLMoE-1B-7B and DeepSeek-MoE-16B, on a 1024-token prompt and batch-1 decoding:

  designs   3D-AIMC (16 tiers share a lane, experts in order, all planes)
            G&Y 2-way / 4-way (load-sorted grouping, all planes)
            + load-sorted grouping on 16 tiers (all planes)
            uniform k-bar = 4 on 16 tiers (load-sorted)
            OC-DPA: 16 tiers, load-sorted, DPA k-bar = 4 in prefill, k = 8 in decode
  budget    TTFT and prefill energy of uniform and DPA allocation for k-bar = 2..8 (16 tiers,
            load-sorted, decode at k = 8)
  sharing   TTFT and decode latency per token for s = 1, 2, 4, 8, 16 tiers per
            ADC lane, with experts in order and with load-sorted grouping
  decode    latency per token split into NPU and experts, experts on UFS flash
            (NPU+flash) versus experts on AIMC

Needs the routing traces and residual tables of scripts/02 and 03, and the
NeuroSim binaries of neurosim_patch/ (NEUROSIM_DIR). Writes one JSON file.

  python scripts/10_hardware.py --out cache/hardware.json
"""
import argparse, json
from dataclasses import replace

import numpy as np

from ocdpa.compare import SPEC, SHARED_ON_AIMC, evaluate, placement
from ocdpa.hw.npu import MODELS, NPU, NPU_CFG, dense_step
from ocdpa.planes import SCALE_BITS
from ocdpa.system import npu_experts, plans

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="OLMoE-1B-7B,DeepSeek-MoE-16B")
ap.add_argument("--kbar", type=int, default=4, help="prefill plane budget of OC-DPA")
ap.add_argument("--long", default="_long", help="suffix of the measurement trace, cache/trace_<tag><suffix>.npz")
ap.add_argument("--out", default="cache/hardware.json")
args = ap.parse_args()

npu = NPU(**NPU_CFG)
out = {}
for m in MODELS:
    if m.name not in args.models.split(","):
        continue
    tag, resid = SPEC[m.name]
    cal = np.load(f"cache/trace_{tag}.npz")["idx"].astype(np.int64)          # 2048-token calibration trace
    tr = np.load(f"cache/trace_{tag}{args.long}.npz")                             # 8192-token measurement trace
    idx, gate = tr["idx"].astype(np.int64), tr["gate"]
    R = np.load(f"cache/{resid}")
    R = R[R.shape[0] - idx.shape[0]:]                                       # drop leading dense layers
    L = idx.shape[0]
    loads = [np.bincount(cal[l].ravel(), minlength=m.experts) for l in range(L)]
    k8 = plans(idx, gate, R, 8, m.topk, False)
    k_uni = plans(idx, gate, R, args.kbar, m.topk, False)
    k_dpa = plans(idx, gate, R, args.kbar, m.topk, True)

    def run(share, grouped, k_pre, k_dec, scale_bits=None):
        slots = [placement(loads[l], share, grouped) for l in range(L)]
        return evaluate(m, idx, k_pre, k_dec, slots, share, npu, scale_bits=scale_bits)

    designs = {
        "3D-AIMC":                       run(16, False, k8, k8),
        "G&Y 2-way":                     run(2, True, k8, k8),
        "G&Y 4-way":                     run(4, True, k8, k8),
        "16 tiers + load-sorted grouping": run(16, True, k8, k8),
        f"16 tiers + uniform k={args.kbar}":  run(16, True, k_uni, k8, SCALE_BITS),   # any-precision encoding: scales stored
        "OC-DPA":                        run(16, True, k_dpa, k8, SCALE_BITS),
    }
    sharing = {}
    for s in (1, 2, 4, 8, 16):
        for grouped in (False, True):
            r = run(s, grouped, k8, k8)
            sharing[f"s{s}_{'grouped' if grouped else 'in_order'}"] = dict(
                ttft_ms=r["prefill"]["ttft_ms"], decode_ms=r["decode"]["ms"])

    budget = {}
    for kb in range(2, 9):
        for name, dpa in (("uniform", False), ("DPA", True)):
            r = run(16, True, plans(idx, gate, R, kb, m.topk, dpa), k8, SCALE_BITS)
            budget[f"{name}_{kb}"] = dict(ttft_ms=r["prefill"]["ttft_ms"], energy_mJ=r["prefill"]["energy_mJ"])

    # decode, experts on flash (all planes, no DRAM expert cache, oracle prefetch) vs on AIMC
    dense_flash = dense_step(m, NPU(), 1, 1024)
    flash_mJ, flash_ms, _ = npu_experts(m, idx, k8, 0, 512)
    on_aimc = SHARED_ON_AIMC and m.shared_inter > 0
    dense_aimc = dense_step(replace(m, shared_inter=0) if on_aimc else m, NPU(), 1, 1024)
    dec_aimc = designs["3D-AIMC"]["decode"]["ms"]
    out[m.name] = dict(
        designs=designs, sharing=sharing, budget=budget,
        decode=dict(flash=dict(npu_ms=dense_flash["latency_ms"], experts_ms=flash_ms,
                               mJ=dense_flash["dram_mJ"] + flash_mJ),
                    aimc=dict(npu_ms=dense_aimc["latency_ms"], experts_ms=dec_aimc - dense_aimc["latency_ms"],
                              mJ=designs["3D-AIMC"]["decode"]["energy_mJ"])))
    print(f"{m.name}", flush=True)
    for name, r in designs.items():
        print(f"  {name:34s} {r['footprint_mm2']:7.0f} mm2  TTFT {r['prefill']['ttft_ms']:7.1f} ms  "
              f"E {r['prefill']['energy_mJ']:6.1f} mJ  decode {r['decode']['tok_s']:5.1f} tok/s", flush=True)
    fl, ai = out[m.name]["decode"]["flash"], out[m.name]["decode"]["aimc"]
    print(f"  decode: experts on AIMC are {(fl['npu_ms'] + fl['experts_ms']) / (ai['npu_ms'] + ai['experts_ms']):.1f}x faster and "
          f"use {fl['mJ'] / ai['mJ']:.1f}x less energy per token than NPU+flash; the NPU takes "
          f"{100 * ai['npu_ms'] / (ai['npu_ms'] + ai['experts_ms']):.0f}% of the AIMC decode latency", flush=True)
    base, ours = designs["3D-AIMC"]["prefill"], designs["OC-DPA"]["prefill"]
    print(f"  experts take {100 * base['expert_mJ'] / base['energy_mJ']:.0f}% of the prefill energy; OC-DPA lowers it by "
          f"{100 * (1 - ours['expert_mJ'] / base['expert_mJ']):.0f}% for the experts and {100 * (1 - ours['energy_mJ'] / base['energy_mJ']):.0f}% in total", flush=True)

json.dump(out, open(args.out, "w"), indent=1, default=float)
print("wrote", args.out)
