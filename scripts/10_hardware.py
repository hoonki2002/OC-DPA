"""Hardware results: the design comparison, the TTFT ablation and the sharing sweep.

For OLMoE-1B-7B and DeepSeek-MoE-16B, on a 1024-token prompt and batch-1 decoding:

  designs   3D-AIMC (16 tiers share a lane, 8:1 column mux, experts in order, all planes)
            + load-sorted grouping on 16 tiers (all planes)
            best baseline: the point of the design space (tiers per lane s, columns per ADC mux, and for models
                           with always-on shared experts the mux of their lanes) with the highest prefill
                           throughput per area, all planes read
            + uniform k-bar = 4 at the configuration of OC-DPA
            OC-DPA: DPA k-bar = 4 in prefill, k = 8 in decode, at the point of the same design space that
                    is best for it
  budget    TTFT and prefill energy of uniform and DPA allocation for k-bar = 2..8 (at the sharing
            of OC-DPA, load-sorted, decode at k = 8)
  sharing   TTFT, decode latency per token and the memory and logic dies for s = 1, 2, 4, 8, 16 tiers per
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

from ocdpa.compare import SPEC, SHARED_ON_AIMC, best, evaluate, grid, placement
from ocdpa.paths import data_file
from ocdpa.hw.npu import MODELS, NPU, NPU_CFG, dense_step
from ocdpa.planes import SCALE_BITS
from ocdpa.system import npu_experts, plans

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="OLMoE-1B-7B,DeepSeek-MoE-16B,Qwen3-30B-A3B")
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
    cal = np.load(data_file(f"trace_{tag}.npz"))["idx"].astype(np.int64)
    tr = np.load(data_file(f"trace_{tag}{args.long}.npz"))
    idx, gate = tr["idx"].astype(np.int64), tr["gate"]
    R = np.load(data_file(resid))
    R = R[R.shape[0] - idx.shape[0]:]
    L = idx.shape[0]
    loads = [np.bincount(cal[l].ravel(), minlength=m.experts) for l in range(L)]
    k8 = plans(idx, gate, R, 8, m.topk, False)
    k_uni = plans(idx, gate, R, args.kbar, m.topk, False)
    k_dpa = plans(idx, gate, R, args.kbar, m.topk, True)

    def run(share, grouped, k_pre, k_dec, scale_bits=None, mux=8, shared_mux=None, P=1024):
        slots = [placement(loads[l], share, grouped) for l in range(L)]
        return evaluate(m, idx, k_pre, k_dec, slots, share, npu, P=P, scale_bits=scale_bits, mux=mux, shared_mux=shared_mux)

    rows = grid(m, idx, loads, npu, {"all planes": (k8, k8, None), "OC-DPA": (k_dpa, k8, SCALE_BITS)})
    b, o = best(rows, "all planes"), best(rows, "OC-DPA")
    cfg = lambda r: dict(share=r["share"], mux=r["mux"], shared_mux=r["shared_mux"])
    at = lambda r: dict(share=r["share"], mux=r["mux"], shared_mux=r["shared_mux"])
    designs = {
        "3D-AIMC":                       run(16, False, k8, k8),
        "16 tiers + load-sorted grouping": run(16, True, k8, k8),
        "Best baseline":                 run(**{"share": b["share"], "grouped": True, "k_pre": k8, "k_dec": k8,
                                                "mux": b["mux"], "shared_mux": b["shared_mux"]}),
        f"Uniform k={args.kbar}":        run(o["share"], True, k_uni, k8, SCALE_BITS, o["mux"], o["shared_mux"]),
        "OC-DPA":                        run(o["share"], True, k_dpa, k8, SCALE_BITS, o["mux"], o["shared_mux"]),
        "Baseline at the OC-DPA point":  run(o["share"], True, k8, k8, None, o["mux"], o["shared_mux"]),
        "OC-DPA at the baseline point":  run(b["share"], True, k_dpa, k8, SCALE_BITS, b["mux"], b["shared_mux"]),
    }
    configs = {"3D-AIMC": dict(share=16, mux=8, shared_mux=None), "Best baseline": cfg(b), "OC-DPA": cfg(o)}
    same_area = min((r for r in rows if r["design"] == "all planes" and r["footprint_mm2"] <= o["footprint_mm2"]),
                    key=lambda r: r["ttft_ms"], default=None)
    same_ttft = min((r for r in rows if r["design"] == "all planes" and r["ttft_ms"] <= o["ttft_ms"]),
                    key=lambda r: r["footprint_mm2"], default=None)

    sharing = {}
    for s in (1, 2, 4, 8, 16):
        for grouped in (False, True):
            r = run(s, grouped, k8, k8, None, b["mux"], b["shared_mux"])
            sharing[f"s{s}_{'grouped' if grouped else 'in_order'}"] = dict(
                ttft_ms=r["prefill"]["ttft_ms"], decode_ms=r["decode"]["ms"], mem_mm2=r["mem_mm2"],
                logic_mm2=r["logic_mm2"], footprint_mm2=r["footprint_mm2"])

    sharing_p = {}
    for P in (1024, 4096):
        for s_ in (1, 2, 4, 8, 16):
            r = run(s_, True, k8, k8, None, b["mux"], b["shared_mux"], P=P)
            sharing_p[f"P{P}_s{s_}"] = dict(ttft_ms=r["prefill"]["ttft_ms"], expert_ms=r["prefill"]["expert_ms"])

    budget = {}
    for kb in range(2, 9):
        for name, dpa in (("uniform", False), ("DPA", True)):
            r = run(o["share"], True, plans(idx, gate, R, kb, m.topk, dpa), k8, SCALE_BITS, o["mux"], o["shared_mux"])
            budget[f"{name}_{kb}"] = dict(ttft_ms=r["prefill"]["ttft_ms"], energy_mJ=r["prefill"]["energy_mJ"])

    prompt = []
    for P in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        slots_b = [placement(loads[l], b["share"], True) for l in range(L)]
        kw = dict(P=P, mux=b["mux"], shared_mux=b["shared_mux"])
        a_ = evaluate(m, idx, k8, k8, slots_b, b["share"], npu, **kw)
        o_ = evaluate(m, idx, k_dpa, k8, slots_b, b["share"], npu, scale_bits=SCALE_BITS, **kw)
        prompt.append(dict(tokens=P, npu_ms=a_["prefill"]["ttft_ms"] - a_["prefill"]["expert_ms"],
                           lanes_ms=a_["prefill"]["expert_ms"], lanes_ocdpa_ms=o_["prefill"]["expert_ms"],
                           ttft_ms=a_["prefill"]["ttft_ms"], ttft_ocdpa_ms=o_["prefill"]["ttft_ms"]))

    dense_flash = dense_step(m, NPU(), 1, 1024)
    flash_mJ, flash_ms, _ = npu_experts(m, idx, k8, 0, 512)
    on_aimc = SHARED_ON_AIMC and m.shared_inter > 0
    dense_aimc = dense_step(replace(m, shared_inter=0) if on_aimc else m, NPU(), 1, 1024)
    dec_aimc = designs["3D-AIMC"]["decode"]["ms"]
    out[m.name] = dict(
        designs=designs, configs=configs, grid=rows, sharing=sharing, sharing_mux=b["mux"], budget=budget, prompt=prompt, sharing_p=sharing_p,
        same_area=same_area, same_ttft=same_ttft,
        decode=dict(flash=dict(npu_ms=dense_flash["latency_ms"], experts_ms=flash_ms,
                               mJ=dense_flash["dram_mJ"] + flash_mJ),
                    aimc=dict(npu_ms=dense_aimc["latency_ms"], experts_ms=dec_aimc - dense_aimc["latency_ms"],
                              mJ=designs["3D-AIMC"]["decode"]["energy_mJ"])))
    print(f"{m.name}", flush=True)
    for name, r in designs.items():
        c = configs.get(name)
        pt = f"  s={c['share']:<2d} mux {c['mux']}:1" + (f" shared {c['shared_mux']}:1" if c["shared_mux"] else "") if c else ""
        print(f"  {name:34s} {r['footprint_mm2']:7.0f} mm2  TTFT {r['prefill']['ttft_ms']:7.1f} ms  "
              f"E {r['prefill']['energy_mJ']:6.1f} mJ  decode {r['decode']['tok_s']:5.1f} tok/s  "
              f"{1024e3 / r['prefill']['ttft_ms'] / r['footprint_mm2']:5.1f} tok/s/mm2{pt}", flush=True)
    print(f"  OC-DPA over the best baseline: {o['tok_s_mm2'] / b['tok_s_mm2']:.2f}x throughput per area", flush=True)
    if same_area:
        print(f"  best baseline of no more area than OC-DPA ({same_area['footprint_mm2']:.0f} mm2, s={same_area['share']} "
              f"mux {same_area['mux']}:1): TTFT {same_area['ttft_ms']:.0f} ms, OC-DPA {o['ttft_ms']:.0f} ms "
              f"({same_area['ttft_ms'] / o['ttft_ms']:.2f}x)", flush=True)
    if same_ttft:
        print(f"  best baseline of no more TTFT than OC-DPA ({same_ttft['ttft_ms']:.0f} ms): {same_ttft['footprint_mm2']:.0f} mm2 "
              f"against {o['footprint_mm2']:.0f} mm2 ({100 * (1 - o['footprint_mm2'] / same_ttft['footprint_mm2']):.0f}% less)", flush=True)
    fl, ai = out[m.name]["decode"]["flash"], out[m.name]["decode"]["aimc"]
    print(f"  decode: experts on AIMC are {(fl['npu_ms'] + fl['experts_ms']) / (ai['npu_ms'] + ai['experts_ms']):.1f}x faster and "
          f"use {fl['mJ'] / ai['mJ']:.1f}x less energy per token than NPU+flash; the NPU takes "
          f"{100 * ai['npu_ms'] / (ai['npu_ms'] + ai['experts_ms']):.0f}% of the AIMC decode latency", flush=True)
    base, ours = designs["Best baseline"]["prefill"], designs["OC-DPA"]["prefill"]
    print(f"  experts take {100 * base['expert_mJ'] / base['energy_mJ']:.0f}% of the prefill energy; OC-DPA lowers it by "
          f"{100 * (1 - ours['expert_mJ'] / base['expert_mJ']):.0f}% for the experts and {100 * (1 - ours['energy_mJ'] / base['energy_mJ']):.0f}% in total", flush=True)

json.dump(out, open(args.out, "w"), indent=1, default=float)
print("wrote", args.out)
