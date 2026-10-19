"""Every hardware number of the paper, from one command.

Runs the hardware half of the repository (it needs no GPU and no model, only the small inputs of data/ and, for new
design points, a built NeuroSim) and writes one report:

  python scripts/13_hardware_report.py [--out cache/hardware]

  11_allocator_ppa.py         area, energy and latency of the on-chip allocator          -> allocator_ppa.json
  10_hardware.py              design space (tiers per ADC lane x columns per ADC), the best baseline and OC-DPA,
                              the sharing sweep, the prompt sweep, the budget sweep       -> hardware.json
  12_precision_alternatives.py  static 4-bit and 4-bit + 8-bit against OC-DPA             -> alternatives.json
  30_make_figures.py          Fig. 1 and Fig. 2                                           -> figs/

and prints, in hardware_report.md and hardware_claims.json, the tables and every figure quoted in the text, so that the
manuscript can be checked line by line. The quality half (KL, accuracy) is separate: scripts/21_kl_sweep.py and
scripts/22_downstream_choice.py / 23_downstream_generation.py.
"""
import argparse, json, math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="cache/hardware", help="directory for the JSON files, the report and the figures")
ap.add_argument("--skip-run", action="store_true", help="only rebuild the report from the JSON files already in --out")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
OLM, DS, QW = "OLMoE-1B-7B", "DeepSeek-MoE-16B", "Qwen3-30B-A3B"
MODELS = (OLM, DS, QW)

from ocdpa.paths import ALLOCATOR, DATA

if not args.skip_run:
    if not ALLOCATOR.exists():                                               # no NeuroSim build: use the shipped allocator PPA
        print("== NeuroSim `allocator` not built: using data/allocator_ppa.json", flush=True)
        (out / "allocator_ppa.json").write_text((DATA / "allocator_ppa.json").read_text())
    for cmd in (([] if not ALLOCATOR.exists() else [["11_allocator_ppa.py", "--out", str(out / "allocator_ppa.json")]]) +
                [["10_hardware.py", "--out", str(out / "hardware.json")],
                ["12_precision_alternatives.py", "--out", str(out / "alternatives.json")],
                ["30_make_figures.py", "--hardware", str(out / "hardware.json"), "--out", str(out / "figs"), "--only", "1,2"]]):
        print("==", " ".join(cmd), flush=True)
        subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], check=True)

hw = json.load(open(out / "hardware.json"))
alt = json.load(open(out / "alternatives.json"))
ppa = json.load(open(out / "allocator_ppa.json"))
claims, md = {}, []


def row(r):
    return r["footprint_mm2"], r["prefill"]["ttft_ms"], r["prefill"]["energy_mJ"], 1024e3 / r["prefill"]["ttft_ms"] / r["footprint_mm2"]


# ---------------------------------------------------------------- Table I
md.append("## Table I: hardware comparison at a 1024-token prefill\n")
md.append("| model | design | s / m (shared m) | mm2 | TTFT (ms) | E_pre (mJ) | tok/s/mm2 |\n|---|---|---|---|---|---|---|")
for model in MODELS:
    h = hw[model]
    for name in ("3D-AIMC", "Best baseline", "OC-DPA"):
        c = h["configs"][name]; f, t, e, tp = row(h["designs"][name])
        pt = f"{c['share']} / {c['mux']}" + (f" ({c['shared_mux']})" if c["shared_mux"] else "")
        md.append(f"| {model} | {name} | {pt} | {f:.0f} | {t:.0f} | {e:.0f} | {tp:.1f} |")
        claims[f"{model}.{name}"] = dict(point=c, mm2=f, ttft_ms=t, energy_mJ=e, tok_s_mm2=tp)

# ---------------------------------------------------------------- claims of the text
md.append("\n## Numbers quoted in the text\n")
for model in MODELS:
    h = hw[model]; base = h["designs"]["Best baseline"]; oc = h["designs"]["OC-DPA"]; aim = h["designs"]["3D-AIMC"]
    fb, tb, eb, tpb = row(base); fo, to, eo, tpo = row(oc); fa, ta, ea, tpa = row(aim)
    g = [r for r in h["grid"] if r["design"] == "all planes"]
    near = [r for r in g if r["ttft_ms"] <= 1.01 * to]                       # "the same TTFT" = within 1 %
    same_ttft = min(near, key=lambda r: r["footprint_mm2"]) if near else None
    same_area = min((r for r in g if r["footprint_mm2"] <= fo * 1.0001), key=lambda r: r["ttft_ms"], default=None)
    sh = h["sharing"]; s1, s16 = sh["s1_grouped"], sh["s16_grouped"]
    p1024 = next(p for p in h["prompt"] if p["tokens"] == 1024)
    cross = None
    for a, b in zip(h["prompt"], h["prompt"][1:]):
        if a["lanes_ms"] < a["npu_ms"] and b["lanes_ms"] >= b["npu_ms"]:        # the lanes overtake the NPU between a and b
            f_a, f_b = math.log(a["lanes_ms"] / a["npu_ms"]), math.log(b["lanes_ms"] / b["npu_ms"])
            cross = math.exp(math.log(a["tokens"]) + (-f_a / (f_b - f_a)) * (math.log(b["tokens"]) - math.log(a["tokens"])))
    dec = h["decode"]
    c = dict(
        footprint_ratio_s1_to_s16=s1["footprint_mm2"] / s16["footprint_mm2"], footprint_s1=s1["footprint_mm2"], footprint_s16=s16["footprint_mm2"],
        logic_die_s1=s1["logic_mm2"], logic_die_s16=s16["logic_mm2"], memory_die_s1=s1["mem_mm2"], memory_die_s16=s16["mem_mm2"],
        ttft_s1=s1["ttft_ms"], ttft_s16=s16["ttft_ms"], ttft_ratio_s16_to_s1=s16["ttft_ms"] / s1["ttft_ms"],
        sharing_sweep_columns_per_adc=h["sharing_mux"],
        aimc3d_to_best_baseline_area_pct=100 * (fb / fa - 1), aimc3d_to_best_baseline_ttft_ratio=ta / tb,
        throughput_per_area_gain=tpo / tpb, ocdpa_ttft_at_baseline_point=h["designs"]["OC-DPA at the baseline point"]["prefill"]["ttft_ms"],
        same_area_baseline=same_area, same_ttft_baseline=same_ttft,
        same_ttft_area_ratio=(same_ttft["footprint_mm2"] / fo) if same_ttft else None,
        same_area_ttft_ratio=(same_area["ttft_ms"] / to) if same_area else None,
        area_saved_at_baseline_ttft_pct=100 * (1 - fo / fb), energy_saving_pct=100 * (1 - eo / eb),
        expert_energy_share_pct=100 * base["prefill"]["expert_mJ"] / eb,
        expert_energy_saving_pct=100 * (1 - oc["prefill"]["expert_mJ"] / base["prefill"]["expert_mJ"]),
        lane_share_of_ttft_at_1024_pct=100 * p1024["lanes_ms"] / p1024["ttft_ms"], lane_halving_at_1024=p1024["lanes_ms"] / p1024["lanes_ocdpa_ms"],
        npu_ms_at_1024=p1024["npu_ms"], lane_ms_at_1024=p1024["lanes_ms"], lane_bound_beyond_tokens=cross,
        decode_speedup_vs_flash=(dec["flash"]["npu_ms"] + dec["flash"]["experts_ms"]) / (dec["aimc"]["npu_ms"] + dec["aimc"]["experts_ms"]),
        decode_energy_ratio_vs_flash=dec["flash"]["mJ"] / dec["aimc"]["mJ"],
        npu_share_of_aimc_decode_pct=100 * dec["aimc"]["npu_ms"] / (dec["aimc"]["npu_ms"] + dec["aimc"]["experts_ms"]),
        budget={k: v["ttft_ms"] for k, v in h["budget"].items()})
    claims[model] = c
    md.append(f"### {model}\n")
    md.append(f"- Fig. 1 (sweep of s at {c['sharing_sweep_columns_per_adc']} columns per ADC): footprint {c['footprint_s1']:.0f} -> {c['footprint_s16']:.0f} mm2 "
              f"({c['footprint_ratio_s1_to_s16']:.1f}x); logic die {c['logic_die_s1']:.0f} -> {c['logic_die_s16']:.0f}, memory die {c['memory_die_s1']:.0f} -> {c['memory_die_s16']:.0f} mm2; "
              f"TTFT {c['ttft_s1']:.0f} -> {c['ttft_s16']:.0f} ms ({c['ttft_ratio_s16_to_s1']:.1f}x)")
    md.append(f"- 3D-AIMC -> best baseline: {c['aimc3d_to_best_baseline_area_pct']:+.0f}% area, TTFT {ta:.0f} -> {tb:.0f} ms ({c['aimc3d_to_best_baseline_ttft_ratio']:.2f}x)")
    md.append(f"- OC-DPA over the best baseline: {c['throughput_per_area_gain']:.2f}x throughput per area; energy {c['energy_saving_pct']:.0f}% lower "
              f"(experts: {c['expert_energy_share_pct']:.0f}% of the baseline's prefill energy, {c['expert_energy_saving_pct']:.0f}% lower)")
    if same_ttft:
        md.append(f"- same TTFT (within 1%): the baseline needs {same_ttft['footprint_mm2']:.0f} mm2 (s={same_ttft['share']}, m={same_ttft['mux']}, TTFT {same_ttft['ttft_ms']:.0f} ms) "
                  f"against {fo:.0f} mm2: {c['same_ttft_area_ratio']:.2f}x the area")
    if same_area:
        md.append(f"- same area: baseline TTFT {same_area['ttft_ms']:.0f} ms against {to:.0f} ms: {c['same_area_ttft_ratio']:.2f}x")
    md.append(f"- prefill at 1024 tokens: NPU+DRAM {c['npu_ms_at_1024']:.0f} ms, lanes {c['lane_ms_at_1024']:.0f} ms ({c['lane_share_of_ttft_at_1024_pct']:.0f}% of the TTFT); "
              f"OC-DPA halves the lane time ({c['lane_halving_at_1024']:.1f}x); lanes bound the prefill beyond about {c['lane_bound_beyond_tokens'] or float('nan'):.0f} tokens")
    md.append(f"- decode: experts on AIMC {c['decode_speedup_vs_flash']:.1f}x faster and {c['decode_energy_ratio_vs_flash']:.1f}x less energy than NPU+flash; NPU {c['npu_share_of_aimc_decode_pct']:.0f}% of the AIMC decode latency\n")

# ---------------------------------------------------------------- Table II
md.append("## Table II: sources of the TTFT reduction and static quantization (OLMoE; the KL column comes from scripts/21_kl_sweep.py)\n")
md.append("| row | s / m | mm2 | memory die | logic die | TTFT (ms) | tok/s/mm2 | E_pre (mJ) |\n|---|---|---|---|---|---|---|---|")
rows = alt[OLM]
for name, r in rows.items():
    md.append(f"| {name} | {r['share']} / {r['mux']} | {r['footprint_mm2']:.0f} | {r['mem_mm2']:.0f} | {r['logic_mm2']:.0f} | {r['ttft_ms']:.0f} | {r['prefill_tok_s_mm2']:.1f} | {r['energy_mJ']:.0f} |")
oc_row = rows["+ DPA (OC-DPA)"]; two = rows["Static 4-bit + 8-bit"]; chk = alt["memory_die_check"]
claims["table2"] = dict(rows=rows, two_copy_area_ratio=two["footprint_mm2"] / oc_row["footprint_mm2"],
                        ocdpa_over_two_copy_throughput=oc_row["prefill_tok_s_mm2"] / two["prefill_tok_s_mm2"],
                        mixed_vs_crossbar_memory_die_pct=100 * (chk["mixed"]["mem_mm2"] / chk["all_crossbar"]["mem_mm2"] - 1),
                        mixed_footprint=chk["mixed"]["footprint_mm2"], crossbar_footprint=chk["all_crossbar"]["footprint_mm2"])
md.append(f"\n- two-copy (4 + 8 bit) area {claims['table2']['two_copy_area_ratio']:.2f}x that of OC-DPA; OC-DPA throughput per area {claims['table2']['ocdpa_over_two_copy_throughput']:.2f}x")
md.append(f"- 1T1R base planes: memory die {chk['mixed']['mem_mm2']:.0f} against {chk['all_crossbar']['mem_mm2']:.0f} mm2 for all-crossbar "
          f"({claims['table2']['mixed_vs_crossbar_memory_die_pct']:+.0f}%); footprint {chk['mixed']['footprint_mm2']:.0f} against {chk['all_crossbar']['footprint_mm2']:.0f} mm2\n")

# ---------------------------------------------------------------- allocator
md.append("## On-chip allocator (marginal-gain rule)\n")
md.append("| model | cycles | latency (ns) | energy (pJ) | datapath (um2) | with table (um2) | table (KB) |\n|---|---|---|---|---|---|---|")
for model, tag in ((OLM, "OLMoE-1B-7B"), (DS, "DeepSeek-MoE-16B"), (QW, "Qwen3-30B-A3B")):
    a = ppa[tag]["marginal gain"]; dp = ppa[tag]["DP, 1 lane"]
    fp = hw[model]["designs"]["OC-DPA"]["footprint_mm2"]
    md.append(f"| {model} | {a['cycles']:.0f} | {a['latency_ns']:.1f} | {a['energy_pJ']:.1f} | {a['area_um2'] - a['lut_um2']:.0f} | {a['area_um2']:.0f} | {a['lut_KB']:.2f} |")
    claims[f"allocator.{model}"] = dict(cycles=a["cycles"], latency_ns=a["latency_ns"], critical_ns=a["critical_ns"], energy_pJ=a["energy_pJ"],
                                        datapath_um2=a["area_um2"] - a["lut_um2"], total_um2=a["area_um2"], table_KB=a["lut_KB"],
                                        fraction_of_footprint=a["area_um2"] * 1e-6 / fp, dp_cycles=dp["cycles"], dp_latency_ns=dp["latency_ns"])
md.append("")

(out / "hardware_report.md").write_text("\n".join(md) + "\n")
json.dump(claims, open(out / "hardware_claims.json", "w"), indent=1, default=float)
print("\n".join(md))
print(f"\nwrote {out / 'hardware_report.md'} and {out / 'hardware_claims.json'}")
