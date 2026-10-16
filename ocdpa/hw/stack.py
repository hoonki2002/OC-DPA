"""Heterogeneous-3D expert store: memory tiers on top, ADCs on the logic tier.

Every expert has to stay resident -- the premise is that neither DRAM nor the
array can be reprogrammed per token -- and a 2D 1T1R array cannot hold them:
OLMoE's experts alone need ~26,000 mm2. NeuroSim's stock H3D does not help
much because each SubArray carries its own ADCs up into the memory tier.

This model splits the SubArray by tier, using probe3d to price every block at
the node it sits on:

  memory tier (22 nm, x T)  cell array, WL/SL switch matrices, column mux,
                            write level shifters
  logic tier  (7 nm, x 1)   SAR ADCs, shift-add

and lets an ADC lane be shared two ways: by `m` columns inside a tier (the
usual column mux) and by `s` tiers stacked over it, through one vertical TSV
that each of those tiers can switch onto. G = T / s independent output buses
then rise through each stack.

Inter-tier sharing costs time only when two tiers of the same group are busy
in the same layer. Tiers are filled with *different experts*, so in a top-k
of E MoE most of a group is idle on any one token; `stack_latency` measures
that collision cost rather than assuming it.
"""
from __future__ import annotations

import json, os, subprocess
from dataclasses import dataclass

import numpy as np

from ocdpa.paths import CACHE as _CACHE_DIR, PROBE3D

CACHE = _CACHE_DIR / "probe3d.json"       # memoised NeuroSim runs

ROWS = COLS = 128
# Vertical interconnect: fine-pitch via-last TSVs in thinned dies, the class
# that can carry one TSV per ADC lane (Samal et al., IEEE S3S 2016, put
# conventional 5 um TSVs at a 30-50 um pitch, far coarser than a subarray).
#   pitch 2 um, diameter 1 um      -- imec, "Extreme wafer thinning and nano-TSV
#                                     processing for 3D heterogeneous
#                                     integration", ECTC 2020
#   capacitance 7 fF per tier      -- coaxial liner capacitance of that via in a
#                                     10 um die (11.9 fF, 0.1 um oxide) scaled by
#                                     0.6, the ratio measured/coaxial of the
#                                     3x30 um (31.1 fF) and 5x50 um (~80 fF) TSVs
#                                     reported by Georgia Tech; sensitivity 3.5-14 fF
# NeuroSim's Heterogeneous3D defaults (1.7 um, 20 fF) remain the conservative case.
TSV_PITCH_UM = 2.0
TSV_CAP_F = 7e-15
TSV_RES = 0.3             # Heterogeneous3D/Param.cpp
# NeuroSim TSVPath at the 7 nm logic tier (Heterogeneous3D/NeuroSIM/tsvprobe),
# 20 fF TSV: 1.528 fJ per digital bit at activity 0.15; scaled to TSV_CAP_F below.
# A digital TSV adds NeuroSim's two inverters (0.029 um2) to its pitch square.
_INV_FF = 2 * 0.3738 + 0.0419                     # driver/receiver caps from tsvprobe
TSV_BIT_FJ = 1.52803 * (_INV_FF + TSV_CAP_F * 1e15) / (_INV_FF + 20.0)
TSV_DIGITAL_UM2 = TSV_PITCH_UM ** 2 + 0.0291
V_READ = 0.5              # Param.cpp readVoltage
VDD_LOGIC = 0.7           # Technology.cpp, 7 nm
INPUT_ACTIVITY = 0.5      # fraction of input lines toggling per input bit


RON = 300000              # RRAM on-resistance (ohm) passed to NeuroSim; override for sensitivity runs


def probe(tech: int, mux: int, access: int = 1, n: int = 128, colcap_ff: float = 0.0) -> dict:
    """Component areas (um2) and energies (pJ per subarray read).

    access 1 = 1T1R, 4 = selector crossbar (NeuroSim's 4F2 cell); n x n array.
    colcap_ff: TSV capacitance hung on each mux output, which NeuroSim adds to
    the column for both settling time and bitline energy.
    """
    key = (f"t{tech}/m{mux}" + ("" if (access, n) == (1, 128) else f"/a{access}/n{n}")
           + (f"/c{colcap_ff:g}" if colcap_ff else "")
           + ("" if RON == 300000 else f"/r{RON}"))
    store = json.load(open(CACHE)) if CACHE.exists() else {}
    if key not in store:
        env = dict(os.environ, NS_TECHNODE=str(tech), NS_SARADC="1",
                   NS_LEVELOUTPUT="256", NS_COLMUXED=str(mux), NS_RON=str(int(RON)),
                   NS_CELLBIT="1", NS_MEMCELL="2", NS_ACCESS=str(access),
                   NS_ROWS=str(n), NS_COLS=str(n), NS_EXTRA_COLCAP_FF=str(colcap_ff))
        out = subprocess.run([str(PROBE3D)], env=env, capture_output=True,
                             text=True, check=True).stdout
        rec = {}
        for line in out.splitlines():
            f = line.split(",")
            if f[0] in ("comp_area_um2", "comp_energy_pJ", "comp_latency_ns"):
                rec[f[0]] = {f[i]: float(f[i + 1]) for i in range(1, len(f) - 1, 2)}
            elif f[0] == "readLatency_ns":
                rec["readLatency_ns"] = float(f[1])
            elif f[0] == "extra_colcap_fF":
                rec["extra_energy_pJ"] = float(f[3])
        store[key] = rec
        CACHE.write_text(json.dumps(store, indent=1))
    return store[key]


@dataclass
class Stack:
    tiers: int = 16          # T memory tiers per stack
    mux: int = 8             # m columns per ADC lane inside a tier
    share: int = 16          # s tiers per ADC lane (1 = no inter-tier sharing)
    write_share: int = 1     # subarrays sharing one set of write level shifters
    input_share: int = 8     # subarrays of a tier fed by one bundle of input TSVs

    @property
    def colcap_ff(self):
        # An output crossing from the top tier passes every tier below it;
        # NeuroSim's TSVPath charges numTSV x tsvCap for that, and the
        # synchronous read clock is set by the slowest (top) tier.
        return self.tiers * TSV_CAP_F * 1e15

    @property
    def buses(self):         # G vertical output buses per stack position
        return self.tiers // self.share

    @property
    def lanes(self):         # ADC lanes on the logic tier per stack position
        return COLS // self.mux * self.buses

    def areas(self):
        mem, log = probe(22, self.mux), probe(7, self.mux)
        a, la = mem["comp_area_um2"], log["comp_area_um2"]
        # digital input TSVs carry NeuroSim's driver/receiver inverters; the
        # analog outputs are bare TSVs into the ADC lane
        tsv = (ROWS / self.input_share * TSV_DIGITAL_UM2 +
               COLS // self.mux * self.buses * TSV_PITCH_UM ** 2)
        mem_tier = (a["array"] + a["wlSwitch"] + a["slSwitch"] + a["mux"] +
                    a["muxDecoder"] + a["levelShift"] / self.write_share + tsv)
        adc_per_lane = la["sarADC"] / (COLS // self.mux)
        sa_per_lane = la["shiftAddWeight"] / (COLS // self.mux)
        logic = self.lanes * (adc_per_lane + sa_per_lane) + tsv
        return dict(mem_tier=mem_tier, logic=logic, tsv=tsv,
                    footprint=max(mem_tier, logic))

    def pj_per_conv(self):
        mem = probe(22, self.mux, colcap_ff=self.colcap_ff)
        log = probe(7, self.mux)
        e, le = mem["comp_energy_pJ"], log["comp_energy_pJ"]
        tsv_out = mem.get("extra_energy_pJ", 0.0) / COLS     # NeuroSim bitline term on the TSV
        local = (e["array"] + e["wlSwitch"] + e["mux"] + e["muxDecoder"]) / COLS - tsv_out
        adc = (le["sarADC"] + le["shiftAddWeight"]) / COLS
        # one input bit per read row, shared by input_share subarrays, over
        # the COLS conversions that read produces
        tsv_in = ROWS / self.input_share * TSV_BIT_FJ * 1e-3 / COLS
        return dict(local=local, adc=adc, tsv_out=tsv_out, tsv_in=tsv_in,
                    total=local + adc + tsv_out + tsv_in)

    def t_read_ns(self):
        # Column settling with the TSV load, as NeuroSim computes it on the
        # 22 nm memory tier.
        return probe(22, self.mux, colcap_ff=self.colcap_ff)["readLatency_ns"]


@dataclass
class Tier:
    """One kind of memory subarray in the stack and how its lanes are shared."""
    access: int = 1          # 1 = 1T1R, 4 = crossbar
    n: int = 128             # n x n subarray
    share: int = 16          # tiers per ADC lane
    planes: int = 8          # bit-planes of each weight stored in this kind


def hetero_area(params: float, tiers_spec: list[Tier], tiers=16, mux=8,
                write_share=16, input_share=1, lane_sets="kind"):
    """Footprint (mm2) of a stack holding `params` weights as 8 bit-planes,
    split across subarray kinds.

    `lane_sets` says how the ADC lanes of the logic tier are provisioned. The timing model
    (compare.layer_cost, Eq. 1 of the paper) reads the planes of an expert one after the other
    through the lanes of its group, so the planes of one kind share one set of lanes through a
    plane-select mux ("kind", the default): a lane set per tile position of each kind. "plane"
    gives every plane its own lane set, which only fits hardware that reads planes in parallel;
    it is how the earlier version of this model counted lanes.

    Every subarray reads 128 rows at a time -- the AnyBCQ group, so each
    column current carries one alpha -- and a 256-row crossbar takes two
    reads, with twice the lanes provisioned so its throughput matches.
    `input_share` subarrays of one tier take the same input slice and share
    one bundle of input TSVs over an in-tier bus.
    """
    mem = logic = 0.0
    parts = {}
    lane_7nm = probe(7, mux)["comp_area_um2"]
    lane_area = (lane_7nm["sarADC"] + lane_7nm["shiftAddWeight"]) / (128 // mux)
    for t in tiers_spec:
        a = probe(22, mux, t.access, t.n)["comp_area_um2"]
        n_sub = params * t.planes / (t.n * t.n)
        n_lane = n_sub if lane_sets == "plane" else params * min(t.planes, 1.0) / (t.n * t.n)
        reads = t.n // 128                                   # sequential 128-row reads
        buses = tiers // t.share * reads
        tsv = (t.n / input_share * TSV_DIGITAL_UM2 +
               t.n // mux * buses * TSV_PITCH_UM ** 2)
        sub = (a["array"] + a["wlSwitch"] + a["slSwitch"] + a["mux"] +
               a["muxDecoder"] + a["levelShift"] / write_share)
        m_tier = n_sub / tiers * (sub + tsv)
        l_tier = n_lane / tiers * (t.n // mux * buses * lane_area + tsv)
        mem += m_tier; logic += l_tier
        parts[f"a{t.access}/n{t.n}"] = dict(mem_mm2=m_tier / 1e6, logic_mm2=l_tier / 1e6,
                                            sub_um2=sub, tsv_um2=tsv)
    return dict(mem_mm2=mem / 1e6, logic_mm2=logic / 1e6,
                footprint_mm2=max(mem, logic) / 1e6, parts=parts)


def stack_latency(st: Stack, experts: int, topk: int, k: np.ndarray | int,
                  routes: np.ndarray, input_bits: int = 8):
    """Mean busy time (ns) of the busiest ADC group, per matrix stage.

    Expert e lives on tier e % T of stack-set e // T, so each sharing group
    holds `s` distinct experts. A group's lane must convert, one after the
    other, every plane of every routed expert it holds.

    routes: (tokens, topk) expert ids; k: planes per routed expert, scalar or
    (tokens, topk).
    """
    k = np.broadcast_to(np.asarray(k), routes.shape)
    group = (routes // st.tiers) * st.buses + (routes % st.tiers) // st.share
    busy = np.zeros((routes.shape[0], experts // st.share + 1))
    np.add.at(busy, (np.arange(routes.shape[0])[:, None], group), k)
    return busy.max(1).mean() * input_bits * st.t_read_ns()


if __name__ == "__main__":
    from ocdpa.hw.npu import MODELS

    rng = np.random.default_rng(0)
    print("NeuroSim 1T1R 128x128, 8b SAR, memory tiers 22 nm, logic tier 7 nm,"
          " TSV 1.7 um / 20 fF\n")
    base = probe(22, 8)
    a2d = sum(base["comp_area_um2"][c] for c in
              ("array", "wlSwitch", "slSwitch", "mux", "muxDecoder", "levelShift",
               "sarADC", "shiftAddWeight"))
    print(f"2D reference subarray: {a2d:.0f} um2, 0.342 pJ/conv, "
          f"{base['readLatency_ns']:.0f} ns/read\n")

    for m in MODELS:
        n_sub = m.n_moe * m.experts * 3 * m.hidden * m.moe_inter * 8 / (ROWS * COLS)
        routes = np.stack([rng.choice(m.experts, m.topk, replace=False)
                           for _ in range(4000)])
        print(f"{m.name}: {n_sub/1e6:.2f} M subarrays, 2D {n_sub*a2d/1e6:,.0f} mm2; "
              f"uniform-random routing, k=4")
        print(f"  {'T':>3} {'m':>3} {'s':>3} {'wr':>3} | {'mem um2':>8} {'logic um2':>9} "
              f"{'footprint mm2':>13} | {'pJ/conv':>7} {'(adc':>6} {'tsv)':>6} | "
              f"{'stage us':>8} {'tok ms':>7} {'uJ/tok':>7}")
        for T, mux, s, w in [(16, 8, 1, 1), (16, 8, 4, 1), (16, 8, 16, 1),
                             (16, 8, 1, 16), (16, 8, 4, 16), (16, 8, 16, 16),
                             (16, 16, 16, 16), (16, 32, 16, 16)]:
            st = Stack(T, mux, s, w)
            ar, en = st.areas(), st.pj_per_conv()
            fp = n_sub / T * ar["footprint"] / 1e6
            lat = stack_latency(st, m.experts, m.topk, 4, routes)
            tok_ms = m.n_moe * 2 * lat / 1e6        # gate|up in parallel, then down
            uj = m.expert_conversions(4) * en["total"] / 1e6
            print(f"  {T:3d} {mux:3d} {s:3d} {w:3d} | {ar['mem_tier']:8.0f} {ar['logic']:9.0f} "
                  f"{fp:13,.0f} | {en['total']:7.3f} {en['adc']:6.3f} "
                  f"{en['tsv_out']+en['tsv_in']:6.3f} | {lat/1e3:8.2f} {tok_ms:7.2f} {uj:7.1f}")
        print()
