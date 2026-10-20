"""The proposal against the two published ways of sharing analog periphery.

  IBM-style   Buchel et al., Nat. Comput. Sci. 2025: one expert per tier, the
              tiers of a tile share its periphery, experts placed in order.
  Gao-Yang    Gao & Yang, ISCAS 2026: crossbars of 2 or 4 experts share
              DAC/ADC/accumulators; experts grouped by load (lowest with
              highest) so every group carries a similar share of tokens.
  Proposal    16-tier sharing, Gao-Yang's load-balanced grouping adopted,
              and phase-aware DPA (prefill kbar 4, decode 8).

Both baselines run every plane (k = 8): neither changes the model. All
three use the same NPU, LPDDR, heterogeneous-cell memory tiers and the same
analog cost per conversion, so the only differences are how lanes are shared,
how experts are placed on them, and how many planes are read.

Load profiles come from the short routing trace (wikitext, 2,048 tokens);
every cost is measured on the long trace (8,192 tokens), so the grouping
never sees the text it is scored on.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ocdpa.hw.stack import Stack, hetero_area, Tier
from ocdpa.planes import GS, KMAX
from ocdpa.hw.npu import NPU, NPU_AREA_MM2, dense_step
from ocdpa.system import ALPHA_PJ, INPUT_BITS
from ocdpa.hw.workload import dense

TIERS = 16
SPEC = {"OLMoE-1B-7B": ("olmoe", "residual_olmoe.npy"),
        "DeepSeek-MoE-16B": ("deepseek", "residual_deepseek.npy"),
        "Qwen3-30B-A3B": ("qwen3", "residual_qwen3.npy")}
HETERO = lambda s: [Tier(1, 128, s, 2), Tier(4, 256, s, 6)]


def placement(load: np.ndarray, share: int, balanced: bool) -> np.ndarray:
    """slot[e] for one layer. Slots s*g .. s*g+s-1 share group g's lanes.

    balanced: experts sorted by load are dealt to groups in a serpentine, so
    each group mixes heavy and light experts -- for groups of two this is
    exactly Gao & Yang's lowest-with-highest pairing.
    """
    E = len(load)
    if not balanced:
        return np.arange(E)
    n_groups = E // share
    order = np.argsort(load)[::-1]
    slot = np.empty(E, dtype=np.int64)
    fill = [0] * n_groups
    for rank, e in enumerate(order):
        rnd, pos = divmod(rank, n_groups)
        g = pos if rnd % 2 == 0 else n_groups - 1 - pos
        slot[e] = g * share + fill[g]
        fill[g] += 1
    return slot


def group_of(slot, share):
    return slot // share


def layer_cost(m, st, idx, k, slot, pipelined):
    """(mJ, ms) of one layer for the rows of idx/k with experts at `slot`."""
    groups_per_expert = 3 * m.hidden * m.moe_inter / 128
    planes = float(k.sum())
    e_mj = (groups_per_expert * INPUT_BITS * planes * st.pj_per_conv()["total"] +
            groups_per_expert * planes * ALPHA_PJ) / 1e9
    g = group_of(slot[idx], st.share)
    ng = m.experts // st.share + 1
    t_read = st.t_read_ns()
    if pipelined:
        load = np.bincount(g.ravel(), weights=k.ravel(), minlength=ng)
        last = np.bincount(g[-1], weights=k[-1], minlength=ng)
        reads = load.max() + last.max()
    else:
        reads = 0.0
        for r in range(idx.shape[0]):
            reads += 2 * np.bincount(g[r], weights=k[r], minlength=ng).max()
        reads /= idx.shape[0]
    return e_mj, reads * INPUT_BITS * t_read / 1e6


SHARED_MUX, SHARED_K = 4, 8
SHARED_ON_AIMC = True

NPU_LOGIC_MM2 = NPU_AREA_MM2


def shared_cost(m, P, pipelined, mux=None):
    """(mJ, ms) of the shared experts of one layer for P tokens."""
    st = Stack(TIERS, mux or SHARED_MUX, 1, 16)
    groups = 3 * m.hidden * m.shared_inter / 128
    planes = P * SHARED_K
    e_mj = (groups * INPUT_BITS * planes * st.pj_per_conv()["total"] +
            groups * planes * ALPHA_PJ) / 1e9
    reads = planes + SHARED_K if pipelined else 2 * SHARED_K
    return e_mj, reads * INPUT_BITS * st.t_read_ns() / 1e6


def scale_storage_mm2(params, scale_bits):
    """Memory-tier area of the 8-bit scales: alpha (8 planes) and beta take
    (KMAX + 1) * scale_bits bits per group of 128 weights, 0.5625 bit per
    weight at 8 bits, held as digital cells of the denser crossbar kind."""
    if not scale_bits:
        return 0.0
    bits_per_weight = (KMAX + 1) * scale_bits / GS
    return hetero_area(params, [Tier(4, 256, 16, bits_per_weight, cells=1)], tiers=TIERS, input_share=8)["mem_mm2"]


def shared_area(m, scale_bits=None, mux=None):
    if not m.shared_inter:
        return dict(mem_mm2=0.0, logic_mm2=0.0)
    params = m.n_moe * 3 * m.hidden * m.shared_inter
    a = hetero_area(params, [Tier(1, 128, 1, 2), Tier(4, 256, 1, 6)], tiers=TIERS,
                    mux=mux or SHARED_MUX, input_share=8)
    a["mem_mm2"] += scale_storage_mm2(params, scale_bits)
    return a


def evaluate(m, idx, k_pre, k_dec, slots, share, npu, P=1024, warm=512, scale_bits=None, mux=8, shared_mux=None):
    """Prefill (TTFT, energy), batch-1 decode and footprint of one design. `scale_bits` is the
    storage width of alpha/beta for designs that use the any-precision encoding (None: plain bit-slicing).
    `mux` is the number of columns per ADC lane of the routed experts, `shared_mux` that of the shared experts."""
    st = Stack(TIERS, mux, share, 16)
    L = idx.shape[0]
    on = SHARED_ON_AIMC and m.shared_inter > 0
    m_npu = replace(m, shared_inter=0) if on else m
    d = dense(m_npu, npu, 1, P, P)
    e = t = 0.0
    sh_e, sh_t = shared_cost(m, P, True, shared_mux) if on else (0.0, 0.0)
    for l in range(L):
        de, dt = layer_cost(m, st, idx[l, P:2 * P], k_pre[l, P:2 * P], slots[l], True)
        e += de + sh_e; t += max(dt, sh_t)
    pre = dict(ttft_ms=d["ms"] + t, ttft_chunk_ms=max(d["ms"], t), expert_ms=t,
               energy_mJ=d["mJ"] + e, expert_mJ=e)
    dd = dense_step(m_npu, NPU(), 1, 1024)
    sh_e, sh_t = shared_cost(m, 1, False, shared_mux) if on else (0.0, 0.0)
    e = t = 0.0
    rows = slice(warm, warm + 512)
    for l in range(L):
        de, dt = layer_cost(m, st, idx[l, rows], k_dec[l, rows], slots[l], False)
        e += de / 512 + sh_e; t += max(dt, sh_t)
    dec = dict(ms=dd["latency_ms"] + t, tok_s=1e3 / (dd["latency_ms"] + t),
               energy_mJ=dd["dram_mJ"] + e)
    params = m.n_moe * m.experts * 3 * m.hidden * m.moe_inter
    area = hetero_area(params, HETERO(share), tiers=TIERS, mux=mux, input_share=8)
    sa = shared_area(m, scale_bits, shared_mux) if on else dict(mem_mm2=0.0, logic_mm2=0.0)
    mem = area["mem_mm2"] + sa["mem_mm2"] + scale_storage_mm2(params, scale_bits)
    adc_logic = area["logic_mm2"] + sa["logic_mm2"]
    logic = adc_logic + NPU_LOGIC_MM2
    return dict(prefill=pre, decode=dec, footprint_mm2=max(mem, logic),
                logic_mm2=logic, adc_logic_mm2=adc_logic, mem_mm2=mem)


GRID_SHARE = (1, 2, 4, 8, 16)
GRID_MUX = (8, 4, 2, 1)
GRID_SHARED_MUX = (4, 2, 1)


def grid(m, idx, loads, npu, designs, P=1024, shares=GRID_SHARE, muxes=GRID_MUX, shared_muxes=GRID_SHARED_MUX):
    """Price every (share, mux, shared_mux) point for each design.

    designs: {name: (k_prefill, k_decode, scale_bits)}. Experts are grouped by load (placement(..., True)).
    Returns a list of dicts with the design name, the point, the dies, TTFT, prefill energy, decode speed and
    `tok_s_mm2` = prompt tokens per second of TTFT per mm2 of footprint.
    """
    on = SHARED_ON_AIMC and m.shared_inter > 0
    rows = []
    for s in shares:
        slots = [placement(loads[l], s, True) for l in range(idx.shape[0])]
        for mux in muxes:
            for sm in (shared_muxes if on else (None,)):
                for name, (kp, kd, sb) in designs.items():
                    r = evaluate(m, idx, kp, kd, slots, s, npu, P=P, scale_bits=sb, mux=mux, shared_mux=sm)
                    t = r["prefill"]["ttft_ms"]
                    rows.append(dict(design=name, share=s, mux=mux, shared_mux=sm, mem_mm2=r["mem_mm2"],
                                     logic_mm2=r["logic_mm2"], footprint_mm2=r["footprint_mm2"], ttft_ms=t,
                                     energy_mJ=r["prefill"]["energy_mJ"], decode_tok_s=r["decode"]["tok_s"],
                                     tok_s_mm2=P * 1e3 / t / r["footprint_mm2"]))
    return rows


def best(rows, design):
    """The point of `design` with the highest prefill throughput per area."""
    return max((r for r in rows if r["design"] == design), key=lambda r: r["tok_s_mm2"])
