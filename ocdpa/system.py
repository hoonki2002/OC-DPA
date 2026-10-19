"""Whole-system decode cost per token: the proposal against an NPU that pages
its experts from flash.

Both systems run every non-expert layer the same way -- an NPU with the dense
weights and KV cache resident in LPDDR4 (npu.dense_step) -- and differ only in
where the routed experts are computed:

  B   NPU. Experts live on UFS 4.0 flash because on-device DRAM cannot hold
      them; a fixed DRAM budget caches recently used bit-planes (LRU over
      (layer, expert, plane)). A miss reads flash, writes DRAM, then the NPU
      reads DRAM. Planes are stored separately, so with an allocator the NPU
      only touches the planes it uses -- this is DPA-MoE's own setting.
  P   The proposal. All experts resident in a 16-tier heterogeneous-3D array,
      ADCs on the logic tier shared by 8 columns and all 16 tiers (h3d.Stack
      with s=16). No expert byte moves.

Routing and gates are real (trace.py); plane counts come from the uniform
allocator or from DPA's marginal-gain allocator on the measured residual
table, so each token gets the plan the hardware would make.

Left out of both sides, and called out where the numbers are printed: NPU MAC
energy (the baseline also runs the expert GEMVs on it, so omitting it favours
the baseline), the per-group scale storage, and prefill.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from ocdpa.allocators import batched_greedy
from ocdpa.hw.stack import Stack, stack_latency
from ocdpa.hw.npu import (NPU, ws_gemm, npu_compute_pj,
                 DRAM_READ_PJ_PER_B, DRAM_WRITE_PJ_PER_B, DRAM_GBPS)

SSD_GBPS = 4.2                              # Samsung UFS 4.0, sequential read
SSD_PJ_PER_B = 2.5 / 6.0e6 * 1e-3 * 1e12    # 6.0 MB/s per mA at VCC 2.5 V
from ocdpa.hw.stack import LOGIC_NODE
# alpha multiply per column and the latency of the multiplier, NeuroSim scaleunit.cpp (SU_TECHNODE) at the logic node
ALPHA_PJ = {22: 0.383892, 7: 0.105007}[LOGIC_NODE]
ALPHA_MULT_NS = {22: 1.5186, 7: 0.6345}[LOGIC_NODE]
INPUT_BITS = 8


def plans(idx, gate, residual, kbar, topk, dpa):
    """(L, T, topk) plane counts."""
    if not dpa:
        return np.full(idx.shape, kbar, dtype=np.int64)
    out = []
    for l in range(idx.shape[0]):
        cost = (torch.as_tensor(gate[l]) ** 2).unsqueeze(-1) * \
               torch.as_tensor(residual[l])[torch.as_tensor(idx[l]).long()]
        out.append(batched_greedy(cost.double(), kbar * topk).numpy())
    return np.stack(out)


def npu_experts(m, idx, k, cache_bytes, warm):
    """Mean per-token flash/DRAM energy (mJ) and latency (ms) of the experts."""
    plane_b = 3 * m.hidden * m.moe_inter / 8          # one binary plane of one expert
    cap = int(cache_bytes // plane_b)
    lru: OrderedDict = OrderedDict()
    L, T, _ = idx.shape
    npu = NPU()
    c_gu, _, _ = ws_gemm(npu, 1, m.moe_inter, m.hidden)
    c_d, _, _ = ws_gemm(npu, 1, m.hidden, m.moe_inter)
    compute_ns = (2 * c_gu + c_d) / npu.clock_ghz      # one expert, planes merged
    mac_pj = (2 * npu_compute_pj(1, m.moe_inter, m.hidden) +
              npu_compute_pj(1, m.hidden, m.moe_inter))
    e = t = 0.0
    hits = refs = 0
    for tok in range(T):
        for l in range(L):
            miss_b = read_b = 0.0
            for j, ex in enumerate(idx[l, tok]):
                for p in range(int(k[l, tok, j])):
                    key = (l, int(ex), p)
                    read_b += plane_b
                    if key in lru:
                        lru.move_to_end(key); h = 1
                    else:
                        miss_b += plane_b; h = 0
                        lru[key] = None
                        if len(lru) > cap:
                            lru.popitem(last=False)
                    if tok >= warm:
                        hits += h; refs += 1
            if tok >= warm:
                e += (miss_b * (SSD_PJ_PER_B + DRAM_WRITE_PJ_PER_B) + read_b * DRAM_READ_PJ_PER_B
                      + m.topk * mac_pj)
                # flash must land before the router's experts can start; the
                # DRAM stream then overlaps the GEMVs of the same layer
                t += miss_b / SSD_GBPS + max(read_b / DRAM_GBPS, m.topk * compute_ns)
    n = T - warm
    return e / n / 1e9, t / n / 1e6, hits / max(refs, 1)


def aimc_experts(m, idx, k, st: Stack, warm):
    """Mean per-token energy (mJ) and latency (ms) of the experts on the stack."""
    groups = 3 * m.hidden * m.moe_inter / 128           # column-groups per expert
    pj = st.pj_per_conv()["total"]
    kk = k[:, warm:]
    conv = groups * INPUT_BITS * kk.sum() / kk.shape[1]
    alpha = groups * kk.sum() / kk.shape[1] * ALPHA_PJ   # one multiply per plane
    e_mj = (conv * pj + alpha) / 1e9
    lat = 0.0
    for l in range(idx.shape[0]):
        # gate and up run side by side, then down: two serial stages per layer
        lat += 2 * stack_latency(st, m.experts, m.topk, k[l, warm:], idx[l, warm:])
    return e_mj, lat / 1e6
