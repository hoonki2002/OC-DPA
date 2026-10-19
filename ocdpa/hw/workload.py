"""The NPU's non-expert work with on-chip accumulation (prefill and decode).

`dense()` prices everything that is not an expert (attention, router, the dense
layers, shared experts that stay on the NPU, the output head) for one step of
`batch` sequences with `seq` new tokens each. It differs from the SCALE-Sim
convention (hw/npu.py `ws_gemm`) in one respect: GEMMs with many rows use
`ws_gemm_tiled`, SCALE-Sim's cycle count with partial sums kept in the OFMAP
SRAM (M is tiled so an M x C tile fits) and each output written once.
SCALE-Sim's convention spills a partial-sum tile per row fold, which put
3.5-13 GB of writes on a 1024-token prefill.
"""
from __future__ import annotations

import math

from ocdpa.hw.npu import (NPU, Model, npu_compute_pj,
                          DRAM_READ_PJ_PER_B, DRAM_WRITE_PJ_PER_B, DRAM_GBPS)

ACC_BYTES = 4


def ws_gemm_tiled(npu: NPU, M, N, K):
    """(cycles, DRAM read B, DRAM write B) with on-chip accumulation, 8-bit operands.

    Three loop orders are considered, each within its own SRAM, and the one
    with the least traffic is taken, as a compiler would pick it:
      output tiles  M_t x N_t partial sums stay in the OFMAP SRAM while K
                    streams; each tile re-reads its A rows and B columns;
      input-resident  M_t full rows of A (M_t x K) stay in the IFMAP SRAM and
                    B streams once per row block;
      weight-resident N_t full columns of B (K x N_t) stay in the FILTER SRAM
                    and A streams once per column block.
    """
    R, C = npu.rows, npu.cols
    rf, cf = math.ceil(K / R), math.ceil(N / C)
    cycles = rf * cf * (2 * R + C + M - 2) - 1
    acc = npu.ofmap_kb * 1024 // ACC_BYTES
    best = None
    for n_t in sorted({min(N, 2 ** i) for i in range(4, 20)}):
        m_t = min(M, max(1, acc // n_t))
        traffic = M * K * math.ceil(N / n_t) + K * N * math.ceil(M / m_t)
        if best is None or traffic < best:
            best = traffic
    m_in = min(M, npu.ifmap_kb * 1024 // K)
    if m_in >= 1:
        best = min(best, M * K + K * N * math.ceil(M / m_in))
    n_w = min(N, npu.filter_kb * 1024 // K)
    if n_w >= 1:
        best = min(best, M * K * math.ceil(N / n_w) + K * N)
    return cycles, best, M * N


def dense(m: Model, npu: NPU, batch: int, seq: int, ctx: int):
    """Non-expert work of one step: `batch` sequences, `seq` new tokens each.

    Projections see all batch*seq rows against one weight; attention runs per
    sequence (and per KV head) against that sequence's own cache.
    Attention is fused (FlashAttention-style): the score tile never leaves the
    chip, so the two attention products move only Q, K, V and the output.
    """
    h, nh, nkv, hd = m.hidden, m.heads, m.kv_heads, m.head_dim
    rep, rows = nh // nkv, batch * seq
    g = []
    for _ in range(m.layers):
        g += [(rows, nh * hd, h, 1, None), (rows, nkv * hd, h, 1, None),
              (rows, nkv * hd, h, 1, None),
              (seq * rep, ctx, hd, nkv * batch, (seq * rep * hd + ctx * hd, 0)),
              (seq * rep, hd, ctx, nkv * batch, (ctx * hd, seq * rep * hd)),
              (rows, h, nh * hd, 1, None), (rows, m.experts, h, 1, None)]
    for li in range(m.layers):
        inter = m.dense_inter if li < m.first_dense else m.shared_inter
        if inter:
            g += [(rows, inter, h, 1, None), (rows, inter, h, 1, None), (rows, h, inter, 1, None)]
    g.append((batch, m.vocab, h, 1, None))
    cyc = rd = wr = 0
    t = comp = 0.0
    for M, N, K, n, fused in g:
        c, r, w = ws_gemm_tiled(npu, M, N, K)
        comp += n * npu_compute_pj(M, N, K, npu.rows, npu.cols)
        if fused is not None:
            r, w = fused
        cyc += n * c; rd += n * r; wr += n * w
        t += n * max(c / npu.clock_ghz, (r + w) / DRAM_GBPS)
    return dict(mJ=(rd * DRAM_READ_PJ_PER_B + wr * DRAM_WRITE_PJ_PER_B + comp) / 1e9,
                compute_mJ=comp / 1e9,
                ms=t / 1e6, compute_ms=cyc / npu.clock_ghz / 1e6)
