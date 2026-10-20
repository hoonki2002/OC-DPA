"""The digital half of the chip: everything that is not an expert.

Only the expert matrices go on the analog array. Attention, the router, the
LM head and the normalisations stay digital, following the heterogeneous
split that Chowdhury et al. argue for -- the dense modules are a small share
of the parameters but see every token, so analog approximation costs more
there than it saves.

That half is modelled as an NPU with its own SRAM in front of an LPDDR4
channel: SCALE-Sim walks each GEMM over a weight-stationary systolic array
and reports cycles and off-chip traffic, and DRAMsim3 prices the traffic.
The two halves are therefore costed by published simulators rather than by
one spreadsheet, and the analog side keeps NeuroSim.

Decode is the case that matters here. With one token in flight every GEMM is
a GEMV, each weight is used exactly once, and nothing is left to amortise the
fetch -- so the dense path's cost is set by how its weights are stored, not
by how fast the array is.
"""
from __future__ import annotations

import csv, math
from dataclasses import dataclass
from pathlib import Path

DRAM_READ_PJ_PER_B = 58.80
DRAM_WRITE_PJ_PER_B = 54.06
DRAM_GBPS = 18.04

NPU_PRIMS = {
    7:  dict(pj_per_mac=0.122233, sram_rd_pj_per_b=0.028123, sram_wr_pj_per_b=0.016171, area_mm2=0.6005 + 3 * 0.2470),
    22: dict(pj_per_mac=0.398704, sram_rd_pj_per_b=0.130264, sram_wr_pj_per_b=0.071002, area_mm2=4.2524 + 3 * 2.4361),
}
from ocdpa.hw.stack import LOGIC_NODE
_P = NPU_PRIMS[LOGIC_NODE]
NPU_PJ_PER_MAC = _P["pj_per_mac"]
SRAM_RD_PJ_PER_B = _P["sram_rd_pj_per_b"]
SRAM_WR_PJ_PER_B = _P["sram_wr_pj_per_b"]
NPU_AREA_MM2 = _P["area_mm2"]


def npu_compute_pj(M, N, K, rows=128, cols=128, acc_bytes=4):
    """MAC and on-chip SRAM energy of one weight-stationary GEMM.

    SRAM traffic follows SCALE-Sim's access counts: every IFMAP element is
    read once per column fold, every weight once, and the partial sums are
    read and written once per row fold.
    """
    rf, cf = math.ceil(K / rows), math.ceil(N / cols)
    sram = (M * K * cf + K * N) * SRAM_RD_PJ_PER_B + \
           M * N * rf * acc_bytes * (SRAM_RD_PJ_PER_B + SRAM_WR_PJ_PER_B)
    return M * N * K * NPU_PJ_PER_MAC + sram


@dataclass
class Model:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    topk: int
    moe_inter: int
    vocab: int
    shared_inter: int = 0
    first_dense: int = 0
    dense_inter: int = 0

    @property
    def n_moe(self):
        return self.layers - self.first_dense

    def dense_gemms(self, seq: int, ctx: int):
        """(name, M, N, K, count) for every non-expert GEMM of one forward step.

        M is the number of tokens in flight: 1 for decode, the prompt length
        for prefill. The two attention products read the KV cache rather than
        a weight matrix; each KV head owns its own cache, so they are issued
        once per KV head with the query heads that share it stacked along M.
        """
        h, nh, nkv, hd = self.hidden, self.heads, self.kv_heads, self.head_dim
        rep = nh // nkv
        g = []
        for li in range(self.layers):
            g += [(f"L{li}.q", seq, nh * hd, h, 1),
                  (f"L{li}.k", seq, nkv * hd, h, 1),
                  (f"L{li}.v", seq, nkv * hd, h, 1),
                  (f"L{li}.scores", seq * rep, ctx, hd, nkv),
                  (f"L{li}.context", seq * rep, hd, ctx, nkv),
                  (f"L{li}.o", seq, h, nh * hd, 1),
                  (f"L{li}.router", seq, self.experts, h, 1)]
            inter = self.dense_inter if li < self.first_dense else self.shared_inter
            if inter:
                g += [(f"L{li}.mlp_gate", seq, inter, h, 1), (f"L{li}.mlp_up", seq, inter, h, 1),
                      (f"L{li}.mlp_down", seq, h, inter, 1)]
        g.append(("lm_head", seq, self.vocab, h, 1))
        return g

    def dense_weight_bytes(self, wbits=8):
        """Weights the dense path must fetch for one forward step."""
        h, nh, nkv, hd = self.hidden, self.heads, self.kv_heads, self.head_dim
        per_layer = h * nh * hd + 2 * h * nkv * hd + nh * hd * h + h * self.experts
        mlp = (self.first_dense * 3 * h * self.dense_inter +
               self.n_moe * 3 * h * self.shared_inter)
        return (self.layers * per_layer + mlp + h * self.vocab) * wbits / 8

    def kv_bytes(self, ctx, wbits=8):
        return self.layers * 2 * ctx * self.kv_heads * self.head_dim * wbits / 8

    def expert_conversions(self, kbar, input_bits=8):
        """Analog conversions for one token, for comparison."""
        groups = 3 * self.topk * self.hidden * self.moe_inter / 128
        return self.n_moe * groups * kbar * input_bits


MODELS = [
    Model("OLMoE-1B-7B", 16, 2048, 16, 16, 128, 64, 8, 1024, 50304),
    Model("DeepSeek-MoE-16B", 28, 2048, 16, 16, 128, 64, 6, 1408, 102400,
          shared_inter=2 * 1408, first_dense=1, dense_inter=10944),
    Model("Qwen3-30B-A3B", 48, 2048, 32, 4, 128, 128, 8, 768, 151936),
]


@dataclass
class NPU:
    """A mobile-class weight-stationary systolic array."""
    rows: int = 128
    cols: int = 128
    ifmap_kb: int = 256
    filter_kb: int = 256
    ofmap_kb: int = 128
    clock_ghz: float = 1.0

    def config(self, run_name, topo):
        cfg = f"""[general]
run_name = {run_name}

[architecture_presets]
ArrayHeight:    {self.rows}
ArrayWidth:     {self.cols}
IfmapSramSzkB:  {self.ifmap_kb}
FilterSramSzkB: {self.filter_kb}
OfmapSramSzkB:  {self.ofmap_kb}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Dataflow:       ws
Bandwidth:      10
MemoryBanks:    1
ReadRequestBuffer: 60
WriteRequestBuffer: 60

[layout]
IfmapCustomLayout: false
FilterCustomLayout: false
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport: false
SparseRep: ellpack_block
OptimizedMapping: false
BlockSize: 4
RandomNumberGeneratorSeed: 0

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: false
"""
        return cfg


def run_scalesim(model: Model, npu: NPU, seq: int, ctx: int, out_dir: str):
    """Returns per-layer cycles and off-chip traffic from SCALE-Sim."""
    from scalesim.scale_sim import scalesim
    work = Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    topo = work / "topo.csv"
    with open(topo, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Layer", "M", "N", "K", ""])
        for name, M, N, K, n in model.dense_gemms(seq, ctx):
            for i in range(n):
                w.writerow([f"{name}.{i}" if n > 1 else name, M, N, K, ""])
    cfg = work / "arch.cfg"
    cfg.write_text(npu.config("dense", str(topo)))
    sim = scalesim(save_disk_space=True, verbose=False, config=str(cfg),
                   topology=str(topo), input_type_gemm=True)
    sim.run_scale(top_path=str(work))
    return work


def ws_gemm(npu: NPU, M: int, N: int, K: int):
    """SCALE-Sim 2.0.2's weight-stationary GEMM, in closed form.

    SCALE-Sim materialises a (cycles x array-width) demand matrix per layer,
    which for a 50k-vocab LM head or a 1024-token prefill runs past 60 GB of
    RAM. Its schedule is deterministic, so the same numbers follow from the
    fold counts. Checked against the simulator on 7 GEMMs (M 1..512) at
    32x32/4 kB and 64x64/16 kB: cycles and DRAM filter/IFMAP reads match
    exactly, OFMAP writes to within the fixed drain tail (<= C-1 words).

    Returns (compute cycles, DRAM read words, DRAM write words). Note the
    simulator spills a partial-sum tile to DRAM for every row fold rather
    than accumulating on chip; that convention is kept here.
    """
    R, C = npu.rows, npu.cols
    rf, cf = math.ceil(K / R), math.ceil(N / C)
    cycles = rf * cf * (2 * R + C + M - 2) - 1
    ifmap_words = npu.ifmap_kb * 1024
    ifmap = M * K * (cf if M * K >= ifmap_words / 2 else 1)
    return cycles, ifmap + K * N, M * N * rf


def dense_step(model: Model, npu: NPU, seq: int, ctx: int, wbytes=1):
    """Cycles, DRAM bytes and DRAM-bounded latency of the dense path.

    Each GEMM takes the longer of its compute time and its DRAM transfer
    time on the measured LPDDR4 channel, so decode comes out memory-bound
    and long prefill compute-bound without either being assumed.
    """
    cyc = rd = wr = 0
    t_ns = 0.0
    comp_pj = 0.0
    for _, M, N, K, n in model.dense_gemms(seq, ctx):
        c, r, w = ws_gemm(npu, M, N, K)
        cyc += n * c; rd += n * r * wbytes; wr += n * w * wbytes
        comp_pj += n * npu_compute_pj(M, N, K, npu.rows, npu.cols)
        t_ns += n * max(c / npu.clock_ghz, (r + w) * wbytes / DRAM_GBPS)
    e_mj = (rd * DRAM_READ_PJ_PER_B + wr * DRAM_WRITE_PJ_PER_B + comp_pj) / 1e9
    return dict(cycles=cyc, rd_B=rd, wr_B=wr, dram_mJ=e_mj, compute_mJ=comp_pj / 1e9,
                compute_ms=cyc / npu.clock_ghz / 1e6, latency_ms=t_ns / 1e6)


if __name__ == "__main__":
    print("  Per-token dense-path weight traffic if it is fetched from DRAM")
    print(f"  (LPDDR4-2400 x64, {DRAM_READ_PJ_PER_B:.1f} pJ/B, {DRAM_GBPS:.1f} GB/s)\n")
    print(f"  {'model':16s} {'dense MB/tok':>13} {'DRAM mJ/tok':>12} {'DRAM ms/tok':>12} "
          f"{'AIMC uJ/tok':>12} {'ratio':>9}")
    for m in MODELS:
        mb = m.dense_weight_bytes() / 1e6
        mj = m.dense_weight_bytes() * DRAM_READ_PJ_PER_B / 1e9
        ms = m.dense_weight_bytes() / (DRAM_GBPS * 1e9) * 1e3
        uj = m.expert_conversions(4) * 0.3421 / 1e6
        print(f"  {m.name:16s} {mb:13.1f} {mj:12.3f} {ms:12.2f} {uj:12.1f} {mj*1e3/uj:8.0f}x")


NPU_CFG = dict(ifmap_kb=2048, filter_kb=2048, ofmap_kb=2048)
