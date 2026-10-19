# DPA-MoE

**DPA-MoE: A Runtime Precision Control Framework for MoE Inference.**

Mixture-of-Experts (MoE) models route each token to a few experts, but every
selected expert still runs at one fixed precision. DPA-MoE makes routed
experts *any-precision*:

```text
The router decides which experts to use.
DPA-MoE decides how many precision chunks of each selected expert to use.
```

Expert FFN weights are encoded once as **prefix-decodable chunks** with
SharedScale AnyBCQ: using `k` chunks reconstructs an expert at `k`-bit
fidelity, and raising precision only appends a chunk. Per-expert
reconstruction residuals are profiled offline, and at inference time a
**Dynamic Precision Allocator (DPA)** combines them with the live router gate
scores to split a global chunk budget across the active experts. One encoded
checkpoint therefore covers many weight-size/quality operating points, with no
retraining or re-encoding.

The single global dial is

```text
KBar = mean chunks per active expert        (2 ≤ KBar ≤ 8 for the released encodings)
```

## How allocation works

For each MoE dispatch the allocator solves

```text
minimize   Σ_e g[e]² · R[layer, e, K[e]]
subject to Σ_e K[e] = round(KBar · n_active),   K_min ≤ K[e] ≤ K_max
```

where `g[e]²` is the squared gate score of active expert `e` (aggregated over
the routed tokens in a batch), `R[layer, e, K]` is the offline residual
profile, and `K[e]` is the chunk count given to expert `e`. This is a
multiple-choice knapsack over the active experts; residual curves are first
projected to be monotone in `K`, and the DP is exact for that projected table.
The router's expert selection is never changed.

The **Uniform-k** baseline uses the same encoded artifact and execution path
but gives every active expert `round(KBar)` chunks, which isolates the effect
of the allocation policy.

## Repository layout

```text
src/dpa_moe/                 encoder: SharedScale AnyBCQ calibration, chunk packing,
                             GGUF assembly (CLI: dpa-moe-encode)
third_party/llama.cpp/       llama.cpp fork with the DPA-MoE runtime (git submodule)
    dpa-moe-ext/core/          runtime, chunked VRAM pool, GGUF stream reader
    dpa-moe-ext/decoder/       AnyBCQ / SharedScale AnyBCQ chunk matmul kernels
    dpa-moe-ext/moe/           MoE scheduler, KBar allocator, per-architecture executors
third_party/anybcq/          vendored AnyBCQ (NAVER Cloud), with the changes listed
                             in third_party/anybcq/NOTICE.md
experiments/                 reproduction pipeline (steps 0–4) and configs
tests/                       unit tests
```

Supported architectures: Qwen3-MoE (Qwen3-30B-A3B) and DeepSeek-MoE
(DeepSeekMoE-16B-Chat).

## Installation

Requirements: Linux, an NVIDIA GPU with a CUDA toolkit (`nvcc`), CMake ≥ 3.18,
Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recursive https://github.com/kimGreg/DPA-MoE.git
cd DPA-MoE
uv sync                                   # Python encoder + dependencies (.venv)
bash experiments/0_build/run.sh           # llama.cpp fork with DPA_MOE_EXT=ON
uv run pytest                             # unit tests
```

## Reproducing the experiments

Each model is described by one config, `experiments/config/<model>/base.yaml`,
plus benchmark settings in `bench.yaml`, `kld.yaml` and `speed.yaml` next to it.

```bash
bash experiments/1_download/run.sh --models qwen3 --baseline   # weights, baselines, datasets
bash experiments/2_encode/run.sh                               # codebooks + chunked GGUF
bash experiments/3_dynamic_precision/run.sh                    # residual profile R[L, e, K]
bash experiments/4_bench/run.sh                                # multiple-choice, KLD, (speed)
```

Select a model with `CONFIG=` (steps 2–3) / `ROOT_CONFIG=` (step 4), e.g.
`CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml`. See
[experiments/README.md](experiments/README.md) for every step, its inputs and
outputs, and the analysis scripts behind the paper's figures and appendices.

The calibration corpora `calibration_v3.txt` / `calibration_v5.txt` are not
redistributed; step 1 warns when they are missing.

## Running a chunked model

The encoder writes a GGUF that the fork's tools (`llama-server`,
`llama-perplexity`, `llama-cli`, …) load directly. Runtime behaviour is set
with environment variables:

```bash
DPA_MOE_KBAR=4                                   # global chunk budget
DPA_MOE_KBAR_ALLOCATOR=profile                   # profile-backed DPA (or: uniform)
DPA_MOE_KBAR_PROFILE_FILE=cache/experiment/<model>/3_dynamic_precision/bin/expert_residuals.bin
DPA_MOE_VRAM_CAP_MB=8000                         # VRAM pool for expert chunks
```

Two execution modes share the same allocation semantics:

- **Full-pinned capture** (`DPA_MOE_PIN_ALL=1 DPA_MOE_ALLOW_CAPTURE=1`): all
  chunks resident in VRAM, CUDA graphs enabled; used for quality and KLD sweeps.
- **Dynamic** (`DPA_MOE_VRAM_CAP_MB=...`): chunks are loaded on demand through
  the VRAM pool; used to measure runtime overhead under a memory cap.

Phase-aware budgets switch `KBar` between prompt processing and generation
(applied once per graph/batch, not per token inside a fused prefill batch):

```bash
DPA_MOE_KBAR_PP_BEGIN=7 DPA_MOE_KBAR_PP_END=6
DPA_MOE_KBAR_TG_BEGIN=4 DPA_MOE_KBAR_TG_END=2
DPA_MOE_KBAR_TG_DECAY_START=64 DPA_MOE_KBAR_TG_DECAY_END=128
```

The allocator keeps per-dispatch precision state that is not request-scoped,
so dynamic and scheduled configurations are benchmarked with one request in
flight (`-np 1`).

## Limitations

The llama.cpp runtime is a research prototype. Allocation and variable chunk
materialization are not fused into the router or MoE kernels, so DPA adds
steady-state throughput and time-to-first-token overhead relative to
Uniform-k; report it as prototype overhead. DPA-MoE is an allocation framework
and is complementary to expert offloading, prefetching and caching systems.

## Citation

```bibtex
@inproceedings{kim2026dpamoe,
  title  = {{DPA-MoE}: A Runtime Precision Control Framework for {MoE} Inference},
  author = {Kim, Jaeyun and Lee, Hoonki and Shin, Susie and Kim, Hyunmin and
            Lee, Dahyun and Kim, Sangyeob},
  year   = {2026}
}
```

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Third-party components keep their
own licenses: AnyBCQ (Apache-2.0) and llama.cpp (MIT); see [NOTICE](NOTICE).
