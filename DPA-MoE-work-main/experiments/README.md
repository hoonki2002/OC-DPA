# DPA-MoE experiments

Numbered pipeline that reproduces the paper's results. Every `run.sh` is
idempotent (it skips work whose outputs exist; `FORCE=1` redoes it).

```text
experiments/
├── reproduce.sh               bash experiments/reproduce.sh [all|0..4]
├── config/<model>/            base.yaml (model, encode, profile) + bench/kld/speed.yaml
├── 0_build/                   build the llama.cpp fork (DPA_MOE_EXT=ON)
├── 1_download/                HF weights, baseline GGUFs, datasets, BF16 GGUF conversion
├── 2_encode/                  SharedScale AnyBCQ calibration → chunked GGUF
├── 3_dynamic_precision/       offline residual profile R[L, e, K] → runtime .bin
│   └── analysis/              cross-term / budget-error diagnostics (paper appendix)
├── 4_bench/                   multiple-choice accuracy, KLD, speed; figure scripts
├── results/                   tracked result tables used for the paper
└── smoke_test_dpa_moe.sh      one-shot llama-server check of an encoded GGUF
```

## Selecting a model

Each model has one config directory. Steps 2–3 take `CONFIG`, step 4 takes
`ROOT_CONFIG` (model) and `BENCH_CONFIG` (benchmark settings, default
`bench.yaml` next to the root config):

```bash
CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml bash experiments/2_encode/run.sh
ROOT_CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml bash experiments/4_bench/run.sh
```

Without overrides every step uses `experiments/config/Qwen3-30B-A3B/`.

## Steps

| # | Step | Reads | Writes |
|---|------|-------|--------|
| 0 | `0_build/` | `third_party/llama.cpp/` | binaries in `third_party/llama.cpp/build-dpa-moe/bin/` |
| 1 | `1_download/` | model keys on the command line | `cache/models/<model>/{hf,baseline}/`, `cache/datasets/` |
| 2 | `2_encode/` | HF weights, calibration corpus | `cache/models/<model>/ss_anybcq_layer/` codebooks, `cache/models/<model>/chunked/ss_anybcq.gguf` |
| 3 | `3_dynamic_precision/` | HF weights, step-2 codebooks, profile corpus | `expert_residuals.npz` and the runtime `expert_residuals.bin` (paths in `base.yaml: profile`) |
| 4 | `4_bench/` | step-2 GGUF, step-3 profile, baseline GGUFs | `experiments/results/<model>/{bench,kld,speed}/` |

`reproduce.sh` runs step 1 without arguments, which downloads datasets only;
download models and baselines first with
`bash experiments/1_download/run.sh --models qwen3 --baseline`.

Not redistributed: the calibration corpora `calibration_v3.txt` and
`calibration_v5.txt` listed (with checksums) in each `base.yaml`.

## Paper figures and appendices

- KLD vs effective weight: `4_bench/analysis/plot.py` (uses
  `4_bench/kld_sweep/baseline.py` for the static-quantization points).
- Accuracy vs effective weight: `4_bench/analysis/plot_bench_scores.py`;
  `4_bench/analysis/aggregate_bench_scores.py` collects all result files into
  `experiments/results/paper_figures/`.
- Cross-term analysis: `3_dynamic_precision/analysis/run_error_diagnostics.sh`.

## Reproducibility notes

- Seeds come from `base.yaml: seed` (profile, benchmarks).
- `base.yaml: hardware` records the GPUs used for the paper runs; GPU
  selections in the configs can be overridden with `GPUS=` / `GPU=`.
- Logs from multi-process steps go to `cache/logs/`.
