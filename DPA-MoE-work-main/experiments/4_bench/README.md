# Step 4 — Benchmarks

```bash
bash experiments/4_bench/run.sh
```

Phases (each runs every configuration listed under `benchmark.config` in the
benchmark config, one process per GPU from `benchmark.gpu`):

| Phase | Script | Measures | Skip with |
|-------|--------|----------|-----------|
| `llama_mc` | `llama_complexity/run_bench.py` | ARC-C/E, PIQA, WinoGrande accuracy (`llama-perplexity --multiple-choice`) | `SKIP_LLAMA_MC=1` |
| `kld` | `kld_sweep/run_wikitext_ppl.py` | KLD / perplexity vs the BF16 reference on wikitext-2 test | `SKIP_KLD=1` |
| `speed` | `speed_sweep/run_sweep.py` | generation TPS and TTFT, single request | on only with `SKIP_SPEED=0` |

The KLD phase first runs `kld_sweep/gen_ref.sh`, which writes the BF16
reference logits once per model.

`ROOT_CONFIG` selects the model (`experiments/config/<model>/base.yaml`) and
`BENCH_CONFIG` the benchmark settings (default: `bench.yaml` next to it):

```bash
# Downstream accuracy
BENCH_CONFIG=experiments/config/Qwen3-30B-A3B/bench.yaml SKIP_KLD=1 bash experiments/4_bench/run.sh
# KLD sweep
BENCH_CONFIG=experiments/config/Qwen3-30B-A3B/kld.yaml SKIP_LLAMA_MC=1 bash experiments/4_bench/run.sh
# Speed sweep
BENCH_CONFIG=experiments/config/Qwen3-30B-A3B/speed.yaml SKIP_LLAMA_MC=1 SKIP_KLD=1 SKIP_SPEED=0 \
    bash experiments/4_bench/run.sh
```

Configurations: `Q2_K` / `Q5_K_M` / `Q8_0` (static baselines),
`uniform_k<k>` (Uniform-k), `dynamic_k<KBar>` (DPA), and `sched_*`
(phase-aware KBar schedules).

Results:

```text
experiments/results/<model>/bench/<config>/llama_mc/
experiments/results/<model>/kld/<config>/data/        kld.json, summary.csv, provenance.json
experiments/results/<model>/speed/<config>/<setting>/
```

Logs: `cache/logs/4_bench/`.

## Inputs

- Multiple-choice data: `cache/datasets/{arc_c,arc_e,piqa,winogrande}.jsonl`,
  built by `.venv/bin/python experiments/4_bench/llama_complexity/prepare_data.py`
  (ARC-Challenge test, ARC-Easy / PIQA / WinoGrande validation; the
  generated `.mc.bin` files are byte-identical to those of the paper runs).
- Dynamic configs read the residual profile at `benchmark.dynamic_profile_file`.

## Figures

```bash
.venv/bin/python experiments/4_bench/kld_sweep/baseline.py          # baseline-quant KLD points
.venv/bin/python experiments/4_bench/analysis/plot.py               # KLD vs effective weight
.venv/bin/python experiments/4_bench/analysis/plot_bench_scores.py  # accuracy vs effective weight
.venv/bin/python experiments/4_bench/analysis/aggregate_bench_scores.py
```

## Notes

Dynamic and scheduled configs run single-flight (`-np 1`): the allocator's
precision state is per dispatch, not per request. The speed sweep also
disables prompt caching; its setting names encode prompt/generation length,
ubatch and VRAM cap (e.g. `p512_g128_ub256_cap6g`).
