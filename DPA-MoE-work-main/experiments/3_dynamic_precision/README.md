# Step 3 — Dynamic precision profile

Builds the offline residual profile that the Dynamic Precision Allocator
combines with live router gate scores at inference time.
[PROBLEM.md](PROBLEM.md) is the specification of the allocation problem.

```bash
bash experiments/3_dynamic_precision/run.sh                          # Qwen3-30B-A3B
CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml bash experiments/3_dynamic_precision/run.sh
```

1. `profiling/profile_residuals.py` measures the gate-free residual
   `R[L, e, K] = E_calib ||ΔE_e^(K)(x)||²` for every layer `L`, expert `e` and
   chunk count `K`, using the step-2 codebooks and routed calibration tokens.
   Output: `expert_residuals.npz` (`profile.residuals_npz` in the config).
2. `profiling/pack.py` converts it to the binary read by the runtime
   (`profile.runtime_bin`), passed as `DPA_MOE_KBAR_PROFILE_FILE`.

KLD, accuracy and speed are measured in step 4.

## Budget

```text
KBar            = mean chunks per active expert
logical chunks  = Σ_e K[e] ≈ KBar × unique active experts
physical chunks = 3 × logical chunks          # gate + up + down
```

Decode processes one token, so `unique active experts ≤ top_k`. Batched
prefill uses the union of experts across the batch, which can approach the
full expert count: prefill at KBar=4 on Qwen3 can need about `3 × 80 × 4 = 960`
chunks per layer, decode about `3 × 8 × 4 = 96`.

## Calibration policy

- The profile is gate-free; gate scores are applied only at allocation time.
- Use a fixed seed (`seed` in the config, passed as `--seed`).
- Use train/dev data only, disjoint from the benchmark splits in step 4.
- For multiple-choice scoring, profile on `prompt + candidate` text for every
  candidate (llama.cpp scores all of them). Build such a corpus with

  ```bash
  .venv/bin/python experiments/3_dynamic_precision/profiling/prepare_mc_calibration.py \
      --seed 0 --out cache/datasets/mc_train_candidates_seed0.txt
  ```

  and add it to `profile.corpus` in the config before profiling.

## Layout

```text
3_dynamic_precision/
├── PROBLEM.md                   allocation problem specification
├── run.sh                       profile → pack
├── _config.py                   config-backed paths (also used by step-4 KLD tools)
├── profiling/
│   ├── profile_residuals.py     R[L, e, K] measurement
│   ├── _collector.py            routed-activation collection hooks
│   ├── _compute.py              chunk unpacking + residual computation
│   ├── merge.py                 merge sharded (token- or layer-parallel) profiles
│   ├── pack.py                  npz → runtime .bin
│   └── prepare_mc_calibration.py
└── analysis/                    diagnostics behind the paper's appendix
    ├── run_error_diagnostics.sh exact vs diagonal routed error per KBar
    ├── cross_term_diagnostic.py exact / diagonal / profile-diagonal error ratios
    ├── budget_error_diagnostic.py same-budget DPA vs Uniform-k residual error
    ├── expert_output_cross_term.py cross-expert terms of BF16 expert outputs
    └── check_allocator.py       sanity checks of a greedy allocator port on a real profile
```

The runtime allocator itself (exact DP over the monotone-projected profile)
lives in `third_party/llama.cpp/dpa-moe-ext/moe/common/moe_scheduler.cpp`; its
unit test is `dpa_moe-kbar-planner-test`.
