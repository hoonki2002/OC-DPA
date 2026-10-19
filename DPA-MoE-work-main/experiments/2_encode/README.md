# Step 2 — Encode

Turns an HF MoE checkpoint into a DPA-MoE GGUF. Routed-expert FFN weights
(gate/up/down) are encoded as prefix-decodable AnyBCQ chunk streams; all other
weights (attention, embeddings, lm_head, router, norms) are written inline
(Q8_0 / F32) during assembly.

```bash
bash experiments/2_encode/run.sh                                     # Qwen3-30B-A3B
CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml bash experiments/2_encode/run.sh
```

`run.sh` calibrates codebooks with `calibrate_parallel.sh`, then runs
`dpa-moe-encode assemble` and `dpa-moe-encode finalize`. The algorithm,
on-disk layout and calibration details are in
[ENCODING_ALGORITHM.md](ENCODING_ALGORITHM.md).

## Variants

| Variant | Algorithm | Storage |
|---------|-----------|---------|
| `ss_anybcq` (default) | SharedScale AnyBCQ: incremental block reconstruction with one shared α/β table; at each precision tier the active α planes are jointly retrained with a previous-K preservation loss. | `FLAG_ANY_PREC=0`; shared α/β + bit planes. |
| `anybcq` | Standard AnyBCQ with per-tier α/β snapshots (storage-overhead comparison). | `FLAG_ANY_PREC=1`; larger GGUF. |

The variant comes from `encode.variant` in the config; override with
`DPA_MOE_VARIANT=anybcq`.

## Calibration modes

- `layer_fp_parallel` (used by `calibrate_parallel.sh`): one process per GPU,
  each calibrating a disjoint range of MoE layers against BF16 cached inputs.
  Fast; an AdaQuant-style approximation of the layer-sequential chain.
- `sequential`: the original AnyBCQ order, where each layer sees the quantized
  outputs of earlier layers. Single GPU:

  ```bash
  CUDA_VISIBLE_DEVICES=0 .venv/bin/dpa-moe-encode calibrate \
      --config experiments/config/Qwen3-30B-A3B/base.yaml --calibration-mode sequential
  ```

Leading dense decoder layers (`first_k_dense_replace` in the HF config, e.g.
DeepSeek-MoE's layer 0) have no experts and are skipped.

## Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONFIG` | Qwen3-30B-A3B `base.yaml` | Model config. |
| `GPUS` | `encode.gpus`, else all visible GPUs | GPU IDs for calibration. |
| `INNER_BS` | `encode.inner_bs` | Inner batch size inside block reconstruction. |
| `FP_CACHING` | `1` | Calibrate against BF16 cached layer inputs. |
| `SKIP_CALIBRATE` / `SKIP_ASSEMBLE` / `FORCE` | `0` | Skip or redo phases. |
| `DPA_MOE_ITERS` | `encode.hyperparams.iters` | Override block-reconstruction iterations. |
| `DPA_MOE_QUANT_ROUNDS` | `20` | BCQ initialization / add-one-bit rounds (vendored AnyBCQ). |
| `DPA_MOE_HF_LOAD_DEVICE` | `cuda:0` | Device for loading the HF model during assembly. |

## Outputs

Paths come from the config's `model.codebooks` / `model.ggufs`, e.g. for Qwen3:

| Variant | Codebooks | Final GGUF |
|---------|-----------|------------|
| `ss_anybcq` | `cache/models/Qwen3-30B-A3B/ss_anybcq_layer/` | `cache/models/Qwen3-30B-A3B/chunked/ss_anybcq.gguf` |
| `anybcq` | `cache/models/Qwen3-30B-A3B/anybcq_layer/` | `cache/models/Qwen3-30B-A3B/chunked/anybcq.gguf` |

Calibration logs: `cache/logs/2_encode/`.
