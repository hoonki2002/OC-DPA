# Step 1 — Download

Pulls HF model weights, the static-quantization baseline GGUFs, and the
datasets used for calibration, KLD and speed measurements.

```bash
bash experiments/1_download/run.sh                                   # datasets only (default)
bash experiments/1_download/run.sh --models qwen3 --baseline         # one model + baselines
bash experiments/1_download/run.sh --all                             # both models + baselines
bash experiments/1_download/run.sh --models deepseek_moe --no-datasets
```

## Models

| key | HF repo | local_dir |
|-----|---------|-----------|
| `qwen3` | `Qwen/Qwen3-30B-A3B` | `cache/models/Qwen3-30B-A3B/hf` |
| `deepseek_moe` | `deepseek-ai/deepseek-moe-16b-chat` | `cache/models/deepseek-moe-16b-chat/hf` |

Baseline GGUFs (`--baseline`):

| key | HF repo | files |
|-----|---------|-------|
| `qwen3` | `unsloth/Qwen3-30B-A3B-GGUF` | full repo (BF16 + quants) |
| `deepseek_moe` | `mradermacher/deepseek-moe-16b-chat-GGUF` | `Q2_K`, `Q5_K_M`, `Q8_0` |

The KLD reference needs a BF16 GGUF at the config's `model.bf16_gguf`. Qwen3's
comes with the unsloth baseline repo; for DeepSeek-MoE convert it from HF:

```bash
bash experiments/1_download/convert_bf16.sh experiments/config/deepseek-moe-16b-chat/base.yaml
```

## Datasets

| Path | Source | Used by |
|------|--------|---------|
| `cache/datasets/wikitext-2-raw-test.txt` | `Salesforce/wikitext` (`wikitext-2-raw-v1/test`) | KLD / perplexity |
| `cache/datasets/wikitext-2-raw-train.txt` | `Salesforce/wikitext` (`wikitext-2-raw-v1/train`) | encoder calibration, residual profile |
| `cache/datasets/c4_{en,ko}_validation.txt` | `allenai/c4` (en, multilingual ko) | speed prompts |
| `cache/datasets/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json` | ShareGPT (HF mirror) | speed prompts |
| `cache/datasets/calibration_v{3,5}.txt` | not redistributed | encoder calibration |

Place the calibration corpora at the paths (and sha256) listed in
`experiments/config/<model>/base.yaml` before step 2; this script warns when
they are missing.

## Idempotency

- Models: skipped if a `*.index.json` weight index exists in `local_dir`.
- Baseline GGUFs: skipped if any `*.gguf` exists in `local_dir`.
- Datasets: skipped if the output file is non-empty (`TPS_FORCE=1` refreshes
  the speed-prompt sets).
