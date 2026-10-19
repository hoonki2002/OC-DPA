#!/usr/bin/env bash
# Step 1 — Download HF model weights, optional baseline GGUFs, and the
# shared calibration / KLD-eval datasets used by the encoder pipeline.
#
# Usage:
#   bash experiments/1_download/run.sh [options]
#
# Options:
#   --models <m1,m2,...>   Comma list. Keys: qwen3, deepseek_moe.
#   --all                  Shortcut for --models qwen3,deepseek_moe --baseline.
#   --baseline             Also pull the static-quantization baseline GGUFs.
#   --datasets             (default) Pull calibration/KLD/TPS text.
#   --no-datasets          Skip dataset step.
#   -h | --help            Print this help.
#
# Idempotent: existing local files are skipped.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/.." && pwd)"
DATASETS_DIR="$ROOT/cache/datasets"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
HF="${HF:-hf}"

# huggingface_hub v1.x removed hf_transfer (HF_HUB_ENABLE_HF_TRANSFER is
# silently ignored).  The current speed knobs live on the Xet backend.
# Enable high-perf mode + a generous request timeout by default.
# Source: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-24}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# ── Model registry ───────────────────────────────────────────────────
# key | hf_repo | local_dir (relative to repo root)
declare -A MODEL_REPO MODEL_DIR
MODEL_REPO[qwen3]="Qwen/Qwen3-30B-A3B"
MODEL_DIR[qwen3]="cache/models/Qwen3-30B-A3B/hf"
MODEL_REPO[deepseek_moe]="deepseek-ai/deepseek-moe-16b-chat"
MODEL_DIR[deepseek_moe]="cache/models/deepseek-moe-16b-chat/hf"

# Baseline GGUFs (public quants used as KLD references).  An
# ``INCLUDE`` pattern restricts the download to a subset of files in
# the repo; leaving it empty pulls the whole repo.
declare -A BASELINE_REPO BASELINE_DIR BASELINE_INCLUDE
BASELINE_REPO[qwen3]="unsloth/Qwen3-30B-A3B-GGUF"
BASELINE_DIR[qwen3]="cache/models/Qwen3-30B-A3B/baseline"
BASELINE_INCLUDE[qwen3]=""
BASELINE_REPO[deepseek_moe]="mradermacher/deepseek-moe-16b-chat-GGUF"
BASELINE_DIR[deepseek_moe]="cache/models/deepseek-moe-16b-chat/baseline"
BASELINE_INCLUDE[deepseek_moe]="*Q2_K.gguf,*Q5_K_M.gguf,*Q8_0.gguf"

MODELS=""
WITH_BASELINE=0
WITH_DATASETS=1

usage() { sed -n '2,18p' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --models)        MODELS="$2"; shift 2;;
        --all)           MODELS="qwen3,deepseek_moe"; WITH_BASELINE=1; shift;;
        --baseline)      WITH_BASELINE=1; shift;;
        --datasets)      WITH_DATASETS=1; shift;;
        --no-datasets)   WITH_DATASETS=0; shift;;
        -h|--help)       usage; exit 0;;
        *) echo "[1_download] unknown arg: $1" >&2; usage; exit 2;;
    esac
done

cd "$ROOT"

# ── Datasets ─────────────────────────────────────────────────────────
download_datasets() {
    mkdir -p "$DATASETS_DIR"

    _json_gz_to_text() {
        local out="$1"; shift
        "$PYTHON" - "$out" "$@" <<'PY'
import gzip
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
inputs = [Path(p) for p in sys.argv[2:]]
out.parent.mkdir(parents=True, exist_ok=True)
n = 0
with out.open("w", encoding="utf-8") as dst:
    for path in inputs:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") if isinstance(obj, dict) else None
                if not text:
                    continue
                text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
                if text:
                    dst.write(text.replace("\n", " ") + "\n")
                    n += 1
print(f"[datasets/c4] wrote {n} rows -> {out}")
PY
    }

    # wikitext-2 raw: test split for KLD/PPL, train split for encoder
    # calibration and the residual profile corpus.
    local split
    for split in test train; do
        local wt="$DATASETS_DIR/wikitext-2-raw-${split}.txt"
        if [ ! -s "$wt" ]; then
            echo "[datasets] wikitext-2-raw-${split}.txt"
            local pq="$DATASETS_DIR/.wt2_${split}.parquet"
            curl -fL --retry 3 -o "$pq" \
                "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/${split}-00000-of-00001.parquet"
            "$PYTHON" - "$pq" "$wt" <<'PY'
import sys, pyarrow.parquet as pq
df = pq.read_table(sys.argv[1]).to_pandas()
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("\n".join(df["text"].tolist()))
PY
            rm -f "$pq"
        else
            echo "[datasets] $wt already present"
        fi
    done

    local c4_en="$DATASETS_DIR/c4_en_validation.txt"
    if [ ! -s "$c4_en" ] || [ "${TPS_FORCE:-0}" = "1" ]; then
        echo "[datasets/tps] c4_en_validation.txt"
        local c4_en_parts=()
        for n in $(seq 0 7); do
            local i
            i="$(printf "%05d" "$n")"
            local gz="$DATASETS_DIR/.c4_en_validation_${i}.json.gz"
            c4_en_parts+=("$gz")
            if [ ! -s "$gz" ] || [ "${TPS_FORCE:-0}" = "1" ]; then
                curl -fL --retry 3 -o "$gz" \
                    "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.${i}-of-00008.json.gz"
            fi
        done
        _json_gz_to_text "$c4_en" "${c4_en_parts[@]}"
    else
        echo "[datasets/tps] $c4_en already present"
    fi

    local c4_ko="$DATASETS_DIR/c4_ko_validation.txt"
    if [ ! -s "$c4_ko" ] || [ "${TPS_FORCE:-0}" = "1" ]; then
        echo "[datasets/tps] c4_ko_validation.txt"
        local c4_ko_gz="$DATASETS_DIR/.c4_ko_validation.00000-00001.json.gz"
        if [ ! -s "$c4_ko_gz" ] || [ "${TPS_FORCE:-0}" = "1" ]; then
            curl -fL --retry 3 -o "$c4_ko_gz" \
                "https://huggingface.co/datasets/allenai/c4/resolve/mC4_3.1.0/multilingual/c4-ko-validation.00000-00001.json.gz"
        fi
        _json_gz_to_text "$c4_ko" "$c4_ko_gz"
    else
        echo "[datasets/tps] $c4_ko already present"
    fi

    # The calibration_v3/v5 corpora are not redistributed. Warn if they are
    # missing so callers know to place files matching
    # the paths and checksums in experiments/config/<model>/base.yaml.
    for name in calibration_v3 calibration_v5; do
        local p="$DATASETS_DIR/${name}.txt"
        if [ ! -s "$p" ]; then
            echo "[datasets] WARN: $p missing"
            echo "    place the configured calibration corpus before encoding"
        else
            echo "[datasets] $p already present"
        fi
    done

    local sharegpt="$DATASETS_DIR/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
    local sharegpt_url="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
    if [ ! -s "$sharegpt" ] || [ "${TPS_FORCE:-0}" = "1" ]; then
        echo "[datasets/tps] ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
        curl -fL --retry 3 -o "$sharegpt" "$sharegpt_url"
    else
        echo "[datasets/tps] $sharegpt already present"
    fi
}

# ── Models ───────────────────────────────────────────────────────────
download_model() {
    local key="$1"
    local repo="${MODEL_REPO[$key]:-}"
    local dir="${MODEL_DIR[$key]:-}"
    if [ -z "$repo" ]; then
        echo "[models] unknown key '$key' — pick one of: ${!MODEL_REPO[*]}" >&2
        return 1
    fi
    mkdir -p "$dir"
    if "$PYTHON" - "$dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    index = root / name
    if not index.exists():
        continue
    data = json.loads(index.read_text())
    files = sorted(set(data.get("weight_map", {}).values()))
    if files and all((root / f).is_file() and (root / f).stat().st_size > 0 for f in files):
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "[models/$key] $repo → $dir (already present)"
        return 0
    fi
    echo "[models/$key] downloading $repo → $dir"
    "$HF" download "$repo" --local-dir "$dir"
}

download_baseline() {
    local key="$1"
    local repo="${BASELINE_REPO[$key]:-}"
    local dir="${BASELINE_DIR[$key]:-}"
    local include="${BASELINE_INCLUDE[$key]:-}"
    if [ -z "$repo" ]; then
        echo "[baseline/$key] no baseline configured — skipping"
        return 0
    fi
    mkdir -p "$dir"
    if [ -z "$include" ] && compgen -G "$dir/*.gguf" > /dev/null; then
        echo "[baseline/$key] $repo → $dir (already has .gguf, skipping)"
        return 0
    fi
    echo "[baseline/$key] $repo → $dir"
    local args=( download "$repo" --local-dir "$dir" )
    if [ -n "$include" ]; then
        IFS=',' read -ra inc <<< "$include"
        for pat in "${inc[@]}"; do
            args+=( --include "$pat" )
        done
    fi
    "$HF" "${args[@]}"
}

if [ "$WITH_DATASETS" -eq 1 ]; then
    download_datasets
fi

if [ -n "$MODELS" ]; then
    IFS=',' read -ra MARR <<< "$MODELS"
    for m in "${MARR[@]}"; do
        download_model "$m"
        if [ "$WITH_BASELINE" -eq 1 ]; then
            download_baseline "$m"
        fi
    done
else
    if [ "$WITH_DATASETS" -eq 0 ]; then
        echo "[1_download] nothing to do — pass --models or --datasets."
        exit 1
    fi
fi

echo "[1_download] done."
