#!/usr/bin/env bash
# Step 2 — Encode DPA-MoE AnyBCQ codebooks and assemble the GGUF.
#
# Calibrates per-layer codebooks with calibrate_parallel.sh, then runs
# `dpa-moe-encode assemble` and `finalize`. The model is selected by
# CONFIG (one config per model).
#
#   bash experiments/2_encode/run.sh
#   CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml bash experiments/2_encode/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

: "${CONFIG:=experiments/config/Qwen3-30B-A3B/base.yaml}"
: "${DPA_MOE_VARIANT:=}"
: "${GPUS:=}"
: "${INNER_BS:=}"
: "${FP_CACHING:=1}"
: "${SKIP_CALIBRATE:=0}"
: "${SKIP_ASSEMBLE:=0}"
: "${FORCE:=0}"

eval "$(.venv/bin/python "$HERE/_encode_config.py" "$CONFIG" "$DPA_MOE_VARIANT")"

echo "==== 2_encode ===="
echo "config       : $CONFIG"
echo "model_key    : $MODEL_KEY"
echo "variant      : $VARIANT"
echo "gpus         : ${GPUS:-${CFG_GPUS:-all visible}}"
echo "inner_bs     : ${INNER_BS:-$CFG_INNER_BS}"
echo "fp_caching   : $FP_CACHING"
echo "codebooks    : $CODEBOOKS_DIR"
echo "final gguf   : $FINAL_GGUF"
echo

if [ "$SKIP_CALIBRATE" != "1" ]; then
    env CONFIG="$CONFIG" \
        DPA_MOE_VARIANT="$VARIANT" \
        GPUS="$GPUS" \
        INNER_BS="$INNER_BS" \
        FP_CACHING="$FP_CACHING" \
        bash "$HERE/calibrate_parallel.sh"
else
    echo "[2_encode] SKIP calibration by request"
fi

if [ "$SKIP_ASSEMBLE" != "1" ]; then
    common=(--config "$CONFIG" --variant "$VARIANT")
    [ "$FORCE" = "1" ] && common+=(--force)
    .venv/bin/dpa-moe-encode assemble "${common[@]}"
    .venv/bin/dpa-moe-encode finalize "${common[@]}"
else
    echo "[2_encode] SKIP assembly by request"
fi

echo "[2_encode] done"
