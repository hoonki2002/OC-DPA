#!/usr/bin/env bash
# Multi-GPU layer-range calibration launcher.
#
# Starts one process per GPU, assigns each process a disjoint range of
# MoE decoder layers, and writes per-layer codebooks into the
# config-selected output dir. Each process runs the original AnyBCQ
# per-layer execution on its own GPU; there is no cross-GPU expert
# placement inside a layer.
#
# Usage:
#   bash experiments/2_encode/calibrate_parallel.sh
#   CONFIG=experiments/config/deepseek-moe-16b-chat/base.yaml GPUS=0,1 \
#       bash experiments/2_encode/calibrate_parallel.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

: "${CONFIG:=experiments/config/Qwen3-30B-A3B/base.yaml}"
: "${DPA_MOE_VARIANT:=}"
: "${DPA_MOE_CODEBOOKS_DIR:=}"
: "${INNER_BS:=}"
: "${FP_CACHING:=1}"
: "${GPUS:=}"
: "${LOG_DIR:=cache/logs/2_encode}"
USER_START_LAYER="${START_LAYER:-}"

eval "$(.venv/bin/python experiments/2_encode/_encode_config.py "$CONFIG" "$DPA_MOE_VARIANT")"
[ -n "$USER_START_LAYER" ] && START_LAYER="$USER_START_LAYER"
[ -n "$DPA_MOE_CODEBOOKS_DIR" ] && CODEBOOKS_DIR="$DPA_MOE_CODEBOOKS_DIR"
[ -z "$INNER_BS" ] && INNER_BS="$CFG_INNER_BS"
if [ -z "$GPUS" ]; then
    GPUS="${CFG_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}
if [ "$N_GPUS" -lt 1 ]; then
    echo "[calibrate-parallel] no GPUs selected (set GPUS=...)" >&2
    exit 2
fi
mkdir -p "$LOG_DIR"

echo "[calibrate-parallel] model_key=$MODEL_KEY variant=$VARIANT"
echo "[calibrate-parallel] layers=$N_LAYERS start_layer=$START_LAYER gpus=$GPUS inner_bs=$INNER_BS fp_caching=$FP_CACHING"
echo "[calibrate-parallel] codebooks_dir=$CODEBOOKS_DIR"

declare -a PIDS
TOTAL_WORK=$((N_LAYERS - START_LAYER))
for i in "${!GPU_LIST[@]}"; do
    gpu=${GPU_LIST[$i]}
    start=$((START_LAYER + i * TOTAL_WORK / N_GPUS))
    end_exclusive=$((START_LAYER + (i + 1) * TOTAL_WORK / N_GPUS))
    if [ "$start" -ge "$end_exclusive" ]; then
        echo "[calibrate-parallel] skip GPU $gpu: empty layer range"
        continue
    fi
    end=$((end_exclusive - 1))
    log="$LOG_DIR/calibrate_${MODEL_KEY}_${VARIANT}_gpu${gpu}.log"
    echo "[calibrate-parallel] GPU $gpu layers ${start}:${end} log=$log"
    cmd=(.venv/bin/dpa-moe-encode calibrate
         --config "$CONFIG"
         --variant "$VARIANT"
         --calibration-mode layer_fp_parallel
         --force
         --codebooks-dir "$CODEBOOKS_DIR")
    CUDA_VISIBLE_DEVICES="$gpu" \
    DPA_MOE_LAYER_RANGE="${start}:${end}" \
    DPA_MOE_INNER_BS="$INNER_BS" \
    DPA_MOE_FP_CACHING="$FP_CACHING" \
        nohup "${cmd[@]}" > "$log" 2>&1 &
    PIDS[$i]=$!
done

echo "[calibrate-parallel] launched PIDs: ${PIDS[*]}"
status=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "[calibrate-parallel] pid $pid finished"
    else
        echo "[calibrate-parallel] pid $pid FAILED (see $LOG_DIR)" >&2
        status=1
    fi
done

if [ -d "$CODEBOOKS_DIR" ]; then
    n=$(find "$CODEBOOKS_DIR" -maxdepth 1 -name 'layer_*.safetensors' | wc -l)
    echo "[calibrate-parallel] total layer codebooks = $n / $N_LAYERS"
fi
exit "$status"
