#!/usr/bin/env bash
# 4_bench — single entry point for the full benchmark sweep.
#
# Phases (each can be skipped independently):
#   llama_mc  llama.cpp multiple-choice → llama_complexity/run_bench.py
#   kld       Wikitext PPL/KLD          → kld_sweep/run_wikitext_ppl.py
#   speed     text generation speed     → speed_sweep/run_sweep.py
#
# Env overrides:
#   ROOT_CONFIG  default: experiments/config/Qwen3-30B-A3B/base.yaml
#   BENCH_CONFIG default: experiments/config/<model>/bench.yaml
#   SKIP_LLAMA_MC set 1 to skip llama.cpp multiple-choice phase
#   SKIP_KLD      set 1 to skip Wikitext PPL/KLD phase
#   SKIP_SPEED    set 1 to skip standalone speed phase
#   FORCE         set 1 to rerun already-complete benchmarks
#
# Examples:
#   bash experiments/4_bench/run.sh
#   SKIP_SPEED=0 bash experiments/4_bench/run.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${LOG_DIR:=$REPO/cache/logs/4_bench}"
mkdir -p "$LOG_DIR"
cd "$REPO"

PYTHON="${PYTHON:-.venv/bin/python}"
ROOT_CONFIG="${ROOT_CONFIG:-$REPO/experiments/config/Qwen3-30B-A3B/base.yaml}"

SKIP_LLAMA_MC="${SKIP_LLAMA_MC:-0}"
SKIP_KLD="${SKIP_KLD:-0}"
SKIP_SPEED="${SKIP_SPEED:-1}"    # speed sweep is opt-in (long-running)
FORCE="${FORCE:-0}"

# Read benchmark config settings from _config.py
read -r MODEL BENCH_CONFIG_RESOLVED CONFIGS_CSV GPU_MAP_JSON < <(ROOT_CONFIG="$ROOT_CONFIG" BENCH_CONFIG="${BENCH_CONFIG:-}" "$PYTHON" - <<'PY'
import sys, json
sys.path.insert(0, "experiments/4_bench")
from _config import BENCHMARK_CONFIGS, BENCH_CONFIG, GPU_MAP, MODEL
print(MODEL, BENCH_CONFIG, ",".join(BENCHMARK_CONFIGS), json.dumps(GPU_MAP))
PY
)
BENCH_CONFIG="$BENCH_CONFIG_RESOLVED"

IFS=',' read -ra CONFIG_LIST <<< "$CONFIGS_CSV"

echo "────────────────────────────────────────────────────────"
echo "  4_bench  MODEL=$MODEL"
echo "  root config: $ROOT_CONFIG"
echo "  bench config: $BENCH_CONFIG"
echo "  configs:  ${CONFIG_LIST[*]}"
echo "  phases:   llama_mc=$([ "$SKIP_LLAMA_MC" = "1" ] && echo skip || echo run)  kld=$([ "$SKIP_KLD" = "1" ] && echo skip || echo run)  speed=$([ "$SKIP_SPEED" = "1" ] && echo skip || echo run)"
echo "────────────────────────────────────────────────────────"
echo ""

_gpu_for() {
    "$PYTHON" -c "
import json, sys
m = json.loads('$GPU_MAP_JSON')
print(m.get('$1', '0'))
"
}

_launch_phase() {
    local script="$1"; shift
    local phase_name="$1"; shift
    local pids=()
    for cfg in "${CONFIG_LIST[@]}"; do
        local gpu; gpu=$(_gpu_for "$cfg")
        local logfile="$LOG_DIR/${phase_name}_${cfg}.log"
        echo "  GPU=$gpu  CONFIG=$cfg  → $logfile"
        env GPU="$gpu" CONFIG="$cfg" MODEL="$MODEL" ROOT_CONFIG="$ROOT_CONFIG" BENCH_CONFIG="$BENCH_CONFIG" \
            FORCE="$FORCE" \
            nohup "$PYTHON" "$script" > "$logfile" 2>&1 &
        pids+=($!)
    done
    echo "  PIDs: ${pids[*]}"
    echo "  Waiting for $phase_name …"
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "  WARN: PID $pid exited non-zero"; ((failed++)); }
    done
    [ "$failed" -eq 0 ] && echo "  $phase_name DONE." || echo "  $phase_name: $failed config(s) failed."
}

# ── llama.cpp multiple-choice phase ───────────────────────────────────────────
if [ "$SKIP_LLAMA_MC" != "1" ]; then
    echo "[llama_mc] launching ${#CONFIG_LIST[@]} configs …"
    _launch_phase experiments/4_bench/llama_complexity/run_bench.py llama_mc
    echo ""
fi

# ── Wikitext PPL/KLD phase ───────────────────────────────────────────────────
if [ "$SKIP_KLD" != "1" ]; then
    echo "[kld] BF16 reference logits (skipped if present) …"
    env CONFIG="$ROOT_CONFIG" GPU="$(_gpu_for "${CONFIG_LIST[0]}")" BATCH=1 PYTHON="$PYTHON" \
        bash experiments/4_bench/kld_sweep/gen_ref.sh
    echo "[kld] launching ${#CONFIG_LIST[@]} configs …"
    _launch_phase experiments/4_bench/kld_sweep/run_wikitext_ppl.py kld
    echo ""
fi


# ── Standalone text speed phase ───────────────────────────────────────────────
if [ "$SKIP_SPEED" != "1" ]; then
    echo "[speed] running standalone text speed sweep …"
    env MODEL="$MODEL" ROOT_CONFIG="$ROOT_CONFIG" BENCH_CONFIG="$BENCH_CONFIG" FORCE="$FORCE" \
        "$PYTHON" experiments/4_bench/speed_sweep/run_sweep.py
    echo ""
fi

echo "────────────────────────────────────────────────────────"
echo "  4_bench complete."
echo "  MC:    experiments/results/$MODEL/bench/"
echo "  KLD:   experiments/results/$MODEL/kld/"
echo "  Speed: experiments/results/$MODEL/speed/"
echo "────────────────────────────────────────────────────────"
