#!/usr/bin/env bash
# Top-level driver for the DPA-MoE reproduction pipeline. Each step's
# run.sh is idempotent: it skips work whose outputs already exist.
#
# Usage:
#   bash experiments/reproduce.sh [all|0|1|2|3|4|<step folder>]
#
# Steps read the model from CONFIG / ROOT_CONFIG (default: Qwen3-30B-A3B);
# step 1 downloads datasets only unless given --models (run it directly).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STEP="${1:-all}"
STEPS=(0_build 1_download 2_encode 3_dynamic_precision 4_bench)

run_one() {
  local s="$1"
  echo "===================================================================="
  echo "=== step: $s   (start $(date -Iseconds))"
  echo "===================================================================="
  bash "$HERE/$s/run.sh"
}

if [ "$STEP" = "all" ]; then
  for s in "${STEPS[@]}"; do run_one "$s"; done
  echo "[reproduce] all steps done  $(date -Iseconds)"
  exit 0
fi

for s in "${STEPS[@]}"; do
  if [ "$s" = "$STEP" ] || [[ "$s" == "${STEP}_"* ]]; then
    run_one "$s"
    exit 0
  fi
done
echo "[reproduce] unknown step: '$STEP'.  Choices: all, ${STEPS[*]}" >&2
exit 2
