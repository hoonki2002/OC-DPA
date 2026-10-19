#!/usr/bin/env bash
# The quality half of the paper on OLMoE-1B-7B in one resumable command: encode the experts with ocdpa/encoder.py, profile the
# residual table, and measure the KL divergences of Fig. 4 (budget sweep, sigma sweeps). Every stage is skipped when its
# output exists (so the script resumes after an interruption or a reboot) and prints its wall-clock time.
#
#   WORK=/path/to/persistent/dir bash scripts/24_quality_run.sh      # results in $WORK/cache, logs in $WORK/logs
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
W="${WORK:-$REPO/work}"; mkdir -p "$W/cache" "$W/logs"; cd "$W"
export PYTHONPATH="$REPO" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0125}"      # the model of the KL and hardware results; its planes are cache/planes_olmoe
TASK="mixed,0.01,0.02,cal"
SIG="mixed,0,0,cal;mixed,0.01,0.02,cal;mixed,0.02,0.04,cal;mixed,0.05,0.10,cal;mixed,0.10,0.20,cal"

stage() {                                    # stage <name> <output file> <command...>
  local name=$1 out=$2; shift 2
  if [ -e "$out" ]; then echo "[skip] $name ($out exists)"; return; fi
  echo "[run ] $name  $(date +%H:%M:%S)"
  local t0=$SECONDS
  "$@" > "logs/$name.log" 2>&1 || { echo "[FAIL] $name, see $W/logs/$name.log"; exit 1; }
  echo "[done] $name  $(date +%H:%M:%S)  ($((SECONDS - t0)) s)"
}

stage irdrop   cache/irdrop_gain.npz            "$PY" "$REPO/scripts/07_calibrate_irdrop.py"
stage encode   cache/planes_olmoe/meta.json     "$PY" "$REPO/scripts/01_encode_experts.py" --model $MODEL --out cache/planes_olmoe
stage residual cache/residual_olmoe.npy         "$PY" "$REPO/scripts/03_profile_residual.py" --model $MODEL --planes cache/planes_olmoe --out cache/residual_olmoe.npy
OL="--model $MODEL --planes cache/planes_olmoe --residual cache/residual_olmoe.npy"
# Table II first: all planes, uniform k = 4 and OC-DPA at k-bar = 4 under the task noise, then the static GPTQ copies
stage kl_table2     cache/kl_olmoe_table2.json      "$PY" "$REPO/scripts/21_kl_sweep.py" $OL --noise "$TASK" --budgets 4 --out cache/kl_olmoe_table2.json
stage gptq          cache/planes_olmoe_gptq4/L15.npz "$PY" "$REPO/scripts/20_encode_native_gptq.py" --model $MODEL --out cache/planes_olmoe_gptq
stage kl_static     cache/kl_olmoe_static.json      "$PY" "$REPO/scripts/21_kl_sweep.py" --model $MODEL --planes cache/planes_olmoe_gptq8 --native-planes cache/planes_olmoe_gptq4 --native-only --home cpu --residual cache/residual_olmoe.npy --noise "$TASK" --budgets 4 --out cache/kl_olmoe_static.json
# then Fig. 4: budget sweep and sigma sweeps
stage kl_task       cache/kl_olmoe_task.json        "$PY" "$REPO/scripts/21_kl_sweep.py" $OL --noise "$TASK" --budgets 2,3,4,5,6,7 --out cache/kl_olmoe_task.json
stage sigma_experts cache/sigma_olmoe_experts.json  "$PY" "$REPO/scripts/21_kl_sweep.py" $OL --noise "$SIG" --budgets 4 --no-uniform --out cache/sigma_olmoe_experts.json
stage sigma_dense   cache/sigma_olmoe_dense.json    "$PY" "$REPO/scripts/21_kl_sweep.py" $OL --noise "$SIG" --all8-only --dense attn,router,head --out cache/sigma_olmoe_dense.json
# the placement of the planes: all eight planes in crossbars (sigma 2 %) instead of planes 1-2 in 1T1R (1 %), Sec. III-B
stage kl_xbar       cache/kl_olmoe_xbar.json        "$PY" "$REPO/scripts/21_kl_sweep.py" $OL --noise "xbar,0.01,0.02,cal" --budgets 4 --no-uniform --out cache/kl_olmoe_xbar.json
echo "ALL DONE $(date +%H:%M:%S)"
