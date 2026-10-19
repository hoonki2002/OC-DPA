#!/usr/bin/env bash
# Reproduce the diagonal error-estimator diagnostics for DPA-MoE
# (paper appendix: Cross-Term Analysis).
#
# This script measures the reviewer-facing quantity:
#
#   exact_mixed = || sum_e g_e * (E_fp(x) - E_k(e)(x)) ||^2
#   diag_mixed  = sum_e g_e^2 * || E_fp(x) - E_k(e)(x) ||^2
#
# For each Kbar budget it first uses the same profile-DP allocator as runtime
# to choose per-expert K[e], then compares exact_mixed against diag_mixed on the
# same sampled Wikitext activations. Integer budgets also include same-budget
# uniform-K exact/diagonal errors.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

: "${CONFIG:=$REPO_ROOT/experiments/config/Qwen3-30B-A3B/base.yaml}"
: "${GPU:=0}"
: "${DEVICE:=cuda:0}"
: "${MAX_TOKENS:=256}"
: "${SEQ_LEN:=$MAX_TOKENS}"
: "${LAYER_START:=0}"
: "${LAYER_END:=4}"
: "${BUDGETS:=3 4 5 6 7}"
: "${OUT:=experiments/results/Qwen3-30B-A3B/bench/budget_error_diag/wikitext_small_L${LAYER_START}_${LAYER_END}.csv}"

read HF_DIR CODEBOOK_DIR PROFILE_NPZ SEED CORPUS < <("$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
model = m.get("model", {})
profile = m.get("profile", {})
hf_dir = model.get("hf", {}).get("local_dir", "cache/models/Qwen3-30B-A3B/hf")
codebook_dir = model.get("codebooks", {}).get("ss_anybcq", "cache/models/Qwen3-30B-A3B/ss_anybcq_layer")
profile_npz = profile.get(
    "residuals_npz",
    "cache/experiment/Qwen3-30B-A3B/layout_solver_wikitext_monotonic/expert_residuals.npz",
)
corpus = profile.get("corpus", ["cache/datasets/wikitext-2-raw-train.txt"])
if isinstance(corpus, str):
    corpus_s = corpus
else:
    corpus_s = ",".join(corpus)
print(
    hf_dir,
    codebook_dir,
    profile_npz,
    m.get("seed", profile.get("seed", 0)),
    corpus_s,
)
PY
)

# Prefer the monotonic Wikitext profile when present; it is the profile used by
# current benchmark dynamic runs. Leave PROFILE_NPZ override available.
if [ -z "${PROFILE_NPZ_OVERRIDE:-}" ] && \
   [ -s "cache/experiment/Qwen3-30B-A3B/layout_solver_wikitext_monotonic/expert_residuals.npz" ]; then
    PROFILE_NPZ="cache/experiment/Qwen3-30B-A3B/layout_solver_wikitext_monotonic/expert_residuals.npz"
else
    PROFILE_NPZ="${PROFILE_NPZ_OVERRIDE:-$PROFILE_NPZ}"
fi

echo "==== DPA-MoE diagonal error estimator ===="
echo "config       : $CONFIG"
echo "gpu          : $GPU"
echo "hf dir       : $HF_DIR"
echo "codebooks    : $CODEBOOK_DIR"
echo "profile npz  : $PROFILE_NPZ"
echo "corpus       : ${CORPUS//,/ }"
echo "layers       : [$LAYER_START, $LAYER_END)"
echo "tokens       : $MAX_TOKENS"
echo "budgets      : $BUDGETS"
echo "out          : $OUT"
echo

[ -d "$HF_DIR" ] || { echo "[error-est] missing HF_DIR=$HF_DIR"; exit 10; }
[ -d "$CODEBOOK_DIR" ] || { echo "[error-est] missing CODEBOOK_DIR=$CODEBOOK_DIR"; exit 11; }
[ -s "$PROFILE_NPZ" ] || { echo "[error-est] missing PROFILE_NPZ=$PROFILE_NPZ"; exit 12; }

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$HERE/budget_error_diagnostic.py" \
    --hf-dir "$HF_DIR" \
    --codebook-dir "$CODEBOOK_DIR" \
    --profile-npz "$PROFILE_NPZ" \
    --corpus ${CORPUS//,/ } \
    --out "$OUT" \
    --device "$DEVICE" \
    --max-tokens "$MAX_TOKENS" \
    --seq-len "$SEQ_LEN" \
    --layer-start "$LAYER_START" \
    --layer-end "$LAYER_END" \
    --budgets $BUDGETS \
    --seed "$SEED"

echo
echo "wrote: $OUT"
