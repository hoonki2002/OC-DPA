#!/usr/bin/env bash
# Dynamic precision profiling driver.
#
# This stage owns only calibration/profile collection and packing the runtime
# residual profile. Benchmark-side KLD, Wikitext PPL, TPS, and plots are owned
# by experiments/4_bench/run.sh and experiments/4_bench/kld_sweep/.
#
# Env overrides:
#   CONFIG                default: experiments/config/Qwen3-30B-A3B/base.yaml
#   GPU                   default: config.profile.gpu
#   HF_DIR/CODEBOOK_DIR   default: config.model + config.profile
#   PROFILE_*             default: config.profile
#   SKIP_PROFILE=1        reuse existing expert_residuals.npz
#   FORCE=1               rerun profile collection
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${CONFIG:=$REPO_ROOT/experiments/config/Qwen3-30B-A3B/base.yaml}"
: "${SKIP_PROFILE:=0}"

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

read MAN_FAMILY MAN_GPU MAN_HF_DIR MAN_CODEBOOK_DIR MAN_PROFILE_NPZ MAN_PROFILE_BIN MAN_PROFILE_MAX_TOKENS MAN_PROFILE_MAX_PER_EXPERT MAN_PROFILE_SEQ_LEN MAN_PROFILE_DEVICE MAN_PROFILE_SEED MAN_PROFILE_CORPUS < <("$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
profile = m["profile"]
model = m["model"]
target = {
    "family": model["family"],
    "hf_dir": model["hf"]["local_dir"],
    "codebooks_dirs": model["codebooks"],
}
def csv(xs):
    if isinstance(xs, str):
        return xs
    return ",".join(str(x) for x in xs)
def first(xs):
    if isinstance(xs, (list, tuple)):
        return xs[0] if xs else ""
    return xs
print(
    target["family"],
    first(profile.get("gpu", m.get("hardware", {}).get("gpu_pin", 0))),
    profile.get("hf_dir", target["hf_dir"]),
    profile.get("codebook_dir", target["codebooks_dirs"].get("ss_anybcq", "")),
    profile.get("residuals_npz", f"cache/experiment/{target['family']}/layout_solver/expert_residuals.npz"),
    profile.get("runtime_bin", f"cache/experiment/{target['family']}/3_dynamic_precision/bin/expert_residuals.bin"),
    profile.get("max_tokens", 16384),
    profile.get("max_per_expert", 256),
    profile.get("seq_len", 2048),
    profile.get("device", "cuda:0"),
    profile.get("seed", m.get("seed", 0)),
    csv(profile.get("corpus", [])),
)
PY
)

: "${GPU:=$MAN_GPU}"
: "${HF_DIR:=$REPO_ROOT/$MAN_HF_DIR}"
: "${CODEBOOK_DIR:=$REPO_ROOT/$MAN_CODEBOOK_DIR}"
: "${RESIDUALS_NPZ:=$REPO_ROOT/$MAN_PROFILE_NPZ}"
: "${RESIDUALS_BIN:=$REPO_ROOT/$MAN_PROFILE_BIN}"
: "${PROFILE_MAX_TOKENS:=$MAN_PROFILE_MAX_TOKENS}"
: "${PROFILE_MAX_PER_EXPERT:=$MAN_PROFILE_MAX_PER_EXPERT}"
: "${PROFILE_SEQ_LEN:=$MAN_PROFILE_SEQ_LEN}"
: "${PROFILE_DEVICE:=${MAN_PROFILE_DEVICE:-cuda:0}}"
: "${PROFILE_SEED:=$MAN_PROFILE_SEED}"
: "${PROFILE_CORPUS:=${MAN_PROFILE_CORPUS:-}}"

echo "==== dynamic precision profile ===="
echo "config       : $CONFIG"
echo "model        : $MAN_FAMILY"
echo "gpu          : $GPU"
echo "profile npz  : $RESIDUALS_NPZ"
echo "runtime bin  : $RESIDUALS_BIN"
echo "profile seed : $PROFILE_SEED"
echo

if [ "$SKIP_PROFILE" != "1" ]; then
    if [ -s "$RESIDUALS_NPZ" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[run] SKIP profile — $RESIDUALS_NPZ exists"
    else
        [ -d "$HF_DIR" ] || { echo "[run] missing HF_DIR=$HF_DIR"; exit 10; }
        [ -d "$CODEBOOK_DIR" ] || { echo "[run] missing CODEBOOK_DIR=$CODEBOOK_DIR"; exit 11; }
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$HERE/profiling/profile_residuals.py" \
            --hf-dir "$HF_DIR" \
            --codebook-dir "$CODEBOOK_DIR" \
            --out "$(dirname "$RESIDUALS_NPZ")" \
            --device "$PROFILE_DEVICE" \
            --max-tokens "$PROFILE_MAX_TOKENS" \
            --max-per-expert "$PROFILE_MAX_PER_EXPERT" \
            --seq-len "$PROFILE_SEQ_LEN" \
            --seed "$PROFILE_SEED" \
            ${PROFILE_CORPUS:+--corpus ${PROFILE_CORPUS//,/ }}
    fi
else
    echo "[run] SKIP profile by request"
fi

[ -s "$RESIDUALS_NPZ" ] || { echo "[run] missing residual profile: $RESIDUALS_NPZ"; exit 12; }

"$PYTHON" "$HERE/profiling/pack.py" \
    --npz "$RESIDUALS_NPZ" \
    --out "$RESIDUALS_BIN"

echo
echo "artifacts:"
echo "  residual profile : $RESIDUALS_NPZ"
echo "  runtime profile  : $RESIDUALS_BIN"
echo
echo "benchmark KLD/TPS/plots are run from experiments/4_bench/run.sh"
