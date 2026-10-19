#!/usr/bin/env bash
# Generate the BF16 reference logits used by the KLD evaluation
# (run_wikitext_ppl.py). Run once per model before 4_bench/run.sh.
#
# The reference must use the same batch/ubatch kernel path as the
# candidate KLD run.
#
# Output: cache/experiment/_shared_<model family>/ref_logits_b${BATCH}.bin
# Idempotent: skips if the file exists.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

: "${PYTHON:=$REPO_ROOT/.venv/bin/python}"

# CONFIG (experiments/config/<model>/base.yaml) selects the model; the
# shared dir is named after the model family so models don't overwrite.
read SHARED BF16_GGUF < <(cd "$REPO_ROOT" && "$PYTHON" -c "
import sys; sys.path.insert(0,'experiments/3_dynamic_precision')
from _config import shared_root, active_variant
v = active_variant()
print(f'{shared_root()} {v[\"bf16_gguf\"]}')
")
mkdir -p "$SHARED"

: "${KLD_EVAL_FILE:=$REPO_ROOT/cache/datasets/wikitext-2-raw-test.txt}"
WIKI="$KLD_EVAL_FILE"
BIN_PPL="$REPO_ROOT/third_party/llama.cpp/build-dpa-moe/bin/llama-perplexity"

: "${GPU:=0}"
: "${CTX_SIZE:=4096}"
: "${REF_CHUNKS:=10}"
: "${BATCH:=1}"
: "${UBATCH:=$BATCH}"

REF="$SHARED/ref_logits_b${BATCH}.bin"

if [ -s "$REF" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[gen_ref] SKIP — $REF already exists ($(stat -c%s "$REF") bytes)"
    exit 0
fi

[ -f "$BF16_GGUF" ] || { echo "[gen_ref] missing $BF16_GGUF"; exit 2; }
[ -s "$WIKI" ]      || { echo "[gen_ref] missing $WIKI"; exit 3; }
[ -x "$BIN_PPL" ]   || { echo "[gen_ref] missing $BIN_PPL"; exit 4; }

echo "[gen_ref] generating $REF (BF16, c=$CTX_SIZE b=$BATCH ub=$UBATCH, chunks=$REF_CHUNKS)"
CUDA_VISIBLE_DEVICES="$GPU" \
    "$BIN_PPL" -m "$BF16_GGUF" -f "$WIKI" \
        --kl-divergence-base "$REF" \
        -c "$CTX_SIZE" -b "$BATCH" -ub "$UBATCH" -ngl 999 -s 0 \
        --no-warmup --chunks "$REF_CHUNKS" --fit off \
        > "$SHARED/gen_ref.out" 2> "$SHARED/gen_ref.err"

[ -s "$REF" ] || { echo "[gen_ref] FAILED — see $SHARED/gen_ref.err"; exit 5; }
echo "[gen_ref] wrote $REF ($(stat -c%s "$REF") bytes)"
