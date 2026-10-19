#!/usr/bin/env bash
# Convert a model's HF checkpoint to the BF16 GGUF used as the KLD
# reference (experiments/4_bench/kld_sweep/gen_ref.sh). Only needed when
# no BF16 GGUF is published for the model (e.g. DeepSeek-MoE-16B; the
# Qwen3 BF16 GGUF comes from the unsloth baseline repo via --baseline).
#
# Usage:
#   bash experiments/1_download/convert_bf16.sh experiments/config/<model>/base.yaml
#
# Input and output paths come from the config: model.hf.local_dir and
# model.bf16_gguf.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
CONVERT="$ROOT/third_party/llama.cpp/convert_hf_to_gguf.py"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

config="${1:?usage: $0 experiments/config/<model>/base.yaml}"
read -r hf out < <("$PYTHON" - "$config" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))["model"]
print(m["hf"]["local_dir"], m["bf16_gguf"])
PY
)

if [ ! -s "$hf/config.json" ]; then
    echo "[convert_bf16] HF dir missing or incomplete: $hf" >&2
    echo "[convert_bf16] run experiments/1_download/run.sh --models <key> first" >&2
    exit 3
fi

mkdir -p "$(dirname "$out")"
if [ -s "$out" ]; then
    echo "[convert_bf16] $out already exists, skipping"
    exit 0
fi

echo "[convert_bf16] $hf  →  $out"
"$PYTHON" "$CONVERT" "$hf" --outtype bf16 --outfile "$out"
echo "[convert_bf16] done ($(du -h "$out" | awk '{print $1}'))"
