#!/usr/bin/env bash
# Smoke-test a dpa_moe-encoded GGUF: launch llama-server, send a one-line
# completion, confirm the dpa_moe sentinel rail engaged.
#
#   $1: path to encoded GGUF (e.g. cache/models/.../chunked/ss_anybcq.gguf)
#   $2: short prompt (default: "The capital of France is")
#   $3: expected gguf_arch token (e.g. "qwen3moe", "deepseek")
#
# Greps stderr for the "gguf_arch=... → activation=..." log line and the
# pool-ready announcement.  Generation isn't asserted, just exercised.

set -euo pipefail

GGUF="${1:?usage: smoke_test_dpa_moe.sh <gguf> [prompt] [expected_arch]}"
PROMPT="${2:-The capital of France is}"
EXPECT_ARCH="${3:-}"
PORT="${PORT:-11434}"
GPU="${GPU:-0}"
VRAM_CAP_MB="${VRAM_CAP_MB:-16384}"
N_PARALLEL="${N_PARALLEL:-1}"
PROMPT2="${PROMPT2:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SERVER=$REPO/third_party/llama.cpp/build-dpa-moe/bin/llama-server
LOG_DIR="${LOG_DIR:-$REPO/cache/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/dpa_moe_smoke_$$.log"

[ -x "$SERVER" ] || { echo "missing $SERVER — build first" >&2; exit 2; }
[ -f "$GGUF" ]  || { echo "missing $GGUF" >&2; exit 2; }

echo "[smoke] launching server on :$PORT for $GGUF"
echo "[smoke] gpu=$GPU vram_cap_mb=$VRAM_CAP_MB n_parallel=$N_PARALLEL"
CUDA_VISIBLE_DEVICES="$GPU" DPA_MOE_VRAM_CAP_MB="$VRAM_CAP_MB" \
  "$SERVER" --model "$GGUF" --port "$PORT" --host 127.0.0.1 \
            -ngl 999 -c 4096 -np "$N_PARALLEL" -fit off --no-warmup --jinja \
  > "$LOG" 2>&1 &
SERVER_PID=$!
trap "kill -INT $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true" EXIT

# wait for /health
echo "[smoke] waiting for /health"
for i in {1..120}; do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[smoke] server up after ${i}s"
        break
    fi
    sleep 1
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[smoke] server died before becoming ready; tail of log:" >&2
        tail -30 "$LOG" >&2
        exit 3
    fi
done

echo "[smoke] waiting for model readiness"
for i in {1..600}; do
    if curl -fsS "http://127.0.0.1:$PORT/dpa_moe/stats" >/dev/null 2>&1; then
        echo "[smoke] model ready after ${i}s"
        break
    fi
    sleep 1
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[smoke] server died during model load; tail of log:" >&2
        tail -60 "$LOG" >&2
        exit 3
    fi
    if [ "$i" -eq 600 ]; then
        echo "[smoke] model did not become ready within 600s; tail of log:" >&2
        tail -60 "$LOG" >&2
        exit 3
    fi
done

send_prompt() {
    local label="$1"
    local prompt="$2"
    echo "[smoke] sending ${label}: $prompt"
    local resp
    resp=$(PROMPT_TEXT="$prompt" python3 - <<'PY'
import json
import os
print(json.dumps({
    "model": "dpa_moe",
    "messages": [{"role": "user", "content": os.environ["PROMPT_TEXT"]}],
    "max_tokens": 32,
    "temperature": 0.0,
    "cache_prompt": False,
}))
PY
)
    resp=$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$resp" || true)
    echo "[smoke] ${label} response: $resp" | head -c 1000
    echo
}

# generation requests
send_prompt "prompt1" "$PROMPT"
if [ -n "$PROMPT2" ]; then
    send_prompt "prompt2" "$PROMPT2"
fi

# log inspections — print install/activation header AND the most recent
# sentinel/forward counters (the very-final stats line emitted at server
# shutdown reflects post-generation totals).
echo
echo "[smoke] === install/activation ==="
grep -E "gguf_arch|activation=|sentinel branch active" "$LOG" | head -8 || true
echo
echo "[smoke] === final dpa_moe routing stats ==="
grep -E "DPAMoE routing:.*forward_moe_layer_calls" "$LOG" | tail -2 || true
echo "[smoke] ================================="

if [ -n "$EXPECT_ARCH" ]; then
    if grep -q "gguf_arch='$EXPECT_ARCH'" "$LOG"; then
        echo "[smoke] PASS — gguf_arch='$EXPECT_ARCH' confirmed"
    else
        echo "[smoke] FAIL — expected gguf_arch='$EXPECT_ARCH' not seen" >&2
        exit 4
    fi
fi

# stats endpoint
STATS=$(curl -fsS "http://127.0.0.1:$PORT/dpa_moe/stats" 2>/dev/null || true)
if [ -n "$STATS" ]; then
    echo "[smoke] === dpa_moe stats ==="
    echo "$STATS"
    STATS_JSON="$STATS" python3 - <<'PY'
import json
import os
import sys

stats = json.loads(os.environ["STATS_JSON"])
routing = stats.get("routing", {})
claims = int(routing.get("sentinel_claims", 0) or 0)
fwd = int(routing.get("forward_moe_layer_calls", 0) or 0)
if claims <= 0 or fwd <= 0:
    print(
        f"[smoke] FAIL -- routing counters did not advance: "
        f"sentinel_claims={claims} forward_moe_layer_calls={fwd}",
        file=sys.stderr,
    )
    sys.exit(5)
print(
    f"[smoke] PASS -- routing counters advanced: "
    f"sentinel_claims={claims} forward_moe_layer_calls={fwd}"
)
PY
else
    echo "[smoke] FAIL -- /dpa_moe/stats was empty" >&2
    exit 5
fi

echo "[smoke] done — full log at $LOG"
