#!/usr/bin/env bash
set -euo pipefail

# Reproducible local-only EvalScope Agent runner.  The default path never uses
# the historical 19329 SSH tunnel and never enables the naive text proxy.
DATASET="${1:-general_fc}"
LIMIT="${2:-1}"
WORK_DIR="${3:-results/evalscope/local-${DATASET}-$(date +%Y%m%d_%H%M%S)}"
MODEL_ALIAS="${RWKV_EVAL_MODEL_ALIAS:-g1h-1.5b}"
BASE_URL="${RWKV_EVAL_BASE_URL:-http://127.0.0.1:19316/v1}"
API_KEY="${HELICOPTER_EVAL_API_KEY:-rwkv-skills}"
MAX_STEPS="${RWKV_EVAL_MAX_STEPS:-3}"
UV_BIN="${UV_BIN:-/home/chase/.local/bin/uv}"
test -x "$UV_BIN"

exec env HELICOPTER_EVAL_API_KEY="$API_KEY" \
  "$UV_BIN" run --no-default-groups --group agent --group eval --no-sync helicopter eval evalscope \
    "$MODEL_ALIAS" "$DATASET" \
    --config configs/example.toml \
    --model-catalog configs/models/local-g1h-single-replica.toml \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --no-server \
    --no-naive-chat-proxy \
    --limit "$LIMIT" \
    --work-dir "$WORK_DIR" \
    --strategy function_calling \
    --agent-environment local \
    --max-steps "$MAX_STEPS"
