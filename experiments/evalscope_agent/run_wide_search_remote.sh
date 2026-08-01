#!/usr/bin/env bash
set -u

ROOT=/home/rwkv/chase/EvalScope
MODEL_ALIAS=${1:?model alias}
BASE_URL=${2:?base URL}
LABEL=${3:?run label}
LIMIT=${4:-}
ROUTER_MODE=${5:-router}

args=(
  --config "$ROOT/configs/example.toml"
  --model-catalog "$ROOT/configs/models/g1h-single-replica.toml"
  --base-url "$BASE_URL"
  --api-key rwkv-skills
  --no-server
  --generation-config "$ROOT/experiments/evalscope_agent/flower-nocot-generation-2048.json"
  --judge-strategy llm
  --judge-model-args "$ROOT/experiments/evalscope_agent/wide-search-judge-13p3b.json"
  --judge-worker-num 1
  --agent-config '{"strategy":"function_calling","tools":["bash"],"environment":"local","max_steps":50}'
  --eval-batch-size 1
  --request-timeout 600
  --work-dir "$ROOT/results/evalscope/$LABEL"
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$ROUTER_MODE" != "no-router" ]]; then
  args+=(
    --parallel-candidate-router
    --candidate-batch-size 4
    --candidate-max-tokens 2048
    --aggregate-max-tokens 2048
    --candidate-max-candidates 8
    --candidate-context-chars 6000
    --candidate-prompt-max-chars 12288
  )
fi

cd "$ROOT"
exec /home/rwkv/.local/bin/uv run --project "$ROOT" helicopter eval evalscope "$MODEL_ALIAS" wide_search "${args[@]}"
