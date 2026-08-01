#!/usr/bin/env bash
set -u

ROOT=/home/rwkv/chase/EvalScope
MODEL_ALIAS=${1:?model alias}
BASE_URL=${2:?base URL}
LABEL=${3:?run label}
LIMIT=${4:-}

args=(
  --config "$ROOT/configs/example.toml"
  --model-catalog "$ROOT/configs/models/g1h-single-replica.toml"
  --base-url "$BASE_URL"
  --api-key rwkv-skills
  --no-server
  --generation-config "$ROOT/experiments/evalscope_agent/flower-nocot-generation-2048.json"
  --agent-config '{"strategy":"swe_bench_toolcall","tools":["bash"],"environment":"docker","max_steps":50}'
  --eval-batch-size 1
  --request-timeout 600
  --work-dir "$ROOT/results/evalscope/$LABEL"
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

cd "$ROOT"
exec /home/rwkv/.local/bin/uv run --project "$ROOT" helicopter eval evalscope "$MODEL_ALIAS" swe_bench_verified_mini_agentic "${args[@]}"
