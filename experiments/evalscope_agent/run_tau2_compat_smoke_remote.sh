#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/rwkv/chase/EvalScope
export PYTHONPATH="$ROOT/experiments/evalscope_agent/tau2-compat"
exec /home/rwkv/.local/bin/uv run --project "$ROOT" helicopter eval evalscope g1h-7.2b tau2_bench \
  --config "$ROOT/configs/example.toml" \
  --model-catalog "$ROOT/configs/models/g1h-single-replica.toml" \
  --base-url http://127.0.0.1:29572/v1 \
  --api-key rwkv-skills \
  --no-server \
  --generation-config "$ROOT/experiments/evalscope_agent/flower-nocot-generation-2048.json" \
  --dataset-args '{"tau2_bench":{"extra_params":{"user_model":"rwkv7-g1h-7.2b-20260710-ctx10240","api_key":"rwkv-skills","api_base":"http://127.0.0.1:29572/v1","generation_config":{"temperature":0.0,"max_tokens":2048}}}}' \
  --request-timeout 120 \
  --work-dir "$ROOT/results/evalscope/tau2-compat-smoke-7p2b" \
  --limit 1
