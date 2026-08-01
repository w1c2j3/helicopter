#!/usr/bin/env bash
set -euo pipefail

# Run one official EvalScope benchmark whose task environment is Docker-backed.
# The script performs a preflight first and never removes images, containers, or
# cache.  Use PREPARE_ONLY=1 to validate the environment without starting a
# benchmark or building task images.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-/home/chase/.local/bin/uv}"
PROFILE="${1:?usage: $0 swebench_verified_mini|terminal_bench_v2_1 MODEL_ALIAS BASE_URL LABEL [LIMIT]}"
MODEL_ALIAS="${2:?usage: $0 swebench_verified_mini|terminal_bench_v2_1 MODEL_ALIAS BASE_URL LABEL [LIMIT]}"
BASE_URL="${3:?usage: $0 swebench_verified_mini|terminal_bench_v2_1 MODEL_ALIAS BASE_URL LABEL [LIMIT]}"
LABEL="${4:?usage: $0 swebench_verified_mini|terminal_bench_v2_1 MODEL_ALIAS BASE_URL LABEL [LIMIT]}"
LIMIT="${5:-}"
API_KEY="${HELICOPTER_EVAL_API_KEY:-rwkv-skills}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

case "$PROFILE" in
  swebench_verified_mini)
    DATASET="swe_bench_verified_mini_agentic"
    CONFIG="$ROOT/configs/evalscope_agent/swebench_verified_mini_docker.toml"
    UV_GROUP="swe-bench"
    AGENT_ARGS=(--agent-config '{"strategy":"swe_bench_toolcall","tools":["bash"],"environment":"docker","max_steps":250}')
    ;;
  terminal_bench_v2_1)
    DATASET="terminal_bench_v2_1"
    CONFIG="$ROOT/configs/evalscope_agent/terminal_bench_v2_1_docker.toml"
    UV_GROUP="terminal-bench"
    AGENT_ARGS=(--no-agent-config)
    ;;
  *)
    echo "unsupported Docker benchmark profile: $PROFILE" >&2
    exit 2
    ;;
esac

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found or not executable: $UV_BIN" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is required for $PROFILE" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is unavailable for $PROFILE" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "profile config not found: $CONFIG" >&2
  exit 2
fi

echo "docker preflight passed: profile=$PROFILE dataset=$DATASET"
docker version --format 'server={{.Server.Version}}'
docker system df || true

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "PREPARE_ONLY=1: dependencies and Docker daemon verified; no benchmark started"
  exit 0
fi

WORK_DIR="${WORK_DIR:-$ROOT/results/evalscope/docker-${LABEL}-$(date +%Y%m%d_%H%M%S)}"
args=(
  --config "$CONFIG"
  --model-catalog "$ROOT/configs/models/g1h-single-replica.toml"
  --base-url "$BASE_URL"
  --api-key "$API_KEY"
  --no-server
  --work-dir "$WORK_DIR"
  --eval-batch-size 1
  "${AGENT_ARGS[@]}"
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

cd "$ROOT"
exec env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  "$UV_BIN" run --no-default-groups --group "$UV_GROUP" --no-sync \
  helicopter eval evalscope "$MODEL_ALIAS" "$DATASET" "${args[@]}"
