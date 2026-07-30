#!/usr/bin/env bash
set -u

# Re-run the low-environment Agent benchmark matrix against one local RWKV
# endpoint.  The script deliberately keeps one benchmark in flight per model
# endpoint, preserves a separate work directory per dataset, and continues
# after a benchmark-specific environment failure so later scores remain
# usable.  Raw traces and reports stay under results/ and are not source data.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-/home/chase/.local/bin/uv}"
MODEL_ALIAS="${1:?usage: $0 MODEL_ALIAS BASE_URL LABEL [DATASET ...]}"
BASE_URL="${2:?usage: $0 MODEL_ALIAS BASE_URL LABEL [DATASET ...]}"
LABEL="${3:?usage: $0 MODEL_ALIAS BASE_URL LABEL [DATASET ...]}"
shift 3

WORK_ROOT="${WORK_ROOT:-$ROOT/results/evalscope/retest-${LABEL}-$(date +%Y%m%d_%H%M%S)}"
GENERATION_CONFIG="${GENERATION_CONFIG:-experiments/evalscope_agent/flower-nocot-generation-2048.json}"
API_KEY="${HELICOPTER_EVAL_API_KEY:-rwkv-skills}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"

EVAL_BATCH_ARGS=()
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
  EVAL_BATCH_ARGS=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi

if (($#)); then
  DATASETS=("$@")
else
  DATASETS=(
    bfcl_v4
    acebench
    gaia
    officeqa
    general_fc
    k2_verifier
    kimi_verifier
    minimax_verifier
    tool_bench
  )
fi

mkdir -p "$WORK_ROOT"
for dataset in "${DATASETS[@]}"; do
  work_dir="$WORK_ROOT/$dataset"
  mkdir -p "$work_dir"
  echo "=== START $dataset model=$MODEL_ALIAS base_url=$BASE_URL work_dir=$work_dir ==="
  adapter_args=()
  case "$dataset" in
    acebench|tool_bench)
      # These EvalScope adapters are static/single-turn adapters.  Injecting
      # the global AgentLoop makes their benchmark-defined API calls execute
      # against the unrelated bash tool and changes the official protocol.
      adapter_args+=(--no-agent-config)
      ;;
  esac
  (
    cd "$ROOT" || exit 1
    env \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
      "$UV_BIN" run --project "$ROOT" helicopter eval evalscope \
        "$MODEL_ALIAS" "$dataset" \
        --config configs/example.toml \
        --model-catalog configs/models/g1h-single-replica.toml \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --no-server \
        --generation-config "$GENERATION_CONFIG" \
        --parallel-candidate-router \
        --candidate-max-tokens 2048 \
        --aggregate-max-tokens 2048 \
        "${EVAL_BATCH_ARGS[@]}" \
        "${adapter_args[@]}" \
        --work-dir "$work_dir"
  )
  rc=$?
  echo "=== END $dataset exit_code=$rc work_dir=$work_dir ==="
done
