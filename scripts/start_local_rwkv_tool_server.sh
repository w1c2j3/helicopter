#!/usr/bin/env bash
set -euo pipefail

# Local-only RWKV7 OpenAI-compatible service.  This intentionally does not
# use the 19329 SSH-forwarded endpoint.
MODEL_PATH="${RWKV_LOCAL_MODEL_PATH:-/home/chase/weights/rwkv7-g1h-1.5b-20260710-ctx10240.pth}"
MODEL_NAME="${RWKV_LOCAL_MODEL_NAME:-rwkv7-g1h-1.5b-20260710-ctx10240}"
PORT="${RWKV_LOCAL_PORT:-19316}"
VLLM_ROOT="${VLLM_RWKV_ROOT:-/home/chase/GitHub/vllm-rwkv}"
VLLM_PYTHON="${VLLM_RWKV_PYTHON:-/home/chase/GitHub/vllm-rwkv/.venv/bin/python}"
COMPAT_SCRIPT="${RWKV_COMPAT_SCRIPT:-/home/chase/GitHub/RWKV-ECRA/scripts/vllm_wsl_compat.py}"

test -f "$MODEL_PATH"
test -x "$VLLM_PYTHON"
test -f "$COMPAT_SCRIPT"

exec env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}" \
  VLLM_USE_RAPID_SAMPLER="${VLLM_USE_RAPID_SAMPLER:-0}" \
  uv run --no-sync --project "$VLLM_ROOT" --python "$VLLM_PYTHON" "$COMPAT_SCRIPT" \
    --model "$MODEL_PATH" \
    --host "${RWKV_LOCAL_HOST:-127.0.0.1}" \
    --port "$PORT" \
    --api-key "${RWKV_LOCAL_API_KEY:-rwkv-skills}" \
    --tokenizer-mode rwkv \
    --max-model-len "${RWKV_LOCAL_MAX_MODEL_LEN:-10240}" \
    --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization "${RWKV_LOCAL_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-num-batched-tokens "${RWKV_LOCAL_MAX_NUM_BATCHED_TOKENS:-16384}" \
    --max-num-seqs "${RWKV_LOCAL_MAX_NUM_SEQS:-64}" \
    --enable-auto-tool-choice \
    --tool-call-parser rwkv
