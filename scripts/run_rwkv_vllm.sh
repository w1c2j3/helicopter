#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_root="${project_root}/src/infer/vllm-rwkv"
vllm_venv="${HELICOPTER_VLLM_VENV:-${project_root}/.venv}"
if [[ "${vllm_venv}" != /* ]]; then
  vllm_venv="${project_root}/${vllm_venv}"
fi
legacy_vllm_venv="${vllm_root}/.venv-rwkv"
if [[ -z "${HELICOPTER_VLLM_VENV:-}" && ! -x "${vllm_venv}/bin/vllm" &&
      -x "${legacy_vllm_venv}/bin/vllm" ]]; then
  vllm_venv="${legacy_vllm_venv}"
fi
model_path="${project_root}/models/rwkv7/rwkv7-g1i-1.5b-20260805-ctx16384.pth"
model_path="${RWKV_MODEL_PATH:-${model_path}}"
host="127.0.0.1"
port="8000"
served_model_name="rwkv7-g1i-1.5b"
max_model_len="16384"
max_num_seqs="16"
max_num_batched_tokens="16384"
gpu_memory_utilization="0.85"
extra_args=()

while (($# > 0)); do
  case "$1" in
    --host|--port|--served-model-name|--max-model-len|--max-num-seqs|--max-num-batched-tokens|--gpu-memory-utilization)
      if (($# < 2)); then
        printf 'missing value for %s\n' "$1" >&2
        exit 2
      fi
      option="${1#--}"
      option="${option//-/_}"
      printf -v "${option}" '%s' "$2"
      shift 2
      ;;
    --host=*|--port=*|--served-model-name=*|--max-model-len=*|--max-num-seqs=*|--max-num-batched-tokens=*|--gpu-memory-utilization=*)
      option="${1%%=*}"
      value="${1#*=}"
      option="${option#--}"
      option="${option//-/_}"
      printf -v "${option}" '%s' "${value}"
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${vllm_venv}/bin/python" || ! -x "${vllm_venv}/bin/vllm" ]]; then
  printf 'RWKV-vLLM environment not found: %s\n' "${vllm_venv}" >&2
  printf 'prepare it with INSTALL_COMPONENTS=vllm-rwkv,lm-eval,dev scripts/install_local.sh\n' >&2
  exit 2
fi

export PATH="${vllm_venv}/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export VLLM_RWKV7_WKV_MODE="${VLLM_RWKV7_WKV_MODE:-fp16}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

if [[ ! -f "${model_path}" ]]; then
  printf 'RWKV model not found: %s\n' "${model_path}" >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  printf 'invalid vLLM port: %s\n' "${port}" >&2
  exit 2
fi
for value in "${max_model_len}" "${max_num_seqs}" "${max_num_batched_tokens}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'vLLM capacity values must be positive integers\n' >&2
    exit 2
  fi
done

client_host="${host}"
if [[ "${client_host}" == "0.0.0.0" ]]; then
  client_host="127.0.0.1"
elif [[ "${client_host}" == "::" || "${client_host}" == "[::]" ]]; then
  client_host="[::1]"
fi

runtime_dir="${HELICOPTER_RUNTIME_DIR:-${project_root}/.tmp/runtime}"
if [[ "${runtime_dir}" != /* ]]; then
  runtime_dir="${project_root}/${runtime_dir}"
fi
manifest_path="${HELICOPTER_VLLM_POOL_MANIFEST:-${runtime_dir}/rwkv-vllm-pool.json}"
if [[ "${manifest_path}" != /* ]]; then
  printf 'HELICOPTER_VLLM_POOL_MANIFEST must be an absolute path\n' >&2
  exit 2
fi
manifest_dir="$(dirname "${manifest_path}")"
mkdir -p "${runtime_dir}" "${manifest_dir}"
if [[ "${runtime_dir}" == "${project_root}/.tmp/runtime" ]]; then
  chmod 700 "${runtime_dir}"
fi

vllm_version="$("${vllm_venv}/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')"
weight_sha256="$(sha256sum "${model_path}" | cut -d ' ' -f 1)"
export HELICOPTER_LOCAL_MANIFEST_PATH="${manifest_path}"
export HELICOPTER_LOCAL_MANIFEST_VERSION="${vllm_version}"
export HELICOPTER_LOCAL_MANIFEST_BASE_URL="http://${client_host}:${port}"
export HELICOPTER_LOCAL_MANIFEST_MAX_CONCURRENCY="${max_num_seqs}"
export HELICOPTER_LOCAL_MANIFEST_MAX_MODEL_LEN="${max_model_len}"
export HELICOPTER_LOCAL_MANIFEST_WEIGHT_SHA256="${weight_sha256}"
export HELICOPTER_LOCAL_MANIFEST_WEIGHT_NAME="$(basename "${model_path}")"
export HELICOPTER_LOCAL_MANIFEST_GLOBAL_STEP="${HELICOPTER_GLOBAL_STEP:-0}"

"${vllm_venv}/bin/python" - <<'PY'
import json
import os
from pathlib import Path
from uuid import uuid4

target = Path(os.environ["HELICOPTER_LOCAL_MANIFEST_PATH"])
payload = {
    "schema_version": 1,
    "global_step": int(os.environ["HELICOPTER_LOCAL_MANIFEST_GLOBAL_STEP"]),
    "wkv_mode": os.environ["VLLM_RWKV7_WKV_MODE"],
    "vllm_version": os.environ["HELICOPTER_LOCAL_MANIFEST_VERSION"],
    "max_model_len": int(os.environ["HELICOPTER_LOCAL_MANIFEST_MAX_MODEL_LEN"]),
    "weight_sha256": os.environ["HELICOPTER_LOCAL_MANIFEST_WEIGHT_SHA256"],
    "weight_display_name": os.environ["HELICOPTER_LOCAL_MANIFEST_WEIGHT_NAME"],
    "replicas": [
        {
            "base_url": os.environ["HELICOPTER_LOCAL_MANIFEST_BASE_URL"],
            "max_concurrency": int(
                os.environ["HELICOPTER_LOCAL_MANIFEST_MAX_CONCURRENCY"]
            ),
        }
    ],
}
temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.chmod(0o600)
temporary.replace(target)
PY

printf 'RWKV-vLLM pool manifest: %s\n' "${manifest_path}"
unset HELICOPTER_LOCAL_MANIFEST_PATH HELICOPTER_LOCAL_MANIFEST_VERSION
unset HELICOPTER_LOCAL_MANIFEST_BASE_URL HELICOPTER_LOCAL_MANIFEST_MAX_CONCURRENCY
unset HELICOPTER_LOCAL_MANIFEST_MAX_MODEL_LEN HELICOPTER_LOCAL_MANIFEST_WEIGHT_SHA256
unset HELICOPTER_LOCAL_MANIFEST_WEIGHT_NAME HELICOPTER_LOCAL_MANIFEST_GLOBAL_STEP

exec "${vllm_venv}/bin/vllm" serve "${model_path}" \
  --host "${host}" \
  --port "${port}" \
  --served-model-name "${served_model_name}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  "${extra_args[@]}"
