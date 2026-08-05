#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

config_path="configs/eval/lm_eval.toml"
if (($# > 0)) && [[ "$1" != -* ]]; then
  config_path="$1"
  shift
fi

eval_cli="${HELICOPTER_CLI:-${project_root}/.venv/bin/helicopter}"
if [[ ! -x "${eval_cli}" ]]; then
  printf 'helicopter CLI not found: %s\n' "${eval_cli}" >&2
  printf 'prepare the project environment with scripts/install_local.sh\n' >&2
  exit 2
fi

# An explicit manifest belongs to an external service lifecycle. In that case the
# evaluator performs the normal pool preflight and this wrapper never starts vLLM.
if [[ -n "${HELICOPTER_VLLM_POOL_MANIFEST:-}" ]]; then
  exec "${eval_cli}" eval --evaluator lm-eval --config "${config_path}" "$@"
fi

runtime_dir="${HELICOPTER_RUNTIME_DIR:-${project_root}/.tmp/runtime}"
if [[ "${runtime_dir}" != /* ]]; then
  runtime_dir="${project_root}/${runtime_dir}"
fi
manifest_path="${runtime_dir}/rwkv-vllm-pool.json"
server_log="${runtime_dir}/rwkv-vllm.log"
export HELICOPTER_VLLM_POOL_MANIFEST="${manifest_path}"
startup_timeout="${HELICOPTER_VLLM_STARTUP_TIMEOUT:-300}"
if [[ ! "${startup_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'HELICOPTER_VLLM_STARTUP_TIMEOUT must be a positive integer\n' >&2
  exit 2
fi
mkdir -p "${runtime_dir}"
if [[ "${runtime_dir}" == "${project_root}/.tmp/runtime" ]]; then
  chmod 700 "${runtime_dir}"
fi
if [[ -L "${server_log}" ]]; then
  printf 'refusing symlinked RWKV-vLLM log: %s\n' "${server_log}" >&2
  exit 2
fi

manifest_is_healthy() {
  [[ -f "${manifest_path}" && ! -L "${manifest_path}" ]] || return 1
  "${project_root}/.venv/bin/python" - "${manifest_path}" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

try:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    base_url = manifest["replicas"][0]["base_url"].rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base_url}/health", timeout=2) as response:
        if response.status != 200:
            raise RuntimeError(f"health returned HTTP {response.status}")
except Exception:
    raise SystemExit(1)
PY
}

server_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}

if ! manifest_is_healthy; then
  : >"${server_log}"
  chmod 600 "${server_log}"
  "${project_root}/scripts/run_rwkv_vllm.sh" >"${server_log}" 2>&1 &
  server_pid=$!
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  deadline=$((SECONDS + startup_timeout))
  printf 'Starting local RWKV-vLLM; log: %s\n' "${server_log}"
  until manifest_is_healthy; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      printf 'RWKV-vLLM exited before becoming ready. Last log lines:\n' >&2
      tail -n 80 "${server_log}" >&2 || true
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      printf 'RWKV-vLLM did not become ready within %s seconds. Last log lines:\n' \
        "${startup_timeout}" >&2
      tail -n 80 "${server_log}" >&2 || true
      exit 1
    fi
    sleep 1
  done
  printf 'Local RWKV-vLLM is ready.\n'
fi

"${eval_cli}" eval --evaluator lm-eval --config "${config_path}" "$@"
