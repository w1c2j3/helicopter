#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-$ROOT/.venv}"
EVAL_VENV="${EVAL_VENV:-$ROOT/.venv-lighteval}"
UV="${UV:-uv}"
INSTALL_COMPONENTS="${INSTALL_COMPONENTS:-rwkv-lm,vllm-rwkv,verl-rwkv,lighteval,dev}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
UPDATE_UV="${UPDATE_UV:-0}"
UV_UPGRADE="${UV_UPGRADE:-0}"
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"
UV_SYNC_INEXACT="${UV_SYNC_INEXACT:-1}"
VERL_REINSTALL="${VERL_REINSTALL:-auto}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
BUILD_TMPDIR="${BUILD_TMPDIR:-}"
UV_INDEX_URL="${UV_INDEX_URL:-${PYPI_INDEX_URL:-}}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
CARGO_REGISTRY_MIRROR="${CARGO_REGISTRY_MIRROR:-}"
CARGO_REGISTRY_MIRROR_NAME="${CARGO_REGISTRY_MIRROR_NAME:-rsproxy-sparse}"
BUN_VERSION="1.3.14"
BUN_LINUX_X64_SHA256="951ee2aee855f08595aeec6225226a298d3fea83a3dcd6465c09cbccdf7e848f"
BUN_LINUX_AARCH64_SHA256="a27ffb63a8310375836e0d6f668ae17fa8d8d18b88c37c821c65331973a19a3b"

export VLLM_BUILD_PROFILE

RWKV_LM="$ROOT/src/train/rwkv-lm"
VERL="$ROOT/src/train/verl-rwkv"

export PATH="$VENV/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_cmd "$@"
  [[ "${DRY_RUN:-0}" == "1" ]] || "$@"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

die() {
  echo "error: $*" >&2
  exit 1
}

component_enabled() {
  local expected="$1" component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  for component in "${components[@]}"; do
    [[ "$component" == "$expected" ]] && return 0
  done
  return 1
}

validate_install_components() {
  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  ((${#components[@]} > 0)) || die "INSTALL_COMPONENTS must select at least one dependency group"
  for component in "${components[@]}"; do
    case "$component" in
      dev | flash-rwkv | fla-rwkv | vllm-rwkv | verl-rwkv | rwkv-lm | verl-liger | lighteval | scoreboard-server | scoreboard-client) ;;
      full)
        die "INSTALL_COMPONENTS=full is disabled; select explicit dependency groups"
        ;;
      *)
        die "unknown INSTALL_COMPONENTS entry '$component'; use a comma-separated subset of dev,flash-rwkv,fla-rwkv,vllm-rwkv,verl-rwkv,rwkv-lm,verl-liger,lighteval,scoreboard-server,scoreboard-client"
        ;;
    esac
  done
}

validate_uv_upgrade() {
  case "$UV_UPGRADE" in
    0 | lock | 1) ;;
    *)
      die "UV_UPGRADE=$UV_UPGRADE is invalid; use 0 for locked sync, lock to refresh lockfiles without a broad upgrade, or 1 for a broad upgrade"
      ;;
  esac
}

append_uv_sync_policy() {
  local -n sync_args_ref="$1"
  case "$UV_UPGRADE" in
    0) sync_args_ref+=(--locked) ;;
    lock) ;;
    1) sync_args_ref+=(--upgrade) ;;
  esac
}

native_component_enabled() {
  component_enabled flash-rwkv || component_enabled vllm-rwkv ||
    component_enabled verl-rwkv ||
    component_enabled rwkv-lm || component_enabled lighteval
}

vllm_package_enabled() {
  component_enabled vllm-rwkv
}

python_component_enabled() {
  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  for component in "${components[@]}"; do
    [[ "$component" == "scoreboard-client" ]] || return 0
  done
  return 1
}

case "${INSTALL_PROFILE:-}" in
  "" | rwkv) ;;
  full) die "INSTALL_PROFILE=full is disabled; use INSTALL_COMPONENTS" ;;
  *) die "INSTALL_PROFILE=${INSTALL_PROFILE} is disabled; use INSTALL_COMPONENTS" ;;
esac
case "${HELICOPTER_VLLM_BUILD_PROFILE:-}" in
  "") ;;
  full) die "HELICOPTER_VLLM_BUILD_PROFILE=full is disabled; use VLLM_BUILD_PROFILE=rwkv" ;;
  *) die "HELICOPTER_VLLM_BUILD_PROFILE is unsupported; use VLLM_BUILD_PROFILE=rwkv" ;;
esac
[[ "$VLLM_BUILD_PROFILE" == "rwkv" ]] ||
  die "VLLM_BUILD_PROFILE=$VLLM_BUILD_PROFILE is disabled; only rwkv is supported"
validate_install_components
validate_uv_upgrade

warn() {
  echo "warning: $*" >&2
}

version_at_least() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

configure_network() {
  [[ -n "$HF_ENDPOINT" ]] && export HF_ENDPOINT
  [[ -n "${UV_LINK_MODE:-}" ]] && export UV_LINK_MODE

  if [[ -n "$CARGO_REGISTRY_MIRROR" ]]; then
    export CARGO_HOME="${CARGO_HOME:-$VENV/.cargo}"
    mkdir -p "$CARGO_HOME"
    cat >"$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = "$CARGO_REGISTRY_MIRROR_NAME"

[source.$CARGO_REGISTRY_MIRROR_NAME]
registry = "$CARGO_REGISTRY_MIRROR"
EOF
  fi
}

configure_build_dirs() {
  if [[ -n "$BUILD_TMPDIR" ]]; then
    if [[ "$BUILD_TMPDIR" != /* ]]; then
      BUILD_TMPDIR="$ROOT/$BUILD_TMPDIR"
    fi
    mkdir -p "$BUILD_TMPDIR"
    BUILD_TMPDIR="$(cd "$BUILD_TMPDIR" && pwd -P)"
    [[ -w "$BUILD_TMPDIR" ]] || die "BUILD_TMPDIR is not writable: $BUILD_TMPDIR"
    export TMPDIR="$BUILD_TMPDIR"
  fi
}

clean_submodule_venvs() {
  native_component_enabled || return 0
  [[ "$CLEAN_SUBMODULE_VENVS" == "1" ]] || return 0

  local env_dir
  for env_dir in \
    "$FLASH_RWKV/.venv" \
    "$VLLM/.venv" \
    "$VERL/.venv" \
    "$RWKV_LM/.venv"; do
    [[ -e "$env_dir" ]] || continue
    [[ "$env_dir" == "$ROOT"/src/*/.venv ]] || die "refusing to remove unexpected venv path: $env_dir"
    run rm -rf "$env_dir"
  done
}

clean_vllm_cmake_cache() {
  [[ "$CLEAN_VLLM_CMAKE_CACHE" == "1" ]] || return 0
  [[ -d "$VLLM/.deps" ]] || return 0

  local subbuild_dir
  while IFS= read -r subbuild_dir; do
    [[ -n "$subbuild_dir" ]] || continue
    [[ "$subbuild_dir" == "$VLLM/.deps/"*-subbuild ]] ||
      die "refusing to remove unexpected CMake cache path: $subbuild_dir"
    run rm -rf "$subbuild_dir"
  done < <(find "$VLLM/.deps" -maxdepth 1 -type d -name '*-subbuild' -print | LC_ALL=C sort)
}

ensure_uv() {
  if ! have "$UV"; then
    have curl || die "uv is missing and curl is not available to install it"
    run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    have "$UV" || UV="$(command -v uv || true)"
    [[ "${DRY_RUN:-0}" == "1" || -n "$UV" ]] || die "uv installation finished but uv is still not on PATH"
  fi

  if [[ "$UPDATE_UV" == "1" ]]; then
    run "$UV" self update || warn "uv self update failed; continuing with installed uv"
  fi
}

ensure_bun() {
  component_enabled scoreboard-client || return 0
  if [[ -x "$VENV/bin/bun" ]] &&
    [[ "$("$VENV/bin/bun" --version)" == "$BUN_VERSION" ]]; then
    return 0
  fi
  have curl || die "curl is required to install Bun $BUN_VERSION"
  have sha256sum || die "sha256sum is required to verify Bun $BUN_VERSION"
  [[ -x "$VENV/bin/python" ]] ||
    die "workspace Python is required before installing Bun"

  local architecture archive_name expected_sha256 download_url
  case "$(uname -m)" in
    x86_64)
      architecture="x64"
      expected_sha256="$BUN_LINUX_X64_SHA256"
      ;;
    aarch64 | arm64)
      architecture="aarch64"
      expected_sha256="$BUN_LINUX_AARCH64_SHA256"
      ;;
    *)
      die "Bun $BUN_VERSION is not pinned for architecture $(uname -m)"
      ;;
  esac
  archive_name="bun-linux-$architecture.zip"
  download_url="https://github.com/oven-sh/bun/releases/download/bun-v$BUN_VERSION/$archive_name"

  local temporary_root archive extracted binary actual_sha256
  temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/helicopter-bun.XXXXXX")"
  archive="$temporary_root/$archive_name"
  extracted="$temporary_root/extracted"
  run curl --fail --location --retry 3 --output "$archive" "$download_url"
  actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    rm -rf -- "$temporary_root"
    die "Bun $BUN_VERSION SHA-256 mismatch for $archive_name"
  fi
  mkdir -p "$extracted"
  run "$VENV/bin/python" -m zipfile -e "$archive" "$extracted"
  binary="$extracted/bun-linux-$architecture/bun"
  [[ -f "$binary" && ! -L "$binary" ]] || {
    rm -rf -- "$temporary_root"
    die "Bun $BUN_VERSION archive does not contain the expected binary"
  }
  run install -m 0755 "$binary" "$VENV/bin/bun"
  rm -rf -- "$temporary_root"
  [[ "$("$VENV/bin/bun" --version)" == "$BUN_VERSION" ]] ||
    die "installed Bun version does not match $BUN_VERSION"
}

install_system_deps() {
  [[ "$INSTALL_SYSTEM_DEPS" == "1" ]] || return 0
  have apt-get || die "INSTALL_SYSTEM_DEPS=1 currently supports apt-get only"
  run sudo apt-get update
  run sudo apt-get install -y --no-install-recommends \
    build-essential curl git ninja-build pkg-config
}

check_compiler_env() {
  native_component_enabled || return 0
  local missing=()
  have cc || missing+=("cc")
  have c++ || missing+=("c++")

  if ((${#missing[@]})); then
    install_system_deps
    missing=()
    have cc || missing+=("cc")
    have c++ || missing+=("c++")
  fi

  ((${#missing[@]} == 0)) || die "missing C/C++ build tools: ${missing[*]}"
}

check_native_env() {
  native_component_enabled || return 0
  local missing=()
  have cmake || missing+=("cmake")
  have ninja || missing+=("ninja")
  ((${#missing[@]} == 0)) || die "missing native build tools after uv sync: ${missing[*]}"

  local cmake_version
  cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
  version_at_least "$cmake_version" "3.26" || die "cmake >= 3.26 is required; found $cmake_version"

  if have g++; then
    local gcc_version
    gcc_version="$(g++ -dumpfullversion -dumpversion)"
    version_at_least "$gcc_version" "11.3" || die "g++ >= 11.3 is required; found $gcc_version"
  fi

  if [[ "${VLLM_REQUIRE_RUST_FRONTEND:-0}" == "1" ]]; then
    have rustc || die "rustc is required when VLLM_REQUIRE_RUST_FRONTEND=1"
    have cargo || die "cargo is required when VLLM_REQUIRE_RUST_FRONTEND=1"
  fi
}

sync_uv_env() {
  local sync_args=(sync)
  [[ -n "$UV_INDEX_URL" ]] && sync_args+=(--index-url "$UV_INDEX_URL")
  [[ "$UV_SYNC_INEXACT" == "1" ]] && sync_args+=(--inexact)
  sync_args+=(--project "$ROOT" --python "$PYTHON_VERSION" --no-default-groups)
  append_uv_sync_policy sync_args

  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  for component in "${components[@]}"; do
    case "$component" in
      lighteval | scoreboard-server | scoreboard-client) ;;
      *) sync_args+=(--group "$component") ;;
    esac
  done

  run "$UV" "${sync_args[@]}"
}

verl_ready() {
  "$VENV/bin/python" - <<'PY' >/dev/null
import verl
PY
}

install_rwkv_lm_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  if [[ -f "$RWKV_LM/pyproject.toml" || -f "$RWKV_LM/setup.py" ]]; then
    run "${pip[@]}" --no-deps -e "$RWKV_LM"
  else
    echo "rwkv-lm has no local package metadata; dependencies are covered by pyproject.toml"
  fi
}

install_verl_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  if [[ "$VERL_REINSTALL" != "1" ]] && verl_ready; then
    echo "verl editable package is already installed; reusing existing install"
    return 0
  fi
  run "${pip[@]}" --no-deps -e "$VERL"
}

check_python_packages() {
  [[ "$RUN_PIP_CHECK" == "1" ]] || return 0

  print_cmd "$UV" pip check --project "$ROOT" --python "$VENV/bin/python"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0

  local check_output filtered_output
  if check_output="$("$UV" pip check --project "$ROOT" --python "$VENV/bin/python" 2>&1)"; then
    printf '%s\n' "$check_output"
    return 0
  fi

  filtered_output="$(printf '%s\n' "$check_output" |
    grep -v -F 'The package `nvidia-cusparselt-cu13` was built for a different platform' |
    grep -v -E '^(Using Python .+|Checked [0-9]+ packages in .+|Found 1 incompatibility)$' || true)"
  if [[ -z "$filtered_output" ]] &&
     [[ "$check_output" == *'The package `nvidia-cusparselt-cu13` was built for a different platform'* ]]; then
    printf '%s\n' "$check_output" >&2
    warn "ignoring uv platform-tag check for nvidia-cusparselt-cu13; NVIDIA publishes the aarch64 wheel with a manylinux2014_sbsa tag"
    return 0
  fi

  printf '%s\n' "$check_output" >&2
  return 1
}

check_lighteval_packages() {
  component_enabled lighteval || return 0
  [[ "$RUN_PIP_CHECK" == "1" ]] || return 0

  print_cmd "$UV" pip check --project "$ROOT" --python "$EVAL_VENV/bin/python"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0

  local check_output filtered_output
  if check_output="$("$UV" pip check --project "$ROOT" --python "$EVAL_VENV/bin/python" 2>&1)"; then
    printf '%s\n' "$check_output"
    return 0
  fi
  filtered_output="$(printf '%s\n' "$check_output" |
    grep -v -F 'The package `nvidia-cusparselt-cu13` was built for a different platform' |
    grep -v -E '^(Using Python .+|Checked [0-9]+ packages in .+|Found 1 incompatibility)$' || true)"
  if [[ -z "$filtered_output" ]] &&
     [[ "$check_output" == *'The package `nvidia-cusparselt-cu13` was built for a different platform'* ]]; then
    printf '%s\n' "$check_output" >&2
    warn "ignoring uv platform-tag check for nvidia-cusparselt-cu13 in the LightEval environment"
    return 0
  fi
  printf '%s\n' "$check_output" >&2
  return 1
}

configure_network
configure_build_dirs
clean_submodule_venvs
python_component_enabled && ensure_uv
check_compiler_env
python_component_enabled && sync_uv_env
sync_lighteval_env
sync_scoreboard_server
sync_scoreboard_client
check_native_env
install_rwkv_lm_package
install_verl_package

if [[ "$RUN_PIP_CHECK" == "1" ]]; then
  run "$UV" pip check --project "$ROOT" --python "$VENV/bin/python"
fi
python_component_enabled && check_python_packages
check_lighteval_packages

clean_submodule_venvs

echo "Environment ready: $VENV"
if component_enabled lighteval; then
  echo "LightEval environment ready: $EVAL_VENV"
fi
