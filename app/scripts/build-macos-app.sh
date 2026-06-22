#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${app_root}/.." && pwd)"
dist_dir="${app_root}/dist"
dist_web_dir="${dist_dir}/web"
frontend_dist_dir="${repo_root}/frontend/dist"
target_dir="${CARGO_TARGET_DIR:-${app_root}/src-tauri/target}"
features="${AMCP_MACOS_FEATURES:-full-runtime-macos}"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
export CMAKE_OSX_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET}}"

set_default_env_path() {
  local name="$1"
  local path="$2"
  if [[ -z "${!name:-}" && -e "${path}" ]]; then
    export "${name}=$(cd "${path}" && pwd)"
  fi
}

initialize_voxtral_patched_llama_env() {
  if [[ "${features}" != *"voxtral"* && "${features}" != *"full-runtime-macos"* ]]; then
    return
  fi

  local patched_root="${repo_root}/.tmp/llama-cpp-voxtral-pr20638"
  local metal_build="${patched_root}/build-amcp-metal-release"
  local default_build="${patched_root}/build"

  set_default_env_path "AMCP_VOXTRAL_PATCHED_LLAMA_DIR" "${patched_root}"
  if [[ -z "${AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR:-}" ]]; then
    if [[ -d "${metal_build}" ]]; then
      export AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR="$(cd "${metal_build}" && pwd)"
    elif [[ -d "${default_build}" ]]; then
      export AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR="$(cd "${default_build}" && pwd)"
    fi
  fi

  if [[ -z "${AMCP_VOXTRAL_PATCHED_LLAMA_DIR:-}" || -z "${AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR:-}" ]]; then
    cat >&2 <<'EOF'
Voxtral macOS build requires patched llama.cpp.

Set these variables, or build the patched tree under .tmp/llama-cpp-voxtral-pr20638:
  AMCP_VOXTRAL_PATCHED_LLAMA_DIR
  AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR

Example:
  cmake -S .tmp/llama-cpp-voxtral-pr20638 -B .tmp/llama-cpp-voxtral-pr20638/build-amcp-metal-release -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
  cmake --build .tmp/llama-cpp-voxtral-pr20638/build-amcp-metal-release --config Release

To compile only Whisper/Qwen Metal while Voxtral is unavailable:
  AMCP_MACOS_FEATURES=desktop,whisper-metal,qwen-metal npm run build:macos:app
EOF
    exit 1
  fi

  if [[ -e "${AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR}/libggml-metal.dylib" || -e "${AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR}/bin/libggml-metal.dylib" ]]; then
    export AMCP_VOXTRAL_PATCHED_LLAMA_LINK_METAL="${AMCP_VOXTRAL_PATCHED_LLAMA_LINK_METAL:-1}"
  fi
}

copy_frontend_dist() {
  if [[ ! -f "${frontend_dist_dir}/index.html" ]]; then
    echo "Frontend build output was not found: ${frontend_dist_dir}" >&2
    exit 1
  fi
  rm -rf "${dist_web_dir}"
  cp -R "${frontend_dist_dir}" "${dist_web_dir}"
}

copy_if_exists() {
  local source="$1"
  local destination_dir="$2"
  if [[ -e "${source}" ]]; then
    mkdir -p "${destination_dir}"
    cp -R "${source}" "${destination_dir}/"
  fi
}

copy_voxtral_runtime_artifacts() {
  local lib_dir="${AMCP_VOXTRAL_PATCHED_LLAMA_LIB_DIR:-}"
  if [[ -z "${lib_dir}" || ! -d "${lib_dir}" ]]; then
    return
  fi

  local app_bundle="${target_dir}/release/bundle/macos/AMCP.app"
  local app_macos_dir="${app_bundle}/Contents/MacOS"
  local destinations=("${dist_dir}")
  if [[ -d "${app_macos_dir}" ]]; then
    destinations+=("${app_macos_dir}")
  fi

  local candidates=(
    "${lib_dir}"
    "${lib_dir}/bin"
    "${lib_dir}/src"
    "${lib_dir}/tools/mtmd"
    "${lib_dir}/ggml/src"
    "${lib_dir}/ggml/src/ggml-metal"
  )
  local artifacts=(
    "libllama.dylib"
    "libmtmd.dylib"
    "libggml.dylib"
    "libggml-base.dylib"
    "libggml-cpu.dylib"
    "libggml-metal.dylib"
    "default.metallib"
  )

  for destination in "${destinations[@]}"; do
    for candidate in "${candidates[@]}"; do
      [[ -d "${candidate}" ]] || continue
      for artifact in "${artifacts[@]}"; do
        copy_if_exists "${candidate}/${artifact}" "${destination}"
      done
    done
  done
}

cd "${app_root}"
initialize_voxtral_patched_llama_env

echo "Building AMCP for macOS with features: ${features}"
npx tauri build --features "${features}"

mkdir -p "${dist_dir}"
source_bin="${target_dir}/release/amcp-desktop"
if [[ -f "${source_bin}" ]]; then
  cp -f "${source_bin}" "${dist_dir}/AMCP-macos"
fi

source_app="${target_dir}/release/bundle/macos/AMCP.app"
if [[ -d "${source_app}" ]]; then
  rm -rf "${dist_dir}/AMCP.app"
  cp -R "${source_app}" "${dist_dir}/AMCP.app"
fi

copy_voxtral_runtime_artifacts
copy_frontend_dist

echo "macOS distributable created:"
echo "  Binary: ${dist_dir}/AMCP-macos"
echo "  App bundle: ${dist_dir}/AMCP.app"
echo "  Frontend: ${dist_web_dir}"
