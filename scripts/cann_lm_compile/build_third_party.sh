#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# build_third_party.sh — 构建 CANN-LM 自带 C++ 三方库 (nlohmann_json + GTest + makeself)
#
# 调用本仓 cmake/third_party/build/CMakeLists.txt (ExternalProject_Add 从 gitcode
# 镜像下载 + 编译), 落 ${PROJECT_ROOT}/output/third_party/{json,gtest,makeself}.
#
# 幂等: 已装则跳 (Find 模块探测到 json.hpp / libgtest.a / makeself.sh 直接 no-op).
# 产物供 build_cpp.sh (csrc backend + tests) 与 package_run.sh (makeself) 使用.
#
# 用法:
#   bash scripts/cann_lm_compile/build_third_party.sh
#   bash scripts/cann_lm_compile/build_third_party.sh --third_party_path=/custom/path
#   bash scripts/cann_lm_compile/build_third_party.sh --jobs=8
#
# 环境覆盖: PROJECT_ROOT / NLOHMANN_JSON_VERSION / GTEST_VERSION / MAKESELF_VERSION
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${REPO}/scripts/cann_lm_compile/build.conf"

DEFAULT_THIRD_PARTY_PATH="${REPO}/output/third_party"
THIRD_PARTY_PATH="${DEFAULT_THIRD_PARTY_PATH}"
BUILD_TYPE="Release"
PARALLEL_JOBS="$(nproc 2>/dev/null || echo 4)"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --third_party_path=PATH   安装路径 (默认: output/third_party)"
    echo "  --build_type=TYPE         Release/Debug (默认 Release)"
    echo "  --jobs=N                  并行数 (默认 auto)"
    echo "  --help                    帮助"
    echo ""
    echo "Libraries built (gitcode 镜像):"
    echo "  - nlohmann_json ${NLOHMANN_JSON_VERSION}"
    echo "  - GoogleTest    ${GTEST_VERSION}"
    echo "  - makeself      ${MAKESELF_VERSION}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --third_party_path=*) THIRD_PARTY_PATH="$(realpath "${1#*=}")"; shift;;
            --build_type=*)        BUILD_TYPE="${1#*=}"; shift;;
            --jobs=*)             PARALLEL_JOBS="${1#*=}"; shift;;
            --help)               usage; exit 0;;
            *) echo "Unknown option: $1" >&2; usage; exit 1;;
        esac
    done
}

check_tools() {
    for tool in cmake make; do
        command -v "$tool" >/dev/null || { echo "Error: '$tool' not found" >&2; exit 1; }
    done
}

build_all() {
    local build_dir="${REPO}/build/third_party"
    local cm_dir="${REPO}/cmake/third_party/build"

    echo "========================================"
    echo "Building third-party libraries (CANN-LM 自带)"
    echo "========================================"
    echo "  Project root : ${REPO}"
    echo "  Install path : ${THIRD_PARTY_PATH}"
    echo "  Build type   : ${BUILD_TYPE}"
    echo "  Parallel jobs: ${PARALLEL_JOBS}"
    echo ""

    mkdir -p "${THIRD_PARTY_PATH}" "${build_dir}"

    local cmake_args=(
        "-B" "${build_dir}"
        "-S" "${cm_dir}"
        "-DCMAKE_THIRD_PARTY_LIB_DIR=${THIRD_PARTY_PATH}"
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DTORCH_DELEGATE_VERSION=${TORCH_DELEGATE_VERSION}"
        "-DNLOHMANN_JSON_VERSION=${NLOHMANN_JSON_VERSION}"
        "-DGTEST_VERSION=${GTEST_VERSION}"
        "-DMAKESELF_VERSION=${MAKESELF_VERSION}"
        "-DBUILD_LIBTORCH_NPU=OFF"
    )

    echo "Running: cmake ${cmake_args[*]}"
    cmake "${cmake_args[@]}"

    echo ""
    echo "Building..."
    cmake --build "${build_dir}" --target nlohmann_json_build gtest_build makeself_build -- -j"${PARALLEL_JOBS}"

    echo ""
    echo "========================================"
    echo "Build Summary"
    echo "========================================"
    echo "Installation path: ${THIRD_PARTY_PATH}"
    echo "  - nlohmann_json: ${THIRD_PARTY_PATH}/json"
    echo "  - gtest:         ${THIRD_PARTY_PATH}/gtest"
    echo "  - makeself:      ${THIRD_PARTY_PATH}/makeself"
}

main() {
    parse_args "$@"
    check_tools
    build_all
}

main "$@"
