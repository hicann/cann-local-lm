#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# build_third_party.sh - Build and install third-party libraries for torch-delegate
#
# Usage:
#   ./build_third_party.sh                     # Install to output/third_party
#   ./build_third_party.sh --third_party_path=/custom/path  # Custom install path
#
# Libraries built:
#   - nlohmann_json - gitcode mirror
#   - GoogleTest - gitcode mirror
#   - libtorch_npu - Ascend C++ NPU plugin
#
# Note: libtorch is now obtained from Python environment (venv) or user-specified Torch_DIR.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CMAKE_THIRD_PARTY_DIR="${PROJECT_ROOT}/cmake/third_party/build"
source "${PROJECT_ROOT}/build.conf"

DEFAULT_THIRD_PARTY_PATH="${PROJECT_ROOT}/output/third_party"
THIRD_PARTY_PATH="${DEFAULT_THIRD_PARTY_PATH}"

BUILD_TYPE="Release"
PARALLEL_JOBS=$(nproc 2>/dev/null || echo 4)
BUILD_LIBTORCH_NPU="OFF"
PYTHON_EXE=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --third_party_path=PATH   Installation path (default: output/third_party)"
    echo "  --build_type=TYPE         Build type: Release/Debug (default: Release)"
    echo "  --jobs=N                  Parallel build jobs (default: auto)"
    echo "  --help                    Show this help message"
    echo ""
    echo "Libraries built:"
    echo "  - nlohmann_json ${NLOHMANN_JSON_VERSION}"
    echo "  - GoogleTest ${GTEST_VERSION}"
    echo "  - libtorch_npu ${LIBTORCH_NPU_RELEASE_TAG}"
    echo ""
    echo "Note: libtorch is obtained from Python environment (venv) or Torch_DIR"
    echo ""
    echo "Note: libtorch is obtained from Python environment (venv) or Torch_DIR"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --third_party_path=*)
                THIRD_PARTY_PATH="$(realpath "${1#*=}")"
                shift
                ;;
            --build_type=*)
                BUILD_TYPE="${1#*=}"
                shift
                ;;
            --jobs=*)
                PARALLEL_JOBS="${1#*=}"
                shift
                ;;
            --with_libtorch_npu)
                BUILD_LIBTORCH_NPU="ON"
                shift
                ;;
            --python=*)
                PYTHON_EXE="${1#*=}"
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

check_tools() {
    local tools=("cmake" "make")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            echo "Error: Required tool '$tool' not found"
            exit 1
        fi
    done
}

build_all() {
    local build_dir="${PROJECT_ROOT}/build/third_party"

    echo "========================================"
    echo "Building third-party libraries via CMake"
    echo "========================================"

    mkdir -p "${build_dir}"

    local cmake_args=(
        "-B" "${build_dir}"
        "-S" "${CMAKE_THIRD_PARTY_DIR}"
        "-DCMAKE_THIRD_PARTY_LIB_DIR=${THIRD_PARTY_PATH}"
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DTORCH_DELEGATE_VERSION=${TORCH_DELEGATE_VERSION}"
        "-DNLOHMANN_JSON_VERSION=${NLOHMANN_JSON_VERSION}"
        "-DGTEST_VERSION=${GTEST_VERSION}"
        "-DBUILD_LIBTORCH_NPU=${BUILD_LIBTORCH_NPU}"
        "-DLIBTORCH_NPU_SRC_URL=${LIBTORCH_NPU_SRC_URL}"
        "-DLIBTORCH_NPU_BRANCH=${LIBTORCH_NPU_BRANCH}"
        "-DLIBTORCH_NPU_RELEASE_TAG=${LIBTORCH_NPU_RELEASE_TAG}"
    )

    if [[ -n "${PYTHON_EXE}" ]]; then
        cmake_args+=("-DPYTHON_EXECUTABLE=${PYTHON_EXE}")
    fi

    echo "Running: cmake ${cmake_args[@]}"
    cmake "${cmake_args[@]}"

    echo ""
    echo "Building..."

    local targets="nlohmann_json_build gtest_build"
    if [[ "${BUILD_LIBTORCH_NPU}" == "ON" ]]; then
        targets="${targets} libtorch_npu_build"
    fi
    cmake --build "${build_dir}" --target ${targets} -- -j"${PARALLEL_JOBS}"

    echo ""
    echo "Third-party libraries built successfully"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "Build Summary"
    echo "========================================"
    echo "Installation path: ${THIRD_PARTY_PATH}"
    echo ""
    echo "Installed libraries:"
    echo "  - nlohmann_json: ${THIRD_PARTY_PATH}/json"
    echo "  - gtest:         ${THIRD_PARTY_PATH}/gtest"
    if [[ "${BUILD_LIBTORCH_NPU}" == "ON" ]]; then
        echo "  - libtorch_npu:   ${THIRD_PARTY_PATH}/libtorch_npu"
    else
        echo "  - libtorch_npu:   (skipped)"
    fi
    echo ""
    echo "Note: libtorch is obtained from Python environment (venv) or Torch_DIR"
    echo ""
    echo "To use these libraries in CMake:"
    echo ""
    echo "  Option 1: Use default path (output/third_party)"
    echo "    ./build.sh"
    echo "    ./build_test.sh"
    echo ""
    echo "  Option 2: Specify CANN_3RD_LIB_PATH"
    echo "    ./build.sh --cann_3rd_lib_path=${THIRD_PARTY_PATH}"
    echo "    ./build_test.sh --cann_3rd_lib_path=${THIRD_PARTY_PATH}"
    echo ""
    echo "  Option 3: CMake variable"
    echo "    cmake -B build -DCANN_3RD_LIB_PATH=${THIRD_PARTY_PATH}"
    echo ""
}

main() {
    parse_args "$@"
    check_tools

    echo "Project root: ${PROJECT_ROOT}"
    echo "Third-party path: ${THIRD_PARTY_PATH}"
    echo "Build type: ${BUILD_TYPE}"
    echo "Parallel jobs: ${PARALLEL_JOBS}"
    echo ""

    mkdir -p "${THIRD_PARTY_PATH}"

    build_all

    print_summary
}

main "$@"
