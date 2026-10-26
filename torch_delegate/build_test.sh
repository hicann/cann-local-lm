#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# build_test.sh - Build and run TorchDelegate tests
#
# Usage:
#   ./build_test.sh                 # Build and run tests (ASAN enabled by default)
#   ./build_test.sh --debug         # Build debug tests
#   ./build_test.sh --release       # Build release tests
#   ./build_test.sh --asan          # Build with ASAN (default)
#   ./build_test.sh --no-asan       # Build without ASAN
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

source "${PROJECT_ROOT}/build.conf"

BUILD_TYPE="Debug"
BUILD_DIR="build_tests"
ENABLE_ASAN="ON"
TORCH_PKG_DIR=""
PYTHON_BIN=""
CANN_3RD_LIB_PATH=""
RUN_TESTS="YES"
SKIP_THIRD_PARTY="NO"
USE_SYSTEM_PYTHON="NO"
BUILD_TYPE_SET=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --debug              Build debug tests (default)"
    echo "  --release            Build release tests"
    echo "  --asan               Build with AddressSanitizer (default)"
    echo "  --no-asan            Build without AddressSanitizer"
    echo "  --cann_3rd_lib_path=PATH  Specify third-party libraries directory (nlohmann_json, gtest)"
    echo "  --skip_third_party   Skip third-party library build"
    echo "  --skip_tests         Skip running tests (only build)"
    echo "  --system_python      Use current python environment (no .venv creation)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./build_test.sh                     # Build and run tests with ASAN"
    echo "  ./build_test.sh --release --no-asan # Build release tests without ASAN"
    echo "  ./build_test.sh --skip_third_party  # Reuse existing third-party libraries"
    echo "  ./build_test.sh --skip_tests        # Build tests without running"
    echo "  ./build_test.sh --cann_3rd_lib_path=/path/to/third_party"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --debug)
                if [[ -n "${BUILD_TYPE_SET}" && "${BUILD_TYPE}" != "Debug" ]]; then
                    echo "Error: --debug and --release are mutually exclusive" >&2
                    usage
                    exit 1
                fi
                BUILD_TYPE="Debug"
                BUILD_TYPE_SET="yes"
                shift
                ;;
            --release)
                if [[ -n "${BUILD_TYPE_SET}" && "${BUILD_TYPE}" != "Release" ]]; then
                    echo "Error: --debug and --release are mutually exclusive" >&2
                    usage
                    exit 1
                fi
                BUILD_TYPE="Release"
                BUILD_TYPE_SET="yes"
                shift
                ;;
            --asan)
                ENABLE_ASAN="ON"
                shift
                ;;
            --no-asan)
                ENABLE_ASAN="OFF"
                shift
                ;;
            --cann_3rd_lib_path=*)
                CANN_3RD_LIB_PATH="$(realpath "${1#*=}")"
                shift
                ;;
            --skip_third_party)
                SKIP_THIRD_PARTY="YES"
                shift
                ;;
            --skip_tests)
                RUN_TESTS="NO"
                shift
                ;;
            --system_python)
                USE_SYSTEM_PYTHON="YES"
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

check_dependencies() {
    echo "Checking dependencies..."

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        echo "Error: CMake not found. Please install CMake >= 3.18"
        exit 1
    fi

    # Check ASCEND_HOME_PATH
    if [[ -z "${ASCEND_HOME_PATH}" ]]; then
        echo "Error: ASCEND_HOME_PATH not set"
        echo "       Please source CANN environment: source /path/to/Ascend/cann/set_env.sh"
        exit 1
    fi

    if [[ ! -d "${ASCEND_HOME_PATH}" ]]; then
        echo "Error: ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        exit 1
    fi

    echo "Dependencies OK"
}

prepare_venv() {
    echo ""
    echo "========================================"
    echo "Preparing Python Environment"
    echo "========================================"

    if [[ "${USE_SYSTEM_PYTHON}" == "YES" ]]; then
        echo "Using current Python environment (--system_python)"
        PYTHON_BIN="$(command -v python3 || true)"
        if [[ -z "${PYTHON_BIN}" ]] || ! ${PYTHON_BIN} -m pip --version &> /dev/null; then
            echo "Error: pip not found in current environment"
            exit 1
        fi
        echo "  python: ${PYTHON_BIN}"

        echo "Checking torch..."
        local installed_ver
        installed_ver=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
        if [[ -z "${installed_ver}" ]]; then
            echo "Error: torch not installed, please install with:"
            printf '    pip install -r requirements.txt \\\n'
            printf '        --index-url https://download.pytorch.org/whl/cpu \\\n'
            printf '        --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple\n'
            exit 1
        elif [[ "${installed_ver%%+*}" != "${TORCH_VERSION%%+*}" ]]; then
            echo "  [!] torch ${installed_ver} != ${TORCH_VERSION}, please fix manually"
            exit 1
        else
            echo "  [+] torch ${installed_ver} (matches ${TORCH_VERSION})"
        fi

        echo "  [!] Ensure deps installed: pip install -r requirements.txt"

    else
        if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
            echo "Activating existing .venv..."
        else
            local sys_python
            sys_python="$(command -v python3 || true)"
            if [[ -z "${sys_python}" ]] || [[ ! -x "${sys_python}" ]]; then
                echo "Error: python3 not found"
                exit 1
            fi
            echo "Creating .venv..."
            ${sys_python} -m venv "${PROJECT_ROOT}/.venv"
        fi
        source "${PROJECT_ROOT}/.venv/bin/activate"
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python3"

        echo "Installing Python dependencies..."
        ${PYTHON_BIN} -m pip install -r "${PROJECT_ROOT}/requirements.txt" \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
    fi

    TORCH_PKG_DIR=$(${PYTHON_BIN} -c "import torch, os; print(os.path.dirname(torch.__file__))")
    echo "  [+] TORCH_PKG_DIR=${TORCH_PKG_DIR}"
    echo ""
    echo "Python environment ready$([ "${USE_SYSTEM_PYTHON}" == "YES" ] && echo " (--system_python)" )"
}

prepare_third_party() {
    echo ""
    echo "========================================"
    echo "Preparing Third-Party Libraries"
    echo "========================================"

    if [[ "${SKIP_THIRD_PARTY}" == "YES" ]]; then
        echo "Skipping third-party build (--skip_third_party)"
    else
        local tp_args=()
        if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
            tp_args+=("--third_party_path=${CANN_3RD_LIB_PATH}")
        fi

        echo "Running: build_third_party.sh ${tp_args[*]}"
        bash "${PROJECT_ROOT}/build_third_party.sh" "${tp_args[@]}"
    fi

    if [[ -z "${CANN_3RD_LIB_PATH}" ]]; then
        CANN_3RD_LIB_PATH="${PROJECT_ROOT}/output/third_party"
    fi
    echo "  [✓] CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
}

build_tests() {
    echo ""
    echo "========================================"
    echo "Building TorchDelegate Tests (${BUILD_TYPE})"
    echo "========================================"

    if [[ "${ENABLE_ASAN}" == "ON" ]]; then
        echo "AddressSanitizer: ENABLED"
    else
        echo "AddressSanitizer: DISABLED"
    fi

    mkdir -p "${BUILD_DIR}"

    local cmake_args=(
        "-S" "${PROJECT_ROOT}"
        "-B" "${BUILD_DIR}"
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DBUILD_DELEGATE=ON"
        "-DBUILD_TESTS=ON"
        "-DENABLE_ASAN=${ENABLE_ASAN}"
        "-DASCEND_INSTALL_PATH=${ASCEND_HOME_PATH}"
    )

    cmake_args+=("-DTorch_DIR=${TORCH_PKG_DIR}/share/cmake/Torch")

    # Add CANN_3RD_LIB_PATH if specified
    if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
        cmake_args+=("-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")
    fi

    echo "Running: cmake ${cmake_args[@]}"
    cmake "${cmake_args[@]}"

    echo ""
    echo "Building..."
    cmake --build "${BUILD_DIR}" -j$(nproc 2>/dev/null || echo 4)
}

run_tests() {
    if [[ "${RUN_TESTS}" == "NO" ]]; then
        echo ""
        echo "Skipping tests (--skip_tests specified)"
        return
    fi

    echo ""
    echo "========================================"
    echo "Running Tests"
    echo "========================================"

    local test_bin="${BUILD_DIR}/tests/torch_delegate_tests"
    if [[ ! -x "${test_bin}" ]]; then
        echo "Error: test binary not found: ${test_bin}"
        exit 1
    fi

    local ge_stub_dir="${BUILD_DIR}/tests/cpp/ge_stub/lib64"
    local cann_stub_dir="${BUILD_DIR}/tests/cpp/cann_stub/lib64"

    export PYTHONUNBUFFERED=1

    local test_ld_path="${ge_stub_dir}:${cann_stub_dir}:${LD_LIBRARY_PATH}"

    local preload_libs=""
    if [[ -f "${cann_stub_dir}/libascendcl.so" ]]; then
        preload_libs="${cann_stub_dir}/libascendcl.so"
    fi

    if [[ "${ENABLE_ASAN}" == "ON" ]]; then
        echo "Running tests with AddressSanitizer..."
        local asan_rt=$(gcc -print-file-name=libasan.so 2>/dev/null || echo "")
        if [[ -z "${asan_rt}" ]] || [[ "${asan_rt}" == "libasan.so" ]]; then
            asan_rt=$(gcc -print-file-name=libasan.so.6 2>/dev/null || echo "")
        fi
        if [[ -n "${asan_rt}" ]] && [[ "${asan_rt}" != "libasan.so"* ]] && [[ -f "${asan_rt}" ]]; then
            echo "  LD_PRELOAD=${asan_rt}:${preload_libs}"
            export LD_PRELOAD="${asan_rt}:${preload_libs}"
        else
            echo "  [!] Warning: libasan.so not found, ASAN may not work correctly"
            export LD_PRELOAD="${preload_libs}"
        fi
        ASAN_OPTIONS=detect_leaks=1 LD_LIBRARY_PATH="${test_ld_path}" "${test_bin}"
    else
        echo "Running tests..."
        echo "  LD_PRELOAD=${preload_libs}"
        LD_PRELOAD="${preload_libs}" LD_LIBRARY_PATH="${test_ld_path}" "${test_bin}"
    fi
}

print_summary() {
    echo ""
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo "Build type: ${BUILD_TYPE}"
    echo "Build directory: ${BUILD_DIR}"
    echo "AddressSanitizer: ${ENABLE_ASAN}"
    echo "CANN: ${ASCEND_HOME_PATH}"
    echo "Torch package directory: ${TORCH_PKG_DIR}"
    echo "Torch cmake config: ${TORCH_PKG_DIR}/share/cmake/Torch"
    if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
        echo "Third-party path: ${CANN_3RD_LIB_PATH}"
    else
        echo "Third-party path: (default: output/third_party)"
    fi
    echo ""
    echo "Tests run: ${RUN_TESTS}"
}

main() {
    parse_args "$@"
    check_dependencies
    prepare_third_party
    prepare_venv
    build_tests
    run_tests
    print_summary
}

main "$@"
