#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# build.sh - Build TorchDelegate and package it as a single platform wheel
#
# Output: output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
#         (Python API + C++ headers / shared library / CMake config package)
#
# Usage:
#   ./build.sh                     # Build release version (default)
#   ./build.sh --debug             # Build debug version
#   ./build.sh --release           # Build release version (explicit)
#   ./build.sh --skip_third_party  # Skip third-party library build
#   ./build.sh --with_libtorch_npu     # Include libtorch_npu build
#   ./build.sh --output_dir=/path  # Custom output directory (default: output/package)
#   ./build.sh --cann_3rd_lib_path=/path  # Custom third-party library directory
#
# The script sources build.conf for all version numbers and download URLs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

source "${PROJECT_ROOT}/build.conf"

# Build configuration
BUILD_TYPE="Release"
BUILD_DIR="build/release"
TORCH_PKG_DIR=""
PYTHON_BIN=""
CANN_3RD_LIB_PATH=""
VERSION="${TORCH_DELEGATE_VERSION:?TORCH_DELEGATE_VERSION not defined in build.conf}"
OUTPUT_DIR="output/package"
WHEEL_FILE=""
SKIP_THIRD_PARTY="NO"
BUILD_LIBTORCH_NPU="OFF"
USE_SYSTEM_PYTHON="NO"
PRECHECK_ONLY="NO"
BUILD_TYPE_SET=""
START_TIME=$(date +%s)

# ============================================================================
# Logging utilities
# ============================================================================
PHASE_COUNTER=0
PHASE_TOTAL=5
CURRENT_PHASE_START=""

log_phase() {
    PHASE_COUNTER=$((PHASE_COUNTER + 1))
    CURRENT_PHASE_START=$(date +%s)
    local total=${PHASE_TOTAL}
    echo ""
    printf '════════════════════════════════════════════════════════════════════════════════\n'
    printf "  ▶ [%s/%s] %s\n" "${PHASE_COUNTER}" "${total}" "$1"
    printf '════════════════════════════════════════════════════════════════════════════════\n'
}

log_step() {
    echo "  ● $1"
}

log_success() {
    local elapsed
    elapsed=$(($(date +%s) - CURRENT_PHASE_START))
    echo "  ✔ $1 (耗时: ${elapsed}s)"
}

log_skip() {
    echo "  ○ $1"
}

log_error() {
    echo "  ✘ $1"
}

log_warn() {
    echo "  [!] $1"
}

# ============================================================================
# Usage
# ============================================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --debug              Build debug version"
    echo "  --release            Build release version (default)"
    echo "  --cann_3rd_lib_path=PATH  Specify third-party libraries directory"
    echo "  --output_dir=PATH    Output directory for packages (default: output/package)"
    echo "  --skip_third_party   Skip third-party library build"
    echo "  --pre_check          Check environment and exit (no build)"
    echo "  --with_libtorch_npu     Build libtorch_npu (default: OFF)"
    echo "  --system_python        Use current python environment (no .venv creation)"
    echo "  --help               Show this help message"
    echo ""
    echo "Version configuration is read from build.conf"
    echo "  TorchDelegate: ${VERSION}"
    echo "  PyTorch:       ${TORCH_VERSION}"
    echo "  CANN:          ${CANN_VERSION}"
    echo ""
    echo "Output (single release artifact):"
    echo "  <output_dir>/torch_delegate-<version>-py3-none-linux_<arch>.whl"
    echo "  Debug builds append the '+debug' local version segment."
    echo ""
    echo "Examples:"
    echo "  ./build.sh"
    echo "  ./build.sh --with_libtorch_npu"
    echo "  ./build.sh --output_dir=/custom/output"
}

# ============================================================================
# Argument parsing
# ============================================================================
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
                BUILD_DIR="build/debug"
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
                BUILD_DIR="build/release"
                BUILD_TYPE_SET="yes"
                shift
                ;;
            --cann_3rd_lib_path=*)
                local val="${1#*=}"
                if [[ -z "$val" ]]; then
                    echo "Error: --cann_3rd_lib_path requires a non-empty path" >&2
                    usage
                    exit 1
                fi
                CANN_3RD_LIB_PATH="$(realpath "$val")"
                shift
                ;;
            --output_dir=*)
                local val="${1#*=}"
                if [[ -z "$val" ]]; then
                    echo "Error: --output_dir requires a non-empty path" >&2
                    usage
                    exit 1
                fi
                OUTPUT_DIR="$val"
                shift
                ;;
            --skip_third_party)
                SKIP_THIRD_PARTY="YES"
                shift
                ;;
            --pre_check)
                PRECHECK_ONLY="YES"
                shift
                ;;
            --with_libtorch_npu)
                BUILD_LIBTORCH_NPU="ON"
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

    # 归一化为绝对路径：相对路径补 PROJECT_ROOT 前缀，去除末尾斜杠
    if [[ "${OUTPUT_DIR}" != /* ]]; then
        OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
    fi
    OUTPUT_DIR="${OUTPUT_DIR%/}"
}

# ============================================================================
# Phase 0: Precheck
# ============================================================================
phase_precheck() {
    log_phase "环境检查"

    # Python version check
    log_step "检查当前系统激活的 Python 环境..."
    PYTHON_BIN="$(command -v python3 || true)"
    if [[ -z "${PYTHON_BIN}" ]] || [[ ! -x "${PYTHON_BIN}" ]]; then
        log_error "python3 not found. 请准备 Python 3.10 ~ 3.13 环境"
        exit 1
    fi
    local py_ver=$(${PYTHON_BIN} --version 2>&1 | awk '{print $2}')
    local py_major=$(echo "$py_ver" | cut -d. -f1)
    local py_minor=$(echo "$py_ver" | cut -d. -f2)
    if [[ "$py_major" -ne 3 || "$py_minor" -lt 10 || "$py_minor" -gt 13 ]]; then
        log_error "python3 版本 ${py_ver} 不在支持范围 (3.10 ~ 3.13)"
        exit 1
    fi
    echo "  [✓] python3 ${py_ver} (${PYTHON_BIN})"

    log_step "检查 python3-venv..."
    if ! ${PYTHON_BIN} -m venv --help &> /dev/null; then
        log_error "python3-venv 不可用. 安装: apt install python3-venv"
        exit 1
    fi
    echo "  [✓] python3-venv"

    # cmake
    log_step "检查 cmake..."
    if ! command -v cmake &> /dev/null; then
        log_error "cmake not found. 请安装 cmake >= 3.18"
        exit 1
    fi
    echo "  [✓] cmake $(cmake --version | head -1 | awk '{print $3}')"

    # gcc
    log_step "检查 gcc..."
    if ! command -v gcc &> /dev/null; then
        log_error "gcc not found"
        exit 1
    fi
    local gcc_ver=$(gcc -dumpversion)
    local gcc_major=$(echo "$gcc_ver" | cut -d. -f1)
    if [[ "$gcc_major" -gt 13 ]]; then
        log_warn "gcc ${gcc_ver} >= 14, 建议使用 gcc 8 ~ 13"
    fi
    echo "  [✓] gcc ${gcc_ver}"

    # CANN
    log_step "检查 CANN..."
    if [[ -z "${ASCEND_HOME_PATH}" ]]; then
        log_error "ASCEND_HOME_PATH not set. 请 source CANN 环境："
        echo "    source /path/to/Ascend/cann/set_env.sh"
        exit 1
    fi
    if [[ ! -d "${ASCEND_HOME_PATH}" ]]; then
        log_error "ASCEND_HOME_PATH directory does not exist: ${ASCEND_HOME_PATH}"
        exit 1
    fi
    echo "  [✓] CANN ${ASCEND_HOME_PATH}"

    log_step "检查 torch..."
    local torch_ver
    torch_ver=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if [[ -n "${torch_ver}" ]]; then
        if [[ "${torch_ver%%+*}" == "${TORCH_VERSION%%+*}" ]]; then
            echo "  [✓] torch ${torch_ver} (匹配 ${TORCH_VERSION})"
        else
            log_warn "torch ${torch_ver} != ${TORCH_VERSION}"
        fi
    else
        echo "  [·] torch 未安装"
    fi

    # disk
    log_step "检查磁盘空间..."
    local available=$(df -m "${PROJECT_ROOT}" | tail -1 | awk '{print $4}')
    if [[ "$available" -lt 5120 ]]; then
        log_warn "磁盘空间不足: ${available} MB (建议 >= 5120 MB)"
    else
        echo "  [✓] 可用 ${available} MB"
    fi

    log_success "环境检查通过"
}

# ============================================================================
# Phase 1: Third-party libraries
# ============================================================================
phase_third_party() {
    log_phase "编译三方库 (nlohmann_json, gtest, libtorch_npu)"

    if [[ "${SKIP_THIRD_PARTY}" == "YES" ]]; then
        log_skip "跳过三方库编译"
        return 0
    fi

    local tp_args=()
    if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
        tp_args+=("--third_party_path=${CANN_3RD_LIB_PATH}")
    fi
    if [[ "${BUILD_LIBTORCH_NPU}" == "ON" ]]; then
        tp_args+=("--with_libtorch_npu")
        if [[ -n "${PYTHON_BIN}" ]]; then
            tp_args+=("--python=${PYTHON_BIN}")
        fi
    fi

    log_step "执行 build_third_party.sh ${tp_args[*]}"

    bash "${PROJECT_ROOT}/build_third_party.sh" "${tp_args[@]}"

    if [[ -z "${CANN_3RD_LIB_PATH}" ]]; then
        CANN_3RD_LIB_PATH="${PROJECT_ROOT}/output/third_party"
    fi

    log_success "三方库编译完成"
}

# ============================================================================
# Phase 2: Python environment
# ============================================================================
phase_venv() {
    if [[ "${USE_SYSTEM_PYTHON}" == "YES" ]]; then
        log_phase "使用当前 Python 环境 (--system_python)"

        if [[ -z "${PYTHON_BIN}" ]] || ! ${PYTHON_BIN} -m pip --version &> /dev/null; then
            log_error "pip not found in current environment"
            exit 1
        fi
        echo "  [✓] ${PYTHON_BIN}"

        log_step "检查 torch..."
        local installed_ver
        installed_ver=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
        if [[ -z "${installed_ver}" ]]; then
            log_error "torch 未安装，请使用当前 Python 安装："
            printf '    pip install -r requirements.txt \\\n'
            printf '        --index-url https://download.pytorch.org/whl/cpu \\\n'
            printf '        --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple\n'
            exit 1
        elif [[ "${installed_ver%%+*}" != "${TORCH_VERSION%%+*}" ]]; then
            log_warn "torch ${installed_ver} != ${TORCH_VERSION}，请手动修正"
        else
            echo "  [✓] torch ${installed_ver} (匹配 ${TORCH_VERSION})"
        fi

        log_warn "使用当前环境，请确保已安装依赖: pip install -r requirements.txt"

    else
        log_phase "准备 Python 环境 (自动创建 .venv)"

        if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
            log_step "激活已有 .venv..."
        else
            log_step "创建 .venv..."
            ${PYTHON_BIN} -m venv "${PROJECT_ROOT}/.venv"
        fi
        source "${PROJECT_ROOT}/.venv/bin/activate"
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python3"

        log_step "安装 Python 依赖..."
        ${PYTHON_BIN} -m pip install -r "${PROJECT_ROOT}/requirements.txt" \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
    fi

    TORCH_PKG_DIR=$(${PYTHON_BIN} -c "import torch, os; print(os.path.dirname(torch.__file__))")
    echo "  [✓] TORCH_PKG_DIR=${TORCH_PKG_DIR}"

    log_success "Python 环境就绪"
}

# ============================================================================
# Phase 3: cmake build
# ============================================================================
phase_cmake() {
    log_phase "编译 C++ delegate 库 (${BUILD_TYPE})"

    mkdir -p "${BUILD_DIR}"

    local cmake_args=(
        "-S" "${PROJECT_ROOT}"
        "-B" "${BUILD_DIR}"
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DBUILD_DELEGATE=ON"
        "-DBUILD_TESTS=OFF"
        "-DASCEND_INSTALL_PATH=${ASCEND_HOME_PATH}"
        "-DTORCH_DELEGATE_PACKAGE_VERSION=${VERSION}"
        "-DTorch_DIR=${TORCH_PKG_DIR}/share/cmake/Torch"
    )

    if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
        cmake_args+=("-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")
    fi

    log_step "cmake 配置..."
    cmake "${cmake_args[@]}"

    log_step "编译..."
    cmake --build "${BUILD_DIR}" -j$(nproc 2>/dev/null || echo 4)

    log_step "安装..."
    cmake --install "${BUILD_DIR}" --prefix "${BUILD_DIR}/install"

    log_success "C++ delegate 库编译完成"
}

# ============================================================================
# Phase 4: Python whl (embeds C++ artifacts)
# ============================================================================
resolve_wheel_version() {
    if [[ "${BUILD_TYPE}" == "Debug" ]]; then
        echo "${VERSION}+debug"
    else
        echo "${VERSION}"
    fi
}

phase_whl() {
    log_phase "构建 Python whl 包 (内嵌 C++ 制品)"

    mkdir -p "${OUTPUT_DIR}"

    local install_dir
    install_dir="$(cd "${BUILD_DIR}/install" 2>/dev/null && pwd)" || true
    if [[ -z "${install_dir}" ]]; then
        log_error "CMake 安装树不存在: ${BUILD_DIR}/install"
        log_error "请先执行 cmake 阶段 (phase_cmake)"
        exit 1
    fi

    local wheel_version
    wheel_version="$(resolve_wheel_version)"

    export TORCH_DELEGATE_CPP_INSTALL_DIR="${install_dir}"
    export TORCH_DELEGATE_PACKAGE_VERSION="${wheel_version}"

    log_step "C++ 安装树: ${TORCH_DELEGATE_CPP_INSTALL_DIR}"
    log_step "whl 版本:   ${TORCH_DELEGATE_PACKAGE_VERSION}"
    log_step "执行: ${PYTHON_BIN} -m pip wheel . --no-build-isolation --no-deps"
    local pip_log="${BUILD_DIR}/pip_wheel.log"
    if ! ${PYTHON_BIN} -m pip wheel "${PROJECT_ROOT}" \
        --wheel-dir "${OUTPUT_DIR}" \
        --no-build-isolation --no-deps > "${pip_log}" 2>&1; then
        tail -30 "${pip_log}"
        log_error "whl 构建失败，详见 ${pip_log}"
        exit 1
    fi
    tail -3 "${pip_log}"
    rm -rf "${PROJECT_ROOT}/torch_delegate.egg-info" 2>/dev/null || true

    local whl_name
    whl_name=$(ls -t "${OUTPUT_DIR}"/torch_delegate-*.whl 2>/dev/null | head -1 | xargs -r basename 2>/dev/null)
    if [[ -z "${whl_name}" ]]; then
        log_error "未找到产出的 whl: ${OUTPUT_DIR}/torch_delegate-*.whl"
        exit 1
    fi
    echo "    产出: ${whl_name}"
    WHEEL_FILE="${OUTPUT_DIR}/${whl_name}"

    log_success "whl 构建完成 → ${OUTPUT_DIR}"
}

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    local total_elapsed
    total_elapsed=$(($(date +%s) - START_TIME))
    echo ""
    printf '════════════════════════════════════════════════════════════════════════════════\n'
    echo "  构建完成  (总耗时: ${total_elapsed}s)"
    printf '────────────────────────────────────────────────────────────────────────────────\n'
    echo "  配置"
    echo "    构建类型:        ${BUILD_TYPE}"
    echo "    构建目录:        ${BUILD_DIR}"
    echo "    输出目录:        ${OUTPUT_DIR}"
    echo "    版本:            $(resolve_wheel_version)"
    echo "    CANN:            ${ASCEND_HOME_PATH}"
    echo ""
    echo "  产物 (唯一发布物)"
    echo "    ${WHEEL_FILE:-${OUTPUT_DIR}/torch_delegate-$(resolve_wheel_version)-py3-none-linux_<arch>.whl}"
    echo ""
    echo "  安装"
    echo "    pip install ${OUTPUT_DIR}/torch_delegate-*.whl"
    echo ""
    echo "  C++ 接入 (前缀发现)"
    echo "    export TORCH_DELEGATE_ROOT=\$(torch-delegate-config --cmake-prefix)"
    echo "    export TORCH_PKG_DIR=\$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
    echo "    cmake -S . -B build -DTorchDelegate_ROOT_DIR=\$TORCH_DELEGATE_ROOT -DTORCH_PKG_DIR=\$TORCH_PKG_DIR"
    echo ""
    echo "  运行前 (手动配置库路径)"
    echo "    source \$ASCEND_HOME_PATH/set_env.sh"
    echo "    export LD_LIBRARY_PATH=\$TORCH_PKG_DIR/lib:\$TORCH_PKG_DIR.libs:\$TORCH_DELEGATE_ROOT/lib:\$ASCEND_HOME_PATH/lib64:\$LD_LIBRARY_PATH"
    printf '════════════════════════════════════════════════════════════════════════════════\n'
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    parse_args "$@"

    # Adjust phase counter based on run mode
    if [[ "${PRECHECK_ONLY}" == "YES" ]]; then
        PHASE_TOTAL=1
    fi

    phase_precheck
    if [[ "${PRECHECK_ONLY}" == "YES" ]]; then
        exit 0
    fi

    phase_venv
    phase_third_party
    phase_cmake
    phase_whl

    print_summary
}

main "$@"
