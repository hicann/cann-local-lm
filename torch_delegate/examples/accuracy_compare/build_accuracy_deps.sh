#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# build_accuracy_deps.sh - 构建 accuracy_compare 用例的专有依赖
#
# 构建内容：
#   1. xllm_ops (从 https://gitcode.com/xLLM-AI/xllm_ops/) 并安装到 CANN
#   2. stub_kernel (从 tests/cpp/stub_kernel)
#
# 注意：libtorch_npu 由公共脚本 examples/common/build_libtorch_npu.sh 构建，
#       产物在 examples/deps/libtorch_npu，本脚本不再重复构建。
#
# 用法：
#   bash build_accuracy_deps.sh [options]
#
# 选项：
#   --skip-xllm-ops      跳过 xllm_ops 构建
#   --skip-stub-kernel   跳过 stub_kernel 构建
#   --help               显示帮助信息
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${EXAMPLES_DIR}/deps"
WORK_DIR="/tmp/torch_delegate_deps_build"

SKIP_XLLM_OPS=0
SKIP_STUB_KERNEL=0

# 从 build.conf 读取版本信息
source "${PROJECT_ROOT}/build.conf"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "构建 accuracy_compare 用例的专有依赖："
    echo "  - xllm_ops (beam_search)"
    echo "  - stub_kernel"
    echo ""
    echo "注意：libtorch_npu 由公共脚本构建到 ${EXAMPLES_DIR}/deps/libtorch_npu"
    echo ""
    echo "选项："
    echo "  --skip-xllm-ops      跳过 xllm_ops 构建"
    echo "  --skip-stub-kernel   跳过 stub_kernel 构建"
    echo "  --help               显示帮助信息"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-xllm-ops)
                SKIP_XLLM_OPS=1
                shift
                ;;
            --skip-stub-kernel)
                SKIP_STUB_KERNEL=1
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo "[Error] 未知选项: $1"
                usage
                exit 1
                ;;
        esac
    done
}

check_env() {
    echo "[Check] 检查环境..."

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "[Error] ASCEND_HOME_PATH 未设置"
        echo "[Error] 请先执行: source /path/to/Ascend/cann/set_env.sh"
        exit 1
    fi
    echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

    if ! command -v python3 &> /dev/null; then
        echo "[Error] python3 未找到"
        exit 1
    fi

    if ! command -v cmake &> /dev/null; then
        echo "[Error] cmake 未找到"
        exit 1
    fi

    if ! command -v git &> /dev/null; then
        echo "[Error] git 未找到"
        exit 1
    fi

    echo "  python3: $(python3 --version), $(which python3)"
    echo "  cmake: $(cmake --version | head -1)"
    echo "  git: $(git --version)"
}

build_xllm_ops() {
    if [[ ${SKIP_XLLM_OPS} -eq 1 ]]; then
        echo "[Skip] xllm_ops"
        return
    fi

    echo ""
    echo "========================================"
    echo "[1/2] 构建 xllm_ops (beam_search)"
    echo "========================================"

    local src_dir="${WORK_DIR}/xllm_ops_src"
    local install_dir="${OUTPUT_DIR}/xllm_ops"

    if [[ -f "${install_dir}/.build_success" ]]; then
        echo "[Info] xllm_ops 已存在，跳过构建"
        echo "  路径: ${install_dir}"
        return
    fi

    mkdir -p "${WORK_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    # Clone xllm_ops repository (pinned to cann-9.0 branch)
    # 固定分支原因：beam_search.cpp/h 存在 const-correctness bug，
    # Init() 函数将 const TopkTiling& 传给非 const 引用参数，导致编译失败。
    # cann-9.0 分支已确认包含此问题，后续版本可能已修复。
    if [[ ! -d "${src_dir}" ]]; then
        echo "[Step] 克隆 xllm_ops 仓库 (branch=cann-9.0)..."
        git clone --depth 1 --branch cann-9.0 https://gitcode.com/xLLM-AI/xllm_ops.git "${src_dir}"
    else
        echo "[Info] xllm_ops 源码已存在，跳过克隆"
    fi

    # Apply beam_search const-correctness patch
    # Patch 原因：beam_search.cpp/h 的 Init() 函数签名存在 const-correctness bug
    # 错误信息：binding reference of type 'TopkTiling' to const TopkTiling drops const qualifier
    # 修复：将 TopkTiling &topkTilingData, TopkTiling &topKTilingData1 改为 const 引用
    # 参见：patches/beam_search_const_tiling.patch
    local patch_file="${SCRIPT_DIR}/patches/beam_search_const_tiling.patch"
    if [[ ! -f "${patch_file}" ]]; then
        echo "[Error] 未找到 patch 文件: ${patch_file}"
        echo "[Error]   该 patch 用于修复 beam_search 的 const-correctness 编译错误"
        echo "[Error]   解决方案：请确认 examples/accuracy_compare/patches/ 目录包含 beam_search_const_tiling.patch"
        cd "${SCRIPT_DIR}"
        exit 1
    fi

    cd "${src_dir}"
    if git apply --check "${patch_file}" 2>/dev/null; then
        echo "[Patch] 应用 beam_search const-correctness 修复..."
        if ! git apply "${patch_file}"; then
            echo "[Error] beam_search patch 应用失败！"
            echo "[Error]   patch 预检 (git apply --check) 通过但实际 git apply 失败，可能文件内容有变化"
            echo "[Error]   解决方案：手动检查 ${src_dir}/xllm_ops/beam_search/ 目录下的 beam_search.{h,cpp}"
            cd "${SCRIPT_DIR}"
            exit 1
        fi
        echo "[OK] beam_search patch 应用成功"
    else
        # git apply --check 失败可能有两种原因：已应用或路径/上下文不匹配
        # 检查是否已经包含 const TopkTiling（即 patch 早已存在）
        local header_file="${src_dir}/xllm_ops/beam_search/beam_search.h"
        if [[ -f "${header_file}" ]] && grep -q "const TopkTiling \&topkTilingData" "${header_file}"; then
            echo "[Skip] beam_search patch 已应用 (检测到 const TopkTiling 修饰符)"
        else
            echo "[Error] beam_search patch 无法应用，源码内容与预期不符"
            echo ""
            echo "[Error]   可能原因："
            echo "[Error]     1. 源码目录结构变化：期望 xllm_ops/beam_search/beam_search.{h,cpp}"
            if [[ ! -f "${src_dir}/xllm_ops/beam_search/beam_search.h" ]]; then
                echo "[Error]        实际文件不存在: ${src_dir}/xllm_ops/beam_search/beam_search.h"
                echo "[Error]        实际目录结构："
                find "${src_dir}" -name "beam_search.h" 2>/dev/null | sed 's|^|[Error]          |'
            fi
            echo "[Error]     2. 源码上下文行（Context lines）与 patch 不匹配，需重新生成 patch"
            echo ""
            echo "[Error]   解决方案："
            echo "[Error]     1. 删除 ${src_dir} 后重新运行 build_accuracy_deps.sh（会重新 clone）"
            echo "[Error]     2. 或手动将 Init() 函数签名中的 TopkTiling & 改为 const TopkTiling &"
            cd "${SCRIPT_DIR}"
            exit 1
        fi
    fi
    cd "${SCRIPT_DIR}"

    # Build xllm_ops
    # build.sh 构建完成后会自动执行 ./xllm_ops.run 安装到 CANN
    # 安装路径由 ASCEND_HOME_PATH 环境变量决定 (build.sh 内部逻辑 line 390 & 453)
    echo "[Step] 执行 build.sh -n beam_search..."
    echo "[Info] build.sh 会自动安装 xllm_ops 到 ${ASCEND_HOME_PATH}/opp/vendors/"
    cd "${src_dir}"
    bash build.sh -n "beam_search"

    # Copy to install directory
    if [[ -d "${src_dir}/build" ]]; then
        echo "[Step] 复制产物到 ${install_dir}..."
        mkdir -p "${install_dir}"
        cp -r "${src_dir}/build"/* "${install_dir}/" 2>/dev/null || true
        touch "${install_dir}/.build_success"
        echo "[OK] xllm_ops 构建完成 (已自动安装到 CANN)"
        echo "  构建产物: ${install_dir}"
    else
        echo "[Error] xllm_ops build 目录不存在，构建可能失败"
        echo "[Error]   源码目录: ${src_dir}"
        echo "[Error]   请检查 ASCEND_HOME_PATH 是否正确设置"
        cd "${SCRIPT_DIR}"
        exit 1
    fi

    # Verify installation
    local cann_beam_search="${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/op_impl/ai_core/tbe/custom_xllm_math_impl/ascendc/beam_search/beam_search.h"
    if [[ -f "${cann_beam_search}" ]] && grep -q "const TopkTiling &topkTilingData" "${cann_beam_search}" 2>/dev/null; then
        echo "[OK] CANN beam_search 已更新为 const-correctness 修复版本"
    else
        echo "[Warning] 未检测到 CANN beam_search const-correctness 修复"
        echo "[Warning]   文件: ${cann_beam_search}"
        echo "[Warning]   运行时可能仍会遇到编译错误"
    fi

    cd "${SCRIPT_DIR}"
}

build_stub_kernel() {
    if [[ ${SKIP_STUB_KERNEL} -eq 1 ]]; then
        echo "[Skip] stub_kernel"
        return
    fi

    echo ""
    echo "========================================"
    echo "[2/2] 构建 stub_kernel"
    echo "========================================"

    local stub_kernel_dir="${PROJECT_ROOT}/tests/cpp/stub_kernel"
    local build_dir="${stub_kernel_dir}/build"
    local libtorch_npu_dir="${OUTPUT_DIR}/libtorch_npu"

    if [[ -f "${build_dir}/libstub_kernel.so" ]]; then
        echo "[Info] stub_kernel 已存在，跳过构建"
        echo "  路径: ${build_dir}/libstub_kernel.so"
        return
    fi

    if [[ ! -d "${libtorch_npu_dir}/lib" ]]; then
        echo "[Error] libtorch_npu 未构建，请先构建 libtorch_npu"
        exit 1
    fi

    # Get TORCH_PKG_DIR from Python environment (Python torch package directory)
    local torch_dir
    torch_dir=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
    if [[ -z "${torch_dir}" ]]; then
        echo "[Error] 无法获取 torch 路径"
        exit 1
    fi
    echo "[Step] 配置 stub_kernel..."
    echo "  TORCH_PKG_DIR: ${torch_dir}"
    echo "  TORCH_NPU_DIR: ${libtorch_npu_dir}"
    echo "  CANN_ROOT: ${ASCEND_HOME_PATH}"

    mkdir -p "${build_dir}"
    cd "${stub_kernel_dir}"

    cmake -B "${build_dir}" \
        -DTORCH_PKG_DIR="${torch_dir}" \
        -DTORCH_NPU_DIR="${libtorch_npu_dir}" \
        -DCANN_ROOT="${ASCEND_HOME_PATH}" \
        -DCMAKE_BUILD_TYPE=Release

    echo "[Step] 构建 stub_kernel..."
    cmake --build "${build_dir}" -j$(nproc)

    if [[ -f "${build_dir}/libstub_kernel.so" ]]; then
        echo "[OK] stub_kernel 构建完成"
        echo "  路径: ${build_dir}/libstub_kernel.so"
    else
        echo "[Error] stub_kernel 构建失败"
        exit 1
    fi

    cd "${SCRIPT_DIR}"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "构建完成"
    echo "========================================"
    echo ""
    echo "输出目录: ${OUTPUT_DIR}"
    echo ""
    echo "  libtorch_npu: ${OUTPUT_DIR}/libtorch_npu (由公共脚本构建)"

    if [[ ${SKIP_XLLM_OPS} -eq 0 ]]; then
        echo "  xllm_ops: ${OUTPUT_DIR}/xllm_ops"
    fi

    if [[ ${SKIP_STUB_KERNEL} -eq 0 ]]; then
        echo "  stub_kernel: ${PROJECT_ROOT}/tests/cpp/stub_kernel/build/libstub_kernel.so"
    fi

    echo ""
    echo "设置环境变量:"
    echo "  export TORCH_NPU_DIR=${OUTPUT_DIR}/libtorch_npu"
}

cleanup() {
    echo ""
    echo "[Cleanup] 清理临时构建目录..."
    echo "  删除: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
    echo "  [OK] 清理完成"
}

main() {
    parse_args "$@"
    check_env

    echo ""
    echo "输出目录: ${OUTPUT_DIR}"
    echo "临时目录: ${WORK_DIR}"
    echo ""

    # 确保 libtorch_npu 已构建（公共依赖）
    source "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh"
    ensure_libtorch_npu || exit 1

    build_xllm_ops
    build_stub_kernel

    cleanup
    print_summary
}

main "$@"
