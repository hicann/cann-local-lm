#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# build_libtorch_npu.sh - 共享的 libtorch_npu 准备脚本
#
# 默认行为：从发行版下载预构建的 libtorch_npu 包
# 选项 --from-source：从源码编译 libtorch_npu
#
# 用法：
#   bash build_libtorch_npu.sh [options]
#
# 选项：
#   --output-dir=PATH    输出目录（默认：$EXAMPLES_DIR/deps）
#   --work-dir=PATH      临时工作目录（默认：/tmp/torch_delegate_libtorch_npu_build）
#   --from-source        从源码编译（默认：下载预构建包）
#   --help               显示帮助信息
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${EXAMPLES_DIR}/deps"
WORK_DIR="/tmp/torch_delegate_libtorch_npu_build"
FROM_SOURCE=false

# 预构建包下载地址
PREBUILT_URL_X86_64="https://gitcode.com/yuchen97/pytorch/releases/download/20260803_v2.9.0/libtorch_npu_x86_64.tar.gz"
PREBUILT_URL_AARCH64="https://gitcode.com/yuchen97/pytorch/releases/download/20260803_v2.9.0/libtorch_npu_aarch64.tar.gz"

source "${PROJECT_ROOT}/build.conf"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "准备 libtorch_npu v${TORCH_VERSION}"
    echo ""
    echo "选项："
    echo "  --output-dir=PATH    输出目录（默认：${OUTPUT_DIR}）"
    echo "  --work-dir=PATH      临时工作目录（默认：${WORK_DIR}）"
    echo "  --from-source        从源码编译（默认：下载预构建包）"
    echo "  --help               显示帮助信息"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir=*)
                OUTPUT_DIR="${1#*=}"
                shift
                ;;
            --work-dir=*)
                WORK_DIR="${1#*=}"
                shift
                ;;
            --from-source)
                FROM_SOURCE=true
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

    if [[ -z "${OUTPUT_DIR}" ]]; then
        echo "[Error] --output-dir 不能为空"
        usage
        exit 1
    fi
}

check_env() {
    echo "[Check] 检查环境..."

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "[Error] ASCEND_HOME_PATH 未设置"
        echo "[Error] 请先执行: source /path/to/Ascend/cann/set_env.sh"
        exit 1
    fi
    echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

    echo "  python3: $(python3 --version), $(which python3)"
    echo "  cmake: $(cmake --version | head -1)"
}

detect_arch() {
    local arch
    arch=$(uname -m)
    case "${arch}" in
        x86_64|amd64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        *)
            echo "unknown:${arch}"
            ;;
    esac
}

get_prebuilt_url() {
    local arch
    arch=$(detect_arch)
    case "${arch}" in
        x86_64)
            echo "${PREBUILT_URL_X86_64}"
            ;;
        aarch64)
            echo "${PREBUILT_URL_AARCH64}"
            ;;
        *)
            echo ""
            ;;
    esac
}

download_libtorch_npu() {
    echo ""
    echo "========================================"
    echo "下载预构建 libtorch_npu"
    echo "========================================"

    local arch
    arch=$(detect_arch)
    local url
    url=$(get_prebuilt_url)

    if [[ -z "${url}" ]]; then
        echo "[Error] 不支持的架构: ${arch}"
        echo "[Error] 请使用 --from-source 从源码编译"
        exit 1
    fi

    echo "  架构: ${arch}"
    echo "  下载 URL: ${url}"

    local install_dir="${OUTPUT_DIR}/libtorch_npu"

    if [[ -d "${install_dir}/lib" ]] && [[ -d "${install_dir}/include" ]]; then
        echo "[Info] libtorch_npu 已存在，跳过下载"
        echo "  路径: ${install_dir}"
        return
    fi

    mkdir -p "${OUTPUT_DIR}"
    local tarball="${WORK_DIR}/libtorch_npu_${arch}.tar.gz"

    mkdir -p "${WORK_DIR}"

    echo "[Step] 下载预构建包..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "${tarball}" "${url}" || true
    elif command -v curl &> /dev/null; then
        curl -fSL --progress-bar -o "${tarball}" "${url}" || true
    else
        echo "[Error] 未找到 wget 或 curl，无法下载"
        exit 1
    fi

    if [[ ! -f "${tarball}" ]]; then
        echo "[Error] 下载失败: ${url}"
        exit 1
    fi

    echo "[Step] 解压预构建包..."
    mkdir -p "${WORK_DIR}/extract"
    tar -xzf "${tarball}" -C "${WORK_DIR}/extract"

    echo "[Step] 复制到 ${install_dir}..."
    mkdir -p "${install_dir}"

    local extract_dir="${WORK_DIR}/extract"
    local entry_count
    entry_count=$(ls -A "${extract_dir}" | wc -l)

    if [[ "${entry_count}" -eq 1 ]] && [[ -d "${extract_dir}/$(ls -A "${extract_dir}")" ]]; then
        local wrapper_dir="${extract_dir}/$(ls -A "${extract_dir}")"
        if [[ -d "${wrapper_dir}/lib" ]] && [[ -d "${wrapper_dir}/include" ]]; then
            cp -r "${wrapper_dir}"/* "${install_dir}/"
        else
            cp -r "${wrapper_dir}/"* "${install_dir}/"
        fi
    else
        cp -r "${extract_dir}"/* "${install_dir}/"
    fi

    if [[ ! -d "${install_dir}/lib" ]] || [[ ! -d "${install_dir}/include" ]]; then
        echo "[Error] 预构建包缺少 lib/ 或 include/ 目录"
        exit 1
    fi

    echo "[OK] libtorch_npu 下载完成"
    echo "  路径: ${install_dir}"
}

build_libtorch_npu_from_source() {
    echo ""
    echo "========================================"
    echo "从源码编译 libtorch_npu v${TORCH_VERSION}"
    echo "========================================"

    if ! command -v git &> /dev/null; then
        echo "[Error] git 未找到，源码编译需要 git"
        exit 1
    fi

    local src_dir="${WORK_DIR}/pytorch_src"
    local install_dir="${OUTPUT_DIR}/libtorch_npu"

    if [[ -d "${install_dir}/lib" ]] && [[ -d "${install_dir}/include" ]]; then
        echo "[Info] libtorch_npu 已存在，跳过构建"
        echo "  路径: ${install_dir}"
        return
    fi

    mkdir -p "${WORK_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    if [[ ! -d "${src_dir}" ]]; then
        echo "[Step] 克隆 pytorch 仓库..."
        git clone --depth 1 --branch "${LIBTORCH_NPU_BRANCH}" \
            "${LIBTORCH_NPU_SRC_URL}" "${src_dir}"

        echo "[Step] 初始化 git submodules..."
        cd "${src_dir}"
        git submodule update --init --recursive
    else
        echo "[Info] pytorch 源码已存在，跳过克隆"
        cd "${src_dir}"
    fi

    echo "[Step] 执行 build_libtorch_npu.py..."
    export _GLIBCXX_USE_CXX11_ABI=1
    python3 build_libtorch_npu.py
    unset _GLIBCXX_USE_CXX11_ABI

    if [[ -d "${src_dir}/libtorch_npu" ]]; then
        echo "[Step] 复制产物到 ${install_dir}..."
        mkdir -p "${install_dir}"
        cp -r "${src_dir}/libtorch_npu"/* "${install_dir}/"
        echo "[OK] libtorch_npu 构建完成"
        echo "  路径: ${install_dir}"
    else
        echo "[Error] libtorch_npu 产物不存在: ${src_dir}/libtorch_npu"
        exit 1
    fi

    cd "${SCRIPT_DIR}"
}

cleanup() {
    echo ""
    echo "[Cleanup] 清理临时构建目录..."
    echo "  删除: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
    echo "  [OK] 清理完成"
}

prepare_libtorch_npu() {
    if [[ "${FROM_SOURCE}" == true ]]; then
        build_libtorch_npu_from_source
    else
        download_libtorch_npu
    fi
}

print_summary() {
    echo ""
    echo "========================================"
    echo "准备完成"
    echo "========================================"
    echo ""
    echo "输出目录: ${OUTPUT_DIR}"
    echo "  libtorch_npu: ${OUTPUT_DIR}/libtorch_npu"
    echo ""
    echo "设置环境变量:"
    echo "  export TORCH_NPU_DIR=${OUTPUT_DIR}/libtorch_npu"
}

main() {
    parse_args "$@"
    check_env

    echo ""
    echo "输出目录: ${OUTPUT_DIR}"
    echo "临时目录: ${WORK_DIR}"
    echo "模式: $(${FROM_SOURCE} && echo '从源码编译' || echo '下载预构建包')"
    echo ""

    prepare_libtorch_npu

    cleanup
    print_summary
}

main "$@"
