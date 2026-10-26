#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ensure_libtorch_npu.sh - 确保 libtorch_npu 已准备，供各用例复用
#
# 用法一（source 方式，兼容已有用例）：
#   source "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh"
#   ensure_libtorch_npu || fail "..."
#
#   设置 LIBTORCH_NPU_FROM_SOURCE=1 可从源码编译：
#     export LIBTORCH_NPU_FROM_SOURCE=1
#     source "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh"
#     ensure_libtorch_npu || fail "..."
#
# 用法二（bash 直接执行，推荐）：
#   export EXAMPLES_DIR=/path/to/examples
#   TORCH_NPU_DIR=$(bash "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh") || exit 1
#   export TORCH_NPU_DIR
#
#   bash "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh" --from-source  # 源码编译
#
# 行为：
#   1. 若 TORCH_NPU_DIR 已设置且有效，直接使用
#   2. 若 examples/deps/libtorch_npu 已构建，设置 TORCH_NPU_DIR
#   3. 否则自动触发 common/build_libtorch_npu.sh（默认下载预构建包）
#
# 依赖环境变量：
#   EXAMPLES_DIR           — examples 目录的绝对路径（由调用方设置）
#   LIBTORCH_NPU_FROM_SOURCE — 设为 1 时从源码编译（source 模式下替代 --from-source）
#
# 导出环境变量：
#   TORCH_NPU_DIR — libtorch_npu 安装目录
#

ensure_libtorch_npu() {
  local shared_deps="${EXAMPLES_DIR}/deps/libtorch_npu"
  local build_args=""

  if [[ "${LIBTORCH_NPU_FROM_SOURCE:-}" == "1" ]]; then
    build_args="--from-source"
  fi

  # 1. 环境变量已设置且有效
  if [[ -n "${TORCH_NPU_DIR:-}" ]] && [[ -d "${TORCH_NPU_DIR}/lib" ]] && [[ -d "${TORCH_NPU_DIR}/include" ]]; then
    echo "  [✓] TORCH_NPU_DIR=${TORCH_NPU_DIR} (环境变量)"
    return 0
  fi

  # 2. 公共 deps 已构建
  if [[ -f "${shared_deps}/lib/libtorch_npu.so" ]] && [[ -d "${shared_deps}/include" ]]; then
    export TORCH_NPU_DIR="${shared_deps}"
    echo "  [✓] TORCH_NPU_DIR=${TORCH_NPU_DIR} (公共 deps)"
    return 0
  fi

  # 3. 自动准备（默认下载预构建包）
  local mode_str="下载预构建包"
  if [[ "${build_args}" == "--from-source" ]]; then
    mode_str="从源码编译"
  fi
  echo "  libtorch_npu 未就绪，开始自动准备（${mode_str}）..."
  local build_script="${EXAMPLES_DIR}/common/build_libtorch_npu.sh"
  if [[ ! -f "${build_script}" ]]; then
    echo "  [✗] 准备脚本不存在: ${build_script}"
    return 1
  fi
  if ! bash "${build_script}" ${build_args} --output-dir="${EXAMPLES_DIR}/deps"; then
    echo "  [✗] libtorch_npu 准备失败"
    return 1
  fi
  export TORCH_NPU_DIR="${shared_deps}"
  echo "  [✓] TORCH_NPU_DIR=${TORCH_NPU_DIR} (自动准备)"
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  for arg in "$@"; do
    case "${arg}" in
      --from-source)
        export LIBTORCH_NPU_FROM_SOURCE=1
        ;;
    esac
  done
  ensure_libtorch_npu >&2 || exit 1
  echo "${TORCH_NPU_DIR}"
fi
