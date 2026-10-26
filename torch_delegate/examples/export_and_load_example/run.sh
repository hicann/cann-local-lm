#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run.sh - export_and_load_example 一键式验证脚本
#
# 前置条件：
#   1. source <cann_path>/cann/set_env.sh
#   2. 当前 Python 环境已安装 torch_delegate whl
#      （TORCH_DELEGATE_ROOT 会自动从已安装包推导，无需 source set_env.sh）
#
# 用法：
#   bash run.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE=0
TOTAL=4

phase() {
  PHASE=$((PHASE + 1))
  echo ""
  echo "========================================"
  echo " [${PHASE}/${TOTAL}] $1"
  echo "========================================"
}

check() {
  echo "  [✓] $1"
}

fail() {
  echo "  [✗] $1"
  exit 1
}

phase "检查环境"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  fail "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
fi
check "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
  TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
  export TORCH_DELEGATE_ROOT
fi
if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
  fail "TORCH_DELEGATE_ROOT 未设置，且当前 Python 未安装 torch_delegate；请先 pip install <torch_delegate whl>"
fi
check "TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT}"

if ! command -v python3 &> /dev/null; then
  fail "python3 未找到"
fi
check "python3 $(python3 --version 2>&1 | awk '{print $2}')"

if ! command -v cmake &> /dev/null; then
  fail "cmake 未找到"
fi
check "cmake $(cmake --version | head -1 | awk '{print $3}')"

phase "生成示例 epair"
python3 "${ROOT_DIR}/export_ep_and_epair.py"
if [[ -f "${ROOT_DIR}/epair_model.epair" ]]; then
  check "epair_model.epair 已生成"
else
  fail "epair_model.epair 未生成"
fi

phase "构建 C++ 示例"
cd "${ROOT_DIR}"
rm -rf build
_TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 TORCH_PKG_DIR"
cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" \
  -DTORCH_PKG_DIR="${_TORCH_PKG_DIR}"
cmake --build build -j$(nproc)
if [[ -f "${ROOT_DIR}/build/export_and_load_example" ]]; then
  check "build/export_and_load_example"
else
  fail "编译失败"
fi

phase "Delegate 整图推理 + 单算子精度比对"
cd "${ROOT_DIR}"
# 子进程（C++ 二进制）需要 torch 运行时库
TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 torch 路径"
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}.libs:${TORCH_PKG_DIR}/lib:${LD_LIBRARY_PATH:-}"
python3 "${ROOT_DIR}/compare_single_op_vs_delegate.py"

echo ""
echo "========================================"
echo " 验证通过"
echo "========================================"
