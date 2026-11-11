#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e

# 激活 conda 环境
source /home/developer/miniforge3/etc/profile.d/conda.sh
conda activate py311gcc13

# 检查 CANN 环境
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "[error] ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
  exit 1
fi

# 确保 python3 在 PATH 中
export PATH="/home/developer/miniforge3/envs/py311gcc13/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "========================================"
echo " Step 3: 构建 C++ Runner + libcustom_ops"
echo "========================================"

cd "${SCRIPT_DIR}"
rm -rf build
mkdir -p build
cd build

TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || {
  echo "[✗] 无法获取 torch 路径"
  exit 1
}

CMAKE_DELEGATE_ARGS=""
if [[ -n "${TORCH_DELEGATE_ROOT:-}" ]] && [[ -d "${TORCH_DELEGATE_ROOT}/lib" ]]; then
  echo "  [✓] TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT} (已安装包)"
  CMAKE_DELEGATE_ARGS="-DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT}"
  DELEGATE_LIB_DIR="${TORCH_DELEGATE_ROOT}/lib"
elif [[ -n "${TORCH_DELEGATE_BUILD_DIR:-}" ]] && [[ -d "${TORCH_DELEGATE_BUILD_DIR}/csrc" ]]; then
  echo "  [✓] TORCH_DELEGATE_BUILD_DIR=${TORCH_DELEGATE_BUILD_DIR} (源码构建)"
  CMAKE_DELEGATE_ARGS="-DTORCH_DELEGATE_BUILD_DIR=${TORCH_DELEGATE_BUILD_DIR}"
  DELEGATE_LIB_DIR="${TORCH_DELEGATE_BUILD_DIR}/csrc"
elif [[ -d "${PROJECT_ROOT}/build/release/csrc" ]]; then
  echo "  [✓] TORCH_DELEGATE_BUILD_DIR=${PROJECT_ROOT}/build/release (源码构建，自动检测)"
  CMAKE_DELEGATE_ARGS="-DTORCH_DELEGATE_BUILD_DIR=${PROJECT_ROOT}/build/release"
  DELEGATE_LIB_DIR="${PROJECT_ROOT}/build/release/csrc"
else
  echo "[✗] 无法找到 torch_delegate 库。请设置 TORCH_DELEGATE_ROOT（已安装包）或 TORCH_DELEGATE_BUILD_DIR（源码构建）"
  exit 1
fi

# 使用已存在的 libtorch_npu
if [[ -z "${TORCH_NPU_DIR:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/deps/libtorch_npu/lib" ]] && [[ -d "${SCRIPT_DIR}/deps/libtorch_npu/include" ]]; then
    export TORCH_NPU_DIR="${SCRIPT_DIR}/deps/libtorch_npu"
  else
    echo "[✗] TORCH_NPU_DIR 未设置且 deps/libtorch_npu 不存在"
    exit 1
  fi
fi

cmake .. \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" \
  -DCMAKE_BUILD_TYPE=Debug \
  ${CMAKE_DELEGATE_ARGS} \
  -DTORCH_PKG_DIR="${TORCH_PKG_DIR}" \
  -DTORCH_NPU_DIR="${TORCH_NPU_DIR}"
cmake --build . -j"$(nproc)"

if [[ ! -f "fallback_runner" ]]; then
  echo "[✗] fallback_runner 编译失败"
  exit 1
fi
echo "[✓] fallback_runner 已编译"

if [[ ! -f "libcustom_ops.so" ]]; then
  echo "[✗] libcustom_ops.so 编译失败"
  exit 1
fi
echo "[✓] libcustom_ops.so 已编译"

echo ""
echo "[✓] 构建完成"
