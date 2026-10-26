#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/../venv_test"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

cd "${PROJECT_ROOT}"

echo "========================================"
echo " Step 3: 构建 C++ Runner + libcustom_ops"
echo "========================================"

# 构建 torch-delegate (debug 模式)
echo "[1/2] 构建 torch-delegate (debug)..."
./build.sh --debug --skip_third_party

# 构建 fallback_runner (debug 模式)
echo "[2/2] 构建 fallback_runner (debug)..."
cd "${SCRIPT_DIR}"

# 检查 CMakeLists.txt 是否存在
if [[ ! -f "CMakeLists.txt" ]]; then
  echo "[!] CMakeLists.txt 不存在，从 fallback_e2e_test 复制..."
  cp ../fallback_e2e_test/CMakeLists.txt .
fi

rm -rf build
mkdir -p build
cd build

# 获取 torch 路径
TORCH_PKG_DIR=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")

cmake .. -DCANN_ROOT=$ASCEND_HOME_PATH \
         -DCMAKE_BUILD_TYPE=Debug \
         -DTORCH_DELEGATE_BUILD_DIR="${PROJECT_ROOT}/build/debug" \
         -DTORCH_PKG_DIR=${TORCH_PKG_DIR} \
         -DTORCH_NPU_DIR=${TORCH_NPU_DIR}
cmake --build . -j$(nproc)

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
