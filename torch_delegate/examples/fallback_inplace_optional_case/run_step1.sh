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

# 确保 python3 在 PATH 中
export PATH="/home/developer/miniforge3/envs/py311gcc13/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "========================================"
echo " Step 1: 检查环境 + 准备 libtorch_npu"
echo "========================================"

# 检查环境
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "[✗] ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
  exit 1
fi
echo "  [✓] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

if ! command -v python3 &> /dev/null; then
  echo "[✗] python3 未找到"
  exit 1
fi
echo "  [✓] python3 $(python3 --version 2>&1 | awk '{print $2}')"

if ! command -v cmake &> /dev/null; then
  echo "[✗] cmake 未找到"
  exit 1
fi
echo "  [✓] cmake $(cmake --version | head -1 | awk '{print $3}')"

# 准备 libtorch_npu
if [[ -n "${TORCH_NPU_DIR:-}" ]] && [[ -d "${TORCH_NPU_DIR}/lib" ]] && [[ -d "${TORCH_NPU_DIR}/include" ]]; then
  echo "  [✓] TORCH_NPU_DIR=${TORCH_NPU_DIR} (已设置)"
elif [[ -d "${SCRIPT_DIR}/deps/libtorch_npu/lib" ]] && [[ -d "${SCRIPT_DIR}/deps/libtorch_npu/include" ]]; then
  export TORCH_NPU_DIR="${SCRIPT_DIR}/deps/libtorch_npu"
  echo "  [✓] TORCH_NPU_DIR=${TORCH_NPU_DIR} (已存在)"
else
  echo "  TORCH_NPU_DIR 未设置或无效，自动构建 libtorch_npu..."
  if [[ -f "${SCRIPT_DIR}/build_deps.sh" ]]; then
    if ! bash "${SCRIPT_DIR}/build_deps.sh" 2>&1; then
      echo "[✗] build_deps.sh 执行失败"
      exit 1
    fi
    export TORCH_NPU_DIR="${SCRIPT_DIR}/deps/libtorch_npu"
  else
    echo "[✗] TORCH_NPU_DIR 未设置且 build_deps.sh 不存在"
    exit 1
  fi
  echo "  [✓] TORCH_NPU_DIR=${TORCH_NPU_DIR} (自动构建)"
fi

echo ""
echo "[✓] 环境检查完成"
