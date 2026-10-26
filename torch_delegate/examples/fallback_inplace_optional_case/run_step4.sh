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

# 设置 CANN 路径
export ASCEND_HOME_PATH=/mnt/workspace/gitCode/cann/cann-9.1.0

# 确保 python3 在 PATH 中
export PATH="/home/developer/miniforge3/envs/py311gcc13/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "========================================"
echo " Step 4: 执行 C++ Runner"
echo "========================================"

cd "${SCRIPT_DIR}"

# 获取 torch 路径
TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")" || {
  echo "[✗] 无法获取 torch 路径"
  exit 1
}

# 使用已存在的 libtorch_npu
if [[ -z "${TORCH_NPU_DIR:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/deps/libtorch_npu/lib" ]] && [[ -d "${SCRIPT_DIR}/deps/libtorch_npu/include" ]]; then
    export TORCH_NPU_DIR="${SCRIPT_DIR}/deps/libtorch_npu"
  else
    echo "[✗] TORCH_NPU_DIR 未设置且 deps/libtorch_npu 不存在"
    exit 1
  fi
fi

# 检测 TorchDelegate 库路径
if [[ -n "${TORCH_DELEGATE_ROOT:-}" ]] && [[ -d "${TORCH_DELEGATE_ROOT}/lib" ]]; then
  DELEGATE_LIB_DIR="${TORCH_DELEGATE_ROOT}/lib"
elif [[ -n "${TORCH_DELEGATE_BUILD_DIR:-}" ]] && [[ -d "${TORCH_DELEGATE_BUILD_DIR}/csrc" ]]; then
  DELEGATE_LIB_DIR="${TORCH_DELEGATE_BUILD_DIR}/csrc"
elif [[ -d "${PROJECT_ROOT}/build/release/csrc" ]]; then
  DELEGATE_LIB_DIR="${PROJECT_ROOT}/build/release/csrc"
else
  echo "[✗] 无法找到 torch_delegate 库"
  exit 1
fi

export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_NPU_DIR}/lib:${DELEGATE_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_PKG_DIR="${TORCH_PKG_DIR}"
export TORCH_NPU_DIR="${TORCH_NPU_DIR}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR="${SCRIPT_DIR}/output_dir"
export TORCH_DELEGATE_KEEP_ARTIFACTS=1
export TORCH_DELEGATE_LOG_LEVEL=INFO
export TASK_QUEUE_ENABLE=0
export DUMP_GRAPH_PATH=./ge_graph
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=1

rm -rf output_dir
mkdir -p ge_dump ge_graph

EPAIRS=""
for f in output/model_inplace_optional.epair; do
  if [[ -f "$f" ]]; then
    EPAIRS="$EPAIRS $f"
  fi
done

if [[ -z "$EPAIRS" ]]; then
  echo "[✗] 未找到任何 epair 文件"
  exit 1
fi

echo "执行模型:${EPAIRS}"
./build/fallback_runner --custom-ops build/libcustom_ops.so ${EPAIRS}

echo ""
echo "检查产物:"
for model_dir in output_dir/*/; do
  if [[ -d "$model_dir" ]]; then
    echo "  模型: $(basename "$model_dir")"
    for f in custom_op.cpp CMakeLists.txt libcust_opapi.so; do
      if [[ -f "${model_dir}${f}" ]]; then
        echo "  [✓] $f ($(stat -c%s "${model_dir}${f}") bytes)"
      else
        echo "  [✗] $f 未生成"
      fi
    done
  fi
done

echo ""
echo "[✓] 执行完成"
