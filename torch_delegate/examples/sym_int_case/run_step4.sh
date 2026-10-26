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

cd "${SCRIPT_DIR}"

echo "========================================"
echo " Step 4: 执行 C++ Runner"
echo "========================================"

# 路径配置
TORCH_PKG_DIR=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
DELEGATE_LIB_DIR="${PROJECT_ROOT}/build/debug/csrc"

export LD_LIBRARY_PATH=${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${DELEGATE_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_PKG_DIR=${TORCH_PKG_DIR}
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR=$(pwd)/output_dir
export TORCH_DELEGATE_KEEP_ARTIFACTS=1
export TORCH_DELEGATE_LOG_LEVEL=INFO

# PyTorch 日志环境变量
# export TORCH_LOG_LEVEL=DEBUG
# export TORCH_SHOW_DISPATCH_TRACE=1

# torch_npu 日志环境变量
# export TORCH_NPU_LOGS_FILTER="+dispatch,+acl,+memory,+npu"

# GE dump 配置
export DUMP_DATA=1
export DUMP_PATH=$(pwd)/ge_dump
export DUMP_LAYER="Sigmoid Tanh Add my_ops__scale_add"
export DUMP_MODE=all

# GE graph dump（查看连边）
export DUMP_GRAPH_PATH=$(pwd)/ge_graph
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=1
export TASK_QUEUE_ENABLE=0

# 确保目录存在
mkdir -p ge_dump ge_graph

echo "环境变量设置:"
echo "  ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "  TORCH_NPU_DIR: $TORCH_NPU_DIR"
echo "  TORCH_DELEGATE_OUTPUT_DIR: $TORCH_DELEGATE_OUTPUT_DIR"
echo ""

# 清理旧的输出
rm -rf output_dir

# 收集 epair 文件
EPAIRS=""
for f in output/model_simple.epair; do
  if [[ -f "$f" ]]; then
    EPAIRS="$EPAIRS $f"
  fi
done

if [[ -z "$EPAIRS" ]]; then
  echo "[✗] 未找到任何 epair 文件，请先执行 run_step2.sh"
  exit 1
fi

echo "执行模型:$EPAIRS"
echo ""
./build/fallback_runner --custom-ops build/libcustom_ops.so $EPAIRS 2>&1
# ./build/fallback_runner --custom-ops build/libcustom_ops.so $EPAIRS

echo ""
echo "========================================"
echo " 检查产物"
echo "========================================"
for model_dir in output_dir/*/; do
  if [[ -d "$model_dir" ]]; then
    echo "模型: $(basename "$model_dir")"
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
