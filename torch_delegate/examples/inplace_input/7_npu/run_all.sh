#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# run_all.sh - Inplace Input NPU 验证一键脚本
#
# 前置条件：
#   1. source <cann_path>/cann/set_env.sh
#   2. 当前 Python 环境已安装 torch_delegate whl
#      （TORCH_DELEGATE_ROOT 会自动从已安装包推导，无需 source set_env.sh）
#
# 用法：
#   bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR=$SCRIPT_DIR/build
GENERATED_DIR=$SCRIPT_DIR/generated

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================
# Step 0: 检查环境
# ============================================
log_info "============================================"
log_info "Step 0: 检查环境"
log_info "============================================"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  log_error "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
  exit 1
fi
log_info "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
  TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
  export TORCH_DELEGATE_ROOT
fi
if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
  log_error "TORCH_DELEGATE_ROOT 未设置，且当前 Python 未安装 torch_delegate；请先 pip install <torch_delegate whl>"
  exit 1
fi
log_info "TORCH_DELEGATE_ROOT: $TORCH_DELEGATE_ROOT"

log_info "Python: $(python3 --version 2>&1), $(which python3)"
log_info "VIRTUAL_ENV: ${VIRTUAL_ENV:-<not set>}"

python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
  log_error "PyTorch 未安装或无法导入"
  exit 1
}

# ============================================
# Step 1: 安装 torch_delegate Python 包
# ============================================
log_info "============================================"
log_info "Step 1: 安装 torch_delegate Python 包"
log_info "============================================"

pip show torch-delegate > /dev/null 2>&1 || {
  log_warn "torch_delegate 包未安装，尝试安装..."
  # TORCH_DELEGATE_ROOT 可能包含 whl 包
  whl=$(ls "$TORCH_DELEGATE_ROOT"/torch_delegate-*.whl 2>/dev/null | head -1)
  if [[ -n "$whl" ]]; then
    pip install "$whl" --quiet
  else
    log_error "未找到 torch_delegate whl 包，请先安装"
    exit 1
  fi
}
log_info "torch_delegate 已就绪"

# ============================================
# Step 2: 构建 C++ inplace_verify
# ============================================
log_info "============================================"
log_info "Step 2: 构建 C++ inplace_verify"
log_info "============================================"

rm -rf "$BUILD_DIR"

log_info "配置 CMake..."
_TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || {
  log_error "无法获取 TORCH_PKG_DIR，请确保 PyTorch 已安装"
  exit 1
}
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
  -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" \
  -DTORCH_PKG_DIR="${_TORCH_PKG_DIR}" || {
  log_error "CMake 配置失败"
  exit 1
}

log_info "编译 inplace_verify..."
cmake --build "$BUILD_DIR" -j$(nproc) || {
  log_error "编译失败"
  exit 1
}
log_info "inplace_verify 编译完成"

# ============================================
# Step 3: 导出 epair 文件
# ============================================
log_info "============================================"
log_info "Step 3: 导出 epair 文件"
log_info "============================================"

mkdir -p "$GENERATED_DIR"

log_info "运行导出脚本..."
python3 "$SCRIPT_DIR/export_epair.py" || {
  log_error "epair 导出失败"
  exit 1
}

# 检查生成的文件
for model in simple chain param; do
  if [ ! -f "$GENERATED_DIR/epair_model_${model}.epair" ]; then
    log_error "epair 文件未生成: epair_model_${model}.epair"
    exit 1
  fi
done
log_info "epair 文件导出完成"

# ============================================
# Step 4: 运行 NPU 验证
# ============================================
log_info "============================================"
log_info "Step 4: 运行 NPU 验证"
log_info "============================================"

# 设置库路径（torch.libs、torch/lib、delegate lib、CANN lib）
TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)"
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}.libs:${TORCH_PKG_DIR}/lib:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

log_info "已设置LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# 运行验证
RESULTS=()
for model in simple chain param; do
  log_info "验证模型: $model"
  EPAIR_FILE=$GENERATED_DIR/epair_model_${model}.epair

  if "$BUILD_DIR/inplace_verify" "$EPAIR_FILE" "$model" 2>&1 | tee /tmp/verify_${model}.log | grep -q "PASS"; then
    RESULTS+=("${model}: PASS")
    log_info "[$model] 验证通过"
  else
    RESULTS+=("${model}: FAIL")
    log_error "[$model] 验证失败"
    cat /tmp/verify_${model}.log
  fi
done

# ============================================
# Step 5: 输出汇总结果
# ============================================
log_info "============================================"
log_info "验证结果汇总"
log_info "============================================"

ALL_PASS=true
for result in "${RESULTS[@]}"; do
  echo "  $result"
  if [[ "$result" == *"FAIL"* ]]; then
    ALL_PASS=false
  fi
done

if $ALL_PASS; then
  log_info "全部验证通过!"
  exit 0
else
  log_error "部分验证失败，请检查日志"
  exit 1
fi
