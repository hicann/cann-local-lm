#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# prepare_env.sh - Examples 验证环境准备脚本
#
# 用法:
#   bash prepare_env.sh
#
# 功能:
#   1. 创建 venv_test 虚拟环境
#   2. 安装 requirements.txt 中的依赖（torch/torch_npu/triton/triton-ascend 已在文件中注释，自动跳过）
#   3. 手动安装:
#      - torch==2.9.0         (from https://download.pytorch.org/whl/cpu)
#      - torch_npu==2.9.0     (from https://repo.huaweicloud.com/repository/pypi/simple)
#      - triton==3.5.0        (from https://repo.huaweicloud.com/repository/pypi/simple)
#      - triton-ascend==3.2.1 (from https://triton-ascend.osinfra.cn/pypi/simple)
#   4. 从华为云 Daily 构建下载 torch_npu 并覆盖安装
#   5. 安装 ../output/package 下的 torch_delegate whl（需先在仓库根目录执行 ./build.sh）
#
# 虚拟环境路径: $(pwd)/venv_test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_test"

# ============================================
# Step 1: 创建虚拟环境
# ============================================
echo "[Step 1] Removing existing virtual environment (if any)..."
rm -rf "${VENV_DIR}"

echo "[Step 1.1] Creating virtual environment at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[Step 1.1] Upgrading pip..."
pip install --upgrade pip --quiet

# ============================================
# Step 2: 安装 requirements.txt
# ============================================
echo "[Step 2] Installing dependencies from requirements.txt..."
# torch/torch_npu/triton/triton-ascend 已在 requirements.txt 中注释，自动跳过
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ============================================
# Step 3: 手动安装 torch-cpu
# ============================================
echo "[Step 3] Installing torch==2.9.0 from PyTorch CPU index..."
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# ============================================
# Step 4: 安装 torch_npu 暂时下载daily构建
# ============================================
#echo "[Step 4] Installing torch_npu==2.9.0 from HuaweiCloud index..."
#pip install torch_npu==2.9.0 --index-url https://repo.huaweicloud.com/repository/pypi/simple

# ============================================
# Step 5: 安装 triton
# ============================================
echo "[Step 5] Installing triton==3.5.0 from HuaweiCloud index..."
pip install triton==3.5.0 --index-url https://repo.huaweicloud.com/repository/pypi/simple

# ============================================
# Step 6: 安装 triton-ascend
# ============================================
echo "[Step 6] Installing triton-ascend==3.2.1 from triton-ascend index..."
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple

# ============================================
# Step 7: 从 Daily 构建下载并覆盖安装 torch_npu
# ============================================
echo "[Step 7] Downloading torch_npu daily build..."

TORCH_VERSION="2.9.0"
DAILY_BASE_URL="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily"
TODAY=$(date +%Y%m%d)
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PKG_NAME="pytorch_v${TORCH_VERSION}_py${PYVER}.tar.gz"
EXTRACT_DIR="/tmp/torch_delegate_deps_build/pytorch_daily"

echo "  Python version: ${PYVER}, package: ${PKG_NAME}"

rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"

DOWNLOADED=0
for BUILD_NUM in 3 2 1; do
  URL="${DAILY_BASE_URL}/v${TORCH_VERSION}/${TODAY}.${BUILD_NUM}/${PKG_NAME}"
  echo "  Trying: ${URL}"
  if curl --progress-bar --fail --location --max-time 300 -o "${EXTRACT_DIR}/${PKG_NAME}" "${URL}"; then
    echo "  [OK] Downloaded (build ${BUILD_NUM})"
    DOWNLOADED=1
    break
  fi
  echo "  Not available, trying previous build..."
done

if [[ ${DOWNLOADED} -ne 1 ]]; then
  echo "  [ERROR] No daily build found for ${TODAY}"
  echo "  [ERROR] URL pattern: ${DAILY_BASE_URL}/v${TORCH_VERSION}/${TODAY}.<N>/${PKG_NAME}"
  exit 1
fi

echo "  Extracting to ${EXTRACT_DIR}..."
tar -xzf "${EXTRACT_DIR}/${PKG_NAME}" -C "${EXTRACT_DIR}"

echo "  Detecting platform architecture..."
ARCH=$(arch)
echo "  Architecture: ${ARCH}"

echo "  Installing torch_npu from daily build..."
NPU_WHL=$(find "${EXTRACT_DIR}" -maxdepth 1 -name "torch_npu-*.whl" | grep "${ARCH}" | head -1)
if [[ -z "${NPU_WHL}" ]]; then
  echo "  [ERROR] torch_npu wheel for ${ARCH} not found in ${EXTRACT_DIR}"
  echo "  [ERROR] Available wheels:"
  find "${EXTRACT_DIR}" -maxdepth 1 -name "torch_npu-*.whl" -exec basename {} \;
  exit 1
fi
echo "  Installing: $(basename "${NPU_WHL}")"
pip install "${NPU_WHL}" --no-deps --force-reinstall

rm -rf "${EXTRACT_DIR}"

# ============================================
# Step 8: 安装 torch_delegate whl（../build.sh 产出）
# ============================================
echo "[Step 8] Installing torch_delegate wheel..."

WHL_DIR="${SCRIPT_DIR}/../output/package"
TD_WHL=$(ls -t "${WHL_DIR}"/torch_delegate-*.whl 2>/dev/null | head -1)
if [[ -z "${TD_WHL}" ]]; then
  echo "  [ERROR] 未找到 torch_delegate whl: ${WHL_DIR}"
  echo "  [ERROR] 请先在仓库根目录执行: ./build.sh"
  exit 1
fi
echo "  Installing: $(basename "${TD_WHL}")"
pip install "${TD_WHL}" --no-deps --force-reinstall

# ============================================
# 完成
# ============================================
echo ""
echo "===================================="
echo " Environment ready!"
echo "===================================="
echo ""
echo "Activate:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Verify:"
echo "  python -c \"import torch; print('torch:', torch.__version__)\""
echo "  python -c \"import torch_npu; print('torch_npu:', torch_npu.__version__)\""
echo "  python -c \"import triton_ascend; print('triton-ascend OK')\""
echo "  python -c \"import torch_delegate; print('torch_delegate:', torch_delegate.__version__)\""
echo ""
echo "C++ 前缀（TORCH_DELEGATE_ROOT，无需 source set_env.sh）:"
echo "  export TORCH_DELEGATE_ROOT=\$(torch-delegate-config --cmake-prefix)"
