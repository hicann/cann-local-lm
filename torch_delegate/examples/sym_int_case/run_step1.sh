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
VENV_DIR="${SCRIPT_DIR}/../venv_test"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

echo "========================================"
echo " Step 1: 检查环境"
echo "========================================"

echo "Python: $(python --version 2>&1)"
echo "GCC: $(gcc --version 2>&1 | head -1)"
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "TORCH: $(python -c 'import torch; print(torch.__version__)')"
echo "torch_npu: $(python -c 'import torch_npu; print("OK")' 2>&1)"
echo "torch_delegate: $(python -c 'import torch_delegate; print("OK")' 2>&1)"

echo ""
echo "[✓] 环境检查完成"
