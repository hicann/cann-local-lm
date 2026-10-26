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

cd "${SCRIPT_DIR}"

echo "========================================"
echo " Step 5: 精度对比 (Python)"
echo "========================================"

# 检查 compare_accuracy.py 是否存在
if [[ ! -f "compare_accuracy.py" ]]; then
  echo "[!] compare_accuracy.py 不存在，从 fallback_e2e_test 复制..."
  cp ../fallback_e2e_test/compare_accuracy.py .
fi

python compare_accuracy.py
