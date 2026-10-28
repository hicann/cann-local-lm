#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# torch_delegate_ut.sh — torch_delegate 子项目 LLT 门禁
#
# Usage:  bash torch_delegate_ut.sh
# Env:    WORKSPACE, ut_type, TARGET_BRANCH
#
# 内容: [1/2] C++ UT  — build_test.sh (GTest + ASAN, cann/ge stub, 无需 NPU)
#       [2/2] Python UT — pytest tests/python/ (需镜像系统 python 预装
#                          torch 2.9.0 + torchair + pytest, 无需 NPU)
# 说明: 三方库由 build_third_party.sh 从 gitcode 镜像自建; 两段独立跑, 汇总失败退出。
set -x
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
    echo "Switched to gcc-16"
elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
    echo "Switched to gcc-15"
elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
    echo "gcc-16/15 not available, fell back to gcc-14"
fi
gcc --version

source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

cd "${WORKSPACE}/torch_delegate" || exit 1

# 环境诊断: Python UT 依赖系统 python 的 torch/torchair (镜像预装)
python3 --version
python3 -c "import torch; print('torch:', torch.__version__)" \
    || echo "WARN: torch 未安装 (build_test.sh --system_python 与 Python UT 将失败)"
python3 -c "import torchair" 2>/dev/null \
    && echo "torchair: OK" \
    || echo "WARN: torchair 缺失 (test_torchair_converter/test_api 等将失败)"

FAIL=0

echo "=== [1/2] torch_delegate C++ UT (build_test.sh, GTest + ASAN) ==="
bash build_test.sh --system_python
ret=$?
if [ $ret -ne 0 ]; then
    echo "C++ UT: FAIL (ret=$ret)"
    FAIL=1
else
    echo "C++ UT: PASS"
fi

echo ""
echo "=== [2/2] torch_delegate Python UT (pytest tests/python/) ==="
timeout 900 python3 -m pytest tests/python/ -q
ret=$?
if [ $ret -ne 0 ]; then
    echo "Python UT: FAIL (ret=$ret)"
    FAIL=1
else
    echo "Python UT: PASS"
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo "torch_delegate UT: 全部通过"
else
    echo "torch_delegate UT: 有失败"
fi
exit $FAIL
