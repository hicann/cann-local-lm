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
#       [2/2] Python UT — pytest tests/python/ (无需 NPU)
# Python 环境自适应:
#   系统 python 有 torch 2.9.x → --system_python 直用 (lv6 镜像预装, 含 torchair 则全量用例)
#   否则 → build_test.sh 自建 .venv + pip 装 torch (网络路径已由 compile 门禁验证)
#   torchair 缺失时自动跳过依赖它的用例 (TORCHAIR_SKIP), 装好后无需改脚本即全量恢复。
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
python3 --version

source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

cd "${WORKSPACE}/torch_delegate" || exit 1

# Python 模式探测: 系统 torch 2.9.x → system_python; 否则 build_test.sh 走 .venv
PY_FLAGS=""
TEST_PY="python3"
if python3 -c "import torch,sys; sys.exit(0 if torch.__version__.split('+')[0]=='2.9.0' else 1)" 2>/dev/null; then
    PY_FLAGS="--system_python"
    echo "Python 模式: system (torch $(python3 -c 'import torch;print(torch.__version__)'))"
else
    TEST_PY="${PWD}/.venv/bin/python3"
    echo "Python 模式: .venv (系统 python 无 torch 2.9.0, build_test.sh 将自建 venv)"
fi

FAIL=0

echo "=== [1/2] torch_delegate C++ UT (build_test.sh, GTest + ASAN) ==="
# timeout 兜底: build_test.sh 内 pip/三方库下载无超时, 网络停滞会无限挂起 (124=超时)
timeout 1800 bash build_test.sh ${PY_FLAGS}
ret=$?
if [ $ret -eq 124 ]; then
    echo "C++ UT: FAIL (超时 30min, 疑似 pip/三方库下载网络停滞)"
    FAIL=1
elif [ $ret -ne 0 ]; then
    echo "C++ UT: FAIL (ret=$ret)"
    FAIL=1
else
    echo "C++ UT: PASS"
fi

echo ""
echo "=== [2/2] torch_delegate Python UT (pytest tests/python/) ==="
if ! "${TEST_PY}" --version >/dev/null 2>&1; then
    echo "Python UT: FAIL (python 不可用: ${TEST_PY}, venv 未创建?)"
    FAIL=1
else
    ${TEST_PY} -m pytest --version >/dev/null 2>&1 \
        || timeout 300 ${TEST_PY} -m pip install -q pytest -i https://repo.huaweicloud.com/repository/pypi/simple
    if ! ${TEST_PY} -m pytest --version >/dev/null 2>&1; then
        echo "Python UT: FAIL (pytest 不可用)"
        FAIL=1
    else
        # torchair 缺失时跳过导入/运行期依赖它的用例 (整文件 4 个 + 文件内散例 5 个)
        TORCHAIR_SKIP=()
        if ! ${TEST_PY} -c "import torchair" 2>/dev/null; then
            echo "  [!] torchair 缺失, 跳过 torchair 依赖用例"
            TORCHAIR_SKIP=(
                --ignore=tests/python/test_api.py
                --ignore=tests/python/test_epair_roundtrip.py
                --ignore=tests/python/test_torchair_converter.py
                --ignore=tests/python/test_lite_context.py
                --deselect tests/python/test_schema.py::TestEpairManifest::test_manifest_from_real_conversion
                --deselect tests/python/test_schema.py::TestEpairContents
                --deselect tests/python/test_schema.py::TestAscendIRFormat
                --deselect tests/python/test_writer.py::TestEpairWriterWrite::test_write_returns_tuple
                --deselect tests/python/test_writer.py::TestEpairWriterWrite::test_write_injects_ascend_ir
            )
        fi
        timeout 900 "${TEST_PY}" -m pytest tests/python/ -q "${TORCHAIR_SKIP[@]}"
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "Python UT: FAIL (ret=$ret)"
            FAIL=1
        else
            echo "Python UT: PASS"
        fi
    fi
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo "torch_delegate UT: 全部通过"
else
    echo "torch_delegate UT: 有失败"
fi
exit $FAIL
