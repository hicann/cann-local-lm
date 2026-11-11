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
# executorch_npu_ut.sh — executorch_npu Python LLT 门禁 (仅 ARM/aarch64)
#
# Usage:  bash executorch_npu_ut.sh
# Env:    WORKSPACE, ut_type, TARGET_BRANCH
#
# 内容: pytest tests/python 全量 (19 文件, P0+P1);
#       设备相关用例内建 skipif(not torch.npu.is_available()) 自动跳过, 无需 NPU;
#       torch_npu 装不上/不可 import 时自动降级跳过 3 个 P1 融合模型文件。
# 说明: 复用 ARM ndk 镜像条目, python 依赖栈运行时安装 (.venv_ci):
#       torch 2.11.0+cpu / executorch 1.5.0 / transformers 5.17.0 / torch-npu 2.11.0
#       / pytest / jsonschema / pydantic / psutil (均已验证华为云或 pytorch 源可得)。
# 预算: pytest ≤8min (480s 兜底); 实测超预算时启用 P0 降级 (见下方注释行)。
set -x
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
python3 --version

# 早退: PR 未触碰 executorch_npu/** 且未触碰 torch_delegate/** 与 .gitcode/** 时跳过。
# 涉及 torch_delegate 时仍要跑: executorch_npu 经 delegate_adapter 依赖 delegate 导出接口,
# 接口漂移需被本 UT 捕获 (filelist 缺失/为空则照常跑, fail-safe 同 ge 惯例)
FILELIST="${WORKSPACE}/pr_filelist.txt"
if [ -s "${FILELIST}" ] && ! grep -qE "executorch_npu/|torch_delegate/|\.gitcode/" "${FILELIST}"; then
    echo "UT_ExecuTorchNPU SKIP: PR 未改动 executorch_npu/**、torch_delegate/** 与 .gitcode/**, 与本 UT 无关"
    # 占位报告: upload report 步骤 (if always) 找不到 report.xml 会置 job FAILED
    cat > "${WORKSPACE}/executorch_npu/report.xml" <<'XML'
<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="executorch_npu_ut" tests="1" skipped="1" failures="0" errors="0"><testcase name="early_exit_skipped"><skipped message="PR not touching executorch_npu/torch_delegate/.gitcode"/></testcase></testsuite></testsuites>
XML
    exit 0
fi

source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

cd "${WORKSPACE}/executorch_npu" || exit 1

HUAWEI_INDEX="https://repo.huaweicloud.com/repository/pypi/simple"

# 1) venv + 依赖栈
python3 -m venv .venv_ci || { echo "deps FAIL: venv"; exit 1; }
VENV_PY="${PWD}/.venv_ci/bin/python3"

timeout 300 "${VENV_PY}" -m pip install -q --upgrade pip setuptools wheel \
    -i "${HUAWEI_INDEX}" || { echo "deps FAIL: pip base"; exit 1; }
# torch 2.11.0+cpu 首选 pytorch cpu index; pytorch.org 不可达时回退华为云 plain 2.11.0
# (PyPI aarch64 torch wheel 即 CPU 构建, 功能等价; executorch/build.sh 不 pin +cpu 本地标签)
# 探测选源 (10s 探测省 300s 超时等待; 2026-11-10 pytorch.org 对 ARM 池全天断实证)
if curl -s -o /dev/null -m 10 --head https://download.pytorch.org/whl/cpu/; then
    TORCH_ARGS="--index-url https://download.pytorch.org/whl/cpu --extra-index-url ${HUAWEI_INDEX}"
else
    echo "WARN: pytorch.org 探测不可达, 主源切 aliyun 镜像"
    TORCH_ARGS="-f https://mirrors.aliyun.com/pytorch-wheels/cpu/ -i ${HUAWEI_INDEX}"
fi
if ! timeout 600 "${VENV_PY}" -m pip install -q "torch==2.11.0+cpu" ${TORCH_ARGS}; then
    echo "WARN: +cpu 安装失败, 兜底华为云 torch==2.11.0 (aarch64 pypi wheel 即 CPU 构建)"
    timeout 600 "${VENV_PY}" -m pip install -q "torch==2.11.0" -i "${HUAWEI_INDEX}" \
        || { echo "deps FAIL: torch (多源均失败)"; exit 1; }
fi
# protobuf: torchair import 时需要其包元数据 (torch_npu 未声明该依赖, 实测缺失致 P1 用例红)
timeout 600 "${VENV_PY}" -m pip install -q \
    "executorch==1.5.0" "transformers==5.17.0" "protobuf<6" pytest jsonschema pydantic psutil \
    -i "${HUAWEI_INDEX}" || { echo "deps FAIL: test stack"; exit 1; }
# torch_npu: P1 融合模型用例 import 需要 (无需设备); 失败不阻塞, 下方自动降级
if ! timeout 600 "${VENV_PY}" -m pip install -q "torch-npu==2.11.0" -i "${HUAWEI_INDEX}"; then
    echo "WARN: torch-npu 安装失败, P1 融合模型用例将被跳过"
fi

# 2) pytest 全量 (设备用例自动 skip)
export PYTHONPATH="${PWD}/python:${PWD}"

TORCH_NPU_SKIP=()
if ! "${VENV_PY}" -c "import torch_npu" >/dev/null 2>&1; then
    echo "  [!] torch_npu 不可用, 跳过 P1 融合模型用例 (qwen3_5_fused/gemma4_fused)"
    TORCH_NPU_SKIP=(
        --ignore=tests/python/test_qwen3_5_fused_model.py
        --ignore=tests/python/test_qwen3_5_fused_gdn.py
        --ignore=tests/python/test_gemma4_fused_model.py
    )
fi

# 全量 P0+P1; 降级预案(实测>8min 时替换下行):
# timeout 300 "${VENV_PY}" -m pytest tests/python -q -k "not qwen3_5 and not gemma4 and not model_loading" --junitxml=report.xml
timeout 480 "${VENV_PY}" -m pytest tests/python -q --junitxml=report.xml "${TORCH_NPU_SKIP[@]}"
ret=$?
if [ $ret -eq 124 ]; then
    echo "executorch_npu UT TIMEOUT (480s), 考虑启用 P0 降级(见脚本内注释)"
    exit 1
fi

echo ""
if [ $ret -eq 0 ]; then
    echo "executorch_npu UT: PASS"
else
    echo "executorch_npu UT: FAIL (ret=$ret)"
fi
exit $ret
