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
# torch_delegate_compile.sh — torch_delegate 子项目编译门禁
#
# Usage:  bash torch_delegate_compile.sh
# Env:    WORKSPACE, task_name, target_branch
#
# 流程: 环境准备(gcc/CANN) → torch_delegate/build.sh (三方库+C++库+whl)
#       → whl 收集到 ${WORKSPACE}/build_out/ (由 workflow obs-upload 传 OBS)
# 说明: 三方库(nlohmann_json/gtest)由 build_third_party.sh 从 gitcode 镜像自建,
#       不走镜像 lib_cache (规避 gtest ABI/布局耦合)。

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
        echo "Switched to gcc-16"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
        echo "Switched to gcc-15"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
        echo "gcc-16/15 not available, fell back to gcc-14"
    fi
    # 仅 X86 用 buildtools python-3.10.2; ARM 用镜像默认 python3(3.12):
    # aarch64 无 torch 2.9.0+cpu 的 cp310 wheel(仅 cp312+), 强制 3.10 会导致 venv pip 必败
    if [[ "${task_name}" == *X86* ]] && [ -d /opt/buildtools/python-3.10.2/bin ]; then
        export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
    fi
fi
gcc --version
python3 --version

source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

cd "${WORKSPACE}/torch_delegate" || exit 1

# Python 模式: 系统 python 已有 torch 2.9.x → --system_python (免下载);
# 否则走 build.sh 默认模式 (自建 .venv + pip 安装 requirements.txt)
PY_FLAGS=""
if python3 -c "import torch,sys; sys.exit(0 if torch.__version__.split('+')[0]=='2.9.0' else 1)" 2>/dev/null; then
    PY_FLAGS="--system_python"
fi

echo "exec cmd: [bash build.sh ${PY_FLAGS}]"
bash build.sh ${PY_FLAGS}
ret=$?
if [ $ret -ne 0 ]; then
    echo "torch_delegate compile fail"
    exit $ret
fi

# 产物收集: whl → build_out/ (workflow 上传 OBS: ${obs_path}/)
mkdir -p "${WORKSPACE}/build_out"
cp output/package/torch_delegate-*.whl "${WORKSPACE}/build_out/" 2>/dev/null
if ! ls "${WORKSPACE}/build_out"/torch_delegate-*.whl >/dev/null 2>&1; then
    echo "whl not found in output/package/"
    exit 1
fi
ls -l "${WORKSPACE}/build_out"/

exit 0
