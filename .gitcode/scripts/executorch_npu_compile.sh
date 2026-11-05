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
# executorch_npu_compile.sh — executorch_npu 子项目编译门禁 (仅 ARM/aarch64)
#
# Usage:  bash executorch_npu_compile.sh
# Env:    WORKSPACE, task_name, target_branch
#
# 流程: CANN 环境 → 自建 .venv_ci 装依赖栈 (torch 2.11.0+cpu / executorch 1.5.0)
#       → executorch_npu/build.sh (自检 → cmake 构建 → ctest C++ LLT 3用例 → wheel)
#       → whl 收集到 ${WORKSPACE}/build_out/ (由 workflow obs-upload 传 OBS)
# 说明: 复用 ARM ndk 镜像条目(无预烘镜像), python 依赖运行时安装;
#       executorch 1.5.0 PyPI aarch64 wheel 已验证含完整 C++ SDK
#       (include/lib/share/cmake, 满足 build.sh 自检); cmake<3.24 时 pip 兜底;
#       OpenSSL dev 缺失时 apt 兜底 (manifest SHA-256 必需)。
set -x
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
python3 --version
gcc --version | head -1

source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

cd "${WORKSPACE}/executorch_npu" || exit 1

HUAWEI_INDEX="https://repo.huaweicloud.com/repository/pypi/simple"

# 1) venv + 基础构建依赖 (build.sh 经 command -v python 取 venv python)
python3 -m venv .venv_ci || { echo "deps FAIL: venv"; exit 1; }
VENV_PY="${PWD}/.venv_ci/bin/python3"
export PATH="${PWD}/.venv_ci/bin:${PATH}"
if ! timeout 300 "${VENV_PY}" -m pip install -q --upgrade pip setuptools wheel packaging \
    -i "${HUAWEI_INDEX}"; then
    echo "deps FAIL: pip base"; exit 1
fi

# 2) cmake >= 3.24 (镜像版本不足时 pip 装, venv/bin 已在 PATH 前)
CMAKE_VER="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)"
CMAKE_OK="$(echo "${CMAKE_VER:-0.0}" | awk -F. '{print ($1>3 || ($1==3 && $2>=24)) ? 1 : 0}')"
if [ "${CMAKE_OK}" != "1" ]; then
    echo "cmake ${CMAKE_VER:-missing} < 3.24, pip 安装 cmake"
    timeout 300 "${VENV_PY}" -m pip install -q "cmake>=3.24" -i "${HUAWEI_INDEX}" \
        || { echo "deps FAIL: cmake"; exit 1; }
fi
cmake --version | head -1

# 3) OpenSSL dev (find_package(OpenSSL REQUIRED COMPONENTS Crypto))
if [ ! -f /usr/include/openssl/evp.h ]; then
    echo "OpenSSL 头文件缺失, 安装 libssl-dev"
    sudo apt-get install -y libssl-dev >/dev/null 2>&1 \
        || { sudo apt-get update -qq >/dev/null 2>&1; sudo apt-get install -y libssl-dev >/dev/null 2>&1; }
    [ -f /usr/include/openssl/evp.h ] || { echo "deps FAIL: libssl-dev"; exit 1; }
fi

# 4) torch 2.11.0+cpu (pytorch cpu index) + executorch 1.5.0 full wheel (huaweicloud)
if ! timeout 600 "${VENV_PY}" -m pip install -q "torch==2.11.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url "${HUAWEI_INDEX}"; then
    echo "deps FAIL: torch (pytorch.org 不可达?)"; exit 1
fi
if ! timeout 600 "${VENV_PY}" -m pip install -q "executorch==1.5.0" -i "${HUAWEI_INDEX}"; then
    echo "deps FAIL: executorch"; exit 1
fi

# 5) 构建 (cmake + ctest C++ LLT + wheel), 840s 硬兜底
timeout 840 bash build.sh
ret=$?
if [ $ret -eq 124 ]; then
    echo "executorch_npu compile TIMEOUT (840s)"
    exit 1
fi
if [ $ret -ne 0 ]; then
    echo "executorch_npu compile FAIL (ret=$ret)"
    exit $ret
fi

# 6) 产物收集: whl → build_out/ (workflow 上传 OBS: ${obs_path}/)
mkdir -p "${WORKSPACE}/build_out"
cp output/package/executorch_npu-*.whl "${WORKSPACE}/build_out/" 2>/dev/null
if ! ls "${WORKSPACE}/build_out"/executorch_npu-*.whl >/dev/null 2>&1; then
    echo "whl not found in output/package/"
    exit 1
fi
ls -l "${WORKSPACE}/build_out/"

exit 0
