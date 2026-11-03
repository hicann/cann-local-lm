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

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
        echo "Switched to gcc-16"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
        echo "Switched to gcc-15"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
        echo "gcc-16/15 not available, fell back to gcc-14"
    fi
    export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

if gcc --version | head -n1 | grep -q "15\."; then
    # 软链覆盖前先把预置完整 lib_cache/gtest 回填到 gcc15 变体 (gcc15 预置 gtest 可能只有空 include 占位).
    # 注意: 必须在 rm -rf lib_cache 之前做, 否则源目录被删, 回填必失效.
    if [ -d /home/jenkins/opensource/gcc15 ] \
       && [ -f /home/jenkins/opensource/lib_cache/gtest/include/gtest/gtest.h ] \
       && [ -f /home/jenkins/opensource/lib_cache/gtest/lib/libgtest.a ] \
       && ! [ -f /home/jenkins/opensource/gcc15/lib_cache/gtest/lib/libgtest.a ]; then
        echo "[gtest-backfill] 补全 gcc15/lib_cache/gtest (来自预置 lib_cache/gtest)"
        mkdir -p /home/jenkins/opensource/gcc15/lib_cache/gtest
        cp -r /home/jenkins/opensource/lib_cache/gtest/. /home/jenkins/opensource/gcc15/lib_cache/gtest/
    fi
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        if [ -f /home/jenkins/opensource/lib_cache/gtest/include/gtest/gtest.h ] \
           && [ -f /home/jenkins/opensource/lib_cache/gtest/lib/libgtest.a ] \
           && ! [ -f /home/jenkins/opensource/gcc15x86/lib_cache/gtest/lib/libgtest.a ]; then
            echo "[gtest-backfill] 补全 gcc15x86/lib_cache/gtest (来自预置 lib_cache/gtest)"
            mkdir -p /home/jenkins/opensource/gcc15x86/lib_cache/gtest
            cp -r /home/jenkins/opensource/lib_cache/gtest/. /home/jenkins/opensource/gcc15x86/lib_cache/gtest/
        fi
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]] && [ "${target_branch}" == master ]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi

if [[ "${target_branch}" == "master" ]] || [[ "${target_branch}" == "develop" ]]; then
    sudo update-alternatives --set lcov /opt/lcov-2.3.2/bin/lcov
    lcov --version
fi

gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

echo "exec cmd: [bash build.sh --pkg --cann_3rd_lib_path="/home/jenkins/opensource"]"
bash build.sh --pkg --cann_3rd_lib_path="/home/jenkins/opensource"
ret=$?

# 编译打包成功后，将build_out/ 内产物压缩为cann_local_lm_<arch>.tar.gz 放回build_out/
# arch 取值: 使用 uname -m 原值
if [[ $ret -eq 0 ]]; then
    _arch="$(uname -m)"
    case "$_arch" in
        aarch64) pkg_arch="aarch64";;
        x86_64)  pkg_arch="x86_64";;
        *)       pkg_arch="$_arch";;
    esac
    pkg_name="cann_local_lm_${pkg_arch}.tar.gz"
    if [[ -d build_out ]]; then
        echo "compress build_out/ -> build_out/${pkg_name}"
        tar -czf "build_out/${pkg_name}" -C build_out --exclude="${pkg_name}" .
        if [[ -f "build_out/${pkg_name}" ]]; then
            echo "compressed: build_out/${pkg_name} ($(du -h "build_out/${pkg_name}" | cut -f1))"
        else
            echo "WARN: compress build_out/ failed, ${pkg_name} not generated" >&2
        fi
    else
        echo "WARN: build_out/ not found, skip compress" >&2
    fi
fi

exit $ret
