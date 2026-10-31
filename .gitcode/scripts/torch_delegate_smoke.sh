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
# torch_delegate_smoke.sh — torch_delegate 示例冒烟 (A2/910B 真机)
#
# Usage:  bash torch_delegate_smoke.sh
# Env:    WORKSPACE, whl_url (ARM compile job 产物的下载 URL, 来自 compile action outputs)
#
# 流程: CANN 环境 → npu-smi → 下载 compile 阶段 whl → examples/prepare_env.sh 建 venv_test
#       (torch 2.9.0 CPU + torch_npu daily + triton + whl)
#       → 跑最小端到端示例 export_and_load_example (导出 epair → C++ 构建 → GE 真机推理 → 精度比对)
#       → 收集设备侧 slog 供 workflow 上传 OBS
# 硬门禁: 以真实结果退出 (exit ${RET}), 冒烟失败阻塞流水线
set -x
RET=0

# CANN 环境 (冒烟镜像约定 /usr/local/Ascend, 兜底 jenkins 路径)
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
else
    echo "SMOKE FAIL: 未找到 CANN 环境 (set_env.sh)"
    RET=1
fi

# 防 stdout 污染: CANN runtime slog 会混进 run.sh 的 $(python3 -c ...) 捕获,
# 导致 TORCH_DELEGATE_ROOT/TORCH_PKG_DIR 带换行垃圾值 → cmake find_package 失败
# (参照 ge pre_smoke.sh: 日志落文件不打 stdout; 不可设 TORCH_DEVICE_BACKEND_AUTOLOAD=0,
#  示例依赖 torch_npu 自动加载注册 .npu() 方法)
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

npu-smi info 2>&1 | head -20 || echo "WARN: npu-smi 不可用"

cd "${WORKSPACE}/torch_delegate" || exit 0

# 1) 下载 compile 阶段 (ARM job) 产出的 whl, 放到 prepare_env.sh 约定的 output/package/
if [ $RET -eq 0 ]; then
    mkdir -p output/package
    WHL_NAME="$(basename "${whl_url%%\?*}")"
    case "${WHL_NAME}" in
        torch_delegate-*.whl) ;;
        *) WHL_NAME="torch_delegate-0.0.0-py3-none-linux_aarch64.whl" ;;
    esac
    if [ -z "${whl_url:-}" ]; then
        echo "SMOKE FAIL: whl_url 为空 (compile action outputs 未解析?)"
        RET=1
    elif ! wget -nv -O "output/package/${WHL_NAME}" "${whl_url}"; then
        echo "SMOKE FAIL: whl 下载失败: ${whl_url}"
        RET=1
    fi
fi

# 2) 仓内 prepare_env.sh 建 venv_test (含 torch/torch_npu daily/triton + 安装 whl)
if [ $RET -eq 0 ]; then
    cd examples || RET=1
    if ! bash prepare_env.sh; then
        echo "SMOKE FAIL: prepare_env.sh 失败"
        RET=1
    fi
fi

# 3) 最小端到端示例 (run_all.sh 首个 enabled 用例)
if [ $RET -eq 0 ]; then
    source venv_test/bin/activate
    cd export_and_load_example || RET=1
    if ! bash run.sh; then
        echo "SMOKE FAIL: export_and_load_example 失败"
        RET=1
    fi
    cd ..
fi

# 4) 收集设备侧日志 (workflow 上传 OBS 便于定位)
if [ -d /root/ascend/log ]; then
    tar -zcf "${WORKSPACE}/slog.tar.gz" -C /root/ascend log 2>/dev/null
fi

echo ""
if [ $RET -eq 0 ]; then
    echo "SMOKE PASS: export_and_load_example (A2/910B)"
else
    echo "SMOKE FAIL: 详见上方日志"
fi
exit ${RET}
