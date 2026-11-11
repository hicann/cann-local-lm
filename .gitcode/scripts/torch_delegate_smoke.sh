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
#       → 门禁侧编排用例 (不修改产品脚本 run_all.sh, 仅解析其 CASES 表作为用例单一来源):
#         prepare_env.sh 建 venv → SKIP_CASES 剔除暂停用例 → 3 路并行跑常规用例
#         → multi_instance 独占末跑 (测 device_parallelism, device 不能有其他负载)
#       → 收集设备侧 slog 供 workflow 上传 OBS
# 硬门禁: 以真实结果退出 (exit ${RET}), 冒烟失败阻塞流水线
set -x
RET=0

# 早退: PR 未触碰 torch_delegate/** 与 .gitcode/** 时, 构建出的 whl 与 master 无差异,
# 冒烟无增量信号, 直接跳过节省 A2 池 (filelist 缺失/为空则照常跑, fail-safe 同 ge 惯例)
FILELIST="${WORKSPACE}/pr_filelist.txt"
if [ -s "${FILELIST}" ] && ! grep -qE "torch_delegate/|\.gitcode/" "${FILELIST}"; then
    echo "SMOKE SKIP: PR 未改动 torch_delegate/** 与 .gitcode/**, 与本 PR 无关"
    # 占位 slog: upload 步骤 (if always) glob slog*.tar.gz 零匹配会置 job FAILED
    echo "SMOKE SKIP: PR not touching torch_delegate/.gitcode" > "${WORKSPACE}/smoke_skip.txt"
    tar -czf "${WORKSPACE}/slog_skip.tar.gz" -C "${WORKSPACE}" smoke_skip.txt 2>/dev/null
    exit 0
fi

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

# 2) 环境准备 (产品脚本 prepare_env.sh, 不修改): venv_test + torch/torch_npu/triton + whl
if [ $RET -eq 0 ]; then
    cd examples || RET=1
    if ! timeout 900 bash prepare_env.sh; then
        echo "SMOKE FAIL: prepare_env.sh 失败"
        RET=1
    fi
fi

# 3) 门禁侧用例编排 (不动 run_all.sh; CASES 表从其中解析, 表变更自动跟随)
#    - SKIP_CASES: 两个 fallback 用例暂停 (2026-11-05 决定; 恢复时清空该变量)
#    - 3 路并行压缩墙钟 (实测 2 实例并发稳定 / 8 实例被 OOM 杀, 取 3 折中;
#      可用 SMOKE_PARALLEL 环境变量调整)
#    - run_model_multi_instance 独占末跑: 其校验 device_parallelism 并发指标,
#      device 上有其他负载会污染测量; CI 适配 INSTANCES=2/TASKS=12 (8 实例在
#      A2 容器被 OOM 杀, 实测 7/8 worker 提前退出)
export INSTANCES="${INSTANCES:-2}"
export TASKS="${TASKS:-12}"
if [ $RET -eq 0 ]; then
    source venv_test/bin/activate
    if [ -z "${TORCH_DELEGATE_ROOT:-}" ]; then
        TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null | tail -1)"
        export TORCH_DELEGATE_ROOT
    fi
    if [ -z "${TORCH_DELEGATE_ROOT:-}" ]; then
        echo "SMOKE FAIL: TORCH_DELEGATE_ROOT 为空"
        RET=1
    fi
fi

if [ $RET -eq 0 ]; then
    SKIP_CASES="${SKIP_CASES:-fallback_inplace_optional_case fallback_dynamic_optional_case}"
    PARALLEL="${SMOKE_PARALLEL:-3}"
    SMOKE_DIR="$(mktemp -d /tmp/td_smoke.XXXXXX)"

    PAR_CASES=(); EXCL_CASES=()
    while IFS='|' read -r cdir cscript cstatus; do
        [ "${cstatus}" = "enabled" ] || continue
        case " ${SKIP_CASES} " in
            *" ${cdir} "*) echo "  [SKIP-GATE] ${cdir} (门禁暂停)"; continue ;;
        esac
        if [ "${cdir}" = "run_model_multi_instance" ]; then
            EXCL_CASES+=("${cdir}")
        else
            PAR_CASES+=("${cdir}")
        fi
    done < <(grep -oE '"[A-Za-z0-9_]+\|run\.sh\|(enabled|disabled|ignored)"' run_all.sh | tr -d '"')

    echo "  并行用例(${#PAR_CASES[@]}, ${PARALLEL}路): ${PAR_CASES[*]}"
    echo "  独占用例(${#EXCL_CASES[@]}): ${EXCL_CASES[*]}"

    run_case() {
        local d="$1"
        local t0=$(date +%s)
        ( cd "${d}" && timeout 600 bash run.sh ) > "${SMOKE_DIR}/${d}.log" 2>&1
        echo $? > "${SMOKE_DIR}/${d}.status"
        echo "  [done] ${d} rc=$(cat "${SMOKE_DIR}/${d}.status") ($(( $(date +%s) - t0 ))s)"
    }

    for d in "${PAR_CASES[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do sleep 3; done
        run_case "${d}" &
    done
    wait

    for d in "${EXCL_CASES[@]}"; do
        run_case "${d}"
    done

    FAILS=0
    echo ""
    echo "===== 用例汇总 ====="
    for f in "${SMOKE_DIR}"/*.status; do
        d="$(basename "${f}" .status)"
        rc="$(cat "${f}")"
        if [ "${rc}" = "0" ]; then
            echo "  PASS  ${d}"
        else
            echo "  FAIL  ${d} (rc=${rc})"
            FAILS=$((FAILS + 1))
        fi
    done
    if [ ${FAILS} -gt 0 ]; then
        RET=1
        for f in "${SMOKE_DIR}"/*.status; do
            d="$(basename "${f}" .status)"
            [ "$(cat "${f}")" = "0" ] && continue
            echo "===== tail -60 ${d}.log ====="
            tail -60 "${SMOKE_DIR}/${d}.log"
        done
    fi
fi

# 3.5) 失败时回显 multi_instance worker 日志
# (worker 日志在 /tmp 临时目录, 随 job 销毁; 不回显进 job log 就无法定位 worker 提前退出原因)
if [ $RET -ne 0 ]; then
    for d in /tmp/td_run_model_multi_*; do
        [ -d "$d" ] || continue
        echo "===== multi_instance workdir: $d ====="
        ls -l "$d" 2>/dev/null
        for f in "$d"/*.log; do
            [ -f "$f" ] || continue
            echo "----- tail -30 $f -----"
            tail -30 "$f"
        done
    done
fi

# 4) 收集设备侧日志 (workflow 上传 OBS 便于定位)
if [ -d /root/ascend/log ]; then
    tar -zcf "${WORKSPACE}/slog.tar.gz" -C /root/ascend log 2>/dev/null
fi

echo ""
if [ $RET -eq 0 ]; then
    echo "SMOKE PASS: 全部编排用例 (A2/910B)"
else
    echo "SMOKE FAIL: 详见上方日志"
fi
exit ${RET}
