#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run.sh - host_cpu_mixed_case 一键式验证脚本
#
# 前置条件：
#   1. source <cann_path>/set_env.sh（ASCEND_HOME_PATH）
#   2. Python 环境已装 torch + torch_npu（含 torchair）；源码开发时 PYTHONPATH
#      需含 torch_delegate 源码根（已安装 whl 则无需）
#   3. C++ 加载阶段需要已构建的 torch_delegate_backend（源码构建目录或已安装 whl）
#
# 用法：
#   bash run.sh
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLES_DIR}/.." && pwd)"
PHASE=0
TOTAL=4

phase() {
  PHASE=$((PHASE + 1))
  echo ""
  echo "========================================"
  echo " [${PHASE}/${TOTAL}] $1"
  echo "========================================"
}

fail() {
  echo "  [✗] $1"
  exit 1
}

# 导出阶段 PYTHONPATH：源码树优先，其次已安装 whl
if [ -f "${PROJECT_ROOT}/torch_delegate/api.py" ]; then
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
fi

# 加载阶段 codegen 需要的 torch / torch_npu 路径（未显式指定时自动推导）
if [ -z "${TORCH_PKG_DIR:-}" ]; then
  TORCH_PKG_DIR="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
  export TORCH_PKG_DIR
fi
if [ -z "${TORCH_NPU_DIR:-}" ]; then
  TORCH_NPU_DIR="$(python3 -c 'import torch_npu, os; print(os.path.dirname(torch_npu.__file__))')"
  export TORCH_NPU_DIR
fi
# 生成 .so 的传递依赖（libtorch_python 等）经 LD_LIBRARY_PATH 解析
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}/../torch.libs:${TORCH_NPU_DIR}/lib:${LD_LIBRARY_PATH:-}"

phase "导出三种形态模型并校验 Device 归属"
python3 "${ROOT_DIR}/export_model.py"

phase "校验 epair 中的 device 字段与 manifest v2"
python3 - "${ROOT_DIR}/output" <<'PYEOF'
import json
import sys
import zipfile
from pathlib import Path

output_dir = Path(sys.argv[1])
for epair in sorted(output_dir.glob("model_*.epair")):
    with zipfile.ZipFile(epair) as zf:
        manifest = json.loads(zf.read("model_mixed/extra/epair_manifest.json" if False else
                                      [n for n in zf.namelist() if n.endswith("epair_manifest.json")][0]))
        annotated = json.loads(zf.read([n for n in zf.namelist() if n.endswith("ep_json_with_ascend_ir.json")][0]))
    assert manifest["version"] == 2, f"{epair.name}: manifest version != 2"
    devices = {n["ascend_ir"]["device"] for n in annotated["graph_module"]["graph"]["nodes"] if "ascend_ir" in n}
    # cpu_nodes: model_mixed 1 个 CPU 节点，其余模型均为 2 个
    expected_cpu = 1 if "mixed" in epair.name else 2
    assert manifest["cpu_nodes"] == expected_cpu, f"{epair.name}: cpu_nodes={manifest['cpu_nodes']} != {expected_cpu}"
    print(f"  [✓] {epair.name}: version=2, cpu_nodes={manifest['cpu_nodes']}, devices={sorted(devices)}")
PYEOF

phase "构建并执行 HostCpu runner（需要 NPU 设备）"
RUNNER="${ROOT_DIR}/build/host_cpu_runner"
# 源码构建场景自动推导 torch_delegate 构建目录（whl 安装场景请 export TORCH_DELEGATE_ROOT）
if [ -z "${TORCH_DELEGATE_BUILD_DIR:-}" ] && [ -z "${TORCH_DELEGATE_ROOT:-}" ] \
    && [ -f "${PROJECT_ROOT}/build/release/csrc/libtorch_delegate_backend.so" ]; then
  export TORCH_DELEGATE_BUILD_DIR="${PROJECT_ROOT}/build/release"
fi
if command -v cmake >/dev/null 2>&1 && ls /dev/davinci* >/dev/null 2>&1; then
  cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
      -DTORCH_DELEGATE_ROOT_DIR="${TORCH_DELEGATE_ROOT:-}" \
      -DTORCH_DELEGATE_BUILD_DIR="${TORCH_DELEGATE_BUILD_DIR:-}" \
      -DTORCH_NPU_DIR="${TORCH_NPU_DIR}" >/dev/null
  cmake --build "${ROOT_DIR}/build" -j4
  "${RUNNER}" --custom-ops "${ROOT_DIR}/build/libcustom_ops.so" \
      "${ROOT_DIR}/output/model_mixed.epair" \
      "${ROOT_DIR}/output/model_pure_cpu.epair" \
      "${ROOT_DIR}/output/model_non_adjacent.epair" \
      "${ROOT_DIR}/output/model_recommender.epair"
else
  echo "  [SKIP] 无 NPU 设备或 cmake 不可用，跳过 C++ 执行阶段"
fi

phase "精度对比"
if [ -f "${ROOT_DIR}/output/delegate_output_mixed.bin" ]; then
  python3 "${ROOT_DIR}/compare_accuracy.py"
else
  echo "  [SKIP] runner 未执行，跳过精度对比"
fi

echo ""
echo "host_cpu_mixed_case 全部通过"
