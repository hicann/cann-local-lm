#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 一键运行所有阶段
#
# 阶段 1-5: CPU 测试（基础功能）
# 阶段 6-7: NPU 测试（epair 功能，需要 torch_npu）
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/generated"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Inplace Input Verification - All Phases"
echo "============================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Phase 1: torch.export FX Graph (CPU)
echo "=== Phase 1: torch.export FX Graph (CPU) ==="
python "${SCRIPT_DIR}/1_export_fx_graph.py" --device cpu --dtype float32
echo ""

# Phase 2: Verify torch.export (CPU, Cross-Process)
echo "=== Phase 2: Verify torch.export Load (CPU, Cross-Process) ==="
python "${SCRIPT_DIR}/2_verify_export.py" --device cpu --dtype float32
echo ""

# Phase 3 & 4: AOTI Compile & Verify (CPU)
# [SKIP] AOTI 依赖 gcc11+ 高版本 GLIBCXX 库，gcc10 及以下平台无法运行，故默认跳过。
# 如需在 gcc11+ 环境手动验证，可将下方 SKIP_AOTI 改为 false。
SKIP_AOTI=true
if [ "${SKIP_AOTI}" == "false" ]; then
    echo "=== Phase 3: AOTI Compile (CPU) ==="
    python "${SCRIPT_DIR}/3_aoti_compile.py" --device cpu --dtype float32
    echo ""

    echo "============================================================"
    echo " Phase 4: Build verify_aoti (C++)"
    echo "============================================================"
    BUILD_AOTI_DIR="${SCRIPT_DIR}/4_verify_aoti/build"
    mkdir -p "${BUILD_AOTI_DIR}"
    TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)"
    cmake -S "${SCRIPT_DIR}/4_verify_aoti" -B "${BUILD_AOTI_DIR}" \
      -DTORCH_PKG_DIR="${TORCH_PKG_DIR}"
    cmake --build "${BUILD_AOTI_DIR}" -j$(nproc)
    echo ""

    echo "============================================================"
    echo " Phase 4: Verify AOTI Load (CPU)"
    echo "============================================================"
    export LD_LIBRARY_PATH="${TORCH_PKG_DIR}.libs:${TORCH_PKG_DIR}/lib:${LD_LIBRARY_PATH:-}"
    python3 "${SCRIPT_DIR}/4_verify_aoti/verify_aoti.py"
    echo ""
else
    echo "[SKIP] Phase 3 & 4: AOTI Compile & Verify (需要 gcc11+ GLIBCXX，默认跳过)"
    echo ""
fi

# Phase 5: torch.compile FX Graph (CPU)
echo "=== Phase 5: torch.compile FX Graph (CPU) ==="
python "${SCRIPT_DIR}/5_compile_fx_graph.py" --device cpu --dtype float32
echo ""

# NPU 测试（需要 torch_npu 和 TORCH_DELEGATE_ROOT）
if [ -z "${TORCH_DELEGATE_ROOT:-}" ]; then
    TORCH_DELEGATE_ROOT="$(python -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
    export TORCH_DELEGATE_ROOT
fi
if [ -z "${TORCH_DELEGATE_ROOT:-}" ]; then
    echo "[SKIP] TORCH_DELEGATE_ROOT 未设置，跳过 NPU 测试"
    echo "       请先 pip install <torch_delegate whl>（或 examples/prepare_env.sh）后再试"
elif python -c "import torch; assert torch.npu.is_available()" 2>/dev/null; then
    echo "============================================================"
    echo " Phase 6/7: epair export + verify (NPU)"
    echo "============================================================"
    bash "${SCRIPT_DIR}/7_npu/run_all.sh"
    echo ""
else
    echo "[SKIP] NPU not available (torch_npu not installed or NPU not detected)"
    echo "       Skipping Phase 6-7 (epair tests)"
fi

echo "============================================"
echo "All Phases Completed!"
echo "============================================"

# 打印生成的文件列表
echo ""
echo "Generated files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}"

echo ""
echo "FX Graph files (for comparison):"
echo "  - export_fx_graph_*.txt    (torch.export captured)"
echo "  - dynamo_fx_graph_*.txt    (torch.compile Dynamo captured)"
echo "  - epair_fx_graph_*.txt     (epair input graph)"
echo ""
echo "Model files:"
echo "  - export_model_*.pt2       (torch.export saved)"
echo "  - aoti_model_*.pt2         (AOTI compiled)"
echo "  - epair_model_*.epair      (torch_delegate epair)"
