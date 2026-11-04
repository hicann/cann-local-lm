#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export a deterministic torchvision MobileNetV2 for ExecuTorch-NPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower
from torchvision.models import mobilenet_v2

from executorch_npu import (
    AscendCompileOptions,
    AscendPartitioner,
    aten_edge_compile_config,
)


def main() -> None:
    # 完整导出流程：
    # mobilenet_v2(weights=None) + example_input
    #         |
    #         v
    # torch.export.export()
    #         |
    #         v
    # ExportedProgram
    #         |
    #         v
    # to_edge_transform_and_lower()
    #         |
    #         +-- Edge 图转换
    #         +-- AscendPartitioner 全图分区
    #         +-- AscendBackend.preprocess()
    #               |
    #               +-- torch_delegate 导出 EPAIR
    #               +-- TorchAir / GE 建图
    #               +-- CANN Offline_compile
    #               +-- 生成 OM
    #               +-- 收集外置权重
    #               +-- 生成 manifest
    #         |
    #         v
    # .to_executorch()
    #         |
    #         +-- model.pte（图、manifest、OM）
    #         +-- ascend.weights.package.ge_aot.ptd（外置权重）
    #         +-- input.bin / expected.bin（端到端测试数据）
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acl-soc-name", default="Ascend910_9362")
    args = parser.parse_args()
    # 构造mobilenet
    torch.manual_seed(0)
    model = mobilenet_v2(weights=None).eval()
    example_input = torch.randn(1, 3, 224, 224)
    # 使用CPU执行一次，生成参考输出
    with torch.no_grad():
        expected = model(example_input).contiguous()
    # 导出program，设置编译配置，使用backend生成编译产物
    exported_program = torch.export.export(model, (example_input,))
    options = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name=args.acl_soc_name,
    )
    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[AscendPartitioner(options.to_compile_specs())],
        compile_config=aten_edge_compile_config(),
    ).to_executorch()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input.bin").write_bytes(example_input.numpy().tobytes())
    (args.output_dir / "expected.bin").write_bytes(expected.numpy().tobytes())
    (args.output_dir / "model.pte").write_bytes(program.buffer)
    program.write_tensor_data_to_file(args.output_dir)
    print(f"Wrote MobileNetV2 artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
