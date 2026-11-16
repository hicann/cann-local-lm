#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fallback Dynamic/Optional 测试 - 模型导出脚本

导出以下场景的模型：
- my_addn: 动态输入 Tensor[]
- my_addmm (bias=None): 可选输入不占位
- my_addmm (bias=value): 可选输入占位（对照组）
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_ops  # noqa: F401

import torch_delegate as td

import logging

try:
    from torchair.core.utils import logger

    logger.setLevel(logging.WARN)
except ImportError:
    pass


def export_model(model_name, model, example_inputs, output_dir):
    print(f"\n{'=' * 60}")
    print(f"Exporting {model_name}")
    print(f"{'=' * 60}")

    print("[1/3] Exporting ExportedProgram...")
    ep = torch.export.export(model, example_inputs)
    ep.graph_module.print_readable()
    ep_path = output_dir / f"{model_name}.pt2"
    torch.export.save(ep, ep_path)
    print(f"  Saved EP: {ep_path}")

    print("[2/3] Exporting epair...")
    epair_content = td.export_to_epair(ep)
    epair_path = output_dir / f"{model_name}.epair"
    td.epair_save(epair_content, epair_path)
    print(f"  Saved epair: {epair_path}")

    print("[3/3] Statistics:")
    total = epair_content.manifest.total_nodes
    converted = epair_content.manifest.converted_nodes
    fallback = total - converted
    print(f"  Total nodes: {total}")
    print(f"  Converted nodes: {converted}")
    print(f"  Fallback nodes: {fallback}")

    fallback_ops = []
    for node in epair_content.ep_json["graph_module"]["graph"]["nodes"]:
        ascend_ir = node.get("ascend_ir", {})
        if ascend_ir.get("source") == "":
            fallback_ops.append(node.get("target", ""))
    if fallback_ops:
        print(f"  Fallback ops ({len(fallback_ops)}):")
        for op in fallback_ops:
            print(f"    - {op}")

    return epair_content


def save_tensor(output_dir, prefix, name, tensor):
    path = output_dir / f"input_{prefix}_{name}.bin"
    shape_path = output_dir / f"input_{prefix}_{name}_shape.txt"
    tensor.cpu().float().detach().numpy().tofile(path)
    with open(shape_path, "w") as f:
        f.write(",".join(str(d) for d in tensor.shape))
    print(f"  Saved {name}: shape={list(tensor.shape)} -> {path.name}")


def main():
    print("=" * 60)
    print("Fallback Dynamic/Optional test - model export")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    class AddnModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.ops.my_ops.my_addn([x, y])

    class AddmmNoBiasModel(torch.nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.my_ops.my_addmm(input, mat1, mat2, None, beta=2.0)

    class AddmmWithBiasModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.randn(5))

        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.my_ops.my_addmm(input, mat1, mat2, self.bias, beta=2.0)

    addn_model = AddnModel().eval()
    x1 = torch.randn(3, 4)
    y1 = torch.randn(3, 4)
    export_model("model_addn", addn_model, (x1, y1), output_dir)
    save_tensor(output_dir, "addn", "x", x1)
    save_tensor(output_dir, "addn", "y", y1)

    addmm_no_bias = AddmmNoBiasModel().eval()
    inp = torch.randn(2, 5)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 5)
    export_model("model_addmm_no_bias", addmm_no_bias, (inp, mat1, mat2), output_dir)
    save_tensor(output_dir, "addmm_no_bias", "input", inp)
    save_tensor(output_dir, "addmm_no_bias", "mat1", mat1)
    save_tensor(output_dir, "addmm_no_bias", "mat2", mat2)

    addmm_with_bias = AddmmWithBiasModel().eval()
    addmm_with_bias.bias.data = torch.randn(5)
    export_model(
        "model_addmm_with_bias", addmm_with_bias, (inp, mat1, mat2), output_dir
    )
    save_tensor(output_dir, "addmm_with_bias", "input", inp)
    save_tensor(output_dir, "addmm_with_bias", "mat1", mat1)
    save_tensor(output_dir, "addmm_with_bias", "mat2", mat2)

    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)
    print("\nOutput files:")
    for f in sorted(output_dir.glob("*.epair")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("input_*.bin")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("input_*.txt")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
