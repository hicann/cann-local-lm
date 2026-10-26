# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
阶段 1: torch.export 导出并保存 FX 图

核心功能：
1. 使用 torch.export.export 导出 ExportedProgram
2. 保存 FX 图结构 (export_fx_graph_{model}.txt)
3. 保存 FX 图代码 (export_fx_code_{model}.py)
4. 保存模型文件 (export_model_{model}.pt2)
"""

import argparse
from pathlib import Path

import torch

from models import MODEL_CONFIGS, get_model
from utils import resolve_device, create_test_inputs, save_fx_graph, save_fx_code

_THIS_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _THIS_DIR / "generated"


def export_model(
    model_name: str, device: torch.device, dtype: torch.dtype, output_dir: Path
):
    """导出单个模型

    Args:
        model_name: 模型名称
        device: 设备
        dtype: 数据类型
        output_dir: 输出目录
    """
    print(f"\n{'=' * 60}")
    print(f"Exporting model: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"{'=' * 60}")

    # 1. 创建模型
    model = get_model(model_name, device, dtype)
    print(f"  Model class: {model.__class__.__name__}")

    # 2. 创建示例输入
    example_inputs = create_test_inputs(model_name, device, dtype)
    print(f"  Input shapes: {[t.shape for t in example_inputs]}")

    # 3. torch.export.export
    print("\n  [Step 1] torch.export.export...")
    ep = torch.export.export(model, example_inputs)
    print("    [OK] Export successful")

    # 打印 Graph Signature
    print(f"\n  Graph Signature:\n{ep.graph_signature}")

    # 4. 保存 FX 图结构
    print("\n  [Step 2] Saving FX graph...")
    graph_path = output_dir / f"export_fx_graph_{model_name}.txt"
    save_fx_graph(ep.graph, graph_path, "export", model_name)

    # 5. 保存 FX 图代码
    print("\n  [Step 3] Saving FX code...")
    code_path = output_dir / f"export_fx_code_{model_name}.py"
    save_fx_code(ep.graph_module.code, code_path, "export", model_name)

    # 6. 保存模型
    print("\n  [Step 4] Saving model...")
    model_path = output_dir / f"export_model_{model_name}.pt2"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, model_path)
    print(f"    [OK] Saved to: {model_path}")

    # 打印 inplace 操作统计
    inplace_count = 0
    for node in ep.graph.nodes:
        if node.op == "call_function":
            target_str = str(node.target)
            if "_.Tensor" in target_str:
                inplace_count += 1
    print(f"\n  Inplace operations in graph: {inplace_count}")

    return ep


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: torch.export and save FX graph"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "npu"),
        default="cpu",
        help="Device for export (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Data type",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_GENERATED_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        choices=("all", "simple", "chain", "param"),
        default="all",
        help="Which model to export",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: torch.export FX Graph")
    print("=" * 60)

    # 选择要导出的模型
    if args.model == "all":
        models_to_export = list(MODEL_CONFIGS.keys())
    else:
        models_to_export = [args.model]

    for model_name in models_to_export:
        export_model(model_name, device, dtype, args.output_dir)

    print("\n" + "=" * 60)
    print("Phase 1 completed!")
    print(f"Generated files in: {args.output_dir}")
    print("=" * 60)

    # 打印生成的文件列表
    print("\nGenerated files:")
    for f in sorted(
        args.output_dir.glob(
            f"export_*_{args.model if args.model != 'all' else '*'}.pt2"
        )
    ):
        print(f"  {f.name}")
    for f in sorted(
        args.output_dir.glob(
            f"export_fx_*_{args.model if args.model != 'all' else '*'}.txt"
        )
    ):
        print(f"  {f.name}")
    for f in sorted(
        args.output_dir.glob(
            f"export_fx_*_{args.model if args.model != 'all' else '*'}.py"
        )
    ):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
