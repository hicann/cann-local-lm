# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
阶段 3: AOTI 编译并保存产物

核心功能：
1. 加载阶段1保存的 .pt2 文件
2. 使用 torch._inductor.aoti_compile_and_package 编译
3. 保存 AOTI 包 (aoti_model_{model}.pt2)
4. 可选：保存编译前的 FX 图
"""

import argparse
from pathlib import Path

import torch

from models import MODEL_CONFIGS
from utils import resolve_device, save_fx_graph

_THIS_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _THIS_DIR / "generated"


def aoti_compile_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    input_dir: Path,
    output_dir: Path,
):
    """AOTI 编译单个模型

    Args:
        model_name: 模型名称
        device: 设备
        dtype: 数据类型
        input_dir: 输入目录（包含 .pt2 文件）
        output_dir: 输出目录
    """
    print(f"\n{'=' * 60}")
    print(f"AOTI Compile: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"{'=' * 60}")

    pt2_path = input_dir / f"export_model_{model_name}.pt2"
    if not pt2_path.exists():
        print(f"  [ERROR] Model file not found: {pt2_path}")
        print("  Run 'python 1_export_fx_graph.py' first")
        return False

    # 1. 加载 ExportedProgram
    print("\n  [Step 1] Loading ExportedProgram...")
    ep = torch.export.load(pt2_path)
    print(f"    [OK] Loaded from: {pt2_path}")

    # 保存编译前的 FX 图
    print("\n  [Step 2] Saving FX graph before AOTI...")
    graph_path = output_dir / f"aoti_fx_graph_{model_name}.txt"
    save_fx_graph(ep.graph, graph_path, "aoti_input", model_name)

    # 2. AOTI 编译
    print("\n  [Step 3] AOTI compiling...")
    aoti_path = output_dir / f"aoti_model_{model_name}.pt2"

    try:
        compiled_path = Path(
            torch._inductor.aoti_compile_and_package(
                ep,
                package_path=str(aoti_path),
                inductor_configs={"cpp_wrapper": True},
            )
        ).resolve()
        print(f"    [OK] AOTI compiled to: {compiled_path}")

    except Exception as e:
        print(f"    [ERROR] AOTI compilation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 3. 验证文件存在
    if compiled_path.exists():
        print(f"    File size: {compiled_path.stat().st_size / 1024:.2f} KB")
        return True
    else:
        print("    [ERROR] Compiled file not found")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 3: AOTI compile and save")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "npu"),
        default="cpu",
        help="Device for compilation",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Data type",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_GENERATED_DIR,
        help="Input directory containing .pt2 files",
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
        help="Which model to compile",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3: AOTI Compile")
    print("=" * 60)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    # 选择要编译的模型
    if args.model == "all":
        models_to_compile = list(MODEL_CONFIGS.keys())
    else:
        models_to_compile = [args.model]

    results = {}
    for model_name in models_to_compile:
        results[model_name] = aoti_compile_model(
            model_name, device, dtype, args.input_dir, args.output_dir
        )

    # 总结
    print("\n" + "=" * 60)
    print("Phase 3 Summary")
    print("=" * 60)

    for model_name, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  {model_name}: {status}")

    all_success = all(results.values())
    print(f"\n  Overall: {'ALL OK' if all_success else 'SOME FAILED'}")

    # 打印生成的文件
    if all_success:
        print("\nGenerated files:")
        for f in sorted(
            args.output_dir.glob(
                f"aoti_model_{args.model if args.model != 'all' else '*'}.pt2"
            )
        ):
            print(f"  {f.name}")

    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
