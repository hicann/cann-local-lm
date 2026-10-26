# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
阶段 5: torch.compile 编译并观察 FX 图

核心功能：
1. 设置 TORCH_COMPILE_DEBUG 环境变量
2. 使用自定义 backend 捕获 Dynamo FX 图
3. 使用 inductor backend 运行，生成 Inductor IR
4. 保存产物：
   - Dynamo FX 图 (dynamo_fx_graph_{model}.txt)
   - Dynamo FX 代码 (dynamo_fx_code_{model}.py)
   - Inductor IR pre/post fusion
"""

import os
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _THIS_DIR / "generated"

# 设置调试环境变量
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
os.environ.setdefault("TORCH_COMPILE_DEBUG_DIR", str(_GENERATED_DIR))

import torch  # noqa: E402

from models import MODEL_CONFIGS, get_model  # noqa: E402
from utils import resolve_device, create_test_inputs, save_fx_graph, save_fx_code  # noqa: E402


def capture_compile_fx_graph(
    model_name: str, device: torch.device, dtype: torch.dtype, output_dir: Path
):
    """捕获 torch.compile 的 FX 图

    Args:
        model_name: 模型名称
        device: 设备
        dtype: 数据类型
        output_dir: 输出目录
    """
    print(f"\n{'=' * 60}")
    print(f"torch.compile FX Graph: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"{'=' * 60}")

    # 1. 创建模型
    model = get_model(model_name, device, dtype)

    # 2. 创建输入
    inputs = create_test_inputs(model_name, device, dtype)
    print(f"  Input shapes: {[t.shape for t in inputs]}")

    # 3. 使用自定义 backend 捕获 Dynamo FX 图
    print("\n  [Step 1] Capturing Dynamo FX graph with custom backend...")

    captured_graphs = []

    def capture_backend(gm, example_inputs):
        """自定义 backend 来捕获 FX Graph"""
        captured_graphs.append(gm)

        # 保存 Dynamo FX 图
        graph_path = output_dir / f"dynamo_fx_graph_{model_name}.txt"
        save_fx_graph(gm.graph, graph_path, "dynamo", model_name)

        # 保存 Dynamo FX 代码
        code_path = output_dir / f"dynamo_fx_code_{model_name}.py"
        try:
            save_fx_code(gm.code, code_path, "dynamo", model_name)
        except Exception:
            pass

        # 返回 forward 函数用于执行
        return gm.forward

    try:
        import torch._dynamo

        torch._dynamo.reset()

        # 编译模型（使用自定义 backend）
        compiled_model = torch.compile(model, backend=capture_backend, fullgraph=False)
        print("    [OK] Compilation configured")

        # 4. 运行触发编译
        print("\n  [Step 2] Running to trigger compilation...")
        cloned_inputs = [t.clone() for t in inputs]

        with torch.inference_mode():
            output = compiled_model(*cloned_inputs)

        print(f"    Output shape: {output.shape}")
        print(f"    Input modified: {not torch.equal(cloned_inputs[0], inputs[0])}")

        # 5. 检查捕获的图
        print(f"\n  Captured graphs count: {len(captured_graphs)}")

        # 6. 对比 Dynamo 和 export 的 FX 图
        print("\n  [Step 3] Comparing Dynamo vs export FX graph...")
        export_graph_path = output_dir / f"export_fx_graph_{model_name}.txt"

        if export_graph_path.exists():
            print(f"    Export graph exists: {export_graph_path}")
            print(
                f"    Dynamo graph saved: {output_dir / f'dynamo_fx_graph_{model_name}.txt'}"
            )

            # 读取并简单对比
            with open(export_graph_path) as f:
                export_content = f.read()
            with open(output_dir / f"dynamo_fx_graph_{model_name}.txt") as f:
                dynamo_content = f.read()

            # 统计 inplace ops
            export_inplace = export_content.count("INPLACE OPERATION")
            dynamo_inplace = dynamo_content.count("INPLACE OPERATION")

            print(f"    Inplace ops in export: {export_inplace}")
            print(f"    Inplace ops in dynamo: {dynamo_inplace}")

            if export_inplace == dynamo_inplace:
                print("    [OK] Inplace ops count matched")
            else:
                print("    [WARNING] Inplace ops count differs")
        else:
            print("    [WARNING] Export graph not found, run phase 1 first")

        return True

    except Exception as e:
        print(f"    [ERROR] Compile/run failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def observe_inductor_ir(
    model_name: str, device: torch.device, dtype: torch.dtype, output_dir: Path
):
    """观察 Inductor 生成的 IR

    Args:
        model_name: 模型名称
        device: 设备
        dtype: 数据类型
        output_dir: 输出目录
    """
    print(f"\n{'=' * 60}")
    print(f"Inductor IR Observation: {model_name}")
    print(f"{'=' * 60}")

    model = get_model(model_name, device, dtype)
    inputs = create_test_inputs(model_name, device, dtype)

    # 设置调试目录
    debug_subdir = output_dir / f"compile_debug_{model_name}"
    debug_subdir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(debug_subdir)

    print(f"  Debug output directory: {debug_subdir}")

    try:
        import torch._dynamo

        torch._dynamo.reset()

        # 使用 inductor backend
        compiled_model = torch.compile(model, backend="inductor", fullgraph=False)

        cloned_inputs = [t.clone() for t in inputs]
        with torch.inference_mode():
            compiled_model(*cloned_inputs)

        print("  [OK] Inductor compilation and run successful")

        # 检查生成的调试文件
        print(f"\n  Checking generated files in: {debug_subdir}")

        ir_files = list(debug_subdir.glob("**/*.txt"))
        for ir_file in ir_files[:5]:  # 只显示前5个
            print(f"    {ir_file.relative_to(debug_subdir)}")

            # 检查是否是 IR 文件
            if "ir_" in ir_file.name:
                # 复制到主输出目录并改名
                content_type = "ir_pre" if "pre" in ir_file.name else "ir_post"
                dest_path = output_dir / f"compile_{content_type}_{model_name}.txt"

                content = ir_file.read_text()
                dest_path.write_text(
                    f"# Inductor {content_type} IR (model={model_name})\n\n{content}"
                )
                print(f"      -> Saved to: {dest_path.name}")

        return True

    except Exception as e:
        print(f"  [ERROR] Inductor observation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: torch.compile FX graph observation"
    )
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
        "--output-dir",
        type=Path,
        default=_GENERATED_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        choices=("all", "simple", "chain", "param"),
        default="all",
        help="Which model to analyze",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 5: torch.compile FX Graph Observation")
    print("=" * 60)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Output directory: {args.output_dir}")
    print(f"TORCH_COMPILE_DEBUG: {os.environ.get('TORCH_COMPILE_DEBUG')}")

    # 选择要分析的模型
    if args.model == "all":
        models_to_analyze = list(MODEL_CONFIGS.keys())
    else:
        models_to_analyze = [args.model]

    results = {}
    for model_name in models_to_analyze:
        # 捕获 Dynamo FX 图
        success_dynamo = capture_compile_fx_graph(
            model_name, device, dtype, args.output_dir
        )

        # 观察 Inductor IR
        success_inductor = observe_inductor_ir(
            model_name, device, dtype, args.output_dir
        )

        results[model_name] = success_dynamo and success_inductor

    # 总结
    print("\n" + "=" * 60)
    print("Phase 5 Summary")
    print("=" * 60)

    for model_name, success in results.items():
        status = "OK" if success else "FAIL"
        print(f"  {model_name}: {status}")

    # 打印生成的文件
    print("\nGenerated files:")
    for f in sorted(
        args.output_dir.glob(
            f"dynamo_fx_*_{args.model if args.model != 'all' else '*'}.txt"
        )
    ):
        print(f"  {f.name}")
    for f in sorted(
        args.output_dir.glob(
            f"dynamo_fx_*_{args.model if args.model != 'all' else '*'}.py"
        )
    ):
        print(f"  {f.name}")
    for f in sorted(
        args.output_dir.glob(
            f"compile_ir_*_{args.model if args.model != 'all' else '*'}.txt"
        )
    ):
        print(f"  {f.name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit(main())
