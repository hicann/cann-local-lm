# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
阶段 2: 跨进程验证 torch.export 加载

核心功能：
1. 新进程加载阶段1保存的 .pt2 文件
2. 重新生成测试输入数据（固定 seed）
3. 运行 eager 模式获取参考
4. 加载 ExportedProgram 并运行
5. 验证 inplace 输入修改是否正确
6. 验证输出精度是否正确
"""

import argparse
from pathlib import Path

import torch

from models import MODEL_CONFIGS, get_model
from utils import (
    resolve_device,
    create_test_inputs,
    verify_inplace_modification,
    compare_outputs,
)

_THIS_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _THIS_DIR / "generated"


def verify_export_load(
    model_name: str, device: torch.device, dtype: torch.dtype, input_dir: Path
):
    """验证单个模型的 export 加载

    Args:
        model_name: 模型名称
        device: 设备
        dtype: 数据类型
        input_dir: 输入目录（包含 .pt2 文件）
    """
    print(f"\n{'=' * 60}")
    print(f"Verifying export load: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"{'=' * 60}")

    pt2_path = input_dir / f"export_model_{model_name}.pt2"
    if not pt2_path.exists():
        print(f"  [ERROR] Model file not found: {pt2_path}")
        print("  Run 'python 1_export_fx_graph.py' first")
        return False

    # 1. 重新生成测试输入（固定 seed）
    print("\n  [Step 1] Creating test inputs (seed=42)...")
    original_inputs = create_test_inputs(model_name, device, dtype)
    print(f"    Input shapes: {[t.shape for t in original_inputs]}")

    # 2. Eager 模式运行获取参考
    print("\n  [Step 2] Running eager mode for reference...")
    model_eager = get_model(model_name, device, dtype)

    eager_inputs = [t.clone() for t in original_inputs]
    original_inputs_clone = [t.clone() for t in original_inputs]

    with torch.inference_mode():
        eager_output = model_eager(*eager_inputs)

    print(f"    Eager output shape: {eager_output.shape}")
    print(
        f"    Input modified in eager: {not torch.equal(eager_inputs[0], original_inputs_clone[0])}"
    )

    # 3. 加载 ExportedProgram
    print("\n  [Step 3] Loading ExportedProgram...")
    ep = torch.export.load(pt2_path)
    print(f"    [OK] Loaded from: {pt2_path}")

    # 4. 运行加载的模型
    print("\n  [Step 4] Running loaded ExportedProgram...")
    export_inputs = [t.clone() for t in original_inputs]

    with torch.inference_mode():
        export_output = ep.module()(*export_inputs)

    print(f"    Export output shape: {export_output.shape}")

    # 5. 验证 inplace 修改
    passed_inplace = verify_inplace_modification(
        "export_load",
        original_inputs_clone,
        export_inputs,
        eager_inputs,
    )

    # 6. 验证输出精度
    passed_output = compare_outputs("export_load", export_output, eager_output)

    # 7. 打印输入值对比（调试）
    print("\n  Debug - Input values (first 5 elements):")
    print(f"    Original: {original_inputs_clone[0].flatten()[:5]}")
    print(f"    Eager after: {eager_inputs[0].flatten()[:5]}")
    print(f"    Export after: {export_inputs[0].flatten()[:5]}")

    # 总结
    all_passed = passed_inplace and passed_output
    print(f"\n  [{model_name}] Verification: {'PASS' if all_passed else 'FAIL'}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Cross-process verify torch.export load"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "npu"),
        default="cpu",
        help="Device for verification",
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
        "--model",
        choices=("all", "simple", "chain", "param"),
        default="all",
        help="Which model to verify",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)

    print("=" * 60)
    print("Phase 2: Verify torch.export Load (Cross-Process)")
    print("=" * 60)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Input directory: {args.input_dir}")

    # 选择要验证的模型
    if args.model == "all":
        models_to_verify = list(MODEL_CONFIGS.keys())
    else:
        models_to_verify = [args.model]

    results = {}
    for model_name in models_to_verify:
        results[model_name] = verify_export_load(
            model_name, device, dtype, args.input_dir
        )

    # 总结
    print("\n" + "=" * 60)
    print("Phase 2 Summary")
    print("=" * 60)

    for model_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {model_name}: {status}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
