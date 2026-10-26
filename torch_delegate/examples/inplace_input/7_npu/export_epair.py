#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
epair 导出脚本

导出三个 inplace 模型的 epair 文件:
- simple: x.add_(y), x.mul_(scale)
- chain: x.add_(1), x.mul_(2), x.div_(4)
- param: x.add_(bias), x.mul_(scale) (带参数)
"""

import os
import sys
import json
import torch
import torch_npu  # noqa: F401

# 确保 torch_delegate 包可用
try:
    import torch_delegate as td
except ImportError:
    # 尝试从项目根目录导入
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    sys.path.insert(0, project_root)
    import torch_delegate as td

# 导入模型定义 - 从上级目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model, get_input_shapes

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")


def export_epair(
    model_name: str, device: str = "npu", dtype: torch.dtype = torch.float32
) -> bool:
    """导出单个模型的 epair 文件"""

    print(f"\n{'=' * 60}")
    print(f"Export epair: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"{'=' * 60}")

    try:
        # 1. 创建模型
        print("  [Step 1] Creating model on NPU...")
        model = get_model(model_name, torch.device(device), dtype)
        print(f"    Model class: {model.__class__.__name__}")

        # 2. 创建示例输入
        print("  [Step 2] Creating example inputs...")
        input_shapes = get_input_shapes(model_name)
        example_inputs = []
        for shape in input_shapes:
            tensor = torch.randn(shape, device=device, dtype=dtype)
            example_inputs.append(tensor)
        print(f"    Input shapes: {[t.shape for t in example_inputs]}")

        # 3. torch.export.export
        print("  [Step 3] torch.export.export...")
        ep = torch.export.export(model, tuple(example_inputs))
        print("    [OK] Export successful")
        print(f"\n    Graph signature:\n{ep.graph_signature}")

        # 4. 导出为 epair
        print("  [Step 4] Exporting to epair...")
        contents = td.export_to_epair(ep)
        print("    [OK] epair export successful")
        print("\n    Manifest:")
        print(f"      Version: {contents.manifest.version}")
        print(f"      Backend: {contents.manifest.backend_name}")
        print(f"      CANN version: {contents.manifest.cann_version}")
        print(
            f"      Converted nodes: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}"
        )

        # 5. 保存 epair 文件
        epair_path = os.path.join(OUTPUT_DIR, f"epair_model_{model_name}.epair")
        print(f"  [Step 5] Saving epair to: {epair_path}")
        td.epair_save(contents, epair_path)
        file_size = os.path.getsize(epair_path)
        print(f"    [OK] Saved, file size: {file_size / 1024:.2f} KB")

        # 6. 保存 manifest JSON
        manifest_path = os.path.join(OUTPUT_DIR, f"epair_manifest_{model_name}.json")
        manifest_dict = {
            "version": contents.manifest.version,
            "backend_name": contents.manifest.backend_name,
            "cann_version": contents.manifest.cann_version,
            "total_nodes": contents.manifest.total_nodes,
            "converted_nodes": contents.manifest.converted_nodes,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)
        print(f"    Manifest saved to: {manifest_path}")
        print(
            f"    Converted: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes"
        )

        return True

    except Exception as e:
        print(f"    [ERROR] Export failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    print(f"\n{'=' * 60}")
    print("epair Export Script")
    print(f"{'=' * 60}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查 NPU 设备
    if torch.npu.is_available():
        print(f"NPU available: {torch.npu.device_count()} devices")
        device = "npu"
    else:
        print("Warning: NPU not available, using CPU")
        device = "cpu"

    # 导出所有模型
    results = {}
    for model_name in ["simple", "chain", "param"]:
        results[model_name] = export_epair(model_name, device=device)

    # 汇总结果
    print(f"\n{'=' * 60}")
    print("Export Summary")
    print(f"{'=' * 60}")

    all_ok = True
    for model_name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {model_name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  Overall: ALL OK")
        print(f"\nGenerated files in {OUTPUT_DIR}:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            print(f"  {f}")
        return 0
    else:
        print("\n  Overall: SOME FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
