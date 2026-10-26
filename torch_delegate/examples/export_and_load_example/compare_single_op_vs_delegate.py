# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
单算子执行 vs Delegate 整图执行 — 精度比对

流程:
  1. PyTorch eager 逐算子执行，输出保存为 eager_output_<idx>.bin
  2. 调用 C++ export_and_load_example 二进制（输出保存为 delegate_output_<idx>.bin）
  3. 加载两个 bin 文件，逐元素比对
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
import torch_delegate as td


ROOT = Path(__file__).resolve().parent


def load_binary_input(path: Path) -> torch.Tensor:
    raw = torch.from_numpy(np.fromfile(str(path), dtype="float32")).reshape(3, 4)
    print(f"  reading input from: {path}")
    print(f"  shape: {list(raw.shape)}")
    print(f"  data:  {raw.flatten().tolist()}")
    return raw


def run_eager_single_ops():
    """PyTorch eager 逐算子执行，输出保存为 eager_output_<idx>.bin。"""
    epair = td.epair_load(str(ROOT / "epair_model.epair"))
    ep = epair.ep

    x = load_binary_input(ROOT / "input_0.bin")

    model = ep.module().npu()
    x_npu = x.npu()

    print("  Running PyTorch eager (single-op)...")
    with torch.no_grad():
        output = model(x_npu)

    output_cpu = output.cpu()
    print(f"  Output shape: {list(output_cpu.shape)}")
    print(f"  Output data:  {output_cpu.flatten().tolist()}")

    # 保存为二进制文件
    for i, t in enumerate([output_cpu]):
        path = ROOT / f"eager_output_{i}.bin"
        t.float().numpy().tofile(path)
        print(f"  wrote {path}")

    return output_cpu.flatten()


def run_delegate_binary(epair_path: str):
    """调用 C++ delegate 二进制执行（输出已由 C++ 保存为 delegate_output_<idx>.bin）。"""
    result = subprocess.run(
        ["./build/export_and_load_example", epair_path],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Delegate binary exited with code {result.returncode}")


def load_delegate_output() -> torch.Tensor:
    """从 delegate_output_<idx>.bin 加载输出。"""
    path = ROOT / "delegate_output_0.bin"
    if not path.exists():
        raise RuntimeError(f"Delegate output not found: {path}")
    data = torch.from_numpy(np.fromfile(str(path), dtype="float32"))
    print(f"  reading delegate output from: {path}")
    print(f"  data:  {data.tolist()}")
    return data


def compare(
    ref: torch.Tensor, test: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-5
):
    if ref.shape != test.shape:
        print(
            f"  [WARN] Shape mismatch: ref {list(ref.shape)} vs test {list(test.shape)}"
        )
        return False

    abs_diff = (ref - test).abs()
    rel_diff = abs_diff / (ref.abs() + 1e-10)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()

    print(f"  max absolute error: {max_abs:.6e}")
    print(f"  mean absolute error: {mean_abs:.6e}")
    print(f"  max relative error: {max_rel:.6e}")

    if max_abs <= atol or max_rel <= rtol:
        print(f"  => PASS (within rtol={rtol}, atol={atol})")
        return True

    mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)
    mismatch_count = mismatch_mask.sum().item()
    if mismatch_count == 0:
        print("  => PASS (no element exceeds both rtol and atol)")
        return True

    print(f"  => FAIL: {mismatch_count}/{ref.numel()} elements exceed tolerance")
    return False


def main():
    epair_path = str(ROOT / "epair_model.epair")
    if not Path(epair_path).exists():
        print(f"[ERROR] epair file not found: {epair_path}")
        print("  Run 'python export_ep_and_epair.py' first.")
        sys.exit(1)

    # [1/2] Eager single-op execution
    print("=" * 40)
    print(" [1/2] Single-operator execution (PyTorch eager)")
    print("=" * 40)
    print()
    ref = run_eager_single_ops()
    print()
    print("[PASS] Single-operator execution completed")
    print()

    # [2/2] Delegate whole-graph execution
    print("=" * 40)
    print(" [2/2] Delegate (whole-graph) execution")
    print("=" * 40)
    print()
    run_delegate_binary(epair_path)

    # Load delegate output from bin file
    print()
    test = load_delegate_output()

    # Compare
    print()
    print("=" * 40)
    print(" Accuracy comparison")
    print("=" * 40)
    print()
    passed = compare(ref, test)
    print()
    if passed:
        print("=" * 40)
        print(" All checks passed")
        print("=" * 40)
        sys.exit(0)
    else:
        print("=" * 40)
        print(" Accuracy check FAILED")
        print("=" * 40)
        sys.exit(1)


if __name__ == "__main__":
    main()
