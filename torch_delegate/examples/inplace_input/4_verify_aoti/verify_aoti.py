# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
单算子 (PyTorch eager) vs AOTI (C++) — 精度比对 + Inplace 校验

流程:
  1. 生成 input_<idx>.bin（seed=42，与 C++ 侧一致）
  2. PyTorch eager 执行，保存 eager_output_0.bin 和 eager_input_<idx>.bin
  3. 调用 C++ verify_aoti 二进制（加载 AOTI 模型，保存 aoti_output/input_<idx>.bin）
  4. 逐元素比对：output 精度 + inplace 修改一致性
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from models import get_model, get_input_shapes  # noqa: E402

BUILD_DIR = THIS_DIR / "build"
GENERATED_DIR = THIS_DIR.parent / "generated"


def make_input_bin(work_dir: Path, model_name: str):
    """生成 <model_name>_input_<idx>.bin（seed=42），返回 inputs list。"""
    torch.manual_seed(42)
    shapes = get_input_shapes(model_name)
    inputs = []
    for i, shape in enumerate(shapes):
        t = torch.randn(*shape, dtype=torch.float32)
        path = work_dir / f"{model_name}_input_{i}.bin"
        t.numpy().tofile(path)
        inputs.append(t)
        print(f"  wrote {path}  shape={list(shape)}")
    return inputs


def run_eager(model_name: str, inputs):
    """PyTorch eager 执行，返回 (output, post_run_inputs)。"""
    model = get_model(model_name, torch.device("cpu"), torch.float32)
    eager_inputs = [t.clone() for t in inputs]
    with torch.no_grad():
        out = model(*eager_inputs)
    return out, eager_inputs


def save_eager_results(work_dir: Path, model_name: str, output, inputs):
    """保存 eager 的 output 和 post-run inputs，文件名带 model_name 前缀。"""
    path = work_dir / f"{model_name}_eager_output_0.bin"
    output.cpu().float().numpy().tofile(path)
    print(f"  wrote {path}")
    for i, t in enumerate(inputs):
        path = work_dir / f"{model_name}_eager_input_{i}.bin"
        t.cpu().float().numpy().tofile(path)
        print(f"  wrote {path}")


def run_cpp_aoti(work_dir: Path, model_name: str):
    """调用 C++ verify_aoti 二进制。"""
    aoti_path = GENERATED_DIR / f"aoti_model_{model_name}.pt2"
    if not aoti_path.exists():
        raise RuntimeError(f"AOTI model not found: {aoti_path}")

    binary = BUILD_DIR / "verify_aoti"
    if not binary.exists():
        raise RuntimeError(f"Binary not found: {binary}. Run cmake --build first.")

    print(f"  calling: {binary}")
    result = subprocess.run(
        [
            str(binary),
            "--model-path",
            str(aoti_path),
            "--model-name",
            model_name,
            "--input-dir",
            str(work_dir),
            "--output-dir",
            str(work_dir),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"C++ binary exited with code {result.returncode}")


def load_bin(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(str(path), dtype="float32"))


def compare_tensors(
    name: str, ref: torch.Tensor, test: torch.Tensor, rtol=1e-3, atol=1e-5
):
    if ref.shape != test.shape:
        print(f"  [{name}] Shape mismatch: {list(ref.shape)} vs {list(test.shape)}")
        return False

    abs_diff = (ref - test).abs()
    rel_diff = abs_diff / (ref.abs() + 1e-10)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    ok = (max_abs <= atol) or (max_rel <= rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{name}] max_abs={max_abs:.6e} max_rel={max_rel:.6e}  [{status}]")
    return ok


def verify_model(model_name: str):
    print(f"\n{'=' * 60}")
    print(f" Verifying: {model_name}")
    print(f"{'=' * 60}")

    work_dir = THIS_DIR / f"verify_work_{model_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Generate input .bin files ----
    print("\n  [Step 1] Generating input files (seed=42)...")
    inputs = make_input_bin(work_dir, model_name)

    # ---- Step 2: Run eager ----
    print("\n  [Step 2] Running PyTorch eager...")
    eager_out, eager_inputs = run_eager(model_name, inputs)
    print(f"  eager output data: {eager_out.flatten().tolist()}")
    save_eager_results(work_dir, model_name, eager_out, eager_inputs)

    # ---- Step 3: Run C++ AOTI binary ----
    print("\n  [Step 3] Running C++ AOTI binary...")
    run_cpp_aoti(work_dir, model_name)

    # ---- Step 4: Compare output ----
    print("\n  [Step 4] Comparing output...")
    eager_out_flat = eager_out.flatten()
    aoti_out = load_bin(work_dir / f"{model_name}_aoti_output_0.bin")
    output_ok = compare_tensors("output", eager_out_flat, aoti_out)

    # ---- Step 5: Compare inplace ----
    print("\n  [Step 5] Comparing inplace modification...")
    inplace_ok = True
    for i in range(len(inputs)):
        eager_in = load_bin(work_dir / f"{model_name}_eager_input_{i}.bin")
        aoti_in = load_bin(work_dir / f"{model_name}_aoti_input_{i}.bin")
        ok = compare_tensors(f"input[{i}]", eager_in, aoti_in)
        if not ok:
            inplace_ok = False

    # ---- Result ----
    all_ok = output_ok and inplace_ok
    print(f"\n  [{model_name}] Verification: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    models = ["simple", "chain", "param"]
    results = {}
    for name in models:
        results[name] = verify_model(name)

    print(f"\n{'=' * 60}")
    print(" Summary")
    print(f"{'=' * 60}")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
