#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Accuracy Comparison: torch.compile(backend='npu') vs C++ epair runner

This script compares numerical accuracy between two execution paths:
1. Reference: torch.compile(model, backend='npu') - GE backend (Python)
2. Epair: C++ td::DelegateSession + ge::GeSession::RunGraph (C++)

Data transfer: Binary files (.bin) with embedded dtype and shape header.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import struct
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torchair
    import torchair.ops._npu_fused_infer_attention_score  # noqa: F401
except ImportError:
    pass

_ROOT_DIR = Path(__file__).resolve().parent

if __package__ in (None, ""):
    sys.path.insert(0, str(_ROOT_DIR))
else:
    pass


# GE DataType mapping (corrected based on actual GE values)
DTYPE_MAP = {
    torch.float32: 0,  # DT_FLOAT
    torch.uint8: 1,  # DT_UINT8
    torch.int32: 3,  # DT_INT32 (GE uses 3, not 4)
    torch.int16: 10,  # DT_INT16 (GE uses 10)
    torch.bfloat16: 27,  # DT_BFLOAT16
}

DTYPE_STR_MAP = {
    0: "torch.float32",
    1: "torch.uint8",
    3: "torch.int32",
    10: "torch.int16",
    27: "torch.bfloat16",
}


def _load_ge_custom_ops() -> None:
    """Load GE custom op library with auto-rebuild on dlopen failure.

    Reference path uses manually prepared custom_op.cpp (not auto-generated).
    """
    cust_opapi_path = _ROOT_DIR / "custom_op/build/libcust_opapi.so"
    custom_op_cpp = _ROOT_DIR / "custom_op/custom_op.cpp"
    cmake_lists = _ROOT_DIR / "custom_op/CMakeLists.txt"

    # 检查源文件存在（半自动：不自动生成）
    if not custom_op_cpp.exists():
        print(f"[Error] custom_op.cpp not found: {custom_op_cpp}")
        print("[Error] Reference path requires manually prepared custom_op.cpp")
        print("[Error] Please copy custom_op.cpp and CMakeLists.txt to output_dir/")
        raise RuntimeError("custom_op.cpp missing - manual preparation required")

    if not cmake_lists.exists():
        print(f"[Error] CMakeLists.txt not found: {cmake_lists}")
        print("[Error] Reference path requires manually prepared CMakeLists.txt")
        print("[Error] Please copy custom_op.cpp and CMakeLists.txt to output_dir/")
        raise RuntimeError("CMakeLists.txt missing - manual preparation required")

    # 如果 .so 不存在，编译
    if not cust_opapi_path.exists():
        print(f"[Warning] libcust_opapi.so not found: {cust_opapi_path}")
        print("[Info] Building from existing custom_op.cpp...")
        _build_cust_opapi()
        return

    # 测试 dlopen
    try:
        ctypes.CDLL(str(cust_opapi_path), ctypes.RTLD_GLOBAL)
        print(f"[GE Op] Loaded: {cust_opapi_path}")
    except Exception as e:
        print(f"[Warning] dlopen failed: {e}")
        print("[Info] Rebuilding libcust_opapi.so...")
        _build_cust_opapi()


def _build_cust_opapi() -> None:
    """Build libcust_opapi.so from existing custom_op.cpp.

    Assumes custom_op.cpp and CMakeLists.txt already exist.
    Uses environment variables from run.sh: TORCH_PKG_DIR, TORCH_NPU_DIR, ASCEND_HOME_PATH.
    """
    output_dir = _ROOT_DIR / "custom_op"

    # cmake configure
    print("[Build] Running cmake configure...")
    result = subprocess.run(
        ["cmake", "-B", "build"],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[Error] cmake configure failed:")
        print(result.stderr)
        print("[Hint] Check environment variables:")
        print(f"  TORCH_PKG_DIR={os.environ.get('TORCH_PKG_DIR', 'NOT SET')}")
        print(f"  TORCH_NPU_DIR={os.environ.get('TORCH_NPU_DIR', 'NOT SET')}")
        print(f"  ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', 'NOT SET')}")
        raise RuntimeError("cmake configure failed")

    # cmake build
    print("[Build] Running cmake build...")
    result = subprocess.run(
        ["cmake", "--build", "build", "-j4"],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[Error] cmake build failed:")
        print(result.stderr)
        raise RuntimeError("cmake build failed")

    # 验证并加载
    cust_opapi_path = output_dir / "build/libcust_opapi.so"
    if not cust_opapi_path.exists():
        print("[Error] libcust_opapi.so not created after build")
        raise RuntimeError("build failed - .so not created")

    ctypes.CDLL(str(cust_opapi_path), ctypes.RTLD_GLOBAL)
    print(f"[GE Op] Loaded after rebuild: {cust_opapi_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accuracy comparison script")
    parser.add_argument(
        "--ep-path",
        default=str(_ROOT_DIR / "generate/sample.rewritten.pt2"),
    )
    parser.add_argument(
        "--epair-path",
        default=str(_ROOT_DIR / "generate/rewritten.epair"),
    )
    parser.add_argument(
        "--output-json",
        default=str(_ROOT_DIR / "generate/accuracy_result.json"),
    )
    parser.add_argument(
        "--epair-runner",
        default=str(_ROOT_DIR / "epair_runner/build/epair_runner"),
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_sample_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    BATCH_SIZE = 128
    SEQ_LEN = 2
    HIDDEN_DIM = 128

    hidden_states = torch.randn(
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=torch.bfloat16
    )
    sin_cache = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    cos_cache = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    causal_mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), 1, dtype=torch.uint8, device=device),
        diagonal=1,
    )
    attn_mask = (
        causal_mask.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)
    )
    prefix_scores = torch.zeros(BATCH_SIZE, device=device)

    return hidden_states, sin_cache, cos_cache, attn_mask, prefix_scores


def _save_tensor_to_bin(tensor: torch.Tensor, path: Path) -> None:
    dtype_code = DTYPE_MAP.get(tensor.dtype, 0)
    shape = list(tensor.shape)
    ndim = len(shape)

    header = struct.pack("<II", dtype_code, ndim)
    header += struct.pack(f"<{ndim}q", *shape)

    tensor_cpu = tensor.cpu()

    if tensor.dtype == torch.bfloat16:
        data_bytes = tensor_cpu.view(torch.uint16).numpy().tobytes()
    elif tensor.dtype == torch.float16:
        data_bytes = tensor_cpu.view(torch.uint16).numpy().tobytes()
    else:
        data_bytes = tensor_cpu.numpy().tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data_bytes)

    print(
        f"[Save] {path.name}: dtype={tensor.dtype}, shape={shape}, size={len(data_bytes)} bytes"
    )


def _load_tensor_from_bin(path: Path, device: torch.device) -> torch.Tensor:
    with open(path, "rb") as f:
        dtype_code, ndim = struct.unpack("<II", f.read(8))
        shape = struct.unpack(f"<{ndim}q", f.read(ndim * 8))
        data_bytes = f.read()

    dtype_str = DTYPE_STR_MAP.get(dtype_code, "torch.float32")
    dtype = getattr(torch, dtype_str.split(".")[-1])

    # Handle bfloat16 specially since numpy doesn't support it
    if dtype == torch.bfloat16:
        # Read as uint16 then reinterpret as bfloat16
        import numpy as np

        arr = np.frombuffer(data_bytes, dtype=np.uint16)
        arr = arr.reshape(shape)
        tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16).to(device)
    else:
        import numpy as np

        np_dtype = dtype_str.replace("torch.", "")
        arr = np.frombuffer(data_bytes, dtype=np.dtype(np_dtype))
        arr = arr.reshape(shape)
        tensor = torch.from_numpy(arr.copy()).to(device)

    print(f"[Load] {path.name}: dtype={dtype}, shape={list(shape)}")
    return tensor


def _save_inputs(inputs: tuple, input_dir: Path) -> list[Path]:
    input_files = []
    names = ["hidden_states", "sin_cache", "cos_cache", "attn_mask", "prefix_scores"]

    input_dir.mkdir(parents=True, exist_ok=True)

    for i, (tensor, name) in enumerate(zip(inputs, names)):
        path = input_dir / f"input_{i}_{name}.bin"
        _save_tensor_to_bin(tensor, path)
        input_files.append(path)

    return input_files


def _save_outputs(outputs: tuple, output_dir: Path, prefix: str) -> list[Path]:
    output_files = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, tensor in enumerate(outputs):
        path = output_dir / f"{prefix}_output_{i}.bin"
        _save_tensor_to_bin(tensor, path)
        output_files.append(path)

    return output_files


def _run_reference(ep_path: Path, inputs: tuple, device: torch.device) -> tuple:
    _load_ge_custom_ops()

    from ge_converters import register_all_converters

    register_all_converters()

    print("\n[Reference] Loading ExportedProgram...")
    ep = torch.export.load(ep_path)

    print("[Reference] Creating module...")
    model = ep.module().to(device)

    print("[Reference] Converting weights to bfloat16...")
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    for name, buf in model.named_buffers():
        if buf.dtype == torch.float32:
            buf.data = buf.data.to(torch.bfloat16)

    print("[Reference] Compiling with torchair backend (fullgraph=True)...")
    import torchair

    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    compiled_model = torch.compile(model, backend=npu_backend, fullgraph=True)

    print("[Reference] Running inference...")
    inputs_clone = tuple(t.clone() for t in inputs)
    with torch.inference_mode():
        outputs = compiled_model(*inputs_clone)

    return outputs


def _run_epair_cpp(
    epair_path: Path,
    input_files: list[Path],
    output_dir: Path,
    epair_runner: Path,
    device_id: str,
) -> list[Path]:
    print("\n[Epair C++] Running epair_runner...")

    cmd = [
        str(epair_runner),
        str(epair_path),
        str(output_dir),
        device_id,
    ]
    cmd.extend(str(p) for p in input_files)

    print(f"[Epair C++] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(f"[Stderr] {result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"epair_runner failed with code {result.returncode}")

    output_files = sorted(output_dir.glob("output_*.bin"))

    # Skip output_0.bin (RopeInPlace placeholder)
    # Epair outputs: [output_0=RopeInPlace, output_1=token_index, output_2=token_ids, output_3=log_probs]
    # Reference outputs: [ref_output_0=token_index, ref_output_1=token_ids, ref_output_2=log_probs]
    beam_search_outputs = [
        f for f in output_files if not f.name.endswith("output_0.bin")
    ]

    print(
        f"[Epair C++] Found {len(output_files)} outputs, using {len(beam_search_outputs)} for comparison"
    )
    print("[Epair C++] Skipping output_0.bin (RopeInPlace placeholder)")

    return beam_search_outputs


def _compare_outputs(
    ref_outputs: tuple,
    epair_output_files: list[Path],
    device: torch.device,
) -> list[dict]:
    output_names = ["token_index", "token_ids", "log_probs"]
    results = []

    print("\n[Compare] Comparing outputs...")
    print("[Compare] Output mapping:")
    print(
        "  Reference: [ref_output_0=token_index, ref_output_1=token_ids, ref_output_2=log_probs]"
    )
    print("  Epair:     [output_1=token_index, output_2=token_ids, output_3=log_probs]")
    print("  (output_0=RopeInPlace placeholder, skipped)")

    for i, (ref, path) in enumerate(zip(ref_outputs, epair_output_files)):
        if i < len(output_names):
            name = output_names[i]
        else:
            name = f"output_{i}"

        epair = _load_tensor_from_bin(path, device)

        print(f"\n  Output {i}: {name}")
        print(f"    Reference: shape={ref.shape}, dtype={ref.dtype}")
        print(f"    Epair:     shape={epair.shape}, dtype={epair.dtype}")

        # Print first few elements for debugging
        num_elements = min(10, ref.numel())
        ref_flat = ref.flatten()[:num_elements]
        epair_flat = epair.flatten()[:num_elements]

        print(f"    First {num_elements} elements:")
        print(f"      Reference: {ref_flat.tolist()}")
        print(f"      Epair:     {epair_flat.tolist()}")

        allclose = torch.allclose(ref, epair, rtol=1e-3, atol=1e-5)
        max_abs_diff = float((ref - epair).abs().max().item())
        mean_abs_diff = float((ref - epair).abs().mean().item())

        ref_abs = ref.abs()
        nonzero_mask = ref_abs > 1e-6
        if nonzero_mask.any():
            rel_diff = ((ref - epair).abs() / ref_abs.clamp(min=1e-6)).mean().item()
            relative_diff_pct = float(rel_diff * 100)
        else:
            relative_diff_pct = 0.0

        result = {
            "output_name": name,
            "output_index": i,
            "shape": list(ref.shape),
            "dtype": str(ref.dtype),
            "allclose": allclose,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "relative_diff_pct": relative_diff_pct,
        }
        results.append(result)

        print(f"    allclose: {allclose}")
        print(f"    max_abs_diff: {max_abs_diff:.6f}")
        print(f"    mean_abs_diff: {mean_abs_diff:.6f}")
        print(f"    relative_diff_pct: {relative_diff_pct:.4f}%")

    return results


def _print_summary(results: list[dict], overall_pass: bool) -> None:
    print("\n" + "=" * 70)
    print("Accuracy Comparison Summary")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["allclose"] else "FAIL"
        print(f"\n[{status}] {r['output_name']}")
        print(f"  Shape: {r['shape']}, Dtype: {r['dtype']}")
        print(f"  Max Abs Diff: {r['max_abs_diff']:.6f}")
        print(f"  Mean Abs Diff: {r['mean_abs_diff']:.6f}")
        print(f"  Relative Diff: {r['relative_diff_pct']:.4f}%")

    print("\n" + "=" * 70)
    overall_status = "PASS" if overall_pass else "FAIL"
    print(f"Overall Result: {overall_status}")
    print("=" * 70)


def _save_results(
    results: list[dict],
    overall_pass: bool,
    config: dict,
    output_path: Path,
) -> None:
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "comparison_results": results,
        "overall_pass": overall_pass,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\n[Save] Results saved to: {output_path}")


def main() -> int:
    args = _parse_args()

    print("=" * 70)
    print("Accuracy Comparison: torch.compile(npu) vs C++ epair runner")
    print("=" * 70)

    ep_path = Path(args.ep_path).resolve()
    epair_path = Path(args.epair_path).resolve()
    output_json = Path(args.output_json).resolve()
    epair_runner = Path(args.epair_runner).resolve()
    device = torch.device(args.device)
    device_id = args.device.split(":")[-1] if ":" in args.device else "0"

    work_dir = _ROOT_DIR / "generate/accuracy_work"
    input_dir = work_dir / "inputs"
    ref_output_dir = work_dir / "ref_outputs"
    epair_output_dir = work_dir / "epair_outputs"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    print(f"\nEP Path: {ep_path}")
    print(f"Epair Path: {epair_path}")
    print(f"Epair Runner: {epair_runner}")
    print(f"Device: {device}")
    print(f"Work Dir: {work_dir}")

    if not ep_path.exists():
        print(f"\n[Error] EP not found: {ep_path}")
        print("Please run: bash run.sh step1 && bash run.sh step2")
        return 1

    if not epair_path.exists():
        print(f"\n[Error] Epair not found: {epair_path}")
        print("Please run: bash run.sh step2")
        return 1

    if not epair_runner.exists():
        print(f"\n[Error] Epair runner not found: {epair_runner}")
        print("Please build it first")
        return 1

    torch.manual_seed(args.seed)
    inputs = _build_sample_inputs(device)

    print(f"\n[Inputs] Generated with seed={args.seed}")
    for i, t in enumerate(inputs):
        print(f"  input[{i}]: shape={t.shape}, dtype={t.dtype}")

    input_files = _save_inputs(inputs, input_dir)

    epair_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ref_outputs = _run_reference(ep_path, inputs, device)
        _save_outputs(ref_outputs, ref_output_dir, "ref")
    except Exception as e:
        print(f"\n[Error] Reference path failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    try:
        epair_output_files = _run_epair_cpp(
            epair_path, input_files, epair_output_dir, epair_runner, device_id
        )
    except Exception as e:
        print(f"\n[Error] Epair C++ path failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    comparison_results = _compare_outputs(ref_outputs, epair_output_files, device)
    overall_pass = all(r["allclose"] for r in comparison_results)

    _print_summary(comparison_results, overall_pass)

    config = {
        "reference_backend": "torch.compile(backend='npu')",
        "epair_backend": "C++ td::DelegateSession + ge::GeSession::RunGraph",
        "random_seed": args.seed,
        "device": str(device),
        "ep_path": str(ep_path),
        "epair_path": str(epair_path),
        "epair_runner": str(epair_runner),
        "operator_dependencies": {
            "rope_inplace_ge_op": str(_ROOT_DIR / "custom_op/build/libcust_opapi.so"),
            "beam_search": "~/Ascend/opp/vendors/xllm/op_api/lib/libcust_opapi.so",
        },
    }

    _save_results(comparison_results, overall_pass, config, output_json)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
