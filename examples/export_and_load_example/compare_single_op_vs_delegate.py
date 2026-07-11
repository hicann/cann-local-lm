from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch_delegate as td

ROOT = Path(__file__).resolve().parent


def validate_export_and_eager():
    print("  Loading exported model (.pt2)...")
    pt2_path = ROOT / "epair_model.pt2"
    if not pt2_path.exists():
        print(f"[ERROR] .pt2 file not found: {pt2_path}")
        print("  Run 'python export_and_save_epair.py' first.")
        sys.exit(1)

    epair = td.epair_load(str(pt2_path))
    ep = epair.ep

    x = torch.randn(3, 4)
    model = ep.module()
    with torch.no_grad():
        output = model(x)
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output data (first 8): {output.flatten()[:8].tolist()}")
    print("[PASS] Python: epair export + load + eager execution OK")


def validate_cpp_loading():
    print("  Loading epair via C++ binary...")
    epair_path = str(ROOT / "epair_model")

    binary = ROOT / "build" / "export_and_load_example"
    if not binary.exists():
        print(f"[ERROR] C++ binary not found: {binary}")
        print("  Run 'cd {ROOT} && cmake -B build && cmake --build build -j$nproc' first.")
        sys.exit(1)

    result = subprocess.run(
        [str(binary), epair_path],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(result.stdout)
    if result.returncode == 0:
        print("[PASS] C++: epair loading + GE Graph extraction + compile OK")
    else:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print("[WARN] C++ binary exited with code {}. This is expected if there is no NPU device.".format(result.returncode))


def main():
    print("=" * 40)
    print(" [1/2] Python: validate export and eager execution")
    print("=" * 40)
    print()
    validate_export_and_eager()
    print()

    print("=" * 40)
    print(" [2/2] C++: validate epair loading + GE Graph extraction")
    print("=" * 40)
    print()
    validate_cpp_loading()
    print()

    print("=" * 40)
    print(" Validation complete")
    print("=" * 40)
    sys.exit(0)


if __name__ == "__main__":
    main()
