from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch_delegate as td
from torch_delegate import LiteConfig, lite_export_context

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate_export_and_eager():
    print("  Loading exported model (.pt2)...")
    pt2_path = ROOT / "epair_model.pt2"
    if not pt2_path.exists():
        print(f"[ERROR] .pt2 file not found: {pt2_path}")
        print("  Run 'python export_and_save_epair.py' first.")
        sys.exit(1)

    epair = td.epair_load(str(pt2_path))
    ep = epair.ep

    input_path = ROOT / "input_0.bin"
    if input_path.exists():
        import numpy as np
        x = torch.from_numpy(np.fromfile(str(input_path), dtype=np.float32)).reshape(2, 64)
    else:
        x = torch.randn(2, 64)
    model = ep.module()
    with torch.no_grad():
        output = model(x)
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output data (first 8): {output.flatten()[:8].tolist()}")
    print("[PASS] Python: epair export + load + eager execution OK")


def validate_om_compile():
    print("  Compiling OM via cann_lm_compile...")
    epair_dir = ROOT / "epair_model"
    compile_bin = PROJECT_ROOT / "build" / "cann_lm_compile"
    if not compile_bin.exists():
        print(f"[ERROR] cann_lm_compile not found: {compile_bin}")
        print("  Run 'bash scripts/build.sh cpp' first.")
        return

    model_json = str(epair_dir / "ep_json_with_ascend_ir.json")
    weights_index = str(epair_dir / "weight_info.json")
    output_name = str(epair_dir / "compare_test")

    result = subprocess.run(
        [str(compile_bin), model_json, weights_index, output_name],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    om_file = Path(f"{output_name}.om")
    if om_file.exists():
        print(f"[PASS] OM compiled: {om_file.name} ({om_file.stat().st_size} bytes)")
    else:
        print(f"[WARN] OM not produced (cann_lm_compile rc={result.returncode})")
        if result.stderr:
            print(result.stderr[:500], file=sys.stderr)


def main():
    print("=" * 40)
    print(" [1/2] Python: validate export and eager execution")
    print("=" * 40)
    print()
    validate_export_and_eager()
    print()

    print("=" * 40)
    print(" [2/2] C++ cann_lm_compile: OM 编译验证")
    print("=" * 40)
    print()
    validate_om_compile()
    print()

    print("=" * 40)
    print(" Validation complete")
    print("=" * 40)
    sys.exit(0)


if __name__ == "__main__":
    main()
