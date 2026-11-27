"""One-shot GPTQ: converter spawns a dopt subprocess to run GPTQ, then swaps in the main process.

See docs/cann_lm_converter/quant/quant_design.md (one-shot quant plan selection).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _dopt_python() -> str:
    """③ NPU 阶段装 torch_npu 的系统 python (与 converter P1 venv 隔离).

    正常由 scripts/cann_lm_compile/_env_common.sh 的 _resolve_dopt_python 自动探测并 export;
    兜底用 PATH 里的 python3 (此时多半未 source env 脚本, subprocess 会失败并报错).
    """
    return os.environ.get("DOPT_PYTHON") or shutil.which("python3") or "python3"


def _run_dopt_subprocess(hf_path: str, device: str) -> str:
    quant_dir = Path(__file__).resolve().parent
    wrapper_path = quant_dir / "dopt_wrapper.py"
    dopt_python = _dopt_python()
    tmp_out = tempfile.mkdtemp(prefix="dopt_gptq_", dir=os.environ.get("DOPT_TMP_DIR"))
    cmd = [
        dopt_python,
        str(wrapper_path),
        "--model-path",
        hf_path,
        "--optimize-config",
        str(quant_dir / "config.yaml"),
        "--output-dir",
        tmp_out,
        "--hf-type-save",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{quant_dir}:{env.get('PYTHONPATH', '')}"
    env.setdefault("DEVICE", device)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "stubs" not in p)
    print(f"      dopt subprocess (DEVICE={env.get('DEVICE')} GPTQ): out={tmp_out}")
    result = subprocess.run(cmd, cwd=str(quant_dir), env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"dopt GPTQ subprocess failed (exit {result.returncode}), temp output: {tmp_out}"
        )
    return tmp_out


def _detect_device_type() -> str:
    """Detect the compute device type: 'npu' / 'cuda' / 'cpu'.

    Reads the parent process DEVICE first (incl. 'cpu'); if unset, spawns a
    detection probe. Falls back to 'cpu' when no NPU/GPU is present (dopt_lm
    SUPPORTED_DEVICE_TYPES includes 'cpu'; CPU GPTQ works but is slow).
    """
    env_dev = os.environ.get("DEVICE")
    if env_dev in ("npu", "cuda", "cpu"):
        return env_dev
    dopt_python = _dopt_python()
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "stubs" not in p)
    # silence SLOG to stdout: torch_npu import otherwise spams CANN logs and corrupts the print(dev) stdout parse
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
    try:
        result = subprocess.run(
            [
                dopt_python,
                "-c",
                "import torch\n"
                "dev='none'\n"
                "try:\n    import torch_npu\n    if torch.npu.is_available(): dev='npu'\nexcept: pass\n"
                "if dev=='none' and torch.cuda.is_available(): dev='cuda'\n"
                "print(dev)",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        out = result.stdout.strip()
        if out == "npu":
            return "npu"
        if out == "cuda":
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def run_auto_quant(hf_path: str) -> str:
    """One-shot GPTQ: detect NPU/GPU/CPU -> plan C (subprocess). Return the dopt_out_dir path.

    CPU fallback does not raise (dopt_lm supports CPU, dispatch_dtype=float32);
    but 7B+ Hessian is very slow, only suitable for small models / debugging.
    Production should use NPU/GPU or --quant-path to consume offline output.
    """
    dev = _detect_device_type()
    if dev == "cpu":
        print(
            "      [warn] CPU GPTQ: dopt_lm supports CPU (float32) but 7B+ Hessian is very slow; "
            "prefer NPU/GPU or --quant-path to consume offline output"
        )
    print(f"   spawn dopt subprocess (device={dev})")
    return _run_dopt_subprocess(hf_path, dev)
