"""一键式 GPTQ 量化: converter spawn dopt subprocess 跑 GPTQ + swap.

运行时检测系统 NPU/GPU:
- 有 NPU/GPU → 方案 C (subprocess 调 dopt CLI, DEVICE 自动 npu/cuda)
- 无 (纯 CPU) → 报错 (提示 --quant-path 消费已量化的 dopt 产物)

见 docs/cann_lm_converter/quant/dopt_design.md.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch


def generate_dopt_config(artifact, quant_bits: str) -> str:
    """遍历 artifact.model 生成 dopt_config (layer_strategy), 返回临时文件路径.

    Linear (非 embed/lm_head) → Quant_act_weight_eco (w4a8)
    Embedding → Quant_Embed_MinMax
    lm_head (Linear) → Quant_lm_head
    """
    linear_strategy = "Quant_act_weight_eco"  # w4a8
    layer_strategy = {}
    for name, m in artifact.model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if "lm_head" in name:
                strategy = "Quant_lm_head"
            elif "embed_tokens" in name:
                continue
            else:
                strategy = linear_strategy
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": strategy}
        elif isinstance(m, torch.nn.Embedding):
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": "Quant_Embed_MinMax"}
    fd, cfg_path = tempfile.mkstemp(prefix="dopt_config_", suffix=".json")
    with open(cfg_path, "w") as f:
        json.dump({"layer_strategy": layer_strategy}, f, indent=2)
    print(f"      Generated dopt_config: {cfg_path} ({len(layer_strategy)} layers)")
    return cfg_path


def _swap_from_dopt_output(artifact, dopt_out: str) -> dict | None:
    """从 dopt 产物 swap QuantLinear + embed 自动检测."""
    from cann_lm.converter.quant import (
        swap_to_quant_linear_from_dopt, swap_embeddings_from_dopt, load_dopt_state_dict,
    )
    info = swap_to_quant_linear_from_dopt(artifact.model, dopt_out)
    artifact.quant_info = info
    print(f"      Quantized {info['n_quantized']} linears (group_size={info['group_size']})")
    sd = load_dopt_state_dict(dopt_out)
    embed_quant = swap_embeddings_from_dopt(sd, artifact.model)
    if embed_quant:
        print(f"      Embedding int8 (auto-detected): {list(embed_quant)}")
    return embed_quant


def _run_dopt_subprocess(hf_path: str, quant_bits: str, artifact, device: str) -> dict | None:
    """方案 C: spawn dopt subprocess (NPU/GPU GPTQ) → 临时盘 → swap."""
    quant_dir = Path(__file__).resolve().parent
    cfg_path = generate_dopt_config(artifact, quant_bits)
    dopt_python = os.environ.get("DOPT_PYTHON", "/opt/buildtools/Python-3.11.4/bin/python3")
    tmp_out = tempfile.mkdtemp(prefix="dopt_gptq_")
    cmd = [
        dopt_python, "-m", "dopt_lm.opt_main",
        "--model-path", hf_path, "--dopt-config", cfg_path,
        "--optimize-config", str(quant_dir / "config.yaml"),
        "--output-dir", tmp_out, "--hf_type_save",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{quant_dir}:{env.get('PYTHONPATH', '')}"
    env["DEVICE"] = device  # npu / cuda (dopt 用)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "stubs" not in p)
    print(f"      dopt subprocess ({device} GPTQ): out={tmp_out}")
    result = subprocess.run(cmd, cwd=str(quant_dir), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"dopt GPTQ subprocess failed (exit {result.returncode}), 临时产物: {tmp_out}")
    return _swap_from_dopt_output(artifact, tmp_out)


def _detect_device_type() -> str | None:
    """检测系统计算设备类型: 'npu' / 'cuda' / None (纯 CPU). spawn 系统 Python, 去 stubs env."""
    dopt_python = os.environ.get("DOPT_PYTHON", "/opt/buildtools/Python-3.11.4/bin/python3")
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "stubs" not in p)
    try:
        result = subprocess.run(
            [dopt_python, "-c",
             "import torch\n"
             "dev='none'\n"
             "try:\n    import torch_npu\n    if torch.npu.is_available(): dev='npu'\nexcept: pass\n"
             "if dev=='none' and torch.cuda.is_available(): dev='cuda'\n"
             "print(dev)"],
            capture_output=True, text=True, timeout=15, env=env,
        )
        out = result.stdout.strip()
        if out == "npu": return "npu"
        if out == "cuda": return "cuda"
        return None
    except Exception:
        return None


def run_auto_quant(hf_path: str, quant_bits: str, artifact) -> dict | None:
    """一键式 GPTQ: 检测 NPU/GPU → 方案 C (subprocess) / 纯 CPU 报错."""
    dev = _detect_device_type()
    if dev:
        print(f"      [方案 C] subprocess 调 dopt (系统 {dev})")
        return _run_dopt_subprocess(hf_path, quant_bits, artifact, dev)
    raise RuntimeError(
        "--quant-method gptq 需 NPU 或 GPU (GPTQ 需计算设备), 当前环境无 NPU/GPU. "
        "纯 CPU 请用 --quant-path <dopt产物> 消费已量化的 dopt 产物."
    )
