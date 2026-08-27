"""一键式 GPTQ 量化: converter spawn dopt subprocess 跑 GPTQ,主进程后做 swap.

见 docs/cann_lm_converter/quant/quant_design.md "方案 C 设计".
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch


_LINEAR_STRATEGY = "Quant_act_weight_eco"
_LM_HEAD_STRATEGY = "Quant_lm_head"
_EMBED_STRATEGY = "Quant_Embed_MinMax"  # dopt 53ad5ec 删了 Quant_Gemma4_Embed, gemma4 embed 也用此


def _build_layer_strategy(model, model_type: str) -> dict:
    """遍历 model.named_modules 赋量化策略 (Linear→Quant_act_weight_eco, lm_head→Quant_lm_head,
    Embedding→Quant_Embed_MinMax). cann_lm 与 transformers 模型层名一致,通用.
    注: dopt 53ad5ec 删了 Quant_Gemma4_Embed, gemma4 embed 也用 Quant_Embed_MinMax."""
    layer_strategy = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if "lm_head" in name:
                strategy = _LM_HEAD_STRATEGY
            elif "embed_tokens" in name:
                continue
            else:
                strategy = _LINEAR_STRATEGY
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": strategy}
        elif isinstance(m, torch.nn.Embedding):
            strategy = _EMBED_STRATEGY
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": strategy}
    return layer_strategy


def generate_dopt_config(artifact, quant_bits: str) -> str:
    """场景 1 离线辅助: 遍历 cann_lm artifact.model 生成 dopt_config,返回临时文件路径.
    仅供场景 1 用户手动跑 dopt GPTQ 前用 (examples/run_pd4_cases.sh:gen_dopt_config 依赖).
    """
    layer_strategy = _build_layer_strategy(artifact.model, artifact.model_type)
    fd, cfg_path = tempfile.mkstemp(prefix="dopt_config_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"layer_strategy": layer_strategy}, f, indent=2)
    print(f"      Generated dopt_config: {cfg_path} ({len(layer_strategy)} layers)")
    return cfg_path


def _run_dopt_subprocess(hf_path: str, quant_bits: str, device: str) -> str:
    """方案 C: spawn dopt_wrapper.py subprocess (NPU/GPU/CPU GPTQ) → 返回产物目录路径."""
    quant_dir = Path(__file__).resolve().parent
    wrapper_path = quant_dir / "dopt_wrapper.py"
    dopt_python = os.environ.get("DOPT_PYTHON", "/opt/buildtools/Python-3.11.4/bin/python3")
    tmp_out = tempfile.mkdtemp(prefix="dopt_gptq_")
    cmd = [
        dopt_python, str(wrapper_path),
        "--model-path", hf_path,
        "--optimize-config", str(quant_dir / "config.yaml"),
        "--output-dir", tmp_out,
        "--hf-type-save",
        "--quant-bits", quant_bits,
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
            f"dopt GPTQ subprocess failed (exit {result.returncode}), 临时产物: {tmp_out}"
        )
    return tmp_out


def _detect_device_type() -> str:
    """检测系统计算设备类型: 'npu' / 'cuda' / 'cpu'.

    优先读父进程 DEVICE (含 'cpu'),未设才 spawn 检测;无 NPU/GPU 时回落 'cpu'
    (dopt_lm SUPPORTED_DEVICE_TYPES 含 'cpu', CPU 跑 GPTQ 慢但可跑).
    """
    env_dev = os.environ.get("DEVICE")
    if env_dev in ("npu", "cuda", "cpu"):
        return env_dev
    dopt_python = os.environ.get("DOPT_PYTHON", "/opt/buildtools/Python-3.11.4/bin/python3")
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld.split(":") if "stubs" not in p)
    # 关 SLOG 日志到 stdout, 防止 torch_npu import 时 CANN 日志污染 print(dev) 的 stdout 解析
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
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
        if out == "npu":
            return "npu"
        if out == "cuda":
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def run_auto_quant(hf_path: str, quant_bits: str) -> str:
    """一键式 GPTQ: 检测 NPU/GPU/CPU → 方案 C (subprocess). 返回 dopt_out_dir 路径.

    CPU 回落不报错 (dopt_lm 支持 CPU, dispatch_dtype=float32); 但 7B+ Hessian 极慢,
    仅小模型/调试适用, 生产建议 NPU/GPU 或 --quant-path 消费离线产物.
    """
    dev = _detect_device_type()
    if dev == "cpu":
        print("      [warn] CPU GPTQ: dopt_lm 支持 CPU (float32) 但 7B+ Hessian 极慢, "
              "建议 NPU/GPU 或 --quant-path 消费离线产物")
    print(f"      [方案 C] spawn dopt subprocess (device={dev})")
    return _run_dopt_subprocess(hf_path, quant_bits, dev)
