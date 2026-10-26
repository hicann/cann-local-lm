"""验证 op.pth 能否驱动 dopt QLinear 跑出 fakequant 结果。

仅依赖 op.pth (10.9MB 单算子参数) + golden, 不需 dopt_config / optimize_model / 完整模型。
策略硬编码 Quant_INT8-4_ECO (bit=4, group_size=128, weight_algo=omni_group_min_max)。
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

OP_PTH = BENCH_DIR / "op.pth"
OP_NAME = "model.layers.0.self_attn.q_proj"
GOLDEN_DIR = BENCH_DIR / "golden"

from dopt_lm.modules.qlinear import QLinear
import dopt_lm.quantizer.qmin_max  # noqa: F401  # 注册 omni_group_min_max 等量化器
import dopt_lm.quantizer.qeval  # noqa: F401  # 注册 dyn_s8 输入量化器

# Quant_INT8-4_ECO 策略硬编码 (来自 get_optimize_cfg("Quant_INT8-4_ECO"))
QUANT_CFG = {
    "weight": {
        "blocksize": 128,
        "group_size": 128,
        "bit": 4,
        "per_channel": True,
        "weight_algo": "omni_group_min_max",
    },
    "input": {
        "bit": 8,
        "per_channel": False,
        "input_algo": "dyn_s8",
        "unsigned_quant": False,
    },
}


def build_qlinear(op_pth, op_name):
    """直接构造 QLinear (硬编码策略), load op.pth (strip 前缀)。"""
    sd = torch.load(str(op_pth), map_location="cpu", weights_only=True)
    w = sd[f"{op_name}.weight"]
    out_f, in_f = w.shape

    linear = nn.Linear(in_f, out_f, bias=False)
    ql = QLinear(linear, QUANT_CFG, layer_name=op_name)
    ql.set_quant_state(weight=True, input=False)
    ql.quant_op.set_calibrate_state(True)

    # strip 前缀 (op.pth key 带完整 path, QLinear state dict 无前缀)
    prefix = op_name + "."
    stripped = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    ql.load_state_dict(stripped, strict=False)
    ql.quant_op.set_calibrate_state(False)
    ql.eval()
    return ql, (in_f, out_f)


def main():
    ql, (in_f, out_f) = build_qlinear(OP_PTH, OP_NAME)
    print(
        f"[op.pth -> QLinear] q_proj: in={in_f} out={out_f} (硬编码 Quant_INT8-4_ECO)"
    )

    safe = OP_NAME.replace(".", "_")
    gd = torch.load(GOLDEN_DIR / "op_io" / f"{safe}.pt", weights_only=True)
    x = gd["inputs"][0].to(torch.float16)
    golden_out = gd["outputs"][0].float()

    with torch.no_grad():
        out = ql(x).float()

    max_diff = (out - golden_out).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten().unsqueeze(0), golden_out.flatten().unsqueeze(0)
    ).item()
    ok = max_diff < 1e-3
    print(f"  forward output: shape={tuple(out.shape)}")
    print(
        f"  vs golden: maxΔ={max_diff:.6e}  cos={cos:.8f}  -> {'PASS (op.pth 可独立驱动 dopt QLinear)' if ok else 'FAIL'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
