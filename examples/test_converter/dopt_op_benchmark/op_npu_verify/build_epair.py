"""算子级 NPU 验证 — Step 1: 构建 1-op QuantLinear 模型 + export epair (converter env)。

从 dopt trained_quant_weight.pth 取一个算子的 weight+scale, materialize (按 target_soc:
kirinx90→int4pack int32 / Ascend910→int8 nibble), 包成 1-QuantLinear 模型, torch.export +
lite_export_context → epair (含单个 cann_lm::quant_matmul → GE QuantBatchMatmulV4 节点)。

env: converter venv (source scripts/cann_lm_compile/set_env_export.sh)。复用 swap 的 materialize + export_epair 的导出模式。
"""

import argparse
import os
from pathlib import Path

import torch
import torch_delegate
from torch_delegate import LiteConfig, lite_export_context

from cann_lm.converter.export.epair_utils import consolidated_weights_save
from cann_lm.converter.export.ge_converters import register_ge_converters
from cann_lm.converter.export.quant_swap.materialize import (
    materialize_int4_per_group_symmetric,
)
from cann_lm.converter.export.quant_swap.quant_linear import QuantLinear


class OpModel(torch.nn.Module):
    """单算子模型: 一个 QuantLinear (forward = dynamic_quant + cann_lm::quant_matmul)。"""

    def __init__(self, ql: QuantLinear):
        super().__init__()
        self.op = ql

    def forward(self, x):
        return self.op(x)


def materialize_op(op_name, target_soc, *, op_pth=None, quant_dir=None):
    """复用 swap 逻辑: dopt weight+scale → materialize (按 SoC) → (weight_q, weight_scale, group_size)。
    数据源: op_pth (单算子 .pth) 优先, 否则从 quant_dir/trained_quant_weight.pth 按 op_name 抽。"""
    if op_pth:
        sd = torch.load(op_pth, map_location="cpu", weights_only=True)
    elif quant_dir:
        sd = torch.load(
            os.path.join(quant_dir, "trained_quant_weight.pth"),
            map_location="cpu",
            weights_only=True,
        )
    else:
        raise ValueError("need --op-pth or --quant-dir")
    w = sd[f"{op_name}.weight"].float()
    s_flat = sd[f"{op_name}.quant_op.weight_quantizer.s"].float()
    out_f, in_f = w.shape
    groups = s_flat.numel() // out_f
    gs = in_f // groups
    scale2d = s_flat.reshape(out_f, groups)
    is_kirinx90 = "kirin" in (target_soc or "").lower()
    packed = "int4pack" if is_kirinx90 else True
    weight_q, weight_scale = materialize_int4_per_group_symmetric(
        w, scale2d, gs, signed=True, nbits=4, packed=packed
    )
    weight_scale = weight_scale.transpose(-1, -2).contiguous()
    if not is_kirinx90:
        weight_q = weight_q.transpose(-1, -2).contiguous()
    return weight_q, weight_scale, gs, (out_f, in_f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--op-name", required=True)
    ap.add_argument(
        "--op-pth",
        default=None,
        help="单算子 .pth (dopt fakequant 参数, 未经 materialize); 优先于 --quant-dir",
    )
    ap.add_argument("--quant-dir", default=os.environ.get("DOPT_QUANT_DIR"))
    ap.add_argument(
        "--target-soc",
        default="Ascend910B",
        help="Ascend910B (int8 nibble) / KirinX90 (int4pack int32)",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=3, help="sample input seq len")
    args = ap.parse_args()
    if not args.op_pth and not args.quant_dir:
        ap.error("--op-pth or --quant-dir required")

    weight_q, weight_scale, gs, (out_f, in_f) = materialize_op(
        args.op_name, args.target_soc, op_pth=args.op_pth, quant_dir=args.quant_dir
    )
    ql = QuantLinear(weight_q, weight_scale, gs, bias=None, act_quant=True)
    model = OpModel(ql).eval()
    print(
        f"[build_epair] {args.op_name}: weight_q {tuple(weight_q.shape)} {weight_q.dtype}, "
        f"scale {tuple(weight_scale.shape)}, target_soc={args.target_soc}",
        flush=True,
    )

    sample = torch.randn(1, args.seq_len, in_f, dtype=torch.float16)
    lite_config = LiteConfig(target_soc=args.target_soc, allow_hf32=False)
    with lite_export_context(lite_config):
        register_ge_converters()  # 必须在 lite_export_context 内 (否则 torchair 提前加载冲突)
        exported = torch.export.export(model, (sample,), strict=False)
        epair = torch_delegate.export_to_epair(exported)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    consolidated_weights_save(epair, out, save_weights=True)
    pt2 = out / "op_model.pt2"
    torch_delegate.epair_save(epair, str(pt2))
    print(f"[build_epair] epair saved: {out} + {pt2}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
