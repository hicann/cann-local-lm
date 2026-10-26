"""算子级 NPU 验证 — 对比 harness (CPU 侧)。

固定一个算子的 (输入激活, dopt fakequant W4A16 输出) [来自 golden op_io] + dopt 权重,
算 Tier 0 W4A8 reference 输出, 再加载 NPU 算子输出 (由 export→OM→NPU 步骤产出, npu_output.pt),
三方余弦对比:

  - Tier0(W4A8) vs dopt golden(W4A16): 跨语义 sanity (~0.99, int8 激活量化差)
  - NPU(W4A8) vs Tier0(W4A8): ④ 干净算子正确性 (cos ≥0.999, 同语义, 只含 int 累积/实现差)
  - NPU(W4A8) vs dopt golden(W4A16): ④ 跨语义 (cos ≥0.99, 混激活量化误差)

NPU 输出由 op_npu_verify/run_op_npu.py (export→OM→NPU) 产出 npu_output.pt; 未跑则只报 CPU 两方。

判据见 VERIFICATION.md: 算子级 cos ≥0.999 (同语义) / ≥0.99 (跨语义)。
"""

import argparse
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(BENCH_DIR))

from ref_w4a8 import ref_dynamic_quant, ref_quant_weight_int4, ref_quant_matmul


def load_op_golden(op_name: str, golden_dir: Path):
    """从 golden/op_io/<op>.pt 加载 (inputs, outputs) — dopt fakequant W4A16 I/O。"""
    safe = op_name.replace(".", "_")
    data = torch.load(golden_dir / "op_io" / f"{safe}.pt", weights_only=True)
    return data["inputs"], data["outputs"]  # list[act], list[act]


def load_op_weight(op_name, *, op_pth=None, quant_dir=None):
    """从 op_pth (单算子 .pth) 或 trained_quant_weight.pth 加载算子 weight + scale (flattened)。"""
    if op_pth:
        sd = torch.load(op_pth, map_location="cpu", weights_only=True)
    elif quant_dir:
        sd = torch.load(
            os.path.join(quant_dir, "trained_quant_weight.pth"),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    else:
        raise ValueError("need op_pth or quant_dir")
    w = sd[f"{op_name}.weight"].float()
    s_flat = sd[f"{op_name}.quant_op.weight_quantizer.s"].float()
    out_f, in_f = w.shape
    groups = s_flat.numel() // out_f
    gs = in_f // groups
    scale2d = s_flat.reshape(out_f, groups)
    return w, scale2d, gs


def tier0_w4a8_output(x, w, scale2d, gs):
    """Tier 0 W4A8 reference: dynamic_quant(x) → quant_matmul(int8 act, int4 weight)。"""
    xq, xscale = ref_dynamic_quant(x)
    wq, _ = ref_quant_weight_int4(w, scale2d, gs)
    return ref_quant_matmul(xq, wq, scale2d, gs, pertoken_scale=xscale, bias=None)


def cos(a, b):
    af = a.flatten().float()
    bf = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        af.unsqueeze(0), bf.unsqueeze(0)
    ).item()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--op-name", required=True, help="e.g. model.layers.0.self_attn.q_proj"
    )
    ap.add_argument(
        "--op-pth",
        default=None,
        help="单算子 .pth (dopt fakequant 参数); 优先于 --quant-dir",
    )
    ap.add_argument("--quant-dir", default=os.environ.get("DOPT_QUANT_DIR"))
    ap.add_argument("--golden-dir", default=str(BENCH_DIR / "golden"))
    ap.add_argument(
        "--npu-output",
        default=str(SCRIPT_DIR / "npu_output.pt"),
        help="NPU 算子输出 (run_op_npu.py 产出); 不存在则只报 CPU 两方",
    )
    ap.add_argument("--io-index", type=int, default=0, help="用第几条 golden I/O")
    args = ap.parse_args()
    if not args.op_pth and not args.quant_dir:
        ap.error("--op-pth or --quant-dir required (或设 DOPT_QUANT_DIR)")

    inputs, outputs = load_op_golden(args.op_name, Path(args.golden_dir))
    w, scale2d, gs = load_op_weight(
        args.op_name, op_pth=args.op_pth, quant_dir=args.quant_dir
    )
    x = inputs[args.io_index].float()
    dopt_out = outputs[args.io_index].float()  # dopt fakequant W4A16 输出 (golden)
    ref_out = tier0_w4a8_output(x, w, scale2d, gs)  # Tier 0 W4A8 reference

    print(
        f"=== 算子级验证: {args.op_name}  io[{args.io_index}]  shape={tuple(x.shape)} ==="
    )
    print(
        f"  [CPU sanity] Tier0(W4A8) vs dopt golden(W4A16): cos={cos(ref_out, dopt_out):.6f}  (跨语义, 预期~0.99, int8 激活量化差)"
    )

    if os.path.exists(args.npu_output):
        npu_out = torch.load(args.npu_output, weights_only=True).float()
        c_ref = cos(npu_out, ref_out)
        c_dopt = cos(npu_out, dopt_out)
        ok_ref = c_ref > 0.999
        ok_dopt = c_dopt > 0.99
        print(
            f"  [④ 同语义]  NPU(W4A8) vs Tier0(W4A8):  cos={c_ref:.6f}  [{'PASS' if ok_ref else 'FAIL'}] (≥0.999)"
        )
        print(
            f"  [④ 跨语义]  NPU(W4A8) vs dopt(W4A16):  cos={c_dopt:.6f}  [{'PASS' if ok_dopt else 'FAIL'}] (≥0.99)"
        )
        print(
            f"\n  {'PASS' if (ok_ref and ok_dopt) else 'FAIL'}: NPU 算子级验证 (cos 判据)"
        )
    else:
        print(
            f"\n  [④ 待跑] NPU 输出未找到 ({args.npu_output}); 跑 run_op_npu.py 产 npu_output.pt 后再比"
        )
        print("           CPU 两方已就绪: Tier0 ref + dopt golden (上方的 cos)")


if __name__ == "__main__":
    raise SystemExit(main())
