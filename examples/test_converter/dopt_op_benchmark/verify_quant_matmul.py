"""验证 converter W4A8 matmul 算子链 (dynamic_quant + quant_matmul + QuantLinear) 对齐 Tier 0。

被测对象 (under test):
  cann_lm::dynamic_quant (dynamic_quant.py:57 CPU impl) — 激活 int8 量化
  cann_lm::quant_matmul  (quant_matmul.py:81 CPU impl) — W4A8 s8s4 matmul
  QuantLinear.forward    (quant_linear.py:66) — dynamic_quant → quant_matmul 链
标准: Tier 0 ref_w4a8 (ref_dynamic_quant / ref_quant_matmul / ref_w4a8_linear)

信任等级 (见 VERIFICATION.md 信任分级):
  A/B: ⚠️仅一致 (CPU impl 与 Tier 0 共享 dequant 公式, spec 循环) + ✅抓 wiring bug (shape/转置/scale展开/pertoken/bias/dtype)
  C: ✅独立交叉 (QuantLinear packed eager vs ref) — surface 真实 wiring 问题

注: clamp bound (-7 vs -8) 在此不裁 (B 用 ref 的 INT4_NEG_CLIP=-7); -8 问题已由 verify_materialize Part5 处理, NPU 内核裁定待 ④。
"""
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import cann_lm.models.fused_ops.ops as _fo  # noqa: register cann_lm:: ops
from cann_lm.converter.export.quant_swap.materialize import materialize_int4_per_group_symmetric
from cann_lm.converter.export.quant_swap.quant_linear import QuantLinear
from ref_w4a8 import ref_dynamic_quant, ref_quant_weight_int4, ref_quant_matmul, ref_w4a8_linear


def _t(x):
    return torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """余弦相似度 (flatten 后), ④ NPU vs reference 的主评判标准 (容忍 W4A16/W4A8 激活量化差 + int 累积差)."""
    af = a.flatten().float()
    bf = b.flatten().float()
    return torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()


def partA_dynamic_quant():
    print("=== A: cann_lm::dynamic_quant vs ref_dynamic_quant [⚠️仅一致+✅wiring] ===")
    cases = [
        ("synthetic_A1", _t([[1.0, 2.0, -1.0, 0.5]])),
    ]
    qdir = os.environ.get("DOPT_QUANT_DIR")
    gdir = os.environ.get("OP_BENCH_GOLDEN_DIR", os.path.join(SCRIPT_DIR, "golden"))
    if qdir and os.path.exists(os.path.join(gdir, "op_io", "model_layers_0_self_attn_q_proj.pt")):
        d = torch.load(os.path.join(gdir, "op_io", "model_layers_0_self_attn_q_proj.pt"), weights_only=True)
        cases.append(("real_qproj_input", d["inputs"][0].float()))
    npass = 0
    for name, x in cases:
        xq_op, sc_op = torch.ops.cann_lm.dynamic_quant(x)
        xq_ref, sc_ref = ref_dynamic_quant(x)
        ok = torch.equal(xq_op, xq_ref) and torch.allclose(sc_op, sc_ref)
        npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: x_q bit-exact={torch.equal(xq_op, xq_ref)} scale_close={torch.allclose(sc_op, sc_ref)} shape={tuple(xq_op.shape)}")
    return npass


def partB_quant_matmul():
    print("\n=== B: cann_lm::quant_matmul (unpacked int4) vs ref_quant_matmul [⚠️仅一致+✅wiring] ===")
    # 合成: M1 夹具
    w, scale, gs = _t([[2.0, -3.0, 1.5, 0.7]]), _t([[1.0]]), 4
    x = _t([[1.0, 2.0, -1.0, 0.5]])
    xq, xsc = ref_dynamic_quant(x)
    wq, _ = ref_quant_weight_int4(w, scale, gs)          # [N,K] int8
    out_ref = ref_quant_matmul(xq, wq, scale, gs, pertoken_scale=xsc, bias=None)
    # cann_lm CPU impl 约定: x2 = unpacked int4 [K,N] (transpose_x2=False), scale [K//g, N]
    out_op = torch.ops.cann_lm.quant_matmul(
        xq, wq.t().contiguous(), scale.t().contiguous(),
        pertoken_scale=xsc, bias=None, group_sizes=[1, 1, gs],
        output_dtype=6, transpose_x1=False, transpose_x2=False)  # 6=float32 精确比
    ok = torch.allclose(out_op, out_ref, atol=1e-5)
    print(f"  [{'PASS' if ok else 'FAIL'}] synthetic_M1: out_op={out_op.item():.6f} out_ref={out_ref.item():.6f} (期望 -698/127≈-5.496063)")
    # 真实: q_proj L0 + 固定激活
    npass = int(ok)
    qdir = os.environ.get("DOPT_QUANT_DIR")
    gdir = os.environ.get("OP_BENCH_GOLDEN_DIR", os.path.join(SCRIPT_DIR, "golden"))
    if qdir and os.path.exists(os.path.join(gdir, "op_io", "model_layers_0_self_attn_q_proj.pt")):
        sd = torch.load(os.path.join(qdir, "trained_quant_weight.pth"), map_location="cpu", mmap=True, weights_only=True)
        w = sd["model.layers.0.self_attn.q_proj.weight"].float()
        s_flat = sd["model.layers.0.self_attn.q_proj.quant_op.weight_quantizer.s"].float()
        out_f, in_f = w.shape; groups = s_flat.numel() // out_f; gs = in_f // groups
        scale2d = s_flat.reshape(out_f, groups)
        d = torch.load(os.path.join(gdir, "op_io", "model_layers_0_self_attn_q_proj.pt"), weights_only=True)
        x = d["inputs"][0].float()  # [1,3,2560]
        xq, xsc = ref_dynamic_quant(x)
        wq, _ = ref_quant_weight_int4(w, scale2d, gs)
        out_ref = ref_quant_matmul(xq, wq, scale2d, gs, pertoken_scale=xsc, bias=None)
        out_op = torch.ops.cann_lm.quant_matmul(
            xq, wq.t().contiguous(), scale2d.t().contiguous(),
            pertoken_scale=xsc, bias=None, group_sizes=[1, 1, gs],
            output_dtype=6, transpose_x1=False, transpose_x2=False)
        d_rel = (out_op - out_ref).abs().max().item() / out_ref.abs().max().item()
        cs = cos_sim(out_op, out_ref)
        ok2 = d_rel < 1e-5 and cs > 0.999
        npass += int(ok2)
        print(f"  [{'PASS' if ok2 else 'FAIL'}] real_qproj_L0: shape={tuple(out_op.shape)} maxRelΔ={d_rel:.2e} cos={cs:.6f}")
    return npass


def partC_quantlinear_packed():
    print("\n=== C: QuantLinear.forward (packed, swap 构造) vs ref [✅独立交叉: surface wiring] ===")
    w, scale, gs = _t([[2.0, -3.0, 1.5, 0.7]]), _t([[1.0]]), 4
    x = _t([[1.0, 2.0, -1.0, 0.5]])
    wq_pack, wsc = materialize_int4_per_group_symmetric(w, scale, gs, signed=True, nbits=4, packed=True)
    ql = QuantLinear(wq_pack.transpose(-1, -2).contiguous(), wsc.transpose(-1, -2).contiguous(), gs, bias=None, act_quant=True)
    ref = ref_w4a8_linear(x, w, scale, gs, bias=None)
    try:
        out = ql(x)
        ok = torch.allclose(out.float(), ref, atol=1e-4)
        print(f"  [{'PASS' if ok else 'FAIL'}] QuantLinear(packed) eager: out={out.item():.6f} ref={ref.item():.6f}")
        return int(ok)
    except Exception as e:
        print(f"  [FINDING] QuantLinear(packed) eager 崩: {type(e).__name__}: {str(e)[:120]}")
        print(f"           原因: QuantLinear 存 packed weight_q [K//2,N] (swap/export 用), 但 cann_lm::quant_matmul")
        print(f"           CPU impl 期望 unpacked [K,N] (w=x2.float()*sc_exp). → QuantLinear eager CPU 路径断裂,")
        print(f"           仅 export 路径可用 (GE _quant_matmul_converter 处理 packed). eager 回归需 unpacked 或修 CPU impl.")
        return 0


def main():
    na = partA_dynamic_quant()
    nb = partB_quant_matmul()
    partC_quantlinear_packed()
    print(f"\nverify_quant_matmul: A={na} B done, C 见 FINDING (见上方各 PASS/FAIL/FINDING)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
