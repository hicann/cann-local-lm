"""Tier 0 reference: W4A8 s8s4 per-group quant matmul, from the math spec.

独立于 cann_local_lm converter / fused_ops 代码实现 (不复用 materialize.py / quant_matmul.py /
dynamic_quant.py)。每步一行，肉眼可审。spec 选择显式写在这里 + bedrock/hand_fixtures.py，
待 NPU 校准 (Tier 2) 后钉 "规格↔硬件"。

spec (本 reference 编码的定义):
  权重 (int4 对称 per-group, W4):
    s_expanded = scale.repeat_interleave(group_size, dim=-1)   # [out, in]
    q_w = clamp(round(w / s_expanded), INT4_NEG_CLIP, 7)       # signed int4 [-8,7] 二补码全范围
    dequant_w = q_w * s_expanded                                # offset=0
  激活 (int8 对称 per-token, A8):
    act_scale = max(|x|, -1) / 127  (along last dim), clamp(min=1e-8)
    x_q = clamp(round(x / act_scale), -128, 127)
    dequant_x = x_q * act_scale
  matmul (s8s4):
    out = dequant_x @ dequant_w.T + bias                        # x:[...,in] w:[out,in] -> out [...,out]

clamp 范围: INT4_NEG_CLIP = -8 (标准 signed int4 二补码 + dopt qmin_max 实际范围).
  materialize P1 修复前用 [-7,7] (丢 -8, 每层 1万~5万个); 修复后 [-8,7] 对齐 dopt/标准.
  ④NPU 校准最后确认 ge.QuantBatchMatmulV4 内核也是 [-8,7] (标准 signed int4, 大概率是).

权威层级: Tier 0 (本文件, 手算夹具钉) > Tier 1 (dopt fakequant W4A16, 权重网格交叉)
         > Tier 2 (NPU ge.QuantBatchMatmulV4, 部署真值, 校准本 reference)。见 VERIFICATION.md。
"""

import torch

from bedrock.hand_fixtures import FIXTURES, to_tensor

# signed int4 负向 clamp. -8 = 标准 signed int4 二补码全范围 + dopt qmin_max 实际范围;
# materialize P1 修复后已对齐 (-8). ④NPU 校准最后确认内核也是 [-8,7] (标准, 大概率是).
INT4_NEG_CLIP = -8


def ref_quant_weight_int4(w: torch.Tensor, scale: torch.Tensor, group_size: int):
    """w: [out, in] fp; scale: [out, in//group] fp -> (q_w int8 [out,in], scale fp [out,in//group])."""
    w = w.float()
    s_exp = scale.float().repeat_interleave(group_size, dim=-1)  # [out, in]
    q = torch.round(w / s_exp).clamp(INT4_NEG_CLIP, 7).to(torch.int8)
    return q, scale.float()


def ref_dequant_weight_int4(
    q: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    s_exp = scale.float().repeat_interleave(group_size, dim=-1)
    return q.float() * s_exp


def ref_dynamic_quant(x: torch.Tensor):
    """x: [..., in] fp -> (x_q int8 [...,in], act_scale fp32 [..., 1])."""
    xf = x.float()
    act_scale = (xf.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
    xq = torch.clamp(torch.round(xf / act_scale), -128, 127).to(torch.int8)
    return xq, act_scale.to(torch.float32)


def ref_quant_matmul(
    x_q, w_q, weight_scale, group_size, pertoken_scale=None, bias=None
):
    """x_q: int8 [..., m, k]; w_q: int8 [out, in]=[n, k]; weight_scale: [n, k//group]; -> out [..., m, n]."""
    s_exp = weight_scale.float().repeat_interleave(group_size, dim=-1)  # [n, k]
    w_fp = w_q.float() * s_exp  # [n, k]
    if pertoken_scale is not None:
        x_fp = x_q.float() * pertoken_scale.float()  # dequant activation
    else:
        x_fp = x_q.float()
    out = x_fp @ w_fp.t()  # [..., m, k] @ [k, n] -> [..., m, n]
    if bias is not None:
        out = out + bias.float()
    return out


def ref_w4a8_linear(x, w, scale, group_size, bias=None):
    """完整 W4A8 reference (对应 QuantLinear.forward): x [...,in], w [out,in], scale [out,in//group]."""
    xq, xscale = ref_dynamic_quant(x)
    wq, _ = ref_quant_weight_int4(w, scale, group_size)
    return ref_quant_matmul(xq, wq, scale, group_size, pertoken_scale=xscale, bias=bias)


def self_check() -> int:
    """跑手算夹具, 验 reference 符合 spec (bedrock)。人算 expected 来自 bedrock/。"""
    npass = 0
    for fx in FIXTURES:
        name = fx["name"]
        if fx["kind"] == "weight":
            w, sc, gs = to_tensor(fx["w"]), to_tensor(fx["scale"]), fx["group_size"]
            q, _ = ref_quant_weight_int4(w, sc, gs)
            exp = torch.tensor(fx["expected_qw"], dtype=torch.int8)
            ok = torch.equal(q, exp)
            npass += ok
            print(
                f"  [{'PASS' if ok else 'FAIL'}] {name}: q_w={q.flatten().tolist()} expected={exp.flatten().tolist()}"
            )
        elif fx["kind"] == "act":
            x = to_tensor(fx["x"])
            xq, xs = ref_dynamic_quant(x)
            exp_q = torch.tensor(fx["expected_xq"], dtype=torch.int8)
            ok_q = torch.equal(xq, exp_q)
            ok_s = abs(xs.item() - fx["expected_scale"]) < fx["tol"]
            ok = ok_q and ok_s
            npass += ok
            print(
                f"  [{'PASS' if ok else 'FAIL'}] {name}: x_q={xq.flatten().tolist()} expected={exp_q.flatten().tolist()} | scale={xs.item():.6f} expected={fx['expected_scale']:.6f}"
            )
        elif fx["kind"] == "matmul":
            w, sc, gs = to_tensor(fx["w"]), to_tensor(fx["scale"]), fx["group_size"]
            x = to_tensor(fx["x"])
            out = ref_w4a8_linear(x, w, sc, gs, bias=fx.get("bias"))
            exp = fx["expected_out"]
            ok = abs(out.item() - exp) < fx["tol"]
            npass += ok
            print(
                f"  [{'PASS' if ok else 'FAIL'}] {name}: out={out.item():.6f} expected={exp:.6f} (={fx['derive']})"
            )
    print(f"\nTier 0 bedrock: {npass}/{len(FIXTURES)} fixtures PASS")
    return 0 if npass == len(FIXTURES) else 1


if __name__ == "__main__":
    raise SystemExit(self_check())
