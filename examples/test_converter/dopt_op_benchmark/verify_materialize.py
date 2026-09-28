"""验证 converter materialize (quant_swap/materialize.py) 对齐 Tier 0 (ref_w4a8.py)。

被测对象 (under test): cann_lm.converter.export.quant_swap.materialize 的
  materialize_int4_per_group_symmetric / pack_int4 / transdata_scale / restore_scale
标准: Tier 0 ref_w4a8 (手算夹具钉死, 独立于 converter 代码)

三层 bit-exact 校验:
  1. materialize 的 q == Tier 0 的 q (两者都 clamp(round(w/s), -7, 7)) —— 合成夹具 + 真实 dopt 权重
  2. pack_int4 round-trip: pack -> unpack == q (bit-exact)
  3. transdata_scale round-trip: transdata -> restore == s 低 13 位尾数清零 (bit-exact, 特征化精度损失)
"""
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from cann_lm.converter.export.quant_swap.materialize import (
    materialize_int4_per_group_symmetric, pack_int4, pack_int4_int32, transdata_scale, restore_scale, _SCALE_MASK,
)
from ref_w4a8 import ref_quant_weight_int4
from bedrock.hand_fixtures import FIXTURES


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """pack_int4 的逆: [n, k//2] int8 (nibble) -> [n, k] int8 (signed int4)."""
    low = (packed & 0x0F)
    high = (packed >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def _t(x):
    return torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x


def part1_synthetic():
    print("=== Part 1: materialize q vs Tier 0 q (合成夹具) [⚠️仅一致: 共享[-8,7]clamp, 验机制不验spec] ===")
    npass = 0
    for fx in FIXTURES:
        if fx["kind"] != "weight":
            continue
        w, sc, gs = _t(fx["w"]), _t(fx["scale"]), fx["group_size"]
        q_conv, _ = materialize_int4_per_group_symmetric(w, sc, gs, signed=True, nbits=4, packed=False)
        q_ref, _ = ref_quant_weight_int4(w, sc, gs)
        ok = torch.equal(q_conv, q_ref)
        npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {fx['name']}: conv={q_conv.flatten().tolist()} ref={q_ref.flatten().tolist()}")
    return npass


def part2_pack_roundtrip():
    print("\n=== Part 2: pack_int4 round-trip [✅真正确: 编码bit-exact, 与量化spec无关] ===")
    q = torch.tensor([[2, -3, 7, -7, 0, 1, -8, 8 - 1]], dtype=torch.int8)  # 含边界值
    q = torch.clamp(q, -8, 7)
    packed = pack_int4(q)
    restored = unpack_int4(packed)
    ok = torch.equal(q, restored)
    print(f"  [{'PASS' if ok else 'FAIL'}] pack->unpack: orig={q.flatten().tolist()} restored={restored.flatten().tolist()}")
    return int(ok)


def part3_transdata_roundtrip():
    print("\n=== Part 3: transdata_scale round-trip [✅真正确: 存储bit-exact, 与量化spec无关] ===")
    s = torch.tensor([1.0, 0.5, 0.015748, 123.456, 2.0 / 127.0, 1e-8], dtype=torch.float32)
    packed = transdata_scale(s)
    restored = restore_scale(packed)
    expected = (s.view(torch.int32) & _SCALE_MASK).view(torch.float32)
    ok = torch.equal(restored, expected)
    n_changed = int((s.view(torch.int32) != expected.view(torch.int32)).sum())
    print(f"  [{'PASS' if ok else 'FAIL'}] transdata->restore == s&mask: restored={restored.tolist()}")
    print(f"         expected(s low-13 cleared)={expected.tolist()} ({n_changed}/{s.numel()} vals lost mantissa bits)")
    return int(ok)


def part4_real_dopt():
    print("\n=== Part 4: 真实 dopt 权重 (q_proj layer 0) materialize vs Tier 0 [⚠️仅一致: 共享[-8,7]clamp] ===")
    qdir = os.environ.get("DOPT_QUANT_DIR")
    if not qdir:
        print("  [SKIP] DOPT_QUANT_DIR 未设置 (设了才能跑真实数据)")
        return 0
    sd = torch.load(os.path.join(qdir, "trained_quant_weight.pth"), map_location="cpu", mmap=True, weights_only=True)
    name = "model.layers.0.self_attn.q_proj"
    w = sd[f"{name}.weight"].float()
    s_flat = sd[f"{name}.quant_op.weight_quantizer.s"].float()
    out_f, in_f = w.shape
    groups = s_flat.numel() // out_f
    gs = in_f // groups
    scale2d = s_flat.reshape(out_f, groups)
    print(f"  {name}: w={tuple(w.shape)} s_flat={s_flat.numel()} -> groups={groups} group_size={gs}")

    q_conv, sc_conv = materialize_int4_per_group_symmetric(w, scale2d, gs, signed=True, nbits=4, packed=False)
    q_ref, sc_ref = ref_quant_weight_int4(w, scale2d, gs)
    neq = int((q_conv != q_ref).sum())
    ok = (neq == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] materialize q == Tier 0 q: total={q_conv.numel()} diff={neq} (bit-exact)")
    if not ok:
        idx = (q_conv != q_ref).nonzero()[:5]
        for i in idx:
            o, k = i[0].item(), i[1].item()
            print(f"      diff @[{o},{k}]: conv={q_conv[o,k].item()} ref={q_ref[o,k].item()} w={w[o,k].item():.4f} s={scale2d[o,k//gs].item():.6f}")
    # 额外: pack round-trip on real q
    packed = pack_int4(q_conv)
    restored = unpack_int4(packed)
    ok2 = torch.equal(q_conv, restored)
    print(f"  [{'PASS' if ok2 else 'FAIL'}] pack_int4 round-trip on real q (diff={int((q_conv!=restored).sum())})")
    return int(ok) + int(ok2)


def part5_dopt_neg8_drop():
    """回归守卫: materialize P1 修复后 (clamp [-8,7]) 应保住 dopt 的 -8, 不再 clamp 成 -7.

    修复前: materialize [-7,7] 把 dopt[-8,7] 的 -8 全 clamp 成 -7 (每层 1万~5万个, 真精度损失).
    修复后: materialize [-8,7] == dopt[-8,7], n_changed 应=0. 此 Part 守 P1 不回归.
    """
    print("\n=== Part 5: materialize 保住 dopt -8 (P1 回归守卫) [✅真正确(独立交叉)] ===")
    qdir = os.environ.get("DOPT_QUANT_DIR")
    if not qdir:
        print("  [SKIP] DOPT_QUANT_DIR 未设置 (设了才能跑真实数据)")
        return 0
    sd = torch.load(os.path.join(qdir, "trained_quant_weight.pth"), map_location="cpu", mmap=True, weights_only=True)
    ops = ["model.layers.0.self_attn.q_proj",
           "model.layers.0.mlp.down_proj",
           "model.layers.21.mlp.down_proj"]
    total_neg8 = total_changed = 0
    for name in ops:
        w = sd[f"{name}.weight"].float()
        s_flat = sd[f"{name}.quant_op.weight_quantizer.s"].float()
        out_f, in_f = w.shape
        groups = s_flat.numel() // out_f
        gs = in_f // groups
        scale2d = s_flat.reshape(out_f, groups)
        s_exp = scale2d.repeat_interleave(gs, dim=-1)
        dopt_q = torch.round(w / s_exp).clamp(-8, 7).to(torch.int8)   # dopt qmin_max clip [-8,7]
        conv_q, _ = materialize_int4_per_group_symmetric(w, scale2d, gs, signed=True, nbits=4, packed=False)  # 修复后 [-8,7]
        n_neg8 = int((dopt_q == -8).sum())
        n_changed = int((dopt_q != conv_q).sum())
        total_neg8 += n_neg8; total_changed += n_changed
        print(f"  {name}: dopt[-8,7] #-8={n_neg8:>6d} | materialize 改动 {n_changed:>6d} 个 (应=0)")
    ok = (total_changed == 0)
    print(f"  [{'PASS' if ok else 'FAIL/REGRESSED'}] 三层共 {total_neg8} 个 -8, materialize 改动 {total_changed} 个 "
          f"(P1 修复后应 0; 非 0 = materialize clamp 又退回 [-7,7])")
    return int(ok)


def part6_int4pack_vs_opsnn_golden():
    """int4pack 布局对齐 ops-nn canonical unpack_int32_to_int4 [✅真正确(独立交叉)].

    pack_int4_int32 (materialize.py) ↔ ops-nn golden unpack (matmul_quant_util.py) round-trip bit-exact,
    含 -8 边界. 这 lifts 了 int4pack "未验" 状态, 也是 kirinx90 int32-int4 路径 (WeightQuantBatchMatmulV2
    + convert_weight_to_int4_pack) 的布局可信度根.
    """
    import numpy as np
    print("\n=== Part 6: pack_int4_int32 ↔ ops-nn golden unpack [✅真正确(独立交叉)] ===")

    # ops-nn canonical unpack (复制自 ops-nn/matmul/common/tests/st/arch35/matmul_quant_util.py)
    def unpack_int32_to_int4(w32, axis=-1):
        orig = w32.shape
        if len(orig) < 2:
            return w32
        w32 = np.ascontiguousarray(w32); axis = len(orig) - 1
        n_packed = orig[axis]; n_unpacked = n_packed * 8
        wf = w32.reshape(-1, n_packed); res = np.zeros((wf.shape[0], n_unpacked), dtype=np.int8)
        for i in range(8):
            v = (wf >> (i * 4)) & 0xF; v = np.where(v >= 8, v - 16, v); res[:, i::8] = v
        ns = list(orig); ns[axis] = n_unpacked; return res.reshape(ns)

    torch.manual_seed(0)
    q = torch.randint(-8, 8, (4, 16), dtype=torch.int8)  # 含 -8 边界, n%8==0
    packed = pack_int4_int32(q)                          # torch int32 [k, n//8]
    unpacked = unpack_int32_to_int4(packed.numpy())      # numpy int8 [k, n]
    ok = np.array_equal(unpacked, q.numpy())
    n_neg8 = int((q == -8).sum()); n_neg8_kept = int((unpacked == -8).sum())
    print(f"  [{'PASS' if ok else 'FAIL'}] round-trip bit-exact={ok} | 原 #-8={n_neg8} unpack #-8={n_neg8_kept} "
          f"(保住={n_neg8==n_neg8_kept}) | packed {tuple(packed.shape)} {packed.dtype} unpack range [{unpacked.min()},{unpacked.max()}]")
    return int(ok)


def main():
    npass = part1_synthetic() + part2_pack_roundtrip() + part3_transdata_roundtrip() + part4_real_dopt()
    part5_dopt_neg8_drop()
    part6_int4pack_vs_opsnn_golden()
    print(f"\nverify_materialize: done (见上方各 PASS/FAIL/FINDING)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
