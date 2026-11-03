"""quant_swap/materialize.py 单元测试 — int4/int8 权重量化 materialization (纯 tensor ops).

测: transdata_scale↔restore_scale round-trip (scale 存储格式), pack_int4 nibble 布局,
materialize_int4/int8 量化正确性 (q=round(w/s) clamp + dequant q*s≈w) + shape/错误处理.
"""

import pytest
import torch

from cann_local_lm.converter.export.quant_swap.materialize import (
    _SCALE_MASK,
    materialize_int4_per_group_symmetric,
    materialize_int8_per_channel_symmetric,
    pack_int4,
    restore_scale,
    transdata_scale,
)


# ── transdata_scale / restore_scale (fp32 scale 存储格式 round-trip) ──


def test_transdata_scale_dtype_shape():
    s = torch.randn(4, 8, dtype=torch.float32)
    out = transdata_scale(s)
    assert out.dtype == torch.int64
    assert out.shape == s.shape


def test_transdata_restore_roundtrip_zeroes_mantissa_low13():
    s = torch.randn(16, dtype=torch.float32)
    rt = restore_scale(transdata_scale(s))
    assert rt.dtype == torch.float32
    assert rt.shape == s.shape
    # mask 清了 fp32 低 13 位 (mantissa 低 13 bit) → round-trip 后低 13 位为 0
    assert torch.all((rt.view(torch.int32) & 0x1FFF) == 0)
    # 数值仍在 (精度损失小, 低 13 mantissa bit ≈ 2^-13 相对误差)
    assert torch.allclose(rt, s, atol=1e-3, rtol=1e-2)


def test_transdata_scale_masks_high32_of_int64():
    s = torch.randn(4, dtype=torch.float32)
    out = transdata_scale(s)
    # int64 高 32 位被 mask 清零 (低 32 位是 fp32 bit-pattern 的低位)
    assert torch.all((out >> 32) == 0)
    assert torch.all((out & ~_SCALE_MASK) == 0)  # mask 外的位全 0


# ── pack_int4 (nibble 打包) ──


def test_pack_int4_layout_known_values():
    # 4 个 int4 值 [1, -1(=0xF), 7, -8(=0x8)] → 2 bytes (int8 有符号, 用 &0xFF 比无符号字节)
    q = torch.tensor([1, -1, 7, -8], dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.shape == (2,)
    assert packed.dtype == torch.int8
    # byte0 = low nibble(1)=0x01 | high nibble(-1&0xF=0xF)<<4 = 0xF1
    assert (packed[0].item() & 0xFF) == ((0x0F << 4) | 0x01)
    # byte1 = low nibble(7)=0x07 | high nibble(-8&0xF=0x8)<<4 = 0x87
    assert (packed[1].item() & 0xFF) == ((0x08 << 4) | 0x07)


def test_pack_int4_roundtrip():
    q = torch.randint(-8, 7, (3, 10), dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.shape == (3, 5)
    # 逆解: low nibble = even idx, high nibble = odd idx (4-bit 符号扩展: 8..15 → -8..-1)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low_s = torch.where(low >= 8, low - 16, low).to(torch.int8)
    high_s = torch.where(high >= 8, high - 16, high).to(torch.int8)
    rec = torch.empty_like(q)
    rec[..., 0::2] = low_s
    rec[..., 1::2] = high_s
    assert torch.equal(rec, q)


def test_pack_int4_odd_last_dim_raises():
    with pytest.raises(ValueError, match="even"):
        pack_int4(torch.zeros(3, dtype=torch.int8))


# ── materialize_int4_per_group_symmetric ──


def test_materialize_int4_shape_unpacked_and_packed():
    out_f, in_f, g = 4, 8, 4
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    s = torch.rand(out_f, in_f // g, dtype=torch.float32) + 0.1
    q, sc = materialize_int4_per_group_symmetric(w, s, group_size=g, packed=False)
    assert q.shape == (out_f, in_f) and q.dtype == torch.int8
    assert sc.shape == s.shape and sc.dtype == torch.float32
    q_p, _ = materialize_int4_per_group_symmetric(w, s, group_size=g, packed=True)
    assert q_p.shape == (out_f, in_f // 2)


def test_materialize_int4_quant_dequant_exact_fakequant():
    # fakequant: w = q0*s (q0 在 [-7,7] ⊂ [-8,7] 不触发 clamp), round(w/s)=q0 精确, dequant=q0*s=w
    out_f, in_f, g = 2, 8, 4
    q0 = torch.randint(-7, 7, (out_f, in_f), dtype=torch.int8)  # 不触发 clamp
    s = (torch.rand(out_f, in_f // g, dtype=torch.float32) + 0.1) * 0.01
    scale_exp = s.repeat_interleave(g, dim=-1)
    w = q0.float() * scale_exp
    q, sc = materialize_int4_per_group_symmetric(w, s, group_size=g)
    assert torch.equal(q, q0)  # 精确重建 (无 clamp)
    assert torch.allclose(q.float() * sc.repeat_interleave(g, dim=-1), w)
    assert q.min() >= -8 and q.max() <= 7  # signed int4 二补码全范围 [-8,7] (P1 修复)


def test_materialize_int4_clamp_beyond_range():
    out_f, in_f, g = 1, 4, 4  # noqa: F841  # out_f/in_f 为文档化维度, 仅 g 参与 materialize
    w = torch.tensor([[10.0, -10.0, 0.5, -0.5]])
    s = torch.tensor([[0.1]])  # w/s = [100,-100,5,-5] → clamp[-8,7]
    q, _ = materialize_int4_per_group_symmetric(w, s, group_size=g)
    # materialize signed int4 clamp [-8,7] (二补码全范围, 保住 -8; P1 修复前是 [-7,7] 丢 -8)
    assert q.tolist() == [[7, -8, 5, -5]]


@pytest.mark.parametrize(
    "bad_in,bad_scale,g,msg",
    [
        (5, (1, 1), 4, "not divisible"),  # in=5 not divisible by group=4
        (8, (1, 3), 4, "scale shape"),  # scale shape mismatch
    ],
)
def test_materialize_int4_errors(bad_in, bad_scale, g, msg):
    w = torch.randn(1, bad_in, dtype=torch.float32)
    s = torch.rand(*bad_scale, dtype=torch.float32) + 0.1
    with pytest.raises(ValueError, match=msg):
        materialize_int4_per_group_symmetric(w, s, group_size=g)


# ── materialize_int8_per_channel_symmetric ──


def test_materialize_int8_shape_and_range():
    vocab, hidden = 6, 4
    w = torch.randn(vocab, hidden, dtype=torch.float32) * 0.1
    s = (torch.rand(vocab, dtype=torch.float32) + 0.1) * 0.05
    q, sc = materialize_int8_per_channel_symmetric(w, s)
    assert q.shape == (vocab, hidden) and q.dtype == torch.int8
    assert sc.shape == (vocab,) and sc.dtype == torch.float32
    assert q.min() >= -128 and q.max() <= 127


def test_materialize_int8_dequant_allclose():
    vocab, hidden = 5, 8
    w = torch.randn(vocab, hidden, dtype=torch.float32) * 0.1
    s = (torch.rand(vocab, dtype=torch.float32) + 0.1) * 0.05
    q, sc = materialize_int8_per_channel_symmetric(w, s)
    dequant = q.float() * sc.unsqueeze(1)
    assert torch.allclose(dequant, w, atol=0.1)


def test_materialize_int8_dopt_fakequant_exact_roundtrip():
    # dopt fakequant: weight 存的已是 q*scale (float), 故 round(w/s)=q 精确 (maxdiff=0)
    vocab, hidden = 4, 6
    q0 = torch.randint(-127, 127, (vocab, hidden), dtype=torch.int8)
    s = (torch.rand(vocab, dtype=torch.float32) + 0.5) * 0.01
    fakequant = q0.float() * s.unsqueeze(1)  # dopt 存的就是这个
    q, _ = materialize_int8_per_channel_symmetric(fakequant, s)
    assert torch.equal(q, q0)  # 精确重建 (clamp 不触发, 因 q0 在 [-127,127])


def test_materialize_int8_scale_shape_error():
    w = torch.randn(3, 4, dtype=torch.float32)
    s = torch.rand(4, dtype=torch.float32) + 0.1  # 应是 (3,) 不是 (4,)
    with pytest.raises(ValueError, match="scale shape"):
        materialize_int8_per_channel_symmetric(w, s)
