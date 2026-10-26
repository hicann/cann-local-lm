import pytest
import torch

from dopt_lm.quantizer.qeval import (
    DynEvalGridS8,
    DynIntSymQuantizer,
    QuantizerFuncFpOffset,
    QuantizerFuncIntOffset,
)
from dopt_lm.quantizer.qutils import (
    RoundWithGrad,
    do_antiquantize,
    do_quantize,
    fake_nf,
    fakequant_forward,
    get_abs_max_per_channel,
    get_clip_range,
    get_grad_max,
    get_grad_min,
    get_min_max,
    get_min_max_per_channel,
)


@pytest.mark.parametrize(
    "bit,unsigned,keep_neg,expected",
    [
        (4, False, True, (-8, 7)),
        (4, False, False, (-7, 7)),
        (4, True, True, (0, 15)),
        (8, False, True, (-128, 127)),
    ],
)
def test_get_clip_range_integer_modes(bit, unsigned, keep_neg, expected):
    negative, positive = get_clip_range(bit, unsigned, keep_neg)
    assert (negative, positive) == expected


def test_get_clip_range_rejects_unsigned_two_point_five_bit():
    with pytest.raises(ValueError, match="only support sym"):
        get_clip_range(2.5, True)


def test_min_max_helpers_cover_global_topk_and_channels():
    x = torch.tensor([[-5.0, 1.0, 4.0], [-2.0, 3.0, 8.0]])
    minimum, maximum = get_min_max(x)
    assert minimum.item() == -5.0
    assert maximum.item() == 8.0
    top_min, top_max = get_min_max(x, mean_topk=2)
    torch.testing.assert_close(top_min, torch.tensor(-3.5))
    torch.testing.assert_close(top_max, torch.tensor(6.0))
    with pytest.raises(NotImplementedError):
        get_min_max(x, mean_topk=0)

    channel_min, channel_max = get_min_max_per_channel(x, 0)
    channel_abs = get_abs_max_per_channel(x, 0)
    torch.testing.assert_close(channel_min, torch.tensor([-5.0, -2.0]))
    torch.testing.assert_close(channel_max, torch.tensor([4.0, 8.0]))
    torch.testing.assert_close(channel_abs, torch.tensor([5.0, 8.0]))


def test_quantize_antiquantize_and_fakequant_broadcast():
    x = torch.tensor([[-1.2, -0.1, 1.4], [-2.0, 0.4, 2.0]])
    scale = torch.tensor([0.5, 1.0])
    offset = torch.tensor([0.0, 0.0])
    qx = do_quantize(x, scale.reshape(-1, 1), offset.reshape(-1, 1), -2, 2)
    restored = do_antiquantize(qx, scale.reshape(-1, 1), offset.reshape(-1, 1))
    fake = fakequant_forward(x, scale, offset, -2, 2)

    torch.testing.assert_close(qx, torch.tensor([[-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]]))
    torch.testing.assert_close(restored, fake)


def test_round_with_grad_and_gradient_helpers():
    x = torch.tensor([-1.2, 0.4, 2.1], requires_grad=True)
    RoundWithGrad.apply(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))

    value = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    scale = torch.tensor(0.25, requires_grad=True)
    max_result = get_grad_max(value, scale)
    min_result = get_grad_min(value, scale)
    assert torch.all(max_result >= 1.0e-8)
    assert torch.all(min_result <= 0.0)


def test_fake_nf_selects_nearest_label_and_preserves_ste_gradient():
    x = torch.tensor([-0.8, 0.2, 0.9], requires_grad=True)
    labels = torch.tensor([-1.0, 0.0, 1.0])
    result = fake_nf(x, torch.tensor(1.0), labels)

    torch.testing.assert_close(result, torch.tensor([-1.0, 0.0, 1.0]))
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_int_offset_quantizer_clamps_scale_and_has_ste():
    x = torch.tensor([-3.0, -0.4, 0.6, 4.0], dtype=torch.float16, requires_grad=True)
    scale = torch.tensor(0.0)
    offset = torch.tensor(0.0)
    result = QuantizerFuncIntOffset.apply(x, scale, offset, -2, 2)

    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()
    result.float().sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_fp_offset_quantizer_matches_reference_and_has_ste():
    x = torch.tensor([-1.1, 0.2, 1.7], requires_grad=True)
    scale = torch.tensor(0.5)
    offset = torch.tensor(0.25)
    result = QuantizerFuncFpOffset.apply(x, scale, offset, -2, 2)
    expected = (
        torch.clamp(torch.round((x.detach() + offset) / scale), -2, 2) * scale - offset
    )

    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dynamic_s8_quantizer_range_scale_dtype_and_zero_input(dtype):
    x = torch.tensor([[[-2.0, 0.0, 1.0, 4.0]]], dtype=dtype, requires_grad=True)
    normalized, scale = DynIntSymQuantizer.apply(x)

    assert normalized.dtype == dtype
    assert scale.dtype == dtype
    assert normalized.abs().max() <= 1.01
    torch.testing.assert_close(
        scale,
        torch.tensor([[[4.0]]], dtype=dtype),
        rtol=1e-3,
        atol=1e-3,
    )
    (normalized.float().sum() + scale.float().sum() * 0).backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))

    zero_out, zero_scale = DynIntSymQuantizer.apply(torch.zeros(2, 3))
    assert torch.isfinite(zero_out).all()
    assert torch.isfinite(zero_scale).all()
    assert torch.all(zero_scale >= 2.0e-5)


def test_dynamic_grid_buffers_init_and_forward():
    quantizer = DynEvalGridS8()
    moved = DynEvalGridS8.init_from_quantizer(quantizer)
    x = torch.tensor([[1.0, -2.0]])

    result, scale = moved(x)

    assert moved.quant_dtype == "dyn_s8"
    assert moved.bit.item() == 8
    assert result.shape == x.shape
    assert scale.shape == (1, 1)
    assert moved.s.device == quantizer.s.device
