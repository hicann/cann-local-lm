import pytest
import torch

from dopt_lm.quantizer.qmin_max import (
    GroupMinMaxGridWeight,
    LSQAutograd,
    MinMaxGrid,
    OmniGroupMinMaxGridWeight,
    OmniMinMaxGrid,
)


@pytest.mark.parametrize("per_channel,shape", [(True, (3,)), (False, ())])
def test_minmax_initial_scale_shape_and_buffers(per_channel, shape):
    layer = torch.nn.Linear(4, 3, bias=False)
    quantizer = MinMaxGrid(layer, bit=4, per_channel=per_channel)

    assert tuple(quantizer.s.shape) == shape
    assert tuple(quantizer.offset.shape) == shape
    assert quantizer.bit.item() == 4
    assert torch.all(quantizer.s > 0)


@pytest.mark.parametrize("unsigned_mode", [0, 1, 2, 3])
def test_minmax_calibration_modes_produce_finite_values(unsigned_mode):
    layer = torch.nn.Linear(4, 2, bias=False)
    quantizer = MinMaxGrid(
        layer,
        bit=4,
        per_channel=True,
        unsigned_quant=unsigned_mode,
    )
    quantizer.set_calibrate_state(True)
    x = torch.tensor([[-2.0, -1.0, 1.0, 3.0], [-4.0, 0.0, 2.0, 4.0]])

    result = quantizer(x)

    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.isfinite(result).all()
    assert torch.all(quantizer.s >= quantizer.min_scale_value)
    torch.testing.assert_close(quantizer.min, torch.tensor([-2.0, -4.0]))
    torch.testing.assert_close(quantizer.max, torch.tensor([3.0, 4.0]))

    quantizer.set_calibrate_state(False)
    second = quantizer(x)
    assert torch.isfinite(second).all()


def test_minmax_one_bit_and_fp16_flag_on_cpu():
    layer = torch.nn.Linear(3, 2, bias=False)
    quantizer = MinMaxGrid(layer, bit=1, per_channel=True, fp16=True)
    quantizer.set_calibrate_state(True)
    x = torch.tensor([[-2.0, 0.0, 1.0], [-1.0, 2.0, 3.0]], dtype=torch.float16)

    result = quantizer(x)

    assert result.dtype == torch.float16
    assert quantizer.s.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_minmax_rejects_unknown_unsigned_mode():
    layer = torch.nn.Linear(3, 2, bias=False)
    quantizer = MinMaxGrid(layer, bit=4, per_channel=True, unsigned_quant=4)
    quantizer.set_calibrate_state(True)
    with pytest.raises(Exception, match="not support unsigned_quant"):
        quantizer(layer.weight)


def test_group_minmax_validates_shape_and_quantizes():
    layer = torch.nn.Linear(4, 3, bias=False)
    quantizer = GroupMinMaxGridWeight(layer, bit=4, group_size=2)
    quantizer.set_calibrate_state(True)

    result = quantizer(layer.weight.detach())

    assert result.shape == layer.weight.shape
    assert result.dtype == layer.weight.dtype
    assert quantizer.groups == 2
    assert quantizer.s.numel() == 6

    with pytest.raises(ValueError, match="only support per_channel"):
        GroupMinMaxGridWeight(layer, bit=4, per_channel=False, group_size=2)
    with pytest.raises(ValueError, match="must be divisible"):
        GroupMinMaxGridWeight(layer, bit=4, group_size=3)


def test_lsq_autograd_clamps_and_computes_gradients():
    x = torch.tensor([[-3.0, -0.2, 1.6, 4.0]], requires_grad=True)
    step = torch.tensor([[1.0]], requires_grad=True)

    result = LSQAutograd.apply(x, step, -2, 2)
    quantized = result[0]
    torch.testing.assert_close(quantized, torch.tensor([[-2.0, 0.0, 2.0, 2.0]]))
    quantized.sum().backward()

    torch.testing.assert_close(x.grad, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
    assert step.grad is not None
    assert torch.isfinite(step.grad).all()


def test_lsq_autograd_no_clamp_without_grad_returns_quantized_tuple():
    result = LSQAutograd.apply(torch.tensor([0.4, 1.6]), torch.tensor(1.0), None, None)
    torch.testing.assert_close(result[0], torch.tensor([0.0, 2.0]))


@pytest.mark.parametrize("unsigned", [False, True])
def test_omni_minmax_calibrates_accumulates_and_backpropagates(unsigned):
    layer = torch.nn.Linear(4, 2, bias=False)
    quantizer = OmniMinMaxGrid(layer, bit=4, per_channel=True, unsigned_quant=unsigned)
    quantizer.set_calibrate_state(True)
    first = torch.tensor([[-1.0, 0.0, 1.0, 2.0], [-2.0, 0.0, 2.0, 3.0]], requires_grad=True)
    second = torch.tensor([[-4.0, 0.0, 1.0, 1.0], [-1.0, 0.0, 2.0, 5.0]])

    result = quantizer(first)
    quantizer(second)
    result.sum().backward()

    assert quantizer.min_iner.tolist() == [-4.0, -2.0]
    assert quantizer.max_iner.tolist() == [2.0, 5.0]
    assert quantizer.quant_alpha.grad is not None
    assert first.grad is not None
    assert torch.isfinite(result).all()

    quantizer.set_calibrate_state(False)
    assert torch.isfinite(quantizer(first.detach())).all()


def test_omni_group_minmax_validates_dimensions_and_forward():
    layer = torch.nn.Linear(4, 3, bias=False)
    quantizer = OmniGroupMinMaxGridWeight(layer, bit=4, group_size=2)
    quantizer.set_calibrate_state(True)
    result = quantizer(layer.weight)

    assert result.shape == layer.weight.shape
    assert quantizer.s.numel() == 6

    conv = torch.nn.Conv2d(2, 3, kernel_size=2)
    with pytest.raises(ValueError, match="Expected 2 dimensions"):
        OmniGroupMinMaxGridWeight(conv, bit=4, group_size=2)
    with pytest.raises(ValueError, match="must be divisible"):
        OmniGroupMinMaxGridWeight(layer, bit=4, group_size=3)
