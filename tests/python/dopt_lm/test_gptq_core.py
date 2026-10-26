import pytest
import torch
import torch.nn as nn

from dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian import modelutils
from dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian.modelutils import find_layers
from dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian.ptq_algo import (
    WeightQuantHessian,
)
from dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian.quant import (
    Quantizer,
    quantize,
)


def test_gptq_quantize_integer_grid_and_trits():
    x = torch.tensor([[-2.0, -0.2, 0.6, 2.0]])
    integer = quantize(x, torch.tensor(0.5), torch.tensor(2.0), torch.tensor(3))
    expected = torch.tensor([[-1.0, 0.0, 0.5, 0.5]])
    torch.testing.assert_close(integer, expected)

    trits = quantize(x, torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(-1))
    torch.testing.assert_close(trits, torch.tensor([[-1.0, 0.0, 1.0, 1.0]]))


def test_quantizer_configuration_ready_identity_and_trits():
    quantizer = Quantizer()
    x = torch.tensor([1.0, 2.0])
    assert quantizer.ready() is torch.tensor(False) or not quantizer.ready()
    assert quantizer.quantize(x) is x

    quantizer.configure(4, perchannel=True, sym=True)
    assert quantizer.maxq.item() == 15
    assert quantizer.enabled()
    quantizer.configure(2, trits=True)
    assert quantizer.maxq.item() == -1
    assert not quantizer.enabled()


@pytest.mark.parametrize(
    "shape,perchannel,weight,expected_shape",
    [
        ((3, 4), True, True, (3, 1)),
        ((3, 4), False, True, (3, 1)),
        ((2, 4), True, False, (1, 4)),
        ((2, 3, 4), True, False, (1, 1, 4)),
        ((2, 3, 2, 2), True, False, (1, 3, 1, 1)),
    ],
)
def test_quantizer_find_params_shapes(shape, perchannel, weight, expected_shape):
    torch.manual_seed(13)
    x = torch.randn(*shape)
    quantizer = Quantizer()
    quantizer.configure(4, perchannel=perchannel, sym=True)

    quantizer.find_params(x, weight=weight)

    assert tuple(quantizer.scale.shape) == expected_shape
    assert tuple(quantizer.zero.shape) == expected_shape
    assert quantizer.ready()
    assert torch.isfinite(quantizer.quantize(x)).all()


def test_quantizer_handles_all_zero_asymmetric_and_mse_search():
    quantizer = Quantizer()
    quantizer.configure(
        3, perchannel=False, sym=False, mse=True, grid=10, maxshrink=0.5
    )
    x = torch.zeros(2, 4)

    quantizer.find_params(x, weight=True)

    assert quantizer.ready()
    assert torch.all(quantizer.scale > 0)
    assert torch.isfinite(quantizer.zero).all()


def test_weight_hessian_linear_batch_accumulation():
    layer = nn.Linear(4, 3, bias=False)
    algorithm = WeightQuantHessian(layer)
    x1 = torch.eye(4).unsqueeze(0)
    x2 = torch.ones(2, 3, 4)

    algorithm.add_batch(x1, layer(x1))
    first = algorithm.hessian_matrix.clone()
    algorithm.add_batch(x2, layer(x2))

    assert algorithm.hessian_matrix.shape == (4, 4)
    assert algorithm.nsamples == 3
    assert torch.isfinite(algorithm.hessian_matrix).all()
    assert not torch.equal(first, algorithm.hessian_matrix)


def test_weight_hessian_conv_batch_shape():
    layer = nn.Conv2d(1, 2, kernel_size=2, bias=False)
    algorithm = WeightQuantHessian(layer)
    x = torch.randn(2, 1, 3, 3)

    algorithm.add_batch(x, layer(x))

    assert algorithm.hessian_matrix.shape == (4, 4)
    assert algorithm.nsamples == 2


@pytest.mark.parametrize(
    "actorder,static_groups",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_fasterquant_writes_weights_and_group_parameters(actorder, static_groups):
    torch.manual_seed(17)
    layer = nn.Linear(4, 3, bias=False)
    original = layer.weight.detach().clone()
    algorithm = WeightQuantHessian(layer)
    algorithm.quantizer = Quantizer()
    algorithm.quantizer.configure(4, perchannel=True, sym=True, mse=False)
    expected_static_scale = []
    expected_static_zero = []
    if static_groups:
        for start in range(0, original.shape[1], 2):
            group_quantizer = Quantizer()
            group_quantizer.configure(4, perchannel=True, sym=True, mse=False)
            group_quantizer.find_params(original[:, start : start + 2], weight=True)
            expected_static_scale.append(group_quantizer.scale)
            expected_static_zero.append(group_quantizer.zero)
    # Full-rank activation covariance avoids a singular Cholesky factor.
    for _ in range(3):
        x = torch.randn(4, 4)
        algorithm.add_batch(x, layer(x))

    algorithm.fasterquant(
        blocksize=2,
        groupsize=2,
        actorder=actorder,
        static_groups=static_groups,
    )

    assert layer.weight.shape == original.shape
    assert torch.isfinite(layer.weight).all()
    assert not torch.equal(layer.weight, original)
    assert algorithm.quantizer.group_scale.shape == (6, 1)
    assert algorithm.quantizer.group_zero.shape == (6, 1)
    if static_groups:
        expected_scale = torch.cat(expected_static_scale, dim=1).flatten().unsqueeze(1)
        expected_zero = torch.cat(expected_static_zero, dim=1).flatten().unsqueeze(1)
        torch.testing.assert_close(algorithm.quantizer.group_scale, expected_scale)
        torch.testing.assert_close(algorithm.quantizer.group_zero, expected_zero)


def test_fasterquant_static_groups_requires_positive_group_size():
    algorithm = WeightQuantHessian(nn.Linear(4, 2, bias=False))
    algorithm.quantizer = Quantizer()
    algorithm.quantizer.configure(4, perchannel=True, sym=True)

    with pytest.raises(ValueError, match="groupsize to be greater than zero"):
        algorithm.fasterquant(static_groups=True, groupsize=-1)


def test_fasterquant_handles_dead_input_column():
    torch.manual_seed(19)
    layer = nn.Linear(4, 2, bias=False)
    algorithm = WeightQuantHessian(layer)
    algorithm.quantizer = Quantizer()
    algorithm.quantizer.configure(4, perchannel=True, sym=True)
    x = torch.randn(8, 4)
    x[:, 3] = 0
    algorithm.add_batch(x, layer(x))

    algorithm.fasterquant(blocksize=4, groupsize=2)

    assert torch.count_nonzero(layer.weight[:, 3]) == 0


def test_weight_hessian_free_releases_matrix(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "dopt_lm.opt_funcs.weight_quant_ptq.ptq_algo_hessian.ptq_algo.empty_device_cache",
        lambda: calls.append(True),
    )
    algorithm = WeightQuantHessian(nn.Linear(2, 2))
    algorithm.free()
    assert algorithm.hessian_matrix is None
    assert calls == [True]


def test_find_layers_default_custom_and_nested_names():
    model = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.ReLU(), nn.Conv2d(1, 1, 1)))
    found = find_layers(model)
    assert set(found) == {"0", "1.1"}
    custom = find_layers(model, layers=[nn.ReLU])
    assert set(custom) == {"1.0"}


def test_cpu_device_helpers_are_noops():
    assert modelutils.DEVICE_TYPE == "cpu"
    assert modelutils.empty_device_cache() is None
    assert modelutils.synchronize_device() is None
