import copy

import pytest
import torch
import torch.nn.functional as F

from dopt_lm.builder import Quantizers
from dopt_lm.modules.qembed import QEmbedding, QGemma4ScaledEmbedding
from dopt_lm.modules.qlinear import QLinear
from dopt_lm.modules.utils import get_quantizer_cfg
from dopt_lm.quantizer.qmanager import QClass


class AddQuantizer(torch.nn.Module):
    def __init__(self, father, delta=0.0, **kwargs):
        super().__init__()
        self.delta = delta
        self.calibrate_states = []
        self.calls = 0

    def set_calibrate_state(self, state):
        self.calibrate_states.append(state)

    def forward(self, value):
        self.calls += 1
        return value + self.delta


def _register_test_quantizer():
    Quantizers.register_func("ut_add_quantizer")(AddQuantizer)


def test_get_quantizer_cfg_handles_none_partial_and_complete():
    assert get_quantizer_cfg(None) == (None, None, None)
    weight = {"bit": 4}
    output = {"bit": 8}
    assert get_quantizer_cfg({"weight": weight, "output": output}) == (weight, None, output)


def test_qclass_constructs_quantizers_without_mutating_config():
    _register_test_quantizer()
    layer = torch.nn.Linear(2, 2)
    weight_cfg = {"weight_algo": "ut_add_quantizer", "delta": 1.0}
    input_cfg = {"input_algo": "ut_add_quantizer", "delta": 2.0, "enable": False}
    output_cfg = {"input_algo": "ut_add_quantizer", "delta": 3.0}
    originals = copy.deepcopy((weight_cfg, input_cfg, output_cfg))

    manager = QClass(layer, weight_cfg, input_cfg, output_cfg)

    assert isinstance(manager.weight_quantizer, AddQuantizer)
    assert isinstance(manager.input_quantizer, AddQuantizer)
    assert isinstance(manager.output_quantizer, AddQuantizer)
    assert manager.do_weight_quant is True
    assert manager.do_input_quant is False
    assert output_cfg == originals[2]
    assert (weight_cfg, input_cfg) == originals[:2]


def test_qclass_forward_state_calibration_and_empty():
    _register_test_quantizer()
    layer = torch.nn.Linear(2, 2)
    manager = QClass(
        layer,
        {"weight_algo": "ut_add_quantizer", "delta": 1.0},
        {"input_algo": "ut_add_quantizer", "delta": 2.0},
        {"input_algo": "ut_add_quantizer", "delta": 3.0},
    )
    weight = torch.zeros(2, 2)
    x = torch.zeros(1, 2)

    quant_weight, quant_x = manager(weight, x)
    manager.set_calibrate_state(True)

    torch.testing.assert_close(quant_weight, torch.ones_like(weight))
    torch.testing.assert_close(quant_x, torch.full_like(x, 2.0))
    assert manager.weight_quantizer.calibrate_states == [True]
    assert manager.input_quantizer.calibrate_states == [True]
    assert manager.output_quantizer.calibrate_states == [True]
    assert manager.is_empty() is False

    manager.set_quant_state(False, False)
    quant_weight, quant_x = manager(weight, x)
    torch.testing.assert_close(quant_weight, weight)
    torch.testing.assert_close(quant_x, x)
    assert QClass(layer).is_empty() is True


def test_qlinear_without_quantization_matches_torch_linear():
    torch.manual_seed(3)
    source = torch.nn.Linear(4, 3, bias=True)
    wrapped = QLinear(source, {}, layer_name="proj")
    x = torch.randn(2, 5, 4)

    actual = wrapped(x)
    expected = F.linear(x, source.weight, source.bias)

    torch.testing.assert_close(actual, expected)
    assert wrapped.weight is source.weight
    assert wrapped.bias is source.bias
    assert wrapped.layer_name == "proj"


def test_qlinear_static_weight_quantization_calibrates_and_preserves_contract():
    torch.manual_seed(5)
    source = torch.nn.Linear(4, 3, bias=False)
    wrapped = QLinear(
        source,
        {"weight": {"bit": 4, "per_channel": True, "weight_algo": "min_max"}},
    )
    wrapped.quant_op.set_calibrate_state(True)
    x = torch.randn(2, 4)

    actual = wrapped(x)

    assert actual.shape == (2, 3)
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    assert torch.all(wrapped.quant_op.weight_quantizer.s > 0)


def test_qlinear_dynamic_s8_is_close_to_reference():
    torch.manual_seed(7)
    source = torch.nn.Linear(8, 4, bias=True)
    wrapped = QLinear(
        source,
        {"input": {"bit": 8, "input_algo": "dyn_s8", "enable": True}},
    )
    x = torch.randn(2, 3, 8)

    actual = wrapped(x)
    expected = source(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qembedding_forces_per_channel_and_clones_weight(monkeypatch):
    warnings = []
    monkeypatch.setattr("dopt_lm.modules.qembed.logger.warning", warnings.append)
    source = torch.nn.Embedding(6, 4, padding_idx=0)
    config = {
        "weight": {
            "bit": 8,
            "per_channel": False,
            "weight_algo": "min_max",
            "tie_lmhead": False,
        }
    }

    wrapped = QEmbedding(source, config, layer_name="embed")
    wrapped.quant_op.set_calibrate_state(True)
    result = wrapped(torch.tensor([[0, 1, 5]]))

    assert config["weight"]["per_channel"] is True
    assert warnings
    assert wrapped.weight.data_ptr() != source.weight.data_ptr()
    assert wrapped.layer_name == "embed"
    assert result.shape == (1, 3, 4)
    assert torch.isfinite(result).all()


def test_qembedding_tied_weight_keeps_parameter_identity():
    source = torch.nn.Embedding(5, 3)
    wrapped = QEmbedding(
        source,
        {"weight": {"bit": 8, "per_channel": True, "weight_algo": "min_max", "tie_lmhead": True}},
    )
    assert wrapped.weight is source.weight
    assert wrapped.tie_lmhead is True


class ScaledEmbedding(torch.nn.Embedding):
    def __init__(self, *args, tensor_scale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar_embed_scale = 2.5
        if tensor_scale:
            self.register_buffer("embed_scale", torch.tensor(3.0))


@pytest.mark.parametrize("tensor_scale,expected_scale", [(True, 3.0), (False, 2.5)])
def test_scaled_embedding_applies_tensor_or_scalar_scale(tensor_scale, expected_scale):
    torch.manual_seed(11)
    source = ScaledEmbedding(5, 3, tensor_scale=tensor_scale)
    wrapped = QGemma4ScaledEmbedding(
        source,
        {"weight": {"bit": 8, "per_channel": True, "weight_algo": "min_max", "tie_lmhead": True}},
    )
    wrapped.set_quant_state(False, False)
    ids = torch.tensor([[1, 4]])

    expected = F.embedding(ids, source.weight) * expected_scale
    torch.testing.assert_close(wrapped(ids), expected)
    assert "embed_scale" in wrapped._non_persistent_buffers_set
