import json

import pytest
import torch

from dopt_lm import do_opt
from dopt_lm.modules.qembed import QEmbedding, QGemma4ScaledEmbedding
from dopt_lm.modules.qlinear import QLinear
from dopt_lm.quantizer.strategy import get_optimize_cfg
from dopt_lm.utils import replace_module_by_names


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.keep = torch.nn.Linear(4, 2)


def write_config(path, layer_strategy):
    path.write_text(json.dumps({"layer_strategy": layer_strategy}))
    return str(path)


def test_strategy_returns_deep_copy_and_group_override():
    first = get_optimize_cfg("Quant_INT8-4_ECO")
    second = get_optimize_cfg("Quant_INT8-4_ECO;4")
    first["weight"]["bit"] = 2

    assert get_optimize_cfg("Quant_INT8-4_ECO")["weight"]["bit"] == 4
    assert second["weight"]["group_size"] == 4
    assert get_optimize_cfg("float") == {}


@pytest.mark.parametrize(
    "strategy,match",
    [
        ("unknown", "not support"),
        ("Quant_lm_head", "not support"),
        ("Quant_INT8-4_ECO;bad", "invalid literal"),
    ],
)
def test_strategy_rejects_invalid_values(strategy, match):
    with pytest.raises(Exception, match=match):
        get_optimize_cfg(strategy)


def test_get_target_op_class_uses_type_name_and_fallback(monkeypatch):
    class Gemma4TextScaledWordEmbedding(torch.nn.Embedding):
        pass

    class Unknown(torch.nn.Module):
        pass

    assert do_opt.get_target_op_cls(torch.nn.Linear(2, 2)) is QLinear
    assert do_opt.get_target_op_cls(torch.nn.Embedding(2, 2)) is QEmbedding
    assert do_opt.get_target_op_cls(Gemma4TextScaledWordEmbedding(2, 2)) is QGemma4ScaledEmbedding
    assert do_opt.get_target_op_cls(Unknown()) is None
    assert do_opt.get_target_op_cls(Unknown(), {"target_op": torch.nn.Linear}) is QLinear


def test_generate_config_file_lists_only_supported_layers(tmp_path):
    model = ToyModel()
    output = tmp_path / "layers.json"

    do_opt.generate_config_file(model, output)
    config = json.loads(output.read_text())

    assert set(config["layer_strategy"]) == {"embed", "block.0", "keep"}
    assert all(item["quant_strategy"] == "float" for item in config["layer_strategy"].values())
    with pytest.raises(Exception, match="exist"):
        do_opt.generate_config_file(model, output)


def test_trim_dict_recursively_removes_empty_values():
    value = {"none": None, "empty": {}, "nested": {"drop": None, "keep": 3}, "zero": 0}
    assert do_opt.trim_dict(value) == {"nested": {"keep": 3}, "zero": 0}
    assert do_opt.trim_dict({}) is None
    assert do_opt.trim_dict("value") == "value"


def test_parse_layer_config_merges_overrides_and_nested_lora():
    layer = {
        "quant_strategy": "Quant_INT8-4_ECO;4",
        "weight": {"bit": 3},
        "custom": "kept",
        "lora": {
            "quant_strategy": "Quant_aigc_ptq",
            "input": {"bit": 6},
        },
    }

    parsed = do_opt.parse_layer_config(layer)

    assert parsed["weight"]["group_size"] == 4
    assert parsed["weight"]["bit"] == 3
    assert parsed["custom"] == "kept"
    lora_cfg = parsed["lora"]["optimize_config"]
    assert lora_cfg["weight"]["weight_algo"] == "min_max"
    assert lora_cfg["input"]["bit"] == 6
    assert layer["weight"] == {"bit": 3}


def test_parse_quant_config_filters_float_and_unknown_layers(tmp_path):
    model = ToyModel()
    config_path = write_config(
        tmp_path / "config.json",
        {
            "embed": {"quant_strategy": "float"},
            "block.0": {"quant_strategy": "Quant_INT8-4_ECO;4"},
            "not.in.model": {"quant_strategy": "Quant_INT8-4_ECO;4"},
        },
    )

    parsed = do_opt.paser_quant_config(model, config_path)

    assert set(parsed) == {"block.0"}
    assert parsed["block.0"]["weight"]["group_size"] == 4
    assert do_opt.paser_quant_config(model, None) == {}


def test_get_optimized_op_override_unknown_and_float_paths():
    linear = torch.nn.Linear(4, 3)
    config = {"quant_strategy": "Quant_INT8-4_ECO;4", "layer_name": "proj"}
    wrapped = do_opt.get_optimized_op(linear, config)
    assert isinstance(wrapped, QLinear)
    assert wrapped.layer_name == "proj"

    explicit = do_opt.get_optimized_op(
        linear,
        {"quant_strategy": "Quant_INT8-4_ECO;4", "target_op": torch.nn.Linear},
    )
    assert isinstance(explicit, QLinear)
    assert do_opt.get_optimized_op(linear, {"quant_strategy": "float"}) is linear

    with pytest.raises(Exception, match="bad input type"):
        do_opt.get_optimized_op(torch.nn.ReLU(), {"quant_strategy": "Quant_Dummy"})


def test_optimize_model_replaces_nested_modules_and_keeps_float(tmp_path):
    model = ToyModel()
    original_keep = model.keep
    config_path = write_config(
        tmp_path / "config.json",
        {
            "embed": {"quant_strategy": "Quant_Embed_MinMax"},
            "block.0": {"quant_strategy": "Quant_INT8-4_ECO;4"},
            "keep": {"quant_strategy": "float"},
        },
    )

    optimized = do_opt.optimize_model(model, config_path)

    assert optimized is model
    assert isinstance(model.embed, QEmbedding)
    assert isinstance(model.block[0], QLinear)
    assert model.block[0].layer_name == "block.0"
    assert model.keep is original_keep
    assert do_opt.optimize_model(model, None) is model


def test_replace_module_by_names_handles_deep_names_and_consumes_matches():
    model = ToyModel()
    replacement = torch.nn.Identity()
    pending = {"block.0": replacement, "missing": torch.nn.Identity()}

    result = replace_module_by_names(model, pending)

    assert result is model
    assert model.block[0] is replacement
    assert set(pending) == {"missing"}


def test_state_helpers_propagate_to_quant_op_and_simulator():
    calls = []

    class StateTarget:
        def set_calibrate_state(self, state):
            calls.append(("calibrate", state))

        def set_quant_state(self, weight, input_):
            calls.append(("quant", weight, input_))

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_op = StateTarget()
            self.npu_simulator = StateTarget()

    model = torch.nn.Sequential(Holder(), torch.nn.ReLU())
    do_opt.set_calibrate_state(model, True)
    assert do_opt.set_quant_state(model, False, True) is model

    assert ("calibrate", True) in calls
    assert calls.count(("quant", False, True)) == 2
