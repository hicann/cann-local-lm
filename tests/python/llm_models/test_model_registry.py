"""model_registry 单元测试 — 模型名→config/model/wrapper 注册表 + alias + 自动检测.

无需 NPU/权重: 只验注册表结构、alias 解析、get_entry/detect_model_type/available_models.
"""

import pytest

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from cann_lm.models.model_registry import (
    REGISTRY,
    available_models,
    detect_model_type,
    get_entry,
)
from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper


def test_available_models_covers_two_models():
    assert set(available_models()) == {"gemma4", "qwen3_5"}
    assert all(k in REGISTRY for k in available_models())


@pytest.mark.parametrize(
    "name,expected_key",
    [
        ("gemma4", "gemma4"),
        ("gemma-4", "gemma4"),
        ("Gemma4", "gemma4"),  # 大小写不敏感
        ("GEMMA-4", "gemma4"),
        ("qwen3_5", "qwen3_5"),
        ("qwen3.5", "qwen3_5"),
        ("Qwen3.5", "qwen3_5"),
    ],
)
def test_get_entry_resolves_aliases(name, expected_key):
    assert get_entry(name) is REGISTRY[expected_key]


@pytest.mark.parametrize("bad_name", ["unknown", "llama", "", "gemma5", "qwen4"])
def test_get_entry_unknown_raises(bad_name):
    with pytest.raises(ValueError, match="Unknown model"):
        get_entry(bad_name)


@pytest.mark.parametrize(
    "name,config_cls,model_cls,wrapper_cls,json_key",
    [
        (
            "gemma4",
            Gemma4TextConfig,
            Gemma4ForCausalLM,
            Gemma4ExportWrapper,
            "text_config",
        ),
        (
            "qwen3_5",
            Qwen3_5TextConfig,
            Qwen3_5ForCausalLM,
            Qwen3_5ExportWrapper,
            "text_config",
        ),
    ],
)
def test_entry_fields(name, config_cls, model_cls, wrapper_cls, json_key):
    e = get_entry(name)
    assert e.config_cls is config_cls
    assert e.model_cls is model_cls
    assert e.wrapper_cls is wrapper_cls
    assert e.config_json_key == json_key
    assert callable(e.load_weights_fn)
    assert e.model_type  # 非空 (gemma4_text / qwen3_5_text)


@pytest.mark.parametrize(
    "model_type,expected_key",
    [
        ("gemma4", "gemma4"),  # gemma4 HF config 顶层 model_type
        ("qwen3_5", "qwen3_5"),  # qwen3.5 HF config 顶层 model_type
    ],
)
def test_detect_model_type_from_hf_config(model_type, expected_key):
    e = detect_model_type({"model_type": model_type})
    assert e is REGISTRY[expected_key]


def test_detect_model_type_missing_field_raises():
    with pytest.raises(ValueError, match="model_type"):
        detect_model_type({})


def test_detect_model_type_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        detect_model_type({"model_type": "llama"})
