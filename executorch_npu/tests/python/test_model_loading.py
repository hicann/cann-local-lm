# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import json

import pytest
import torch


pytest.importorskip(
    "transformers.models.qwen3_5",
    reason="Transformers Qwen3.5 implementation is not importable",
)

from examples.auto import AutoConfig, AutoModelForCausalLM
from examples.gemma4 import (
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Gemma4TextModel,
)
from examples.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)


def _tiny_gemma_text_config():
    return Gemma4TextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
    )


def _tiny_gemma_composite_config():
    return Gemma4Config(
        text_config=_tiny_gemma_text_config().to_dict(),
        vision_config=None,
        audio_config=None,
    )


def _tiny_text_config():
    return Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 2,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention", "full_attention"],
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
        },
    ).text_config


def _tiny_composite_config():
    return Qwen3_5Config(
        text_config=_tiny_text_config().to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0},
            "head_dim": 8,
        },
    )


def test_composite_model_uses_transformers_submodels() -> None:
    config = _tiny_composite_config()
    model = Qwen3_5Model(config)

    assert type(config).__module__ == "examples.qwen3_5.model.configuration"
    assert type(model.visual).__name__ == Qwen3_5VisionModel.__name__
    assert type(model.language_model).__name__ == Qwen3_5TextModel.__name__
    assert type(model.visual).__module__.startswith("transformers.models.qwen3_5")
    assert type(model.language_model).__module__.startswith(
        "transformers.models.qwen3_5"
    )


def test_auto_config_uses_local_gemma4_config(tmp_path) -> None:
    _tiny_gemma_composite_config().save_pretrained(tmp_path)

    loaded = AutoConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, Gemma4Config)
    assert isinstance(loaded.text_config, Gemma4TextConfig)
    assert type(loaded).__module__ == "examples.gemma4.model.configuration"
    assert type(loaded.text_config).__module__ == "examples.gemma4.model.configuration"


def test_auto_model_loads_text_weights_from_composite_gemma4_checkpoint(
    tmp_path,
) -> None:
    torch.manual_seed(0)
    original = Gemma4ForCausalLM(_tiny_gemma_text_config()).eval()
    _tiny_gemma_composite_config().save_pretrained(tmp_path)
    state_dict = {
        (
            f"model.language_model.{name.removeprefix('model.')}"
            if name.startswith("model.")
            else name
        ): value
        for name, value in original.state_dict().items()
    }
    state_dict["model.vision_tower.unused.weight"] = torch.zeros(1)
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    loaded, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        local_files_only=True,
        use_safetensors=False,
        strict=True,
        output_loading_info=True,
    )

    assert isinstance(loaded, Gemma4ForCausalLM)
    assert isinstance(loaded.config, Gemma4TextConfig)
    assert isinstance(loaded.model, Gemma4TextModel)
    assert type(loaded).__module__ == "examples.gemma4.model.modeling"
    assert type(loaded.config).__module__ == "examples.gemma4.model.configuration"
    assert all(not loading_info[name] for name in loading_info)
    for name, expected in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected), name


def test_auto_model_loads_sharded_qwen_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    original = Qwen3_5ForCausalLM(_tiny_text_config()).eval()
    original.save_pretrained(
        tmp_path,
        safe_serialization=True,
        max_shard_size="20KB",
    )

    loaded, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        local_files_only=True,
        strict=True,
        output_loading_info=True,
    )

    assert isinstance(loaded, Qwen3_5ForCausalLM)
    assert isinstance(loaded.config, Qwen3_5TextConfig)
    assert isinstance(loaded.model, Qwen3_5TextModel)
    assert type(loaded).__module__ == "examples.qwen3_5.model.modeling"
    assert type(loaded.config).__module__ == "examples.qwen3_5.model.configuration"
    assert not loaded.training
    assert all(not loading_info[name] for name in loading_info)
    for name, expected in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected), name


def test_auto_model_loads_text_weights_from_composite_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    text_config = _tiny_text_config()
    original = Qwen3_5ForCausalLM(text_config).eval()
    composite_config = _tiny_composite_config()
    composite_config.save_pretrained(tmp_path)
    state_dict = {
        (
            f"model.language_model.{name.removeprefix('model.')}"
            if name.startswith("model.")
            else name
        ): value
        for name, value in original.state_dict().items()
    }
    state_dict["model.visual.unused.weight"] = torch.zeros(1)
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    loaded, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        local_files_only=True,
        use_safetensors=False,
        strict=True,
        output_loading_info=True,
    )

    assert isinstance(loaded, Qwen3_5ForCausalLM)
    assert all(not loading_info[name] for name in loading_info)
    for name, expected in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected), name


def test_strict_loading_rejects_missing_weights(tmp_path) -> None:
    model = Qwen3_5ForCausalLM(_tiny_text_config()).eval()
    model.config.save_pretrained(tmp_path)
    state_dict = model.state_dict()
    missing_name = next(iter(state_dict))
    state_dict.pop(missing_name)
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    with pytest.raises(RuntimeError, match="missing_keys"):
        AutoModelForCausalLM.from_pretrained(
            tmp_path,
            dtype=torch.float32,
            local_files_only=True,
            use_safetensors=False,
            strict=True,
        )


def test_auto_config_rejects_unsupported_model_type(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "unsupported_for_executorch_npu"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="does not recognize this architecture|Unsupported model type",
    ):
        AutoConfig.from_pretrained(tmp_path, local_files_only=True)
