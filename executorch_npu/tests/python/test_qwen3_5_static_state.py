# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import torch

from examples.common import Qwen3_5CustomKVCache
from examples.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
from examples.qwen3_5.export.adapter import ExportWrapper


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
            # The production fused GDN path requires 128-wide state heads.
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
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


def _cache(max_seq_len: int = 16) -> Qwen3_5CustomKVCache:
    return Qwen3_5CustomKVCache(
        _tiny_text_config(),
        max_batch_size=1,
        max_context_length=max_seq_len,
        device="cpu",
        dtype=torch.float32,
    )


def test_cache_is_the_only_registered_state_owner() -> None:
    cache = _cache()
    named_buffers = list(cache.named_buffers(remove_duplicate=False))

    assert {name for name, _ in named_buffers} == {
        "registered_layers.0.conv_state_0",
        "registered_layers.0.recurrent_state_0",
        "registered_layers.1.keys",
        "registered_layers.1.values",
        "registered_layers.1.cumulative_length",
    }
    assert len({id(buffer) for _, buffer in named_buffers}) == len(named_buffers)
    assert cache.layers[0] is cache.registered_layers["0"]
    assert cache.layers[1] is cache.registered_layers["1"]


def test_export_wrapper_does_not_register_buffer_aliases() -> None:
    model = Qwen3_5ForCausalLM(_tiny_text_config()).eval()
    cache = _cache()
    wrapper = ExportWrapper(model, cache)

    assert not wrapper._buffers
    assert wrapper.cache is cache
    assert all(
        name.startswith("cache.")
        for name, _ in wrapper.named_buffers()
        if "registered_layers" in name
    )


def test_full_attention_cache_cpu_update_uses_requested_positions() -> None:
    cache = _cache(max_seq_len=8)
    key = torch.arange(1 * 2 * 3 * 8, dtype=torch.float32).reshape(1, 2, 3, 8)
    value = key + 1000

    returned_key, returned_value = cache.layers[1].update(
        key,
        value,
        cache_position=torch.tensor([2, 3, 4]),
    )

    assert torch.equal(returned_key[:, :, 2:5], key)
    assert torch.equal(returned_value[:, :, 2:5], value)
    assert cache.layers[1].get_seq_length().item() == 5


def test_decode_linear_cache_can_reuse_previous_state() -> None:
    cache = _cache()
    for state_idx in cache.layers[0].has_previous_state:
        cache.layers[0].has_previous_state[state_idx] = True

    assert all(cache.layers[0].has_previous_state.values())


def test_exported_cache_state_survives_aot_decomposition() -> None:
    model = Qwen3_5ForCausalLM(_tiny_text_config()).to(torch.bfloat16).eval()
    wrapper = ExportWrapper(model, _cache(max_seq_len=8).to(torch.bfloat16)).eval()

    exported = torch.export.export(
        wrapper,
        args=(),
        kwargs={
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "cache_position": torch.arange(4),
        },
        dynamic_shapes=None,
        strict=False,
    )
    decomposed = exported.run_decompositions({})

    assert len(decomposed.graph_signature.buffers_to_mutate) > 0


def test_cpu_cache_export_keeps_native_scatter_and_state_mutations():
    class CacheUpdate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = _cache(max_seq_len=8)

        def forward(self, key, value, positions):
            return self.cache.layers[1].update(key, value, cache_position=positions)

    key = torch.randn(1, 2, 3, 8)
    exported = torch.export.export(
        CacheUpdate(), (key, key + 1, torch.arange(2, 5)), strict=False
    )
    targets = [
        str(node.target) for node in exported.graph.nodes if node.op == "call_function"
    ]
    assert targets.count("npu.scatter_update_.default") == 2
    assert exported.run_decompositions({}).graph_signature.buffers_to_mutate
