# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export-friendly cache implementations shared by CANN model ports."""

from __future__ import annotations

from typing import Any

import torch
import torch.library
import torch_npu
from torch import nn
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicCache,
    LinearAttentionCacheLayerMixin,
    StaticCache,
)


def _scatter_update_cpu(
    input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int
) -> torch.Tensor:
    start = int(indices.reshape(-1)[0])
    length = updates.shape[axis]
    positions = torch.arange(
        start, start + length, dtype=torch.long, device=input.device
    )
    return input.index_copy_(axis, positions, updates)


def _scatter_update_out_cpu(
    input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, axis: int
) -> torch.Tensor:
    return _scatter_update_cpu(input.clone(), indices, updates, axis)


# Native scatter has Meta/PrivateUse1 kernels; supply real CPU execution too.
try:
    torch.library.impl("npu::scatter_update_", "CPU")(_scatter_update_cpu)
    torch.library.impl("npu::scatter_update", "CPU")(_scatter_update_out_cpu)
except RuntimeError:
    pass


class CustomFullAttentionKVCache(nn.Module, CacheLayerMixin):
    is_compileable = True
    supports_early_init = False
    is_sliding = False

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "npu",
    ) -> None:
        nn.Module.__init__(self)
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_context_length
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        cache_shape = (max_batch_size, n_heads, max_context_length, head_dim)
        self.register_buffer(
            "keys", torch.zeros(cache_shape, dtype=dtype, device=device)
        )
        self.register_buffer(
            "values", torch.zeros(cache_shape, dtype=dtype, device=device)
        )
        self.register_buffer(
            "cumulative_length",
            torch.zeros((), dtype=torch.long, device=device),
        )
        self.is_initialized = True

    @property
    def k_cache(self) -> torch.Tensor:
        return self.keys

    @property
    def v_cache(self) -> torch.Tensor:
        return self.values

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_position is None:
            cache_position = self.cumulative_length + torch.arange(
                key_states.shape[-2],
                dtype=torch.long,
                device=key_states.device,
            )

        if self.keys.device.type == "cpu" and not torch.compiler.is_compiling():
            # Eager CPU execution supplies the reference cache semantics.
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        else:
            # Fake/Meta export keeps the native operator in the captured graph.
            torch_npu.scatter_update_(self.keys, cache_position[:1], key_states, 2)
            torch_npu.scatter_update_(self.values, cache_position[:1], value_states, 2)
        self.cumulative_length.copy_(cache_position[-1] + 1)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.max_cache_len, 0

    def get_seq_length(self) -> int | torch.Tensor:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.cumulative_length.zero_()


class _Qwen3_5LinearAttentionLayer(nn.Module, LinearAttentionCacheLayerMixin):
    is_compileable = True
    supports_early_init = False

    def __init__(
        self,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_kernel_size: int,
    ) -> None:
        nn.Module.__init__(self)
        self.register_buffer("conv_state_0", conv_state)
        self.register_buffer("recurrent_state_0", recurrent_state)
        self.number_of_states = 1
        self.is_conv_states_initialized = {0: True}
        self.is_recurrent_states_initialized = {0: True}
        self.has_previous_state = {0: False}
        self.conv_kernel_size = {0: conv_kernel_size}
        self.device = conv_state.device
        self.dtype = conv_state.dtype
        self.record_past = False

    @property
    def conv_states(self) -> dict[int, torch.Tensor]:
        return {0: self.conv_state_0}

    @property
    def recurrent_states(self) -> dict[int, torch.Tensor]:
        return {0: self.recurrent_state_0}

    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
    ) -> None:
        if state_idx != 0:
            raise ValueError("Qwen3.5 only uses state index 0")

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if state_idx != 0:
            raise ValueError("Qwen3.5 only uses state index 0")
        if not self.has_previous_state[0]:
            full_conv_states = conv_states
            self.has_previous_state[0] = True
            if full_conv_states.shape[-1] < self.conv_kernel_size[0]:
                padding = self.conv_kernel_size[0] - full_conv_states.shape[-1]
                full_conv_states = torch.nn.functional.pad(
                    full_conv_states, (padding, 0)
                )
        else:
            full_conv_states = torch.cat((self.conv_state_0, conv_states), dim=-1)
        self.conv_state_0.copy_(full_conv_states[..., -self.conv_kernel_size[0] :])
        return full_conv_states

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        state_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        if state_idx != 0:
            raise ValueError("Qwen3.5 only uses state index 0")
        # Keep the BF16 state dtype explicit across export functionalization.
        self.recurrent_state_0.copy_(
            recurrent_states.to(dtype=self.recurrent_state_0.dtype)
        )
        return self.recurrent_state_0

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return -1, 0

    def get_seq_length(self) -> int:
        return 0

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        self.conv_state_0.zero_()
        self.recurrent_state_0.zero_()
        self.has_previous_state[0] = False


class Qwen3_5CustomKVCache(nn.Module, DynamicCache):
    """Qwen3.5 cache whose tensors have one registered module owner."""

    def __init__(
        self,
        config: Any,
        *,
        max_batch_size: int,
        max_context_length: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "npu",
    ) -> None:
        nn.Module.__init__(self)
        DynamicCache.__init__(self, config=config)
        self.max_seq_len = max_context_length
        self.registered_layers = nn.ModuleDict()

        text_config = config.get_text_config(decoder=True)
        key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        value_dim = (
            text_config.linear_num_value_heads * text_config.linear_value_head_dim
        )
        conv_dim = key_dim * 2 + value_dim

        for layer_idx, layer_type in enumerate(text_config.layer_types):
            if layer_type == "full_attention":
                layer_config = text_config.per_layer_config[layer_idx]
                layer = CustomFullAttentionKVCache(
                    max_batch_size=max_batch_size,
                    max_context_length=max_context_length,
                    n_heads=layer_config.num_key_value_heads,
                    head_dim=layer_config.head_dim,
                    dtype=dtype,
                    device=device,
                )
                self.layers[layer_idx] = layer
                self.registered_layers[str(layer_idx)] = layer
            elif layer_type == "linear_attention":
                layer = _Qwen3_5LinearAttentionLayer(
                    conv_state=torch.zeros(
                        max_batch_size,
                        conv_dim,
                        text_config.linear_conv_kernel_dim,
                        dtype=dtype,
                        device=device,
                    ),
                    # Keep recurrent state in the native GDN layout [B, H, V, K].
                    recurrent_state=torch.zeros(
                        max_batch_size,
                        text_config.linear_num_value_heads,
                        text_config.linear_value_head_dim,
                        text_config.linear_key_head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    conv_kernel_size=text_config.linear_conv_kernel_dim,
                )
                self.layers[layer_idx] = layer
                self.registered_layers[str(layer_idx)] = layer
            else:
                raise ValueError(f"unsupported Qwen3.5 layer type: {layer_type}")


class Gemma4CustomKVCache(nn.Module, StaticCache):
    """Static K/V cache shared by Gemma4 prefill and decode."""

    def __init__(
        self,
        config: Any,
        *,
        max_batch_size: int,
        max_cache_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "npu",
    ) -> None:
        nn.Module.__init__(self)
        StaticCache.__init__(self, config=config, max_cache_len=max_cache_len)
        text_config = config.get_text_config(decoder=True)
        for layer_idx in range(len(self.layers)):
            layer_config = text_config.per_layer_config[layer_idx]
            self.layers[layer_idx] = CustomFullAttentionKVCache(
                max_batch_size=max_batch_size,
                max_context_length=max_cache_len,
                n_heads=layer_config.num_key_value_heads,
                head_dim=layer_config.head_dim,
                dtype=dtype,
                device=device,
            )
            self.layers[layer_idx].is_sliding = (
                text_config.layer_types[layer_idx] == "sliding_attention"
            )
        self.layers = nn.ModuleList(self.layers)
