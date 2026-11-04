# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Gemma4 graph wrappers and attention-mask policy."""

import torch
from torch import nn

from examples.common import Gemma4CustomKVCache


def build_mask(
    length: int,
    kv_len: int,
    window: int | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key_index = torch.arange(kv_len, device=device).unsqueeze(0)
    query_index = torch.arange(length, device=device).unsqueeze(1)
    allowed = key_index <= query_index
    if window:
        allowed = allowed & (key_index + window > query_index)
    mask = torch.zeros((1, 1, length, kv_len), dtype=dtype, device=device)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, cache: nn.Module) -> None:
        super().__init__()
        parameter = next(model.parameters())
        model.model.rotary_emb.prepare_cache(
            cache.layers[0].max_cache_len, parameter.device, parameter.dtype
        )
        self.model = model
        self.cache = cache
        text_config = model.config.get_text_config(decoder=True)
        self.layer_types = sorted(set(text_config.layer_types))

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        sliding_mask: torch.Tensor | None = None,
        full_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_cache in self.cache.layers:
            layer_cache.cumulative_length.copy_(cache_position[0])
        # External masks avoid GE's dynamic causal/sliding composition error.
        attention_mask = None
        if sliding_mask is not None or full_mask is not None:
            attention_mask = {
                layer_type: sliding_mask
                if layer_type == "sliding_attention"
                else full_mask
                for layer_type in self.layer_types
            }
        logits = self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            attention_mask=attention_mask,
        ).logits
        # Split export fixes batch size to one and scores the last token only.
        # Remove both singleton axes so GE exposes a fixed-size vocabulary output.
        return logits.select(0, 0).select(0, 0)


class UnifiedExportWrapper(nn.Module):
    """Legacy single-method export contract."""

    def __init__(self, model: nn.Module, cache: nn.Module) -> None:
        super().__init__()
        parameter = next(model.parameters())
        model.model.rotary_emb.prepare_cache(
            cache.layers[0].max_cache_len, parameter.device, parameter.dtype
        )
        self.model = model
        self.cache = cache

    def forward(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        for layer_cache in self.cache.layers:
            layer_cache.cumulative_length.copy_(cache_position[0])
        return self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        ).logits


def create_cache(
    config, *, max_seq_len: int, device: torch.device, dtype: torch.dtype
) -> Gemma4CustomKVCache:
    return Gemma4CustomKVCache(
        config=config,
        max_batch_size=1,
        max_cache_len=max_seq_len,
        dtype=dtype,
        device=device,
    )
