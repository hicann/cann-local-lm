# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Qwen3.5 graph wrapper, causal mask, and mixed-layer Cache policy."""

import torch
from torch import nn

from examples.common import Qwen3_5CustomKVCache


def build_causal_mask(
    length: int, kv_len: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key_index = torch.arange(kv_len, device=device).unsqueeze(0)
    query_index = torch.arange(length, device=device).unsqueeze(1)
    allowed = key_index <= query_index
    mask = torch.zeros((1, 1, length, kv_len), dtype=dtype, device=device)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, cache: nn.Module) -> None:
        super().__init__()
        model.model.prepare_for_inference()
        self.model = model
        self.cache = cache
        self.layer_types = tuple(model.config.layer_types)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        full_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                self.cache.layers[layer_idx].cumulative_length.copy_(cache_position[0])
        attention_mask = (
            {"full_attention": full_mask, "linear_attention": None}
            if full_mask is not None
            else None
        )
        # Text-only M-RoPE positions follow the explicit cache positions.
        position_ids = cache_position.reshape(1, 1, -1).expand(
            3, input_ids.shape[0], -1
        )
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
            attention_mask=attention_mask,
        ).logits
        # Split export uses batch size one and scores the last token only.
        return logits.select(0, 0).select(0, 0)


def create_cache(
    config,
    *,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    decode: bool = False,
) -> Qwen3_5CustomKVCache:
    cache = Qwen3_5CustomKVCache(
        config=config,
        max_batch_size=1,
        max_context_length=max_seq_len,
        dtype=dtype,
        device=device,
    )
    if decode:
        text_config = config.get_text_config(decoder=True)
        for layer_idx, layer_type in enumerate(text_config.layer_types):
            if layer_type != "linear_attention":
                continue
            for state_idx in cache.layers[layer_idx].has_previous_state:
                cache.layers[layer_idx].has_previous_state[state_idx] = True
    return cache
