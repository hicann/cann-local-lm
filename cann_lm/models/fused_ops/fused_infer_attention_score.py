# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cann_lm::fused_infer_attention_score - fused attention op."""

import torch

from ._library import LIBRARY

LIBRARY.define(
    "fused_infer_attention_score("
    "Tensor query, Tensor key, Tensor value, *, "
    "Tensor? pse_shift=None, Tensor? atten_mask=None, "
    "int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None, "
    "Tensor? dequant_scale1=None, Tensor? quant_scale1=None, "
    "Tensor? dequant_scale2=None, Tensor? quant_scale2=None, "
    "Tensor? quant_offset2=None, Tensor? antiquant_scale=None, "
    "Tensor? antiquant_offset=None, Tensor? block_table=None, "
    "SymInt? query_padding_size=None, SymInt? kv_padding_size=None, "
    "Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, "
    "Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, "
    "Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, "
    "SymInt? actual_shared_prefix_len=None, Tensor? query_rope=None, "
    "Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, "
    "int num_heads=1, float scale=1.0, int pre_tokens=2147483647, "
    "int next_tokens=2147483647, str input_layout=\"BSH\", "
    "int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, "
    "int block_size=0, int antiquant_mode=0, bool softmax_lse_flag=False, "
    "int key_antiquant_mode=0, int value_antiquant_mode=0) -> (Tensor, Tensor)"
)


@torch.library.register_fake("cann_lm::fused_infer_attention_score")
def _fused_infer_attention_score_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    pse_shift: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
    actual_seq_lengths: list[int] | None = None,
    actual_seq_lengths_kv: list[int] | None = None,
    dequant_scale1: torch.Tensor | None = None,
    quant_scale1: torch.Tensor | None = None,
    dequant_scale2: torch.Tensor | None = None,
    quant_scale2: torch.Tensor | None = None,
    quant_offset2: torch.Tensor | None = None,
    antiquant_scale: torch.Tensor | None = None,
    antiquant_offset: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_padding_size: int | None = None,
    kv_padding_size: int | None = None,
    key_antiquant_scale: torch.Tensor | None = None,
    key_antiquant_offset: torch.Tensor | None = None,
    value_antiquant_scale: torch.Tensor | None = None,
    value_antiquant_offset: torch.Tensor | None = None,
    key_shared_prefix: torch.Tensor | None = None,
    value_shared_prefix: torch.Tensor | None = None,
    actual_shared_prefix_len: int | None = None,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    key_rope_antiquant_scale: torch.Tensor | None = None,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    antiquant_mode: int = 0,
    softmax_lse_flag: bool = False,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_heads <= 0:
        raise ValueError(f"num_heads should be greater than 0, but got {num_heads}!")
    del (
        key,
        value,
        pse_shift,
        atten_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        dequant_scale1,
        quant_scale1,
        dequant_scale2,
        quant_offset2,
        antiquant_scale,
        antiquant_offset,
        block_table,
        query_padding_size,
        kv_padding_size,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        key_shared_prefix,
        value_shared_prefix,
        actual_shared_prefix_len,
        key_rope,
        key_rope_antiquant_scale,
        num_heads,
        scale,
        pre_tokens,
        next_tokens,
        input_layout,
        num_key_value_heads,
        sparse_mode,
        inner_precise,
        block_size,
        antiquant_mode,
        key_antiquant_mode,
        value_antiquant_mode,
    )
    if quant_scale2 is not None:
        attention_out = torch.empty_like(query, dtype=torch.int8)
    elif query.dtype == torch.int8:
        if query_rope is not None:
            attention_out = torch.empty_like(query, dtype=query_rope.dtype)
        else:
            attention_out = torch.empty_like(query, dtype=torch.float16)
    else:
        attention_out = torch.empty_like(query, dtype=query.dtype)
    if softmax_lse_flag:
        lse_shape = (query.size(0), query.size(1), query.size(2))
        lse_out = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
    else:
        lse_out = torch.empty([0], dtype=torch.float32, device=query.device)
    return attention_out, lse_out
