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

"""npu::npu_fused_infer_attention_score_v2 GE converter (从 ge_converter.py 抽出).

gemma4 attention 融合算子. cann_lm 自注册到 npu 命名空间 (见
``fused_ops/fused_infer_attention_score.py``); torchair 的 fused_infer_attention_score_v2 converter
在 air 命名空间 (不适用), 故 cann_lm 自己注册 npu 命名空间 converter.

调 ``ge.FusedInferAttentionScore`` (GE IR), 参数映射: num_query_heads->num_heads,
softmax_scale->scale, actual_seq_qlen->actual_seq_lengths, return_softmax_lse->softmax_lse_flag.
简化: 只传 gemma4 用的参数, quant/antiquant/rope/shared_prefix 等不传 (None).

注册由 ``ge_converter.py:register_ge_converters`` 调
``register_fx_node_ge_converter(torch.ops.npu.npu_fused_infer_attention_score_v2.default)(convert_fused_infer_attention_score_v2)``.
"""
from typing import Any


def convert_fused_infer_attention_score_v2(
    query: Any,
    key: Any,
    value: Any,
    *,
    atten_mask: Any = None,
    actual_seq_qlen: Any = None,
    actual_seq_kvlen: Any = None,
    block_table: Any = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    softmax_scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "BSH",
    sparse_mode: int = 0,
    block_size: int = 0,
    inner_precise: int = 0,
    return_softmax_lse: bool = False,
    meta_outputs: Any = None,
) -> Any:
    """npu_fused_infer_attention_score_v2 -> ge.FusedInferAttentionScore (只传 gemma4 用的参数)."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge
    out, lse = ge.FusedInferAttentionScore(
        query,
        [key],
        [value],
        pse_shift=None,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        dequant_scale1=None,
        quant_scale1=None,
        dequant_scale2=None,
        quant_scale2=None,
        quant_offset2=None,
        antiquant_scale=None,
        antiquant_offset=None,
        block_table=block_table,
        query_padding_size=None,
        kv_padding_size=None,
        key_antiquant_scale=None,
        key_antiquant_offset=None,
        value_antiquant_scale=None,
        value_antiquant_offset=None,
        key_shared_prefix=None,
        value_shared_prefix=None,
        actual_shared_prefix_len=None,
        query_rope=None,
        key_rope=None,
        key_rope_antiquant_scale=None,
        dequant_scale_query=None,
        learnable_sink=None,
        q_start_idx=None,
        kv_start_idx=None,
        num_heads=num_query_heads,
        scale=softmax_scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=0,
        softmax_lse_flag=return_softmax_lse,
        key_antiquant_mode=0,
        value_antiquant_mode=0,
        query_quant_mode=0,
        pse_type=0,
        out_dtype=0,
    )
    return out, lse
