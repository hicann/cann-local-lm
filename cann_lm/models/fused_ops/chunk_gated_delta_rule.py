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

"""cann_lm::chunk_gated_delta_rule - chunked gated delta rule op (schema + meta + CPU impl).

Qwen3.5 线性注意力 (GatedDeltaNet) 的 prefill + decode 核心算子. 融合 chunked gated delta
rule 全流程 (L2norm + chunk decay + intra-chunk attention + inter-chunk recurrence +
state update), 替代纯 torch 的 ~100 行 Python for 循环 (会导致 torch.export 展开约
19418 节点 -> export_to_epair GE 转换图爆炸 segfault).

GE 映射: ge.ChunkGatedDeltaRule (手动定义在 ge_converter.py, 因 torchair auto_generated
未生成此算子 + npu_chunk_gated_delta_rule converter 直接 raise RuntimeError).
CANN op_info: aic-ascend910_93-ops-info-transformer.json "ChunkGatedDeltaRule"
  inputs: query/key/value/beta/initial_state(bf16) + actual_seq_lengths(int32) + g(float32, optional)
  outputs: out(bf16)/final_state(bf16)
  attr: scale_value(float, optional, default 1)

CPU impl 复用 torch_chunk_gated_delta_rule (chunked 矩阵化, 比 recurrent 逐 token 快),
做 BHSD<->BSHD + state 维度转换. prefill + decode 共用 (decode seq_len=1 时 chunk 内 pad,
浪费 63x 计算但结果正确).
"""

import torch

from ._library import LIBRARY

LIBRARY.define(
    "chunk_gated_delta_rule("
    "Tensor query, Tensor key, Tensor value, Tensor beta, "
    "Tensor initial_state, Tensor actual_seq_lengths, "
    "Tensor? g=None, *, float scale_value=1.0"
    ") -> (Tensor out, Tensor final_state)"
)


@torch.library.register_fake("cann_lm::chunk_gated_delta_rule")
def _chunk_gated_delta_rule_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    scale_value: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del key, beta, actual_seq_lengths, g, scale_value
    # out: [B, H, seq, v_dim] (query 前几维 + value 最后一维)
    out = query.new_empty(query.shape[:-1] + (value.shape[-1],))
    # final_state: like initial_state [B, H, v_dim, k_dim]
    final_state = initial_state.new_empty(initial_state.shape)
    return out, final_state


@torch.library.impl("cann_lm::chunk_gated_delta_rule", "cpu")
def _chunk_gated_delta_rule_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    scale_value: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU eager: 转换 BHSD -> BSHD 调用纯 torch torch_chunk_gated_delta_rule.

    NPU 算子签名 (BHSD): query/key/value [B,H,seq,dim], beta/g [B,H,seq],
    initial_state [B,H,v_dim,k_dim]. 纯 torch torch_chunk 期望 BSHD ([B,seq,H,dim]) +
    initial_state [B,H,k_dim,v_dim], 故需 transpose.
    """
    del actual_seq_lengths, scale_value
    from cann_lm.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    # BHSD -> BSHD (transpose dim 1,2)
    q_bshd = query.transpose(1, 2)
    k_bshd = key.transpose(1, 2)
    v_bshd = value.transpose(1, 2)
    beta_bshd = beta.transpose(1, 2)
    g_bshd = g.transpose(1, 2) if g is not None else None
    # initial_state NPU [B,H,v_dim,k_dim] -> torch_chunk [B,H,k_dim,v_dim]
    init_bshd = initial_state.transpose(-1, -2)

    out_bshd, final_state = torch_chunk_gated_delta_rule(
        q_bshd, k_bshd, v_bshd, g=g_bshd, beta=beta_bshd,
        initial_state=init_bshd,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # out BSHD [B,seq,H,v_dim] -> BHSD [B,H,seq,v_dim]
    out = out_bshd.transpose(1, 2).contiguous()
    # final_state torch_chunk [B,H,k_dim,v_dim] -> NPU [B,H,v_dim,k_dim]
    final_state_npu = final_state.transpose(-1, -2).contiguous() if final_state is not None else initial_state
    return out, final_state_npu
