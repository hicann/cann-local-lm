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

from ._library import NPU_LIB, _HAS_TORCH_NPU, register_cpu_impl

if NPU_LIB is not None:
    NPU_LIB.define(
        "npu_chunk_gated_delta_rule("
        "Tensor query, Tensor key, Tensor value, *, "
        "Tensor? beta=None, Tensor? initial_state=None, "
        "Tensor? actual_seq_lengths=None, float? scale=None, Tensor? g=None"
        ") -> (Tensor out, Tensor final_state)"
    )

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("npu::npu_chunk_gated_delta_rule")
    def _chunk_gated_delta_rule_meta(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        actual_seq_lengths: torch.Tensor | None = None,
        scale: float | None = None,
        g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del key, beta, actual_seq_lengths, g, scale
        # out: THD (T, Nv, Dv) — T from query[0], Nv/Dv from value
        out = query.new_empty((query.shape[0], value.shape[1], value.shape[2]))
        # final_state: like initial_state (B, Nv, Dv, Dk)
        final_state = initial_state.new_empty(initial_state.shape)
        return out, final_state


@register_cpu_impl("npu::npu_chunk_gated_delta_rule")
def _chunk_gated_delta_rule_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    scale: float | None = None,
    g: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU eager: 转换 THD -> BSHD 调用纯 torch torch_chunk_gated_delta_rule.

    算子签名 (THD, 与 npu_recurrent_gated_delta_rule 一致):
      query/key (T, Nk, Dk), value (T, Nv, Dv), beta/g (T, Nv),
      initial_state (B, Nv, Dv, Dk). T = sum batch seq len (B*seq 拼接).
    纯 torch torch_chunk 期望 BSHD ([B,seq,H,dim]) + initial_state [B,H,k_dim,v_dim],
    故 THD -> BSHD reshape (T=B*seq) + initial_state transpose(-1,-2).
    """
    del scale
    from cann_lm.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    # THD (T,N,D) -> BSHD (B,seq,N,D): T=B*seq, 从 actual_seq_lengths 推 B
    b = actual_seq_lengths.shape[0] if actual_seq_lengths is not None else 1
    t = query.shape[0]
    seq = t // b
    q_bshd = query.reshape(b, seq, query.shape[1], query.shape[2])
    k_bshd = key.reshape(b, seq, key.shape[1], key.shape[2])
    v_bshd = value.reshape(b, seq, value.shape[1], value.shape[2])
    beta_bshd = beta.reshape(b, seq, beta.shape[1]) if beta is not None else None
    g_bshd = g.reshape(b, seq, g.shape[1]) if g is not None else None
    # initial_state NPU (B,Nv,Dv,Dk) -> torch_chunk (B,Nv,Dk,Dv)
    init_bshd = initial_state.transpose(-1, -2)

    out_bshd, final_state = torch_chunk_gated_delta_rule(
        q_bshd, k_bshd, v_bshd, g=g_bshd, beta=beta_bshd,
        initial_state=init_bshd,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        apply_scale=True,
    )

    # out BSHD (B,seq,Nv,Dv) -> THD (T,Nv,Dv)
    out = out_bshd.reshape(t, out_bshd.shape[2], out_bshd.shape[3])
    # final_state torch_chunk (B,Nv,Dk,Dv) -> NPU (B,Nv,Dv,Dk)
    final_state_npu = final_state.transpose(-1, -2).contiguous() if final_state is not None else initial_state
    return out, final_state_npu
