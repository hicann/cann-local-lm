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

"""npu::npu_apply_rotary_pos_emb - fused q+k paired rotate-half RoPE (schema + meta + CPU eager kernel).

注册到 npu 命名空间 (torch.ops.npu.npu_apply_rotary_pos_emb), torchair custom/ 的
npu_apply_rotary_pos_emb converter 自动注册 (ge.ApplyRotaryPosEmb). stubs 环境
(无 torch_npu) 自定义 schema + fake + CPU impl; 真实 torch_npu 由 torch_npu 提供 schema.

与 cann_lm::rotary_mul 的差异: 本算子 q+k 配对 (返回 query_out, key_out 二元组),
in-place 风格 (torchair converter 内部 ge.TensorMove 保护输入); layout 属性决定 NPU
kernel 的张量排布解释 (BSH/BSND/SBND/BNSD/TND), CPU eager 无视 layout 直接 element-wise.
用于 gemma4 full 层 proportional partial rotary (q_rot/k_rot cat/拆装后配对调用).
"""

import torch

from ._library import NPU_LIB, _HAS_TORCH_NPU, register_cpu_impl

if NPU_LIB is not None:
    NPU_LIB.define(
        "npu_apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, "
        "str layout='BSH') -> (Tensor query_out, Tensor key_out)"
    )

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("npu::npu_apply_rotary_pos_emb")
    def _apply_rotary_pos_emb_meta(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: str = "BSH",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cos, sin, layout
        return torch.empty_like(query), torch.empty_like(key)

    @register_cpu_impl("npu::npu_apply_rotary_pos_emb")
    def _apply_rotary_pos_emb_cpu(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layout: str = "BSH",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layout
        q_out = _rope_half(query, cos, sin)
        k_out = _rope_half(key, cos, sin)
        return q_out.to(query.dtype), k_out.to(key.dtype)


def _rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """half mode RoPE: out = x*cos + rotate_half(x)*sin; rotate_half = cat(-x2, x1).

    与 cann_lm::rotary_mul._rotary_mul_cpu 语义一致 (CPU eager 路径).
    layout 不影响 CPU 计算语义 (只影响 NPU kernel 的张量排布解释), 故无视.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)
