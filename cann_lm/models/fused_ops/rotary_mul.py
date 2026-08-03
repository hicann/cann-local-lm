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

"""npu::npu_rotary_mul - fused rotate-half RoPE (schema + meta + CPU eager kernel).

注册到 npu 命名空间 (torch.ops.npu.npu_rotary_mul), torchair custom/ 的
npu_rotary_mul converter 自动注册 (arch35 -> ge.RotaryPositionEmbedding,
旧架构 -> ge.RotaryMul). cann_lm CPU impl 仅支持 rotary_mode='half'.
"""

import torch

from ._library import NPU_LIB, _HAS_TORCH_NPU

if NPU_LIB is not None:
    NPU_LIB.define(
        "npu_rotary_mul(Tensor self, Tensor r1, Tensor r2, str rotary_mode=\"half\") -> Tensor out"
    )

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("npu::npu_rotary_mul")
    def _rotary_mul_meta(
        self: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        rotary_mode: str = "half",
    ) -> torch.Tensor:
        del r1, r2, rotary_mode
        return torch.empty_like(self)

    @torch.library.impl("npu::npu_rotary_mul", "cpu")
    def _rotary_mul_cpu(
        self: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        rotary_mode: str = "half",
    ) -> torch.Tensor:
        if rotary_mode != "half":
            raise NotImplementedError(
                f"rotary_mode={rotary_mode!r} not supported on CPU; only 'half'"
            )
        x1 = self[..., : self.shape[-1] // 2]
        x2 = self[..., self.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        out = (self * r1) + (rotated * r2)
        return out.to(self.dtype)
