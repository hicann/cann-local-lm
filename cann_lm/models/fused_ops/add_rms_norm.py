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

"""npu::npu_add_rms_norm - fused add + RMSNorm op (schema + meta + CPU eager kernel).

注册到 npu 命名空间 (torch.ops.npu.npu_add_rms_norm), torchair custom/ 的
npu_add_rms_norm converter 自动注册 (ge.AddRmsNorm).
"""

import torch

from ._library import NPU_LIB, safe_define, _HAS_TORCH_NPU, register_cpu_impl

if NPU_LIB is not None:
    safe_define(
        "npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-6) "
        "-> (Tensor y, Tensor rstd, Tensor residual)"
    )

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("npu::npu_add_rms_norm")
    def _add_rms_norm_meta(
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del x2, gamma, epsilon
        y = torch.empty_like(x1)
        rstd = x1.new_empty(x1.shape[:-1] + (1,), dtype=torch.float32)
        residual = torch.empty_like(x1)
        return y, rstd, residual

    @register_cpu_impl("npu::npu_add_rms_norm")
    def _add_rms_norm_cpu(
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = x1 + x2
        ms = s.float().pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(ms + epsilon)
        y = ((s.float() * rstd) * gamma.float()).to(s.dtype)
        return y, rstd, s
