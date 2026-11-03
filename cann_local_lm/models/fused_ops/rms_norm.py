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

"""cann_local_lm::rms_norm - RMSNorm op (schema + meta + CPU eager kernel).

注册到 npu 命名空间 (torch.ops.cann_local_lm.rms_norm), torchair custom/ 的
npu_rms_norm converter 自动注册 (ge.RmsNorm). stubs 环境 (无 torch_npu)
自定义 schema + fake + CPU impl; 真实 torch_npu 由 torch_npu 提供 schema.
"""

import torch

from ._library import NPU_LIB, safe_define, _HAS_TORCH_NPU, register_cpu_impl

if NPU_LIB is not None:
    safe_define(
        "rms_norm(Tensor self, Tensor gamma, float epsilon=1e-6) -> (Tensor y, Tensor rstd)"
    )

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("cann_local_lm::rms_norm")
    def _rms_norm_meta(
        self: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del gamma, epsilon
        y = torch.empty_like(self)
        rstd = self.new_empty(self.shape[:-1] + (1,), dtype=torch.float32)
        return y, rstd

    @register_cpu_impl("cann_local_lm::rms_norm")
    def _rms_norm_cpu(
        self: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ms = self.float().pow(2).mean(dim=-1, keepdim=True) + epsilon
        rstd = torch.rsqrt(ms)
        y = ((self.float() * rstd) * gamma.float()).to(self.dtype)
        return y, rstd
