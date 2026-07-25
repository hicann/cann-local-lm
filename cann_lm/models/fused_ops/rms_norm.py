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

"""cann_lm::rms_norm - RMSNorm op (schema + meta + CPU eager kernel)."""

import torch

from ._library import LIBRARY

LIBRARY.define(
    "rms_norm(Tensor x, Tensor gamma, *, float epsilon=1e-6) -> (Tensor y, Tensor rstd)"
)


@torch.library.register_fake("cann_lm::rms_norm")
def _rms_norm_meta(
    x: torch.Tensor,
    gamma: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gamma, epsilon
    y = torch.empty_like(x)
    rstd = x.new_empty(x.shape[:-1] + (1,), dtype=torch.float32)
    return y, rstd


@torch.library.impl("cann_lm::rms_norm", "cpu")
def _rms_norm_cpu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    ms = x.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + epsilon)
    y = ((x.float() * rstd) * gamma.float()).to(x.dtype)
    return y, rstd
