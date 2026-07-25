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

"""cann_lm::rotary_mul - fused rotate-half RoPE (single tensor, 4D BSND).

Mirrors torch_npu.npu_rotary_mul(x, cos, sin, *, rotary_mode). Expects 4D
``[B, S, N, D]`` with cos/sin broadcast over the head axis ``[B, S, 1, D]``.
``rotary_mode='half'`` is the GPT-NeoX / Llama rotate-half pairing.
"""

import torch

from ._library import LIBRARY

LIBRARY.define(
    "rotary_mul(Tensor x, Tensor cos, Tensor sin, *, "
    "str rotary_mode=\"half\") -> Tensor out"
)


@torch.library.register_fake("cann_lm::rotary_mul")
def _rotary_mul_meta(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_mode: str = "half",
) -> torch.Tensor:
    del cos, sin, rotary_mode
    return torch.empty_like(x)


@torch.library.impl("cann_lm::rotary_mul", "cpu")
def _rotary_mul_cpu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_mode: str = "half",
) -> torch.Tensor:
    if rotary_mode != "half":
        raise NotImplementedError(
            f"rotary_mode={rotary_mode!r} not supported on CPU; only 'half'"
        )
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    out = (x * cos) + (rotated * sin)
    return out.to(x.dtype)
