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

"""cann_lm::rope_in_place - fused rotary positional embedding (in-place)."""

import torch

from ._library import LIBRARY

LIBRARY.define(
    "rope_in_place(Tensor(a!) input, Tensor sin_cache, Tensor cos_cache) -> ()"
)


@torch.library.register_fake("cann_lm::rope_in_place")
def _rope_in_place_meta(
    input: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
):
    pass
