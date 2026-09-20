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

"""cann_lm::rotary_mul GE converter -> ge.RotaryMul / ge.RotaryPositionEmbedding.

Mirrors torchair custom/npu_rotary_mul.py: arch35 (davinci 3.5) -> ge.RotaryPositionEmbedding
(mode), else -> ge.RotaryMul. cann_lm::rotary_mul migrated from cann_lm::rotary_mul (avoid
torch_npu duplicate-registration); CPU impl only supports rotary_mode='half'.
"""
from typing import Any


def convert_rotary_mul(
    self: Any,
    r1: Any,
    r2: Any,
    rotary_mode: str = "half",
    *,
    meta_outputs: Any = None,
) -> Any:
    """cann_lm::rotary_mul(Tensor self, Tensor r1, Tensor r2, str rotary_mode) -> Tensor out."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._utils.check_platform import is_arch35
    if is_arch35():
        modes = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
        if rotary_mode not in modes:
            raise NotImplementedError("rotary_mode only support half/interleave/quarter/interleave-half now!")
        return ge.RotaryPositionEmbedding(self, r1, r2, mode=modes[rotary_mode])
    return ge.RotaryMul(self, r1, r2)
