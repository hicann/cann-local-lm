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

"""cann_lm::rms_norm GE converter -> ge.RmsNorm.

Mirrors torchair custom/npu_rms_norm.py (ge.RmsNorm) so cann_lm::rms_norm (migrated
from cann_lm::rms_norm to avoid torch_npu duplicate-registration) maps identically.
"""

from typing import Any


def convert_rms_norm(
    self: Any,
    gamma: Any,
    epsilon: float = 1e-6,
    *,
    meta_outputs: Any = None,
) -> Any:
    """cann_lm::rms_norm(Tensor self, Tensor gamma, float epsilon) -> (Tensor y, Tensor rstd)."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge

    return ge.RmsNorm(self, gamma, epsilon=epsilon)
