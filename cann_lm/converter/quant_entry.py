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

"""Single quant entry-point facade for the converter CLI.

Re-exports the two quantization entry points consumed by
``cann_lm_converter.convert()`` so the CLI imports quant from one module:

- ``run_auto_quant`` (scene 2, ``--quant-method gptq``): spawn dopt subprocess
  to run GPTQ on NPU/GPU/CPU. Implementation: ``cann_lm/converter/quant/auto_quant.py``.
- ``swap_from_dopt`` (scene 1/2, ``--quant-path`` or post-auto-quant): load dopt
  GPTQ product, materialize int4, swap nn.Linear -> QuantLinear. Implementation:
  ``cann_lm/converter/swap.py``.

Laziness note: ``cann_lm_converter`` imports these inside ``convert()``'s quant
conditionals, so this facade (and its eager re-exports) is only loaded when a
quant scenario is selected — no-quant runs do not trigger the npu op
registration side effect of ``swap`` (via ``quant_linear`` -> ``fused_ops``).

See docs/cann_lm_converter/quant/quant_design.md.
"""

from cann_lm.converter.quant.auto_quant import run_auto_quant
from cann_lm.converter.swap import swap_from_dopt

__all__ = ["run_auto_quant", "swap_from_dopt"]
