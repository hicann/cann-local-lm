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

"""GE operator mapping converters (torch.export FX node -> epair ascend_ir).

Centralized location for cann_local_lm's GE converters that map ``torch.ops`` nodes
to ``ge.*`` ops during ``torch.export -> epair`` IR conversion.

- ``ge_converter.register_ge_converters()``: the registrar (imports + registers
  all sub-converters via ``register_fx_node_ge_converter``). Call before
  ``torch_delegate.export_to_epair``.
- ``_*_converter.py``: per-op converters (quant_matmul / chunk_gated_delta_rule
  / conv1d / index_copy / index_put / scatter_nd_update /
  fused_infer_attention_score_v2).

See docs/cann_lm_converter/export/export_design.md "GE 算子注册两层".
"""

from .ge_converter import register_ge_converters

__all__ = ["register_ge_converters"]
