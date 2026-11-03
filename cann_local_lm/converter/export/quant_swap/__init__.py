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

"""Quantization materialization internals (W4A8 s8s4 / W4A16).

Holds the **materialization layer** consumed by the swap orchestrator at
``cann_local_lm/converter/swap.py`` (which moved to the converter root as a peer of
``loader.py`` / ``saver.py`` in the load→swap→save pipeline).

- materialize_int4_per_group_symmetric: fp weight -> (nibble-packed int4-as-int8, scale),
  linear symmetric (offset=0), uses dopt GPTQ scale
- materialize_int8_per_channel_symmetric: embed int8 per-channel (host gather)
- transdata_scale / restore_scale: fp32 <-> int64 bit-pattern (weight_scale storage fmt)
- pack_int4: nibble-pack signed int4 (stored as int8)
- QuantLinear: export-path Linear whose forward calls cann_local_lm::quant_matmul
  (W4A8 also uses npu_dynamic_quant for act quant)

The swap orchestrator (swap_from_dopt /
swap_embeddings_from_dopt / load_dopt_state_dict) lives at
``cann_local_lm.converter.swap``. GPTQ (dopt_lm) runs offline on NPU/GPU/CPU and
produces the dopt output; the converter loads it via --quant-path, materializes
(via this package), swaps (via cann_local_lm.converter.swap), exports. See
docs/cann_lm_converter/quant/quant_design.md.
"""

from .materialize import (
    materialize_int4_per_group_symmetric,
    materialize_int8_per_channel_symmetric,
    transdata_scale,
    restore_scale,
    pack_int4,
)
from .quant_linear import QuantLinear

__all__ = [
    "materialize_int4_per_group_symmetric",
    "materialize_int8_per_channel_symmetric",
    "transdata_scale",
    "restore_scale",
    "pack_int4",
    "QuantLinear",
]
