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

"""Quantization entrypoints for the converter (W4A8 s8s4 / W4A16).

- materialize_int4_per_group_symmetric: fp16 weight -> (int4-as-int8, scale), 线性对称 (offset=0), 用 dopt GPTQ scale
- materialize_int8_per_channel_symmetric: embed int8 per-channel (host gather)
- QuantLinear: export-path Linear whose forward calls npu::npu_quant_matmul
  (W4A8 also uses npu_dynamic_quant for act quant)
- swap_to_quant_linear_from_dopt: port dopt GPTQ output (GPTQ weight + group_scale)
  -> materialize int4 -> swap QuantLinear (legacy, scenes 1 offline use; internally
  load_dopt_state_dict + _swap_linear_from_sd)
- swap_from_dopt: one-shot load sd + Linear swap + embed 抽取 (主进程场景 1/2 用,
  避免 swap_to_quant_linear_from_dopt + swap_embeddings_from_dopt 二次 load IO)
- swap_embeddings_from_dopt: port dopt embed int8+scale for host gather
- load_dopt_state_dict: load dopt GPTQ output (safetensors/.pth/dir)

GPTQ (dopt_lm) runs offline on NPU/GPU and produces the dopt output; the
converter loads it via --quant-path, materializes, swaps, exports. See
docs/cann_lm_converter/quant_design.md.
"""

from .materialize import materialize_int4_per_group_symmetric, materialize_int8_per_channel_symmetric
from .quant_linear import QuantLinear
from .swap import (
    load_dopt_state_dict,
    swap_to_quant_linear_from_dopt,
    swap_from_dopt,
    swap_embeddings_from_dopt,
)

__all__ = [
    "materialize_int4_per_group_symmetric",
    "materialize_int8_per_channel_symmetric",
    "QuantLinear",
    "swap_to_quant_linear_from_dopt",
    "swap_from_dopt",
    "swap_embeddings_from_dopt",
    "load_dopt_state_dict",
]
