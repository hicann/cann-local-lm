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

"""npu::npu_dynamic_quant - per-token dynamic activation quant op (W4A8 act side).

Mirrors torch_npu.npu_dynamic_quant (schema identical, so real torch_npu provides
the op on NPU; stubs env defines it). Output: (y:int8, scale:fp32) per-token along
the last dim. GE converter (converter/export/ge_converter.py) maps it to
``ge.DynamicQuant``.

Used by QuantLinear (W4A8) to produce the int8 activation + per-token scale that
feeds ``npu_quant_matmul``, replacing the aten.abs/amax/clamp/div chain (which had
no GE converter and blocked export).
"""

import torch

from ._library import NPU_LIB, safe_define, _HAS_TORCH_NPU, register_cpu_impl

if NPU_LIB is not None:
    safe_define(
        "npu_dynamic_quant(Tensor input, *, "
        "Tensor? smooth_scales=None, Tensor? group_index=None, "
        "int? dst_type=None, str quant_mode=\"pertoken\", "
        "float dst_type_max=0.) -> (Tensor y, Tensor scale)"
    )


if not _HAS_TORCH_NPU:

    @torch.library.register_fake("npu::npu_dynamic_quant")
    def _npu_dynamic_quant_meta(
        input: torch.Tensor,
        *,
        smooth_scales: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        dst_type: int | None = None,
        quant_mode: str = "pertoken",
        dst_type_max: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del smooth_scales, group_index, dst_type, quant_mode, dst_type_max
        y = input.new_empty(input.shape, dtype=torch.int8)
        scale = input.new_empty(input.shape[:-1], dtype=torch.float32)
        return y, scale

    @register_cpu_impl("npu::npu_dynamic_quant")
    def _npu_dynamic_quant_cpu(
        input: torch.Tensor,
        *,
        smooth_scales: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        dst_type: int | None = None,
        quant_mode: str = "pertoken",
        dst_type_max: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token symmetric int8: scale = max|x|/127 along last dim;
        y = clamp(round(x/scale), -128, 127) as int8."""
        del smooth_scales, group_index, dst_type, quant_mode, dst_type_max
        xf = input.float()
        scale = (xf.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
        y = torch.clamp(torch.round(xf / scale), -128, 127).to(torch.int8)
        return y, scale.to(torch.float32)
