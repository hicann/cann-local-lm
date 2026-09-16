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

"""Register cann_lm GE converters (torch.export -> epair IR)."""

import os
from typing import Any

import torch

from cann_lm.models.fused_ops import ops as _backend_ops  # Registers cann_lm ops.


def register_ge_converters() -> None:
    # lite export 模式: 确保 torchair 从 vendored path 加载 (fake C extension 已安装后).
    # full 模式: active_config() 为 None, prepare_lite_torchair 直接 return, 无副作用.
    from torch_delegate.lite_context import prepare_lite_torchair
    prepare_lite_torchair()

    from torchair import register_fx_node_ge_converter
    # Import torchair custom/ first (registers torchair's npu converters); cann_lm converters below override them.
    from torchair._ge_concrete_graph.ge_converter import custom  # noqa: F401
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import custom_op

    # Monkey-patch is_cann_compat: the stub GE rejects ops not in its registered set (DynamicQuant,
    # Bitcast). Bypassing lets torchair's own converters emit full IR on the CPU/stubs env (else
    # "OperatorFactory find optype DynamicQuant failed" -> export fails).
    import torchair._ge_concrete_graph.compat_ir as _compat_ir
    _compat_ir.is_cann_compat = lambda *a, **k: ""

    from ._quant_matmul_converter import convert_quant_matmul
    register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)(convert_quant_matmul)

    # softplus: torchair placeholder -> ge.SoftplusV2. reg_register.h must register SoftplusV2
    # (else GetGEGraph heap corruption). beta/threshold float() (aten Scalar is int, ge .f is float).
    @register_fx_node_ge_converter(torch.ops.aten.softplus.default)
    def _convert_softplus(self: Any, beta: Any = 1, threshold: Any = 20, *, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge.SoftplusV2(self, beta=float(beta), threshold=float(threshold))

    # l2norm -> ge.L2Normalize. reg_register.h must register L2Normalize.
    @register_fx_node_ge_converter(torch.ops.cann_lm.l2norm.default)
    def _convert_l2norm(x: Any, dim: int = -1, eps: float = 1e-6, *, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge.L2Normalize(x, axis=[dim], eps=float(eps))

    from ._chunk_gated_delta_rule_converter import convert_chunk_gated_delta_rule
    register_fx_node_ge_converter(torch.ops.npu.npu_chunk_gated_delta_rule.default)(convert_chunk_gated_delta_rule)

    from ._conv1d_converter import convert_conv1d
    register_fx_node_ge_converter(torch.ops.aten.conv1d.default)(convert_conv1d)

    from ._index_copy_converter import convert_index_copy
    register_fx_node_ge_converter(torch.ops.aten.index_copy.default)(convert_index_copy)

    from ._scatter_nd_update_converter import convert_scatter_nd_update, convert_scatter_nd_update_
    register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update.default)(convert_scatter_nd_update)
    register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update_.default)(convert_scatter_nd_update_)

    from ._index_put_converter import convert_index_put
    register_fx_node_ge_converter(torch.ops.aten.index_put.default)(convert_index_put)

    from ._fused_infer_attention_score_v2_converter import convert_fused_infer_attention_score_v2
    register_fx_node_ge_converter(torch.ops.npu.npu_fused_infer_attention_score_v2.default)(convert_fused_infer_attention_score_v2)

    # KirinX90 converter patches (last-write-wins, override after all above).
    # 环境变量 KIRINX90_CONVERTER_PATCHES=1 时加载，用等价算子替换 kirinx90 不支持的 op。
    if os.environ.get("KIRINX90_CONVERTER_PATCHES", "") == "1":
        from ._kirinx90_converter_patches import register_kirinx90_patches
        register_kirinx90_patches()
