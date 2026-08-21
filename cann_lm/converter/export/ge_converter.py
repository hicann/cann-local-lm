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

"""TorchAir converters for the three demo custom ops."""

from __future__ import annotations

from typing import Any

import torch

from cann_lm.models.fused_ops import ops as _backend_ops  # Registers cann_lm ops.


def register_ge_converters() -> None:
    from torchair import register_fx_node_ge_converter
    # 先 import torchair custom/ (注册 torchair 的 npu converter, 含 raise 的 npu_chunk_gated_delta_rule
    # + import torch_npu 的 quantmatmul). 后续 cann_lm converter 覆盖之.
    from torchair._ge_concrete_graph.ge_converter import custom  # noqa: F401
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import custom_op

    # Monkey-patch is_cann_compat to bypass the stub GE's check_cann_compat which
    # rejects ops not in the stub's registered set (DynamicQuant, Bitcast, etc.).
    # This lets torchair's own converters (e.g. custom/npu_dynamic_quant.py which
    # uses ge.DynamicQuant with full IR) work on the CPU/stubs env. Without this,
    # ge.DynamicQuant -> check_cann_compat -> "OperatorFactory find optype DynamicQuant
    # failed" -> export fails. With this, the ep_json gets the full IR (IrDef) for
    # DynamicQuant -> the TBE compiler (aclgrphBuildModel) can select the kernel.
    import torchair._ge_concrete_graph.compat_ir as _compat_ir
    _compat_ir.is_cann_compat = lambda *a, **k: ""

    # quant_matmul: 注册到 npu (覆盖 torchair quantmatmul.py 的 import torch_npu converter,
    # 用简化版 _quant_matmul_converter 不依赖 torch_npu)
    from ._quant_matmul_converter import convert_quant_matmul
    register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)(convert_quant_matmul)

    # npu_dynamic_quant: torchair ships its own converter (custom/npu_dynamic_quant.py,
    # pertoken -> ge.DynamicQuant / else ge.DynamicQuantV2). It auto-registers when
    # torchair custom/ is imported above. With the is_cann_compat monkey-patch,
    # ge.DynamicQuant passes the check on the stub env -> full IR in ep_json ->
    # TBE compiler can select the kernel. Do NOT register a duplicate here.

    # @register_fx_node_ge_converter(torch.ops.aten.softplus.default)
    # def _convert_softplus(x: Any, meta_outputs: Any = None) -> Any:
    #     del meta_outputs
    #     return ge.Softplus(x)

    # @register_fx_node_ge_converter(torch.ops.aten.eye.default)
    # def _convert_eye(n: Any, m: Any = None, *, dtype: Any = None, device: Any = None,
    #                  pin_memory: bool = False, meta_outputs: Any = None) -> Any:
    #     del device, pin_memory, meta_outputs
    #     import torch as _torch
    #     rows = int(n)
    #     cols = int(m) if m is not None else rows
    #     dt = _torch.float32 if dtype is None else dtype
    #     eye_tensor = _torch.eye(rows, cols, dtype=dt)
    #     return ge.Const(eye_tensor)

    # npu_chunk_gated_delta_rule: qwen3.5 linear-attn 融合算子, torchair auto_generated 未生成
    # (其 npu_chunk_gated_delta_rule converter 直接 raise RuntimeError). 实现抽到单独模块.
    from ._chunk_gated_delta_rule_converter import convert_chunk_gated_delta_rule
    register_fx_node_ge_converter(torch.ops.npu.npu_chunk_gated_delta_rule.default)(convert_chunk_gated_delta_rule)

    # aten.conv1d: 无 decomposition, 必须自定义 converter (手动填 outputDesc 避开 GE infershape 空 desc;
    # 走 whitelist 分解会 infershape 失败). 实现抽到 _conv1d_converter.py (单独模块).
    from ._conv1d_converter import convert_conv1d
    register_fx_node_ge_converter(torch.ops.aten.conv1d.default)(convert_conv1d)

    # @register_fx_node_ge_converter(torch.ops.aten.type_as.default)

    # @register_fx_node_ge_converter(torch.ops.aten.type_as.default)
    # def _convert_type_as(x: Any, dtype_or_tensor: Any, *, meta_outputs: Any = None) -> Any:
    #     del meta_outputs
    #     import torch as _torch
    #     if isinstance(dtype_or_tensor, _torch.Tensor):
    #         dt = dtype_or_tensor.dtype
    #     else:
    #         dt = dtype_or_tensor
    #     _GE_DTYPE_MAP = {
    #         _torch.float16: 1, _torch.float32: 0, _torch.bfloat16: 27,
    #         _torch.int64: 9, _torch.int32: 3, _torch.bool: 12,
    #     }
    #     return ge.Cast(x, dst_type=_GE_DTYPE_MAP.get(dt, 0))

    # aten.index_copy: KV cache scatter-write. 无 decomposition (whitelist 空操作), torchair 占位
    # NotImplementedError. 实现抽到 _index_copy_converter.py (InplaceUpdate + Transpose).
    from ._index_copy_converter import convert_index_copy
    register_fx_node_ge_converter(torch.ops.aten.index_copy.default)(convert_index_copy)

    # aten.index_put: qwen3.5 past_key_values.update 用 index_put (indices list). 无 decomposition,
    # torchair 占位 NotImplementedError. 实现抽到 _index_put_converter.py (找非 None dim + InplaceUpdate+Transpose).
    from ._index_put_converter import convert_index_put
    register_fx_node_ge_converter(torch.ops.aten.index_put.default)(convert_index_put)

    # npu_fused_infer_attention_score_v2: gemma4 attention 融合算子. cann_lm 自注册 npu 命名空间
    # (见 fused_ops/fused_infer_attention_score.py); torchair 的 converter 在 air 命名空间 (不适用).
    # 实现抽到 _fused_infer_attention_score_v2_converter.py (调 ge.FusedInferAttentionScore).
    from ._fused_infer_attention_score_v2_converter import convert_fused_infer_attention_score_v2
    register_fx_node_ge_converter(torch.ops.npu.npu_fused_infer_attention_score_v2.default)(convert_fused_infer_attention_score_v2)
