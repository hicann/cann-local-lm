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

    @register_fx_node_ge_converter(torch.ops.aten.softplus.default)
    def _convert_softplus(x: Any, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge.Softplus(x)

    @register_fx_node_ge_converter(torch.ops.aten.eye.default)
    def _convert_eye(n: Any, m: Any = None, *, dtype: Any = None, device: Any = None,
                     pin_memory: bool = False, meta_outputs: Any = None) -> Any:
        del device, pin_memory, meta_outputs
        import torch as _torch
        rows = int(n)
        cols = int(m) if m is not None else rows
        dt = _torch.float32 if dtype is None else dtype
        eye_tensor = _torch.eye(rows, cols, dtype=dt)
        return ge.Const(eye_tensor)

    @register_fx_node_ge_converter(torch.ops.npu.npu_chunk_gated_delta_rule.default)
    def _convert_chunk_gated_delta_rule(
        query: Any,
        key: Any,
        value: Any,
        *,
        beta: Any = None,
        initial_state: Any = None,
        actual_seq_lengths: Any = None,
        scale: float | None = None,
        g: Any = None,
        meta_outputs: Any = None,
    ) -> Any:
        # 手动创建 ge.ChunkGatedDeltaRule (torchair auto_generated 未生成此算子,
        # npu_chunk_gated_delta_rule converter 直接 raise RuntimeError, 此处覆盖).
        # CANN op_info: aic-ascend910_93-ops-info-transformer.json "ChunkGatedDeltaRule"
        # inputs: query/key/value/beta/initial_state/actual_seq_lengths + g(optional)
        # outputs: out/final_state, attr: scale_value
        from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name, Tensor, get_invalid_desc
        op = get_default_ge_graph().op.add()
        op.type = "ChunkGatedDeltaRule"
        op.name = next_unique_name(None, "ChunkGatedDeltaRule")
        input_list = [
            (query, "query"),
            (key, "key"),
            (value, "value"),
            (beta, "beta"),
            (initial_state, "initial_state"),
            (actual_seq_lengths, "actual_seq_lengths"),
            (g, "g"),
        ]
        for item_t, item_n in input_list:
            if item_t is not None:
                op.input.append(item_t.tensor)
                op.input_desc.add().CopyFrom(item_t.desc)
                op.input_desc[-1].name = item_n
            else:
                op.input.append('')
                op.input_desc.add().CopyFrom(get_invalid_desc())
                op.input_desc[-1].name = item_n
        op.attr["scale_value"].f = scale if scale is not None else 1.0
        output_index = 0
        op.output_desc.add().name = "out"
        out = Tensor(op, output_index)
        output_index += 1
        op.output_desc.add().name = "final_state"
        final_state = Tensor(op, output_index)
        # 填 output desc shape/dtype (从 meta_outputs), 避 GE infershape 不推空 desc
        mo_list = list(meta_outputs) if isinstance(meta_outputs, (list, tuple)) else [meta_outputs]
        if len(mo_list) >= 1 and mo_list[0] is not None:
            mo0 = mo_list[0]
            del out.desc.shape.dim[:]
            out.desc.shape.dim.extend(list(getattr(mo0, "size", []) or []))
            out.desc.dtype = query.desc.dtype
        if len(mo_list) >= 2 and mo_list[1] is not None:
            mo1 = mo_list[1]
            del final_state.desc.shape.dim[:]
            final_state.desc.shape.dim.extend(list(getattr(mo1, "size", []) or []))
            final_state.desc.dtype = initial_state.desc.dtype
        return out, final_state

    @register_fx_node_ge_converter(torch.ops.aten.conv1d.default)
    def _convert_conv1d(
        x: Any,
        weight: Any,
        bias: Any = None,
        stride: Any = (1,),
        padding: Any = (0,),
        dilation: Any = (1,),
        groups: int = 1,
        *,
        meta_outputs: Any = None,
    ) -> Any:
        from torchair._ge_concrete_graph.ge_converter.aten.convolution import _npu_convolution
        s = stride[0] if isinstance(stride, (list, tuple)) and stride else 1
        p = padding[0] if isinstance(padding, (list, tuple)) and padding else 0
        d = dilation[0] if isinstance(dilation, (list, tuple)) and dilation else 1
        x4 = ge.Unsqueeze(x, axes=[2])       # [B,C,L] -> [B,C,1,L]
        w4 = ge.Unsqueeze(weight, axes=[2])  # [O,C,K] -> [O,C,1,K]
        # 手动填 Unsqueeze outputDesc 的 shape/dtype, 让下游 Conv2D inputDesc (CopyFrom) 非空,
        # 避开 GE infershape 不推空 desc (torchair auto_generated 只设 name="y")
        w_dims = list(weight.desc.shape.dim)  # [O,C,K] (constant 有 shape.dim)
        # x.desc.shape.dim 空 (boundary 只有 _meta), 从 meta_outputs.size [B,C,L_out] 反推 L_in
        mo = meta_outputs[0] if isinstance(meta_outputs, (list, tuple)) else meta_outputs
        mo_size = list(getattr(mo, "size", []) or [])
        x_dims = []
        if len(mo_size) == 3 and len(w_dims) == 3:
            B, C, L_out = mo_size[0], mo_size[1], mo_size[2]
            K = w_dims[2]
            L_in = (L_out - 1) * s + K - 2 * p
            x_dims = [B, C, L_in]
        # 填 x4 (Unsqueeze output) shape [B,C,1,L_in] + dtype (layout 由 _npu_conv2d 的 specific_op_input_layout 设)
        if len(x_dims) == 3:
            del x4.desc.shape.dim[:]
            x4.desc.shape.dim.extend([x_dims[0], x_dims[1], 1, x_dims[2]])
            x4.desc.dtype = x.desc.dtype
            x4.desc.layout = "NCHW"
        # 填 w4 (Unsqueeze output) shape [O,C,1,K] + dtype
        if len(w_dims) == 3:
            del w4.desc.shape.dim[:]
            w4.desc.shape.dim.extend([w_dims[0], w_dims[1], 1, w_dims[2]])
            w4.desc.dtype = weight.desc.dtype
            w4.desc.layout = "NCHW"
        out4 = _npu_convolution(x4, w4, bias, [1, s], [0, p], [1, d], groups, meta_outputs)
        # 填 Conv2D outputDesc shape [B,O,1,L_out] + dtype
        if len(mo_size) == 3 and len(w_dims) == 3:
            del out4.desc.shape.dim[:]
            out4.desc.shape.dim.extend([mo_size[0], w_dims[0], 1, mo_size[2]])
            out4.desc.dtype = x.desc.dtype
            out4.desc.layout = "NCHW"
        return ge.Squeeze(out4, axis=[2])

    @register_fx_node_ge_converter(torch.ops.aten.type_as.default)

    @register_fx_node_ge_converter(torch.ops.aten.type_as.default)
    def _convert_type_as(x: Any, dtype_or_tensor: Any, *, meta_outputs: Any = None) -> Any:
        del meta_outputs
        import torch as _torch
        if isinstance(dtype_or_tensor, _torch.Tensor):
            dt = dtype_or_tensor.dtype
        else:
            dt = dtype_or_tensor
        _GE_DTYPE_MAP = {
            _torch.float16: 1, _torch.float32: 0, _torch.bfloat16: 27,
            _torch.int64: 9, _torch.int32: 3, _torch.bool: 12,
        }
        return ge.Cast(x, dst_type=_GE_DTYPE_MAP.get(dt, 0))

    @register_fx_node_ge_converter(torch.ops.aten.index_copy.default)
    def _convert_index_copy(
        self: Any,
        dim: int,
        index: Any,
        source: Any,
        meta_outputs: Any = None,
    ) -> Any:
        """aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor

        Implemented via InplaceUpdate + transpose: ``InplaceUpdate`` only supports
        dim=0, so transpose self/source to move ``dim`` to axis 0, InplaceUpdate
        (replace indices), then transpose back. Supports arbitrary 1D index
        (not limited to contiguous arange).
        """
        del meta_outputs
        from torchair.ge._ge_graph import DataType
        # InplaceUpdate requires DT_INT32 indices (see REG_OP InplaceUpdate input)
        if index.dtype != DataType.DT_INT32:
            index = ge.Cast(index, dst_type=int(DataType.DT_INT32))
        ndim = self.rank
        if dim < 0:
            dim += ndim
        # perm moves `dim` to position 0: [dim, 0, 1, ..., dim-1, dim+1, ..., ndim-1]
        perm = [dim] + [i for i in range(ndim) if i != dim]
        inverse_perm = [0] * ndim
        for i, p in enumerate(perm):
            inverse_perm[p] = i
        # TensorMove to avoid in-place precision issue (same pattern as index_add.py)
        moved = ge.TensorMove(self)
        transposed_self = ge.TransposeD(moved, perm=perm)
        transposed_source = ge.TransposeD(source, perm=perm)
        # Force dtype alignment: InplaceUpdate requires x and v to have the same dtype.
        # Use self.dtype (known) rather than transposed_self.dtype (UNDEFINED before infershape).
        transposed_source = ge.Cast(transposed_source, dst_type=int(self.dtype))
        updated = ge.InplaceUpdate(transposed_self, index, transposed_source)
        return ge.TransposeD(updated, perm=inverse_perm)
