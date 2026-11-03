# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Adapted from torchair (Copyright (c) 2023, Huawei Technologies Co., Ltd, BSD License):
#   - torchair/python/torchair/_ge_concrete_graph/ge_converter/custom/npu_chunk_gated_delta_rule.py
#   - torchair/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py
# and torch_npu (Copyright (c) Huawei Technologies Co., Ltd, BSD License).
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

"""cann_local_lm::chunk_gated_delta_rule GE converter -> ge node ChunkGatedDeltaRule.

torchair auto_generated didn't generate this op (its converter raises RuntimeError), so this
overrides it (manually built, fills output desc). CANN op_info dim convention is THD 3D
(same as npu_recurrent_gated_delta_rule); modeling already passes THD, passed through as-is.
Registered by ge_converter.register_ge_converters.
"""

from typing import Any


def convert_chunk_gated_delta_rule(
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
    """npu_chunk_gated_delta_rule -> ge node ChunkGatedDeltaRule (manually built; fill output desc to avoid empty)."""
    from torchair.ge._ge_graph import (
        get_default_ge_graph,
        next_unique_name,
        Tensor,
        get_invalid_desc,
    )

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
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = item_n
    op.attr["scale_value"].f = scale if scale is not None else 1.0
    output_index = 0
    op.output_desc.add().name = "out"
    out = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "final_state"
    final_state = Tensor(op, output_index)
    # Fill output desc shape/dtype (from meta_outputs) to avoid GE infershape skipping empty desc
    mo_list = (
        list(meta_outputs)
        if isinstance(meta_outputs, (list, tuple))
        else [meta_outputs]
    )
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
