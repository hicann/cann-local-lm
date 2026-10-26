# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tiny placeholder-to-platform-op rewrite pass."""

from __future__ import annotations

from typing import Any

import torch


def _target_name(target: Any) -> str:
    if hasattr(target, "name"):
        name = str(target.name())
        if name == "air::npu_fused_infer_attention_score":
            return "torchair.ops.npu_fused_infer_attention_score"
        return name
    return str(target)


def _torchair_attention_op() -> Any:
    try:
        import torch_npu  # noqa: F401
        import torchair.ops._npu_fused_infer_attention_score  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torchair is required because fused_infer_attention_score is rewritten "
            "to torchair.ops.npu_fused_infer_attention_score."
        ) from exc
    return torch.ops.air.npu_fused_infer_attention_score.default


def _target_map() -> dict[Any, Any]:
    return {
        torch.ops.xllm_placeholder.rope_in_place.default: (
            torch.ops.xllm_ascend.rope_in_place.default
        ),
        torch.ops.xllm_placeholder.fused_infer_attention_score.default: (
            _torchair_attention_op()
        ),
        torch.ops.xllm_placeholder.beam_search.default: (
            torch.ops.xllm_ascend.beam_search.default
        ),
    }


def rewrite_placeholder_to_custom(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """Rewrites demo placeholder ops and prints each replacement."""

    replacements = _target_map()
    graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}

    for node in exported_program.graph.nodes:
        new_target = replacements.get(node.target)
        if new_target is None:
            new_node = graph.node_copy(node, lambda old_node: env[old_node])
        else:
            print(
                f"[rewrite] {node.name}: "
                f"{_target_name(node.target)} -> {_target_name(new_target)}"
            )
            new_node = graph.call_function(
                new_target,
                args=torch.fx.map_arg(node.args, lambda old_node: env[old_node]),
                kwargs=torch.fx.map_arg(node.kwargs, lambda old_node: env[old_node]),
            )
        new_node.meta = dict(node.meta)
        env[node] = new_node

    graph.lint()
    rewritten_program = torch.export.ExportedProgram(
        exported_program.graph_module,
        graph,
        exported_program.graph_signature,
        exported_program.state_dict,
        exported_program.range_constraints,
        exported_program.module_call_graph,
        example_inputs=exported_program.example_inputs,
        constants=exported_program.constants,
        verifiers=exported_program.verifiers,
    )
    rewritten_program.validate()
    return rewritten_program
