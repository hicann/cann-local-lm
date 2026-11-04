# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


def fold_scalar_constant_casts(exported_program: ExportedProgram) -> int:
    """Preconvert lifted FP32 scalar constants used only through BF16 casts.

    Restrict folding to export-owned constants and literal buffers lifted by
    ExecuTorch. Never freeze user inputs, parameters, or mutable state. Cast
    results must have non-aliasing, functional consumers so sharing a constant
    cannot expose the removal of the cast's fresh allocation.
    """
    graph = exported_program.graph_module.graph
    nodes = {node.name: node for node in graph.nodes}
    mutated = set(exported_program.graph_signature.buffers_to_mutate.values())
    state_dict = dict(exported_program.state_dict)
    constants = dict(exported_program.constants)
    folded = 0

    for spec in exported_program.graph_signature.input_specs:
        if spec.kind is not InputKind.CONSTANT_TENSOR and not (
            spec.kind is InputKind.BUFFER
            and isinstance(spec.target, str)
            and spec.target.startswith("_lifted_tensor_constant")
        ):
            continue
        if spec.target in mutated:
            continue
        source = nodes.get(spec.arg.name)
        storage = constants if spec.target in constants else state_dict
        value = storage.get(spec.target)
        if (
            source is None
            or not isinstance(value, torch.Tensor)
            or isinstance(value, FakeTensor)
            or value.device.type != "cpu"
            or value.ndim != 0
            or value.dtype is not torch.float32
            or value.requires_grad
        ):
            continue
        checks = [
            user
            for user in source.users
            if user.op == "call_function"
            and user.target is torch.ops.aten._assert_tensor_metadata.default
        ]
        if any(
            check.kwargs.get("dtype", check.args[3] if len(check.args) > 3 else None)
            not in (None, torch.float32)
            for check in checks
        ):
            continue
        casts = [user for user in source.users if user not in checks]
        if not casts or not all(_is_safe_bf16_cast(cast, source) for cast in casts):
            continue

        with unset_fake_temporarily(), torch.no_grad():
            storage[spec.target] = value.to(dtype=torch.bfloat16)
        source.meta["val"] = casts[0].meta["val"]
        source.meta.pop("tensor_meta", None)
        if "tensor_meta" in casts[0].meta:
            source.meta["tensor_meta"] = casts[0].meta["tensor_meta"]
        for check in checks:
            if "dtype" in check.kwargs and check.kwargs["dtype"] is not None:
                check.update_kwarg("dtype", torch.bfloat16)
            elif len(check.args) > 3 and check.args[3] is not None:
                check.update_arg(3, torch.bfloat16)
        for cast in casts:
            cast.replace_all_uses_with(source)
            graph.erase_node(cast)
            folded += 1

    if folded:
        # Other methods may share these dictionaries. Replace the mappings,
        # without changing the original tensors or another method's metadata.
        exported_program._state_dict = state_dict
        exported_program._constants = constants
        graph.lint()
        exported_program.graph_module.recompile()
    return folded


def _is_safe_bf16_cast(cast: torch.fx.Node, source: torch.fx.Node) -> bool:
    value = cast.meta.get("val")
    if (
        cast.op != "call_function"
        or cast.target is not torch.ops.aten._to_copy.default
        or cast.args != (source,)
        or dict(cast.kwargs) != {"dtype": torch.bfloat16}
        or not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype is not torch.bfloat16
        or value.ndim != 0
        or not cast.users
    ):
        return False
    for user in cast.users:
        schema = getattr(user.target, "_schema", None)
        if (
            user.op != "call_function"
            or schema is None
            or schema.is_mutable
            or any(result.alias_info is not None for result in schema.returns)
        ):
            return False
    return True
