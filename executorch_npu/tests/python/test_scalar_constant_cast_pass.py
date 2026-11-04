# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import pytest
import torch
from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch_npu.compiler import pipeline
from executorch_npu.compiler.compile_spec import AscendCompileOptions
from executorch_npu.passes import fold_scalar_constant_casts


class ScalarCast(torch.nn.Module):
    def __init__(self, value=1e-6, *, persistent=True, literal=False):
        super().__init__()
        tensor = torch.tensor(value, dtype=torch.float32)
        if literal:
            self.constant = tensor
        else:
            self.register_buffer(
                "_lifted_tensor_constant0", tensor, persistent=persistent
            )

    def forward(self, value):
        constant = (
            self.constant
            if hasattr(self, "constant")
            else self._lifted_tensor_constant0
        )
        return value + constant.to(torch.bfloat16)


def _cast_count(exported):
    return sum(
        node.op == "call_function" and node.target is torch.ops.aten._to_copy.default
        for node in exported.graph.nodes
    )


@pytest.mark.parametrize(
    "constant", [1.0, 1e-6, 1.00390625, -1e-6, float("inf"), float("nan")]
)
@pytest.mark.parametrize("storage", ["persistent", "nonpersistent", "literal"])
def test_folds_constants_without_changing_bf16_results(constant, storage):
    model = ScalarCast(
        constant, persistent=storage != "nonpersistent", literal=storage == "literal"
    )
    values = torch.arange(8, dtype=torch.bfloat16)
    exported = torch.export.export(model, (values,)).run_decompositions({})
    expected = exported.module()(values)
    signature = str(exported.graph_signature)

    assert _cast_count(exported) == 1
    assert fold_scalar_constant_casts(exported) == 1
    exported.validate()
    actual = exported.module()(values)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert str(exported.graph_signature) == signature
    assert _cast_count(exported) == 0
    assert fold_scalar_constant_casts(exported) == 0
    original = (
        model.constant if storage == "literal" else model._lifted_tensor_constant0
    )
    assert original.dtype is torch.float32


def test_preserves_shared_state_dict_for_other_methods():
    values = torch.ones(4, dtype=torch.bfloat16)
    model = ScalarCast()
    first = torch.export.export(model, (values,)).run_decompositions({})
    second = torch.export.export(model, (values,)).run_decompositions({})
    second._state_dict = first.state_dict

    assert fold_scalar_constant_casts(first) == 1
    second.validate()
    assert second.state_dict["_lifted_tensor_constant0"].dtype is torch.float32
    assert _cast_count(second) == 1
    assert torch.equal(first.module()(values), second.module()(values))


@pytest.mark.parametrize(
    "case",
    [
        "mixed_users",
        "runtime_input",
        "ordinary_buffer",
        "vector",
        "device_cast",
        "output",
        "alias",
        "mutation",
    ],
)
def test_skips_casts_that_cannot_share_an_immutable_scalar(case):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            name = "scale" if case == "ordinary_buffer" else "_lifted_tensor_constant0"
            shape = [1.0] if case == "vector" else 1.0
            self.register_buffer(name, torch.tensor(shape, dtype=torch.float32))

        def forward(self, value):
            constant = (
                self.scale
                if case == "ordinary_buffer"
                else self._lifted_tensor_constant0
            )
            if case == "runtime_input":
                constant = value
            if case == "device_cast":
                converted = torch.ops.aten._to_copy.default(
                    constant, dtype=torch.bfloat16, device=torch.device("cpu")
                )
            else:
                converted = constant.to(torch.bfloat16)
            if case == "mixed_users":
                return value + converted, value + constant
            if case == "output":
                return converted
            if case == "alias":
                return converted.view(1) + value
            if case == "mutation":
                converted.add_(value.sum())
            return value + converted

    values = torch.ones(4, dtype=torch.float32)
    exported = torch.export.export(Model(), (values,))
    before = exported.graph_module.code

    assert fold_scalar_constant_casts(exported) == 0
    assert exported.graph_module.code == before
    exported.validate()


def test_preserves_mutated_lifted_buffers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_lifted_tensor_constant0", torch.tensor(1.0))

        def forward(self, value):
            self._lifted_tensor_constant0.add_(1)
            return value + self._lifted_tensor_constant0.to(torch.bfloat16)

    exported = torch.export.export(
        Model(), (torch.ones(4, dtype=torch.bfloat16),)
    ).run_decompositions({})
    assert exported.graph_signature.buffers_to_mutate
    assert fold_scalar_constant_casts(exported) == 0
    exported.validate()


def test_partition_compiler_folds_before_delegate_conversion(monkeypatch, tmp_path):
    values = torch.ones(4, dtype=torch.bfloat16)
    exported = torch.export.export(ScalarCast(), (values,)).run_decompositions({})
    observed = []

    def lower(program, specs, path):
        observed.append(_cast_count(program))
        program.validate()
        path.write_bytes(b"epair")
        return object()

    class Compiler:
        def compile(self, graph, output, **kwargs):
            output.with_suffix(".om").write_bytes(b"om")

    expected = object()
    monkeypatch.setattr(pipeline, "lower_to_ge_graph", lower)
    monkeypatch.setattr(pipeline, "offline_compiler", Compiler())
    monkeypatch.setattr(pipeline, "build_manifest", lambda **kwargs: expected)
    specs = AscendCompileOptions(
        method_name="decode", delegate_index=0, acl_soc_name="Ascend910B3"
    )

    result = pipeline.compile_partition(exported, specs, NamedDataStore(), tmp_path)

    assert result is expected
    assert observed == [0]
