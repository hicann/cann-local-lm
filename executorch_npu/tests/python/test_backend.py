# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from conftest import make_manifest, read_manifest_payload
from executorch.exir import to_edge, to_edge_transform_and_lower
from torch.export.graph_signature import OutputKind

import executorch_npu.edge_compile_config as executorch_compat
from executorch_npu import aten_edge_compile_config, Manifest
from executorch_npu.artifacts.io_contract import build_io_contract
from executorch_npu.backend import AscendBackend
from executorch_npu.compiler import (
    compatibility,
    delegate_adapter,
    pipeline as partition_compiler,
)
from executorch_npu.compiler.compatibility import (
    _patch_torchair_deterministic_options,
)
from executorch_npu.compiler.compile_spec import AscendCompileOptions
from executorch_npu.partitioner import AscendPartitioner
from executorch_npu.passes import (
    lower_edge_copy_ops,
    lower_edge_ops_to_aten,
    lower_static_split_with_sizes_copy,
)


def _io_contract_data(exported, source_contract=None):
    io, state = build_io_contract(exported, source_contract)
    return io.model_dump(exclude_none=True), state.model_dump(exclude_none=True)


class Add(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def _fake_torchair_backend(monkeypatch, options):
    backend = ModuleType("torchair.core._backend")
    backend._get_global_op_compile_config = lambda: dict(options)
    core = ModuleType("torchair.core")
    core._backend = backend
    torchair = ModuleType("torchair")
    torchair.core = core
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair.core", core)
    monkeypatch.setitem(sys.modules, "torchair.core._backend", backend)
    return backend


def test_torchair_options_are_compatible_without_torch_npu(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "0")
    backend = _fake_torchair_backend(
        monkeypatch,
        {"ge.deterministic": "1", "ge.deterministicLevel": "0"},
    )
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    _patch_torchair_deterministic_options()

    assert backend._get_global_op_compile_config() == {
        "ge.deterministic": "0",
        "ge.deterministicLevel": "0",
    }


def test_torchair_options_are_preserved_with_torch_npu(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "0")
    backend = _fake_torchair_backend(
        monkeypatch,
        {"ge.deterministic": "1", "ge.deterministicLevel": "2"},
    )
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    _patch_torchair_deterministic_options()

    assert backend._get_global_op_compile_config() == {
        "ge.deterministic": "1",
        "ge.deterministicLevel": "2",
    }


def test_torchair_compatibility_patch_is_idempotent(monkeypatch) -> None:
    backend = _fake_torchair_backend(monkeypatch, {"option": "value"})
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    _patch_torchair_deterministic_options()
    patched = backend._get_global_op_compile_config
    _patch_torchair_deterministic_options()

    assert backend._get_global_op_compile_config is patched


def test_dynamic_shape_contract_preserves_ranges_for_cann() -> None:
    class Identity(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

    batch = torch.export.Dim("batch", min=2, max=4)
    exported = torch.export.export(
        Identity(),
        (torch.ones(2, 4),),
        dynamic_shapes={"value": {0: batch}},
    )

    assert delegate_adapter.dynamic_input_shape_option(exported) == "2~4,4"
    contract, state = _io_contract_data(exported)
    assert state == {"bindings": []}
    for binding in contract["inputs"] + contract["outputs"]:
        assert binding["shape"] == [-1, 4]
        assert binding["shape_range"] == [[2, 4], [4, 4]]


def test_io_contract_preserves_repeated_output_slots() -> None:
    class Repeated(torch.nn.Module):
        def forward(self, value):
            output = value + 1
            return output, output

    exported = torch.export.export(Repeated(), (torch.ones(1, 4),))
    io, state = build_io_contract(exported)
    assert state.bindings == []
    assert [binding.et_arg_index for binding in io.outputs] == [1, 2]
    assert io.outputs[0].ascend_name == io.outputs[1].ascend_name


def test_io_contract_includes_buffer_mutation_outputs() -> None:
    class Stateful(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("cache", torch.zeros(1, 4), persistent=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.cache.copy_(value)
            return value + 1

    exported = torch.export.export(Stateful(), (torch.ones(1, 4),)).run_decompositions(
        {}
    )
    contract, state = _io_contract_data(exported)

    assert state == {
        "bindings": [
            {
                "state_id": "cache",
                "current_input_arg": 2,
                "next_output_arg": 3,
                "dtype": "float32",
                "shape": [1, 4],
                "reset_policy": "never",
                "reset_owner": False,
                "host_visibility": "device_only",
            }
        ]
    }
    assert [binding["semantic_role"] for binding in contract["inputs"]] == [
        "state_current",
        "data",
    ]
    assert [binding["ascend_name"] for binding in contract["outputs"]] == [
        *exported.graph_signature.buffers_to_mutate,
        *exported.graph_signature.user_outputs,
    ]
    assert [binding["semantic_role"] for binding in contract["outputs"]] == [
        "state_next",
        "data",
    ]
    assert [binding["et_arg_index"] for binding in contract["outputs"]] == [3, 1]


def test_io_contract_matches_reordered_buffer_mutations_by_identity() -> None:
    class Stateful(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("wide", torch.zeros(1, 4), persistent=False)
            self.register_buffer(
                "counter", torch.zeros((), dtype=torch.int64), persistent=False
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.wide.copy_(value)
            self.counter.add_(1)
            return value + 1

    exported = torch.export.export(Stateful(), (torch.ones(1, 4),)).run_decompositions(
        {}
    )
    mutation_specs = [
        spec
        for spec in exported.graph_signature.output_specs
        if spec.kind is OutputKind.BUFFER_MUTATION
    ]
    user_specs = [
        spec
        for spec in exported.graph_signature.output_specs
        if spec.kind is not OutputKind.BUFFER_MUTATION
    ]
    exported.graph_signature.output_specs = [
        *reversed(mutation_specs),
        *user_specs,
    ]

    contract, state = _io_contract_data(exported)
    io_by_arg = {
        binding["et_arg_index"]: binding
        for direction in ("inputs", "outputs")
        for binding in contract[direction]
    }
    for binding in state["bindings"]:
        current = io_by_arg[binding["current_input_arg"]]
        next_state = io_by_arg[binding["next_output_arg"]]
        assert current["dtype"] == binding["dtype"] == next_state["dtype"]
        assert current["shape"] == binding["shape"] == next_state["shape"]
    assert {binding["state_id"]: binding["shape"] for binding in state["bindings"]} == {
        "wide": [1, 4],
        "counter": [],
    }
    assert [binding["semantic_role"] for binding in contract["outputs"]] == [
        "state_next",
        "state_next",
        "data",
    ]
    assert [binding["dtype"] for binding in contract["outputs"]] == [
        "float32",
        "int64",
        "float32",
    ]
    assert [binding["shape"] for binding in contract["outputs"]] == [
        [1, 4],
        [],
        [1, 4],
    ]


@dataclass
class _FakeManifest:
    total_nodes: int = 1
    converted_nodes: int = 1


class _FakeLoader:
    active = False

    def __init__(self, path: Path) -> None:
        assert path.is_file()
        assert os.environ["TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"] == "0"

    def __enter__(self):
        type(self).active = True
        return self

    def __exit__(self, *args) -> None:
        type(self).active = False
        return None

    def get_ge_graph(self) -> object:
        return object()


class _FakeOfflineCompiler:
    def compile(
        self,
        graph,
        output,
        *,
        soc,
        options,
    ) -> None:
        assert graph is not None
        assert not _FakeLoader.active
        assert soc == "Ascend910_9362"
        self.build_options = dict(options)
        output.with_suffix(".om").write_bytes(b"fake-om")
        (output.parent / "weight" / "weight_0").write_bytes(b"fake-weight")


def _fake_torch_delegate():
    return SimpleNamespace(
        export_to_epair=lambda ep: SimpleNamespace(manifest=_FakeManifest()),
        epair_save=lambda contents, path: Path(path).write_bytes(b"epair"),
        EpairModelLoader=_FakeLoader,
    )


def _mock_ge_initialization(monkeypatch) -> None:
    """Keep fake partition tests from requiring a real TorchAir/GE runtime."""
    monkeypatch.setattr(delegate_adapter, "_initialize_ge", lambda soc: None)


def test_preprocess_builds_manifest_and_named_data(monkeypatch) -> None:
    _mock_ge_initialization(monkeypatch)
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    monkeypatch.setattr(partition_compiler, "offline_compiler", _FakeOfflineCompiler())
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    specs = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
    ).to_compile_specs()

    result = AscendBackend.preprocess(exported, specs)
    manifest = read_manifest_payload(result.processed_bytes)
    assert manifest["method"] == "forward"
    assert manifest["delegate_index"] == 0
    assert manifest["role"] == "inference"
    assert len(manifest["io_contract"]["inputs"]) == 2
    assert len(manifest["io_contract"]["outputs"]) == 1
    assert result.data_store_output is not None
    assert len(result.data_store_output.pte_data) == 1
    assert len(result.data_store_output.external_data) == 1


def test_compilation_failure_restores_delegate_environment(monkeypatch) -> None:
    _mock_ge_initialization(monkeypatch)
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    fallback = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
    monkeypatch.setenv(fallback, "1")
    outputs = []

    class FailingCompiler:
        def compile(self, graph, output, **kwargs):
            assert not _FakeLoader.active
            assert os.environ[fallback] == "1"
            output.with_suffix(".om").write_bytes(b"partial-model")
            (output.parent / "weight" / "weight_0").write_bytes(b"partial-weight")
            outputs.append(output)
            raise RuntimeError("simulated offline compilation failure")

    monkeypatch.setattr(partition_compiler, "offline_compiler", FailingCompiler())
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    specs = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
    ).to_compile_specs()
    with pytest.raises(RuntimeError, match="simulated offline compilation failure"):
        AscendBackend.preprocess(exported, specs)
    assert os.environ[fallback] == "1"
    assert not _FakeLoader.active
    assert outputs and not outputs[0].parent.exists()


def test_preprocess_adds_backend_owned_dynamic_input_shape(monkeypatch) -> None:
    _mock_ge_initialization(monkeypatch)
    compiler = _FakeOfflineCompiler()
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    monkeypatch.setattr(partition_compiler, "offline_compiler", compiler)

    class Identity(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

    batch = torch.export.Dim("batch", min=2, max=4)
    exported = torch.export.export(
        Identity(),
        (torch.ones(2, 4),),
        dynamic_shapes={"value": {0: batch}},
    )
    specs = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
    ).to_compile_specs()

    result = AscendBackend.preprocess(exported, specs)

    assert compiler.build_options == {"input_shape": "2~4,4"}
    manifest = read_manifest_payload(result.processed_bytes)
    assert manifest["io_contract"]["inputs"][0]["shape"] == [-1, 4]
    assert manifest["io_contract"]["inputs"][0]["shape_range"] == [
        [2, 4],
        [4, 4],
    ]


def test_multimethod_uses_shared_weight_data(monkeypatch) -> None:
    _mock_ge_initialization(monkeypatch)
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    monkeypatch.setattr(partition_compiler, "offline_compiler", _FakeOfflineCompiler())
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    results = AscendBackend.preprocess_multimethod(
        {"prefill": [exported], "decode": [exported]},
        {
            method_name: [
                AscendCompileOptions(
                    method_name=method_name,
                    delegate_index=0,
                    acl_soc_name="Ascend910_9362",
                ).to_compile_specs()
            ]
            for method_name in ("prefill", "decode")
        },
    )
    output = results["prefill"][0].data_store_output
    assert output is results["decode"][0].data_store_output
    assert output is not None
    assert len(output.buffers) == 2
    assert len(output.pte_data) == 2
    assert len(output.external_data["ascend.weights.package.ge_aot"]) == 1


def test_partial_epair_conversion_fails(monkeypatch) -> None:
    _mock_ge_initialization(monkeypatch)
    fake = _fake_torch_delegate()
    fake.export_to_epair = lambda ep: SimpleNamespace(
        manifest=_FakeManifest(total_nodes=2, converted_nodes=1)
    )
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", lambda: fake)
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    specs = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
    ).to_compile_specs()
    try:
        AscendBackend.preprocess(exported, specs)
    except RuntimeError as error:
        assert "did not fully convert" in str(error)
    else:
        raise AssertionError("partial conversion was accepted")


def test_missing_example_inputs_are_reconstructed() -> None:
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    exported._example_inputs = None

    delegate_adapter.reconstruct_example_inputs(exported)

    args, kwargs = exported.example_inputs
    assert kwargs == {}
    assert len(args) == 2
    assert all(tuple(value.shape) == (1, 4) for value in args)


def test_torch_delegate_controls_device_initialization_options(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "0")
    backend = _fake_torchair_backend(monkeypatch, {})
    calls = []
    configured_options = None

    def initialize_graph_engine(options=None):
        nonlocal configured_options
        options = dict(options or {})
        if configured_options is None:
            configured_options = options
        elif any(
            key not in configured_options or configured_options[key] != value
            for key, value in options.items()
        ):
            raise ValueError("conflicting TorchAir initialization")
        calls.append(options)

    backend.initialize_graph_engine = initialize_graph_engine
    ge_targets = []
    monkeypatch.setattr(
        delegate_adapter,
        "offline_compiler",
        SimpleNamespace(initialize=lambda soc: ge_targets.append(soc)),
    )
    delegate = _fake_torch_delegate()
    export_to_epair = delegate.export_to_epair
    compile_options = {"ge.exec.enableEngineParallel": "0"}

    def export_with_config(program):
        # TorchAir receives these options from the converter, not from the loader.
        backend.initialize_graph_engine(compile_options)
        return export_to_epair(program)

    delegate.export_to_epair = export_with_config
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", lambda: delegate)
    program = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    specs = AscendCompileOptions(
        method_name="forward", delegate_index=0, acl_soc_name="Ascend910B3"
    )
    delegate_adapter.lower_to_ge_graph(program, specs, tmp_path / "model.epair")

    assert calls[0] == compile_options
    assert ge_targets == ["Ascend910B3"]


def test_initialize_ge_is_host_only_without_a_device(monkeypatch) -> None:
    from executorch_npu.compiler.offline import OfflineCompiler

    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "1")
    calls = []
    ge_offline = ModuleType("ge.offline_compile")
    ge_offline.build_initialize = lambda options: calls.append(dict(options))
    ge_offline.build_finalize = lambda: calls.append("close")
    ge = ModuleType("ge")
    ge.offline_compile = ge_offline
    monkeypatch.setitem(sys.modules, "ge", ge)
    monkeypatch.setitem(sys.modules, "ge.offline_compile", ge_offline)
    monkeypatch.setitem(
        sys.modules, "torchair.core._backend", ModuleType("torchair.core._backend")
    )
    compiler = OfflineCompiler()
    monkeypatch.setattr(delegate_adapter, "offline_compiler", compiler)
    try:
        delegate_adapter._initialize_ge("Ascend910B3")
        delegate_adapter._initialize_ge("Ascend910B3")
        assert len(calls) == 1
        assert calls[0]["ge.socVersion"] == "Ascend910B3"
        assert calls[0]["ge.graphRunMode"] == "0"
        assert Path(calls[0]["ge.externalWeightDir"]).is_dir()
    finally:
        compiler.close()
    assert calls[-1] == "close"


def test_state_bindings_are_remapped_to_delegated_signature() -> None:
    class Stateful(torch.nn.Module):
        def forward(self, data, state_a, state_b):
            return data, state_b + 1, state_a + 1

    exported = torch.export.export(
        Stateful(),
        (
            torch.ones(1, 4),
            torch.ones(1, 2),
            torch.ones(1, 3),
        ),
    )
    # Simulate source-export positions after delegated graph extraction drops
    # a leading input; the graph signature presents state tensors at 1 and 2.
    source_contract = {
        "bindings": [
            {
                "state_id": "state-a",
                "current_input_arg": 2,
                "next_output_arg": 5,
                "dtype": "float32",
                "shape": [1, 2],
                "reset_policy": "on_prefill",
                "reset_owner": True,
                "host_visibility": "synchronized",
                "source_input_name": "state_a",
                "source_output_arg": 2,
            },
            {
                "state_id": "state-b",
                "current_input_arg": 3,
                "next_output_arg": 6,
                "dtype": "float32",
                "shape": [1, 3],
                "reset_policy": "on_prefill",
                "reset_owner": True,
                "host_visibility": "synchronized",
                "source_input_name": "state_b",
                "source_output_arg": 1,
            },
        ]
    }

    io_contract, state = _io_contract_data(exported, source_contract)

    assert [binding["state_id"] for binding in state["bindings"]] == [
        "state-a",
        "state-b",
    ]
    assert [binding["current_input_arg"] for binding in state["bindings"]] == [
        1,
        2,
    ]
    assert [binding["next_output_arg"] for binding in state["bindings"]] == [
        5,
        4,
    ]
    assert [binding["semantic_role"] for binding in io_contract["inputs"]] == [
        "data",
        "state_current",
        "state_current",
    ]
    assert [binding["semantic_role"] for binding in io_contract["outputs"]] == [
        "data",
        "state_next",
        "state_next",
    ]


def test_state_binding_remap_rejects_dtype_or_shape_mismatch() -> None:
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    contract = {
        "bindings": [
            {
                "state_id": "state",
                "current_input_arg": 0,
                "next_output_arg": 2,
                "dtype": "float16",
                "shape": [1, 4],
                "reset_policy": "on_prefill",
                "reset_owner": True,
                "host_visibility": "synchronized",
                "source_input_name": "x",
                "source_output_arg": 0,
            }
        ]
    }

    with pytest.raises(ValueError, match="state input metadata mismatch"):
        _io_contract_data(exported, contract)


def test_state_bindings_use_signature_identity_for_same_shape_states() -> None:
    class SameShapeStateful(torch.nn.Module):
        def forward(self, data, first, second):
            return data, second + 1, first + 1

    exported = torch.export.export(
        SameShapeStateful(),
        (torch.ones(1, 4), torch.ones(1, 2), torch.ones(1, 2)),
    )
    contract = {
        "bindings": [
            {
                "state_id": "first",
                "current_input_arg": 1,
                "next_output_arg": 5,
                "dtype": "float32",
                "shape": [1, 2],
                "reset_policy": "on_prefill",
                "reset_owner": True,
                "host_visibility": "synchronized",
                "source_input_name": "first",
                "source_output_arg": 2,
            },
            {
                "state_id": "second",
                "current_input_arg": 2,
                "next_output_arg": 4,
                "dtype": "float32",
                "shape": [1, 2],
                "reset_policy": "on_prefill",
                "reset_owner": True,
                "host_visibility": "synchronized",
                "source_input_name": "second",
                "source_output_arg": 1,
            },
        ]
    }

    _, state = _io_contract_data(exported, contract)

    assert [binding["current_input_arg"] for binding in state["bindings"]] == [1, 2]
    assert [binding["next_output_arg"] for binding in state["bindings"]] == [5, 4]


def test_static_split_with_sizes_copy_is_lowered_to_slices() -> None:
    class Split(torch.nn.Module):
        def forward(self, value):
            first, second, third = torch.ops.aten.split_with_sizes_copy.default(
                value, [2, 2, 2], -1
            )
            return first + second + third

    exported = torch.export.export(Split(), (torch.randn(1, 6),))

    expected = exported.module()(torch.arange(6, dtype=torch.float32).view(1, 6))

    lower_static_split_with_sizes_copy(exported)

    actual = exported.module()(torch.arange(6, dtype=torch.float32).view(1, 6))

    targets = {
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.split_with_sizes_copy.default not in targets
    assert torch.ops.aten.slice_copy.Tensor in targets
    torch.testing.assert_close(actual, expected)


def test_edge_copy_ops_are_lowered_to_torchair_targets() -> None:
    class CopyOps(torch.nn.Module):
        def forward(self, value):
            value = torch.ops.aten.view_copy.default(value, [2, 3])
            value = torch.ops.aten.unsqueeze_copy.default(value, 0)
            value = torch.ops.aten.expand_copy.default(value, [2, 2, 3])
            value = torch.ops.aten.permute_copy.default(value, [1, 0, 2])
            value = torch.ops.aten.select_copy.int(value, 0, 0)
            value = torch.ops.aten.slice_copy.Tensor(value, 1, 0, 2)
            value = torch.ops.aten.squeeze_copy.dims(value, [0])
            return torch.ops.aten.alias_copy.default(value)

    exported = torch.export.export(CopyOps(), (torch.randn(2, 3),))

    input_value = torch.arange(6, dtype=torch.float32).view(2, 3)
    expected = exported.module()(input_value)

    lower_edge_copy_ops(exported)

    actual = exported.module()(input_value)

    targets = {
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert targets == {
        torch.ops.aten.alias.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten.clone.default,
    }
    assert (
        sum(
            node.op == "call_function" and node.target is torch.ops.aten.clone.default
            for node in exported.graph_module.graph.nodes
        )
        == 8
    )
    torch.testing.assert_close(actual, expected)
    assert actual.data_ptr() != input_value.data_ptr()


def test_copy_default_is_preserved_for_torchair() -> None:
    class Copy(torch.nn.Module):
        def forward(self, destination, source):
            return torch.ops.aten.copy.default(destination, source)

    exported = torch.export.export(
        Copy(), (torch.zeros(2), torch.ones(2)), strict=False
    )

    lower_edge_copy_ops(exported)

    targets = {
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.copy.default in targets
    assert torch.ops.aten.alias.default not in targets


def test_edge_ops_are_lowered_to_aten_before_torch_delegate() -> None:
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    edge_program = to_edge(exported).exported_program()

    lower_edge_ops_to_aten(edge_program)
    delegate_adapter.validate_aten_targets(edge_program, "forward", 0)

    assert all(
        not type(node.target).__module__.startswith("executorch.exir.dialects.edge")
        for node in edge_program.graph_module.graph.nodes
        if node.op == "call_function"
    )


def test_aten_edge_compile_config_supports_known_api_shapes(monkeypatch) -> None:
    class LegacyConfig:
        def __init__(self, _use_edge_ops=True) -> None:
            self.use_edge_ops = _use_edge_ops

    monkeypatch.setattr(executorch_compat, "EdgeCompileConfig", LegacyConfig)
    assert executorch_compat.aten_edge_compile_config().use_edge_ops is False

    class CurrentConfig:
        def __init__(self, _skip_dim_order=False) -> None:
            self.skip_dim_order = _skip_dim_order

    monkeypatch.setattr(executorch_compat, "EdgeCompileConfig", CurrentConfig)
    assert executorch_compat.aten_edge_compile_config().skip_dim_order is True


def test_aten_edge_compile_config_uses_defaults_for_unknown_api(monkeypatch) -> None:
    class UnknownConfig:
        def __init__(self, unrelated=False) -> None:
            self.unrelated = unrelated

    monkeypatch.setattr(executorch_compat, "EdgeCompileConfig", UnknownConfig)
    assert executorch_compat.aten_edge_compile_config().unrelated is False


def test_identity_dim_order_clone_is_lowered_to_aten_clone() -> None:
    class DimOrderClone(torch.nn.Module):
        def forward(self, value):
            return torch.ops.dim_order_ops._clone_dim_order.default(
                value, dim_order=[0, 1]
            )

    exported = torch.export.export(DimOrderClone(), (torch.randn(2, 3),))
    input_value = torch.arange(6, dtype=torch.float32).view(2, 3)
    expected = exported.module()(input_value)

    lower_edge_copy_ops(exported)

    actual = exported.module()(input_value)
    targets = {
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.clone.default in targets
    assert all("clone_dim_order" not in str(target) for target in targets)
    torch.testing.assert_close(actual, expected)
    assert actual.data_ptr() != input_value.data_ptr()


def test_squeeze_copy_lowering_rejects_noncontiguous_input() -> None:
    class NoncontiguousSqueeze(torch.nn.Module):
        def forward(self, value):
            transposed = torch.ops.aten.transpose.int(value, 0, 1)
            expanded = torch.ops.aten.unsqueeze.default(transposed, 0)
            return torch.ops.aten.squeeze_copy.dims(expanded, [0])

    exported = torch.export.export(NoncontiguousSqueeze(), (torch.randn(2, 3),))

    with pytest.raises(ValueError, match="contiguous input tensor"):
        lower_edge_copy_ops(exported)


def test_partitioner_preserves_matmul_introduced_by_decomposition() -> None:
    linear = torch.nn.Linear(8, 4).eval()
    exported = torch.export.export(linear, (torch.randn(2, 8),))
    options = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
    )
    partitioner = AscendPartitioner(options.to_compile_specs())

    preserved, predicate = partitioner.ops_to_not_decompose(exported)

    assert predicate is None
    assert preserved == [torch.ops.aten.matmul.default]


def test_fallback_codegen_is_disabled_and_environment_restored(
    monkeypatch,
) -> None:
    name = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TORCH_PKG_DIR", raising=False)
    monkeypatch.delenv("TORCH_NPU_DIR", raising=False)

    with delegate_adapter._fallback_codegen_disabled():
        assert os.environ[name] == "0"
        assert "TORCH_PKG_DIR" not in os.environ
        assert "TORCH_NPU_DIR" not in os.environ

    assert name not in os.environ
    assert "TORCH_PKG_DIR" not in os.environ
    assert "TORCH_NPU_DIR" not in os.environ

    monkeypatch.setenv(name, "1")
    with delegate_adapter._fallback_codegen_disabled():
        assert os.environ[name] == "0"
    assert os.environ[name] == "1"


def test_public_flow_writes_one_pte_and_one_ptd(monkeypatch, tmp_path) -> None:
    _mock_ge_initialization(monkeypatch)
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    monkeypatch.setattr(partition_compiler, "offline_compiler", _FakeOfflineCompiler())
    # Source-tree ExecuTorch does not install its FlatBuffer schema resources.
    # An installed package contains them; route only the source-tree test to
    # the canonical schema directory.
    from executorch.exir._serialize import _flatbuffer

    original_read_binary = _flatbuffer.importlib.resources.read_binary
    executorch_root = Path(_flatbuffer.__file__).parents[2]

    def read_binary(package, name):
        schema_path = executorch_root / "schema" / name
        if name in {"program.fbs", "scalar_type.fbs"} and schema_path.is_file():
            return schema_path.read_bytes()
        return original_read_binary(package, name)

    monkeypatch.setattr(_flatbuffer.importlib.resources, "read_binary", read_binary)
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    options = AscendCompileOptions(
        method_name="forward",
        delegate_index=0,
        acl_soc_name="Ascend910_9362",
    )

    program = to_edge_transform_and_lower(
        exported,
        partitioner=[AscendPartitioner(options.to_compile_specs())],
        compile_config=aten_edge_compile_config(),
    ).to_executorch()
    pte = tmp_path / "model.pte"
    with pte.open("wb") as output:
        output.write(program.buffer)
    program.write_tensor_data_to_file(tmp_path)
    ptd = tmp_path / "ascend.weights.package.ge_aot.ptd"

    assert pte.name == "model.pte"
    assert pte.stat().st_size > 0
    assert ptd.name == "ascend.weights.package.ge_aot.ptd"
    assert ptd.stat().st_size > 0
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".ptd", ".pte"]


def test_unused_state_keeps_no_output_sentinel() -> None:
    class ReadOnlyState(torch.nn.Module):
        def forward(self, data, cache):
            return data + 1

    exported = torch.export.export(
        ReadOnlyState(), (torch.ones(1, 4), torch.zeros(1, 2))
    )
    contract, state = build_io_contract(
        exported,
        {
            "bindings": [
                {
                    "state_id": "cache",
                    "dtype": "float32",
                    "shape": [1, 2],
                    "reset_policy": "never",
                    "reset_owner": False,
                    "host_visibility": "device_only",
                    "source_input_name": "cache",
                }
            ]
        },
    )
    assert state.bindings[0].current_input_arg == 1
    assert state.bindings[0].next_output_arg == -1
    assert contract.inputs[1].semantic_role == "state_current"
    assert all(binding.semantic_role == "data" for binding in contract.outputs)


def test_multimethod_checks_typed_state_before_serializing(monkeypatch) -> None:
    import executorch_npu.backend as backend_module

    serialized = []
    real_to_bytes = Manifest.to_bytes

    def to_bytes(manifest):
        serialized.append(manifest.method)
        return real_to_bytes(manifest)

    def compile_partition(program, specs, store, staging):
        data = make_manifest()
        data.update(method=specs.method_name, role=specs.role)
        data["io_contract"]["inputs"][0]["semantic_role"] = "state_current"
        data["io_contract"]["outputs"][0]["semantic_role"] = "state_next"
        data["state"]["bindings"] = [
            {
                "state_id": "cache",
                "current_input_arg": 0,
                "next_output_arg": 1,
                "dtype": "float32",
                "shape": [1, 4],
                "reset_policy": "on_prefill",
                "reset_owner": False,
                "host_visibility": "device_only",
            }
        ]
        return Manifest.model_validate(data)

    monkeypatch.setattr(backend_module, "compile_partition", compile_partition)
    monkeypatch.setattr(Manifest, "to_bytes", to_bytes)
    with pytest.raises(ValueError, match="exactly one prefill reset owner"):
        AscendBackend.preprocess_multimethod(
            {"prefill": [None], "decode": [None]},
            {
                method_name: [
                    AscendCompileOptions(
                        method_name=method_name,
                        delegate_index=0,
                        acl_soc_name="Ascend910_9362",
                    ).to_compile_specs()
                ]
                for method_name in ("prefill", "decode")
            },
        )
    assert serialized == []


def test_lower_to_ge_graph_returns_graph_after_closing_loader(monkeypatch, tmp_path):
    _mock_ge_initialization(monkeypatch)
    name = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
    monkeypatch.setenv(name, "1")
    graph = object()

    class Loader(_FakeLoader):
        def get_ge_graph(self):
            assert self.active
            assert os.environ[name] == "0"
            return graph

    fake = _fake_torch_delegate()
    fake.EpairModelLoader = Loader
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", lambda: fake)
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    specs = AscendCompileOptions(
        method_name="forward", delegate_index=0, acl_soc_name="Ascend910_9362"
    )

    result = delegate_adapter.lower_to_ge_graph(
        exported, specs, tmp_path / "model.epair"
    )

    assert result is graph
    assert not Loader.active
    assert os.environ[name] == "1"


@pytest.mark.parametrize("stage", ["load", "copy"])
def test_graph_loading_failure_restores_environment_and_closes_loader(
    monkeypatch, tmp_path, stage
):
    _mock_ge_initialization(monkeypatch)
    name = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
    monkeypatch.delenv(name, raising=False)
    closed = []

    class Loader(_FakeLoader):
        def __init__(self, path):
            super().__init__(path)
            if stage == "load":
                raise RuntimeError("loading failed")

        def get_ge_graph(self):
            assert os.environ[name] == "0"
            raise RuntimeError("copy failed")

        def __exit__(self, *args):
            super().__exit__(*args)
            closed.append(True)

    fake = _fake_torch_delegate()
    fake.EpairModelLoader = Loader
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", lambda: fake)
    exported = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
    specs = AscendCompileOptions(
        method_name="forward", delegate_index=0, acl_soc_name="Ascend910_9362"
    )

    with pytest.raises(RuntimeError, match="failed"):
        delegate_adapter.lower_to_ge_graph(exported, specs, tmp_path / "model.epair")

    assert name not in os.environ
    assert not Loader.active
    assert closed == ([True] if stage == "copy" else [])


def test_dynamo_timer_compatibility_preserves_metrics_and_existing_references(
    monkeypatch,
):
    from torch._dynamo import utils as dynamo_utils

    monkeypatch.setattr(dynamo_utils, "compilation_time_metrics", {"existing": [1.0]})
    compatibility._patch_dynamo_compilation_metrics()
    metrics = dynamo_utils.compilation_time_metrics
    metrics["new"].append(2.0)

    compatibility._patch_dynamo_compilation_metrics()

    assert dynamo_utils.compilation_time_metrics is metrics
    assert metrics == {"existing": [1.0], "new": [2.0]}


def test_linear_decomposition_validates_without_global_matmul_exception(monkeypatch):
    _mock_ge_initialization(monkeypatch)
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    monkeypatch.setattr(partition_compiler, "offline_compiler", _FakeOfflineCompiler())
    model = torch.nn.Linear(8, 4, bias=False).eval()
    exported = torch.export.export(model, (torch.randn(2, 3, 8),))
    options = AscendCompileOptions(
        method_name="forward", delegate_index=0, acl_soc_name="Ascend910_9362"
    )
    config = aten_edge_compile_config()
    if hasattr(config, "_core_aten_ops_exception_list"):
        assert torch.ops.aten.matmul.default not in config._core_aten_ops_exception_list
    to_edge_transform_and_lower(
        exported,
        partitioner=[AscendPartitioner(options.to_compile_specs())],
        compile_config=config,
    )
