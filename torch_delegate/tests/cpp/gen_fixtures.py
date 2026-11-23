#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Generate .epair test fixtures for the C++ unit tests.

Produces files in build/tests/cpp/fixtures/ (by default):
  all_npu.epair         — mm + abs, both nodes have ascend_ir (all NPU)
  mixed.epair           — mm (NPU) + relu (CPU), two segments
  all_cpu.epair         — relu + relu, no ascend_ir (all CPU)
  empty_compute.epair   — identity f(x)=x, no compute nodes (only placeholder/output)
  npu_cpu_npu.epair     — abs (NPU) + relu (CPU) + abs (NPU), three alternating segments
  shared_input.epair    — mm (NPU) + relu (CPU), both consume the same graph input "x"
  non_adjacent.epair    — mm (NPU) + relu (CPU) + abs (NPU), unit 0 output consumed by unit 2
  no_model_json.epair   — valid epair structure but missing models/model.json entry
  annotated_mm_add.epair — annotated graph source in extra/ep_json_with_ascend_ir.json
  annotated_mixed.epair  — annotated graph source containing both NPU and CPU nodes
  annotated_attr_types.epair — annotated graph source covering supported attr payload kinds
  missing_graph_source.epair — valid archive missing extra/ep_json_with_ascend_ir.json

Run from the repository root:
    python tests/cpp/gen_fixtures.py
    python tests/cpp/gen_fixtures.py --output-dir /path/to/build/tests/cpp/fixtures
"""

import argparse
import base64
import json
import os
import struct
import sys

from torch.export.pt2_archive import PT2ArchiveWriter

# Ensure the package is importable when run from the repo root
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
FIXTURES_DIR = os.path.join(_REPO_ROOT, "build", "tests", "cpp", "fixtures")

# ---------------------------------------------------------------------------
# Minimal manifest written into every fixture so EpairArchiveReader can open
# the file without issues (model.json presence is checked separately).
# ---------------------------------------------------------------------------
_MANIFEST_PATH = "extra/epair_manifest.json"


ArchiveWriter = PT2ArchiveWriter


def _write_manifest(
    writer: ArchiveWriter,
    total_nodes: int = 0,
    converted: int = 0,
    version: int = 1,
    cpu_nodes: int | None = None,
    cann_version: str = "9.2.0",
) -> None:
    manifest = {
        "version": version,
        "backend_name": "ascend_ge",
        "total_nodes": total_nodes,
        "converted_nodes": converted,
    }
    if version >= 2:
        manifest["cann_version"] = cann_version
    if cpu_nodes is not None:
        manifest["cpu_nodes"] = cpu_nodes
    writer.write_string(_MANIFEST_PATH, json.dumps(manifest))


# ---------------------------------------------------------------------------
# Helpers: build minimal EP JSON dicts without a real ExportedProgram
# ---------------------------------------------------------------------------


def _node(
    target: str, inputs: list, outputs: list, ascend_ir: dict | None = None
) -> dict:
    """Build a single EP JSON node dict."""
    n: dict = {"target": target, "inputs": inputs, "outputs": outputs}
    if ascend_ir is not None:
        n["ascend_ir"] = ascend_ir
    return n


def _as_tensor(name: str) -> dict:
    return {"as_tensor": {"name": name}}


def _ep_json(
    graph_inputs: list[str], graph_outputs: list[str], nodes: list[dict]
) -> dict:
    """Wrap nodes into the EP JSON envelope."""
    return {
        "graph_module": {
            "graph": {
                "inputs": [_as_tensor(n) for n in graph_inputs],
                "outputs": [_as_tensor(n) for n in graph_outputs],
                "nodes": nodes,
            }
        }
    }


def _arg(name: str, tensor_name: str) -> dict:
    """Build an ARG-kind input referencing a tensor."""
    return {"name": name, "arg": _as_tensor(tensor_name), "kind": "ARG"}


def gen_torch_export_fixtures() -> None:
    nodes = [
        _node(
            "torch.ops.aten.mm.default",
            inputs=[_arg("self", "x"), _arg("mat2", "y")],
            outputs=[_as_tensor("mm_1")],
            ascend_ir={"op_type": "MatMul", "attrs": {}},
        ),
        _node(
            "torch.ops.aten.abs.default",
            inputs=[_arg("self", "mm_1")],
            outputs=[_as_tensor("abs_1")],
            ascend_ir={"op_type": "Abs", "attrs": {}},
        ),
    ]
    ep_json_dict = _ep_json(["x", "y"], ["abs_1"], nodes)
    out = os.path.join(FIXTURES_DIR, "all_npu.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=2, converted=2)
    print("Generated: all_npu.epair")

    nodes = [
        _node(
            "torch.ops.aten.mm.default",
            inputs=[_arg("self", "x"), _arg("mat2", "y")],
            outputs=[_as_tensor("mm_1")],
            ascend_ir={"op_type": "MatMul", "attrs": {}},
        ),
        _node(
            "torch.ops.aten.relu.default",
            inputs=[_arg("self", "mm_1")],
            outputs=[_as_tensor("relu_1")],
        ),
    ]
    ep_json_dict = _ep_json(["x", "y"], ["relu_1"], nodes)
    out = os.path.join(FIXTURES_DIR, "mixed.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=2, converted=1)
    print("Generated: mixed.epair")

    nodes = [
        _node(
            "torch.ops.aten.relu.default",
            inputs=[_arg("self", "x")],
            outputs=[_as_tensor("relu_1")],
        ),
        _node(
            "torch.ops.aten.relu.default",
            inputs=[_arg("self", "relu_1")],
            outputs=[_as_tensor("relu_2")],
        ),
    ]
    ep_json_dict = _ep_json(["x"], ["relu_2"], nodes)
    out = os.path.join(FIXTURES_DIR, "all_cpu.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=2, converted=0)
    print("Generated: all_cpu.epair")


# ---------------------------------------------------------------------------
# Fixtures built from hand-crafted EP JSON (new)
# ---------------------------------------------------------------------------


def gen_empty_compute() -> None:
    """No compute nodes — only placeholder and output are present in the graph.
    The model.json has a graph with zero nodes in the 'nodes' array.
    GraphPartitioner must produce an empty execution_units list with EMPTY_GRAPH warning.
    """
    ep_json_dict = _ep_json(
        graph_inputs=["x"],
        graph_outputs=["x"],
        nodes=[],
    )
    out = os.path.join(FIXTURES_DIR, "empty_compute.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer)
    print("Generated: empty_compute.epair")


def gen_npu_cpu_npu() -> None:
    """Three nodes in NPU→CPU→NPU order.
    abs_1 (NPU) → relu_1 (CPU) → abs_2 (NPU)
    Produces 3 ExecutionUnits alternating device.
    """
    nodes = [
        _node(
            "torch.ops.aten.abs.default",
            inputs=[_arg("x", "x")],
            outputs=[_as_tensor("abs_1")],
            ascend_ir={"op_type": "Abs", "attrs": {}},
        ),
        _node(
            "torch.ops.aten.relu.default",
            inputs=[_arg("self", "abs_1")],
            outputs=[_as_tensor("relu_1")],
            # No ascend_ir → CPU
        ),
        _node(
            "torch.ops.aten.abs.default",
            inputs=[_arg("self", "relu_1")],
            outputs=[_as_tensor("abs_2")],
            ascend_ir={"op_type": "Abs", "attrs": {}},
        ),
    ]
    ep_json_dict = _ep_json(["x"], ["abs_2"], nodes)
    out = os.path.join(FIXTURES_DIR, "npu_cpu_npu.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=3, converted=2)
    print("Generated: npu_cpu_npu.epair")


def gen_shared_input() -> None:
    """Graph input 'x' is consumed by both an NPU unit (mm) and a CPU unit (relu).
    mm (NPU, inputs: x, y) → relu (CPU, inputs: x)
    Both units must list 'x' in their inputs.
    """
    nodes = [
        _node(
            "torch.ops.aten.mm.default",
            inputs=[_arg("self", "x"), _arg("mat2", "y")],
            outputs=[_as_tensor("mm_1")],
            ascend_ir={"op_type": "MatMul", "attrs": {}},
        ),
        _node(
            "torch.ops.aten.relu.default",
            inputs=[_arg("self", "x")],
            outputs=[_as_tensor("relu_1")],
            # No ascend_ir → CPU
        ),
    ]
    ep_json_dict = _ep_json(["x", "y"], ["mm_1", "relu_1"], nodes)
    out = os.path.join(FIXTURES_DIR, "shared_input.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=2, converted=1)
    print("Generated: shared_input.epair")


def gen_non_adjacent() -> None:
    """Unit 0 (NPU: mm) output 'mm_1' is consumed by unit 2 (NPU: abs).
    Unit 1 (CPU: relu) sits between them and does NOT consume 'mm_1'.
    mm (NPU) → relu (CPU, input: x) → abs (NPU, input: mm_1)
    'mm_1' must appear in unit 0's outputs AND unit 2's inputs,
    but NOT in unit 1's inputs.
    """
    nodes = [
        _node(
            "torch.ops.aten.mm.default",
            inputs=[_arg("self", "x"), _arg("mat2", "y")],
            outputs=[_as_tensor("mm_1")],
            ascend_ir={"op_type": "MatMul", "attrs": {}},
        ),
        _node(
            "torch.ops.aten.relu.default",
            inputs=[_arg("self", "x")],
            outputs=[_as_tensor("relu_1")],
            # No ascend_ir → CPU; deliberately does NOT consume mm_1
        ),
        _node(
            "torch.ops.aten.abs.default",
            inputs=[_arg("self", "mm_1")],
            outputs=[_as_tensor("abs_1")],
            ascend_ir={"op_type": "Abs", "attrs": {}},
        ),
    ]
    ep_json_dict = _ep_json(["x", "y"], ["relu_1", "abs_1"], nodes)
    out = os.path.join(FIXTURES_DIR, "non_adjacent.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=3, converted=2)
    print("Generated: non_adjacent.epair")


def gen_no_model_json() -> None:
    """A valid epair archive that intentionally omits models/model.json.
    EpairPartitioner::Partition must throw PartitionError(MISSING_MODEL_JSON).
    """
    out = os.path.join(FIXTURES_DIR, "no_model_json.epair")
    with ArchiveWriter(out) as writer:
        # Write only the manifest; deliberately skip models/model.json
        _write_manifest(writer)
    print("Generated: no_model_json.epair")


def gen_missing_op_type() -> None:
    """A valid epair whose model.json contains a node with ascend_ir but no op_type.
    EpairPartitioner::Partition must throw PartitionError containing MISSING_OP_TYPE.
    """
    nodes = [
        _node(
            "torch.ops.aten.mm.default",
            inputs=[_arg("self", "x"), _arg("mat2", "y")],
            outputs=[_as_tensor("mm_1")],
            ascend_ir={"attrs": {}},  # op_type deliberately omitted
        ),
    ]
    ep_json_dict = _ep_json(["x", "y"], ["mm_1"], nodes)
    out = os.path.join(FIXTURES_DIR, "missing_op_type.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(ep_json_dict))
        _write_manifest(writer, total_nodes=1, converted=0)
    print("Generated: missing_op_type.epair")


def _tensor_meta(shape: list[int], requires_grad: bool, dtype: int = 7) -> dict:
    strides = []
    running = 1
    for size in reversed(shape):
        strides.append(running)
        running *= size
    strides.reverse()
    return {
        "dtype": dtype,
        "sizes": [{"as_int": v} for v in shape],
        "requires_grad": requires_grad,
        "device": {"type": "cpu", "index": None},
        "strides": [{"as_int": v} for v in strides],
        "storage_offset": {"as_int": 0},
        "layout": 7,
    }


def _float_bytes(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def _int64_bytes(values: list[int]) -> bytes:
    return struct.pack("<" + "q" * len(values), *values)


def _const_tensor_attr_int64(values: list[int]) -> dict:
    return {
        "value": {
            "t": {
                "desc": {
                    "dtype": "DT_INT64",
                    "shape": {"dim": [str(len(values))]},
                    "layout": "ND",
                },
                "data": base64.b64encode(_int64_bytes(values)).decode("ascii"),
            }
        }
    }


def _const_tensor_attr_float_scalar(value: float) -> dict:
    return {
        "value": {
            "t": {
                "desc": {
                    "dtype": "DT_FLOAT",
                    "shape": {},
                    "layout": "ND",
                },
                "data": base64.b64encode(_float_bytes([value])).decode("ascii"),
            }
        },
        "_readable_value": {
            "s": base64.b64encode(str(value).encode("utf-8")).decode("ascii"),
        },
    }


def gen_annotated_mm_add() -> None:
    graph = {
        "inputs": [_as_tensor("p_weight"), _as_tensor("b_bias"), _as_tensor("x")],
        "outputs": [_as_tensor("add")],
        "nodes": [
            {
                "target": "torch.ops.aten.mm.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "mat2", "arg": _as_tensor("p_weight"), "kind": 1},
                ],
                "outputs": [_as_tensor("mm")],
                "metadata": {"torch_fn": "mm_1;builtin_function_or_method.mm"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.add.Tensor",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("mm"), "kind": 1},
                    {"name": "other", "arg": _as_tensor("b_bias"), "kind": 1},
                ],
                "outputs": [_as_tensor("add")],
                "metadata": {"torch_fn": "add_1;method_descriptor.add"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "p_weight": _tensor_meta([3, 3], True),
            "b_bias": _tensor_meta([3], False),
            "x": _tensor_meta([3, 3], False),
            "mm": _tensor_meta([3, 3], False),
            "add": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
            {
                "buffer": {
                    "arg": {"name": "b_bias"},
                    "buffer_name": "bias",
                    "persistent": True,
                }
            },
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("add")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "MatMul",
                "type": "MatMul",
                "input": ["x", "p_weight", ""],
                "attr": {
                    "transpose_x1": {"b": False},
                    "transpose_x2": {"b": False},
                },
                "inputDesc": [
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {
                        "name": "x2",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {"name": "bias", "attr": {"_is_unfed_optional": {"i": "1"}}},
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"mm": "MatMul:0"}},
    }
    annotated_json["graph_module"]["graph"]["nodes"][1]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "Add",
                "type": "Add",
                "input": ["mm", "b_bias"],
                "attr": {},
                "inputDesc": [
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                    {
                        "name": "x2",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {"mm": "MatMul:0"}, "outputs": {"add": "Add:0"}},
    }

    weights_config = {
        "config": {
            "weight": {
                "path_name": "weight_0",
                "is_param": True,
                "use_pickle": False,
                "tensor_meta": _tensor_meta([3, 3], True),
            },
            "bias": {
                "path_name": "weight_1",
                "is_param": False,
                "use_pickle": False,
                "tensor_meta": _tensor_meta([3], False),
            },
        }
    }
    constants_config = {"config": {}}

    def _write(filename: str, annotated: dict) -> None:
        out = os.path.join(FIXTURES_DIR, filename)
        with ArchiveWriter(out) as writer:
            writer.write_string("models/model.json", json.dumps(model_json))
            writer.write_string(
                "extra/ep_json_with_ascend_ir.json", json.dumps(annotated)
            )
            writer.write_string(
                "data/weights/model_weights_config.json", json.dumps(weights_config)
            )
            writer.write_bytes(
                "data/weights/weight_0",
                _float_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            )
            writer.write_bytes("data/weights/weight_1", _float_bytes([0.5, 1.5, 2.5]))
            writer.write_string(
                "data/constants/model_constants_config.json",
                json.dumps(constants_config),
            )
            _write_manifest(writer, total_nodes=2, converted=2)
        print(f"Generated: {filename}")

    _write("annotated_mm_add.epair", annotated_json)
    # 畸形 shape.dim 触发 std::stoll 异常，用于验证 C ABI 异常边界。
    malformed_json = json.loads(json.dumps(annotated_json))
    malformed_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0][
        "inputDesc"
    ][0]["shape"]["dim"][0] = "abc"
    _write("annotated_mm_add_bad_shape_dim.epair", malformed_json)


def gen_annotated_mixed() -> None:
    graph = {
        "inputs": [_as_tensor("p_weight"), _as_tensor("x")],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.mm.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "mat2", "arg": _as_tensor("p_weight"), "kind": 1},
                ],
                "outputs": [_as_tensor("mm")],
                "metadata": {"torch_fn": "mm_1;builtin_function_or_method.mm"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("mm"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "p_weight": _tensor_meta([3, 3], True),
            "x": _tensor_meta([3, 3], False),
            "mm": _tensor_meta([3, 3], False),
            "relu": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "MatMul",
                "type": "MatMul",
                "input": ["x", "p_weight", ""],
                "attr": {
                    "transpose_x1": {"b": False},
                    "transpose_x2": {"b": False},
                },
                "inputDesc": [
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {
                        "name": "x2",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {"name": "bias", "attr": {"_is_unfed_optional": {"i": "1"}}},
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"mm": "MatMul:0"}},
    }

    weights_config = {
        "config": {
            "weight": {
                "path_name": "weight_0",
                "is_param": True,
                "use_pickle": False,
                "tensor_meta": _tensor_meta([3, 3], True),
            },
        }
    }
    out = os.path.join(FIXTURES_DIR, "annotated_mixed.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps(weights_config)
        )
        writer.write_bytes(
            "data/weights/weight_0",
            _float_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=2, converted=1)
    print("Generated: annotated_mixed.epair")


def gen_annotated_host_cpu() -> None:
    """mm（converter/NPU）+ relu（fallback/CPU）混合图，manifest v2。

    relu 节点带 ascend_ir（source="" + device="cpu"），验证三分派构图与
    HostCpu 能力校验。
    """
    graph = {
        "inputs": [_as_tensor("p_weight"), _as_tensor("x")],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.mm.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "mat2", "arg": _as_tensor("p_weight"), "kind": 1},
                ],
                "outputs": [_as_tensor("mm")],
                "metadata": {"torch_fn": "mm_1;builtin_function_or_method.mm"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("mm"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "p_weight": _tensor_meta([3, 3], True),
            "x": _tensor_meta([3, 3], False),
            "mm": _tensor_meta([3, 3], False),
            "relu": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "MatMul",
                "type": "MatMul",
                "input": ["x", "p_weight", ""],
                "attr": {
                    "transpose_x1": {"b": False},
                    "transpose_x2": {"b": False},
                },
                "inputDesc": [
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {
                        "name": "x2",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {"name": "bias", "attr": {"_is_unfed_optional": {"i": "1"}}},
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"mm": "MatMul:0"}},
    }
    # 场景 3：CPU 节点（HostCpu 后端 fallback）
    annotated_json["graph_module"]["graph"]["nodes"][1]["ascend_ir"] = {
        "source": "",
        "device": "cpu",
        "ops": [{"attr": {}}],
        "mapping": {"inputs": {}, "outputs": {"relu": "aten__relu:0"}},
    }

    weights_config = {
        "config": {
            "weight": {
                "path_name": "weight_0",
                "is_param": True,
                "use_pickle": False,
                "tensor_meta": _tensor_meta([3, 3], True),
            },
        }
    }
    # Device 归属判定记录已在 v2 中移除：C++ 加载期由 ascend_ir.device
    # 与节点 target namespace 直接推断，不再落盘 extra/epair_device_decisions.json。
    out = os.path.join(FIXTURES_DIR, "annotated_host_cpu.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps(weights_config)
        )
        writer.write_bytes(
            "data/weights/weight_0",
            _float_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=2, converted=2, version=2, cpu_nodes=1)
    print("Generated: annotated_host_cpu.epair")


def gen_annotated_host_cpu_non4byte_inputs() -> None:
    """两个 CPU fallback 节点分别消费 int64 / fp16 用户 Data 输入。

    回归用例：GE runtime 的 D2H staging 按 desc dtype 计算长度，HostCpu 用户
    Data 输入不应再被“非 4 字节 dtype”门禁拒绝（历史实现对 8/2 字节 dtype
    直接返回 UNIMPLEMENTED）。
    """
    graph = {
        "inputs": [_as_tensor("x_i64"), _as_tensor("x_f16")],
        "outputs": [_as_tensor("relu_i64"), _as_tensor("relu_f16")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x_i64"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu_i64")],
                "metadata": {"torch_fn": "relu_i64;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x_f16"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu_f16")],
                "metadata": {"torch_fn": "relu_f16;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            # pt2 serde dtype：5 = Long(int64)，6 = Half(fp16)，7 = Float。
            "x_i64": _tensor_meta([3, 3], False, dtype=5),
            "relu_i64": _tensor_meta([3, 3], False, dtype=5),
            "x_f16": _tensor_meta([3, 3], False, dtype=6),
            "relu_f16": _tensor_meta([3, 3], False, dtype=6),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x_i64")}},
            {"user_input": {"arg": _as_tensor("x_f16")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu_i64")}},
            {"user_output": {"arg": _as_tensor("relu_f16")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    for node, out_name in zip(
        annotated_json["graph_module"]["graph"]["nodes"], ["relu_i64", "relu_f16"]
    ):
        # mapping 别名在全图内必须唯一，否则两个 CPU 节点会在 BindDeferredOutputs
        # 争用同一个别名（alias already bound）。
        node["ascend_ir"] = {
            "source": "",
            "device": "cpu",
            "ops": [{"attr": {}}],
            "mapping": {
                "inputs": {},
                "outputs": {out_name: f"aten__relu_{out_name}:0"},
            },
        }

    out = os.path.join(FIXTURES_DIR, "annotated_host_cpu_non4byte_inputs.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=2, converted=2, version=2, cpu_nodes=2)
    print("Generated: annotated_host_cpu_non4byte_inputs.epair")


def gen_annotated_host_cpu_default_args() -> None:
    """CPU fallback 节点省略中间默认参数（beta/alpha），仅传 self/mat1/mat2/use_gelu。

    回归 0012：输入必须按 schema 参数名对齐 IR 槽位；use_gelu 落在自己的槽位，
    beta/alpha 保留空槽，Execute 侧再物化 Scalar/bool 默认值。
    """
    graph = {
        "inputs": [_as_tensor("self"), _as_tensor("mat1"), _as_tensor("mat2")],
        "outputs": [_as_tensor("addmm_out")],
        "nodes": [
            {
                "name": "addmm_act",
                "target": "torch.ops.aten._addmm_activation.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("self"), "kind": 1},
                    {"name": "mat1", "arg": _as_tensor("mat1"), "kind": 1},
                    {"name": "mat2", "arg": _as_tensor("mat2"), "kind": 1},
                    {"name": "use_gelu", "arg": {"as_bool": False}, "kind": 1},
                ],
                "outputs": [_as_tensor("addmm_out")],
                "metadata": {
                    "torch_fn": "addmm_act;builtin_function_or_method._addmm_activation"
                },
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "self": _tensor_meta([4, 4], False),
            "mat1": _tensor_meta([4, 8], False),
            "mat2": _tensor_meta([8, 4], False),
            "addmm_out": _tensor_meta([4, 4], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("self")}},
            {"user_input": {"arg": _as_tensor("mat1")}},
            {"user_input": {"arg": _as_tensor("mat2")}},
        ],
        "output_specs": [{"user_output": {"arg": _as_tensor("addmm_out")}}],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "",
        "device": "cpu",
        "ops": [{"attr": {}}],
        "mapping": {
            "inputs": {},
            "outputs": {"addmm_out": "aten___addmm_activation:0"},
        },
    }

    out = os.path.join(FIXTURES_DIR, "annotated_host_cpu_default_args.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1, version=2, cpu_nodes=1)
    print("Generated: annotated_host_cpu_default_args.epair")


def gen_annotated_assert_tensor_metadata() -> None:
    """Graph containing torch.export-inserted _assert_tensor_metadata guard nodes.

    Reproduces the EP JSON structure that torchair produces for
    ``aten._assert_tensor_metadata.default``: an empty ``source`` ascend_ir
    with a single op that has only ``attr`` (no name/type), empty outputs,
    and empty mapping.  Without PlanNodes filtering, this node falls into
    the fallback branch and fails the build.
    """
    graph = {
        "inputs": [_as_tensor("p_weight"), _as_tensor("x")],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.mm.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "mat2", "arg": _as_tensor("p_weight"), "kind": 1},
                ],
                "outputs": [_as_tensor("mm")],
                "metadata": {"torch_fn": "mm_1;builtin_function_or_method.mm"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten._assert_tensor_metadata.default",
                "inputs": [
                    {"name": "a", "arg": _as_tensor("mm"), "kind": 1},
                    {"name": "size", "arg": {"as_none": True}, "kind": 1},
                    {"name": "stride", "arg": {"as_none": True}, "kind": 1},
                    {"name": "dtype", "arg": {"as_scalar_type": 7}, "kind": 1},
                    {
                        "name": "device",
                        "arg": {"as_device": {"type": "cpu", "index": None}},
                        "kind": 2,
                    },
                    {"name": "layout", "arg": {"as_layout": 7}, "kind": 2},
                ],
                "outputs": [],
                "metadata": {
                    "torch_fn": "_assert_tensor_metadata.default_0;OpOverload._assert_tensor_metadata.default"
                },
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("mm"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "p_weight": _tensor_meta([3, 3], True),
            "x": _tensor_meta([3, 3], False),
            "mm": _tensor_meta([3, 3], False),
            "relu": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"parameter": {"arg": {"name": "p_weight"}, "parameter_name": "weight"}},
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "MatMul",
                "type": "MatMul",
                "input": ["x", "p_weight", ""],
                "attr": {
                    "transpose_x1": {"b": False},
                    "transpose_x2": {"b": False},
                },
                "inputDesc": [
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {
                        "name": "x2",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                    {"name": "bias", "attr": {"_is_unfed_optional": {"i": "1"}}},
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"mm": "MatMul:0"}},
    }
    annotated_json["graph_module"]["graph"]["nodes"][1]["ascend_ir"] = {
        "source": "",
        "ops": [{"attr": {}}],
        "mapping": {"inputs": {}, "outputs": {}},
    }
    annotated_json["graph_module"]["graph"]["nodes"][2]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "Relu",
                "type": "Relu",
                "input": ["mm"],
                "attr": {},
                "inputDesc": [
                    {
                        "name": "x",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["3", "3"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {"mm": "MatMul:0"}, "outputs": {"relu": "Relu:0"}},
    }

    weights_config = {
        "config": {
            "weight": {
                "path_name": "weight_0",
                "is_param": True,
                "use_pickle": False,
                "tensor_meta": _tensor_meta([3, 3], True),
            },
        }
    }
    out = os.path.join(FIXTURES_DIR, "annotated_assert_tensor_metadata.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps(weights_config)
        )
        writer.write_bytes(
            "data/weights/weight_0",
            _float_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=3, converted=2)
    print("Generated: annotated_assert_tensor_metadata.epair")


def gen_annotated_converter_no_ops() -> None:
    graph = {
        "inputs": [_as_tensor("x"), _as_tensor("z")],
        "outputs": [_as_tensor("y")],
        "nodes": [
            {
                "target": "torch.ops.aten.expand.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "size", "arg": {"as_ints": [3, 3]}, "kind": 1},
                ],
                "outputs": [_as_tensor("expand_out")],
                "metadata": {"torch_fn": "expand_1;builtin_function_or_method.expand"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.slice.Tensor",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("expand_out"), "kind": 1},
                    {"name": "dim", "arg": {"as_ints": [0]}, "kind": 1},
                    {"name": "start", "arg": {"as_ints": [0]}, "kind": 1},
                    {"name": "end", "arg": {"as_ints": [3]}, "kind": 1},
                    {"name": "step", "arg": {"as_ints": [1]}, "kind": 1},
                ],
                "outputs": [_as_tensor("slice_out")],
                "metadata": {"torch_fn": "slice_1;builtin_function_or_method.slice"},
                "is_hop_single_tensor_return": None,
            },
            {
                "target": "torch.ops.aten.copy.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("slice_out"), "kind": 1},
                    {"name": "src", "arg": _as_tensor("z"), "kind": 1},
                ],
                "outputs": [_as_tensor("y")],
                "metadata": {"torch_fn": "copy_1;builtin_function_or_method.copy"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([3, 3], False),
            "z": _tensor_meta([3, 3], False),
            "expand_out": _tensor_meta([3, 3], False),
            "slice_out": _tensor_meta([3, 3], False),
            "y": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
            {"user_input": {"arg": _as_tensor("z")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("y")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [],
        "mapping": None,
    }
    annotated_json["graph_module"]["graph"]["nodes"][1]["ascend_ir"] = {
        "source": "converter",
        "ops": [],
        "mapping": None,
    }
    annotated_json["graph_module"]["graph"]["nodes"][2]["ascend_ir"] = {
        "source": "converter",
        "ops": [],
        "mapping": {"inputs": {}, "outputs": {}},
    }
    out = os.path.join(FIXTURES_DIR, "annotated_converter_no_ops.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=3, converted=3)
    print("Generated: annotated_converter_no_ops.epair")


def gen_annotated_converter_noop_copy() -> None:
    graph = {
        "inputs": [_as_tensor("slice_3"), _as_tensor("cat")],
        "outputs": [_as_tensor("copy")],
        "nodes": [
            {
                "target": "torch.ops.aten.copy.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("slice_3"), "kind": 1},
                    {"name": "src", "arg": _as_tensor("cat"), "kind": 1},
                ],
                "outputs": [_as_tensor("copy")],
                "metadata": {"torch_fn": "copy_1;builtin_function_or_method.copy"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "slice_3": _tensor_meta([2, 4], False),
            "cat": _tensor_meta([2, 4], False),
            "copy": _tensor_meta([2, 4], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("slice_3")}},
            {"user_input": {"arg": _as_tensor("cat")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("copy")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [],
        "mapping": {"inputs": {}, "outputs": {}},
    }
    out = os.path.join(FIXTURES_DIR, "annotated_converter_noop_copy.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_converter_noop_copy.epair")


def gen_annotated_bad_converter_noop_meta() -> None:
    graph = {
        "inputs": [_as_tensor("slice_3"), _as_tensor("cat")],
        "outputs": [_as_tensor("copy")],
        "nodes": [
            {
                "target": "torch.ops.aten.copy.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("slice_3"), "kind": 1},
                    {"name": "src", "arg": _as_tensor("cat"), "kind": 1},
                ],
                "outputs": [_as_tensor("copy")],
                "metadata": {"torch_fn": "copy_1;builtin_function_or_method.copy"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "slice_3": _tensor_meta([2, 4], False),
            "cat": _tensor_meta([2, 4], False),
            "copy": _tensor_meta([2, 5], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("slice_3")}},
            {"user_input": {"arg": _as_tensor("cat")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("copy")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [],
        "mapping": {"inputs": {}, "outputs": {}},
    }
    out = os.path.join(FIXTURES_DIR, "annotated_bad_converter_noop_meta.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_bad_converter_noop_meta.epair")


def gen_annotated_attr_types() -> None:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("y")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("y")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([2], False),
            "y": _tensor_meta([2], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("y")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "TDAttrCoverage",
                "type": "TDAttrCoverage",
                "input": ["x"],
                "attr": {
                    "int_attr": {"i": "1"},
                    "float_attr": {"f": 1.25},
                    "string_attr": {
                        "s": base64.b64encode("hello".encode("utf-8")).decode("ascii")
                    },
                    "bool_attr": {"b": True},
                    "dtype_attr": {"dt": "1"},
                    "list_int_attr": {
                        "list": {"i": ["0", "2", "-1"], "valType": "VT_LIST_INT"}
                    },
                    "list_float_attr": {
                        "list": {"f": [0.5, -1.0], "valType": "VT_LIST_FLOAT"}
                    },
                    "list_bool_attr": {
                        "list": {"b": [True, False, True], "valType": "VT_LIST_BOOL"}
                    },
                    "list_string_attr": {
                        "list": {
                            "s": [
                                base64.b64encode("alpha".encode("utf-8")).decode(
                                    "ascii"
                                ),
                                base64.b64encode("beta".encode("utf-8")).decode(
                                    "ascii"
                                ),
                            ],
                            "valType": "VT_LIST_STRING",
                        }
                    },
                    "list_type_attr": {
                        "list": {"dt": ["1", "7"], "valType": "VT_LIST_DATA_TYPE"}
                    },
                    "empty_list_attr": {"list": {"valType": "VT_LIST_INT"}},
                },
                "inputDesc": [
                    {
                        "name": "x",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["2"]},
                        "layout": "ND",
                        "deviceType": "CPU",
                    },
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["2"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"y": "TDAttrCoverage:0"}},
    }
    out = os.path.join(FIXTURES_DIR, "annotated_attr_types.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_attr_types.epair")


def gen_annotated_fallback_with_attrs() -> None:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("topk_values"), _as_tensor("topk_indices")],
        "nodes": [
            {
                "target": "torch.ops.aten.topk.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("topk_values"), _as_tensor("topk_indices")],
                "metadata": {"torch_fn": "topk_1;builtin_function_or_method.topk"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([3, 3], False),
            "topk_values": _tensor_meta([3, 1], False),
            "topk_indices": _tensor_meta([3, 1], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": False,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("topk_values")}},
            {"user_output": {"arg": _as_tensor("topk_indices")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "",
        "ops": [
            {
                "attr": {
                    "k": {"i": "1"},
                    "dim": {"i": "1"},
                    "largest": {"b": True},
                    "sorted": {"b": True},
                }
            }
        ],
    }
    out = os.path.join(FIXTURES_DIR, "annotated_fallback_with_attrs.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=0)
    print("Generated: annotated_fallback_with_attrs.epair")


def _annotated_dynamic_concat_model_json() -> dict:
    graph = {
        "inputs": [_as_tensor("x0"), _as_tensor("x1"), _as_tensor("axis")],
        "outputs": [_as_tensor("cat_out")],
        "nodes": [
            {
                "target": "torch.ops.aten.cat.default",
                "inputs": [
                    {"name": "input0", "arg": _as_tensor("x0"), "kind": 1},
                    {"name": "input1", "arg": _as_tensor("x1"), "kind": 1},
                    {"name": "dim", "arg": _as_tensor("axis"), "kind": 1},
                ],
                "outputs": [_as_tensor("cat_out")],
                "metadata": {"torch_fn": "cat_1;builtin_function_or_method.cat"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "x0": _tensor_meta([1, 1, 1], False),
            "x1": _tensor_meta([1, 1, 1], False),
            "axis": _tensor_meta([], False),
            "cat_out": _tensor_meta([1, 1, 2], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x0")}},
            {"user_input": {"arg": _as_tensor("x1")}},
            {"user_input": {"arg": _as_tensor("axis")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("cat_out")}},
        ],
    }
    return {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }


def gen_annotated_dynamic_concat() -> None:
    model_json = _annotated_dynamic_concat_model_json()
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "TDConcatV2",
                "type": "TDConcatV2",
                "input": ["x0", "x1", "axis"],
                "attr": {"N": {"i": "2"}},
                "inputDesc": [
                    {
                        "name": "x0",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["1", "1", "1"]},
                        "layout": "ND",
                    },
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["1", "1", "1"]},
                        "layout": "ND",
                    },
                    {
                        "name": "concat_dim",
                        "dtype": "DT_INT64",
                        "shape": {},
                        "layout": "ND",
                    },
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["1", "1", "2"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"cat_out": "TDConcatV2:0"}},
    }
    out = os.path.join(FIXTURES_DIR, "annotated_dynamic_concat.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_dynamic_concat.epair")


def gen_annotated_dynamic_concat_bad_prefix() -> None:
    model_json = _annotated_dynamic_concat_model_json()
    annotated_json = json.loads(json.dumps(model_json))
    annotated_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"] = {
        "source": "converter",
        "ops": [
            {
                "name": "TDConcatV2",
                "type": "TDConcatV2",
                "input": ["x0", "x1", "axis"],
                "attr": {"N": {"i": "2"}},
                "inputDesc": [
                    {
                        "name": "x1",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["1", "1", "1"]},
                        "layout": "ND",
                    },
                    {
                        "name": "x2",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["1", "1", "1"]},
                        "layout": "ND",
                    },
                    {
                        "name": "concat_dim",
                        "dtype": "DT_INT64",
                        "shape": {},
                        "layout": "ND",
                    },
                ],
                "outputDesc": [
                    {
                        "name": "y",
                        "dtype": "DT_FLOAT",
                        "shape": {"dim": ["1", "1", "2"]},
                        "layout": "ND",
                    },
                ],
            }
        ],
        "mapping": {"inputs": {}, "outputs": {"cat_out": "TDConcatV2:0"}},
    }
    out = os.path.join(FIXTURES_DIR, "annotated_dynamic_concat_bad_prefix.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json", json.dumps(annotated_json)
        )
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_dynamic_concat_bad_prefix.epair")


def gen_annotated_auto_functionalized() -> None:
    graph = {
        "inputs": [
            _as_tensor("hidden_states"),
            _as_tensor("sin_cache"),
            _as_tensor("cos_cache"),
        ],
        "outputs": [_as_tensor("updated_x")],
        "nodes": [
            {
                "target": "torch.ops.higher_order.auto_functionalized_v2",
                "inputs": [
                    {
                        "name": "op",
                        "arg": {
                            "as_operator": "torch.ops.xllm_ascend.rope_in_place.default"
                        },
                        "kind": 1,
                    },
                    {"name": "sin_cache", "arg": _as_tensor("sin_cache"), "kind": 2},
                    {"name": "cos_cache", "arg": _as_tensor("cos_cache"), "kind": 2},
                    {"name": "_self_base_index", "arg": {"as_int": 0}, "kind": 2},
                    {
                        "name": "_all_bases",
                        "arg": {"as_tensors": [{"name": "hidden_states"}]},
                        "kind": 2,
                    },
                ],
                "outputs": [{"as_none": True}, _as_tensor("updated_x")],
                "metadata": {"torch_fn": "auto_functionalized_v2;higher_order"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {"source": "", "ops": [{"attr": {}}]},
            },
        ],
        "tensor_values": {
            "hidden_states": _tensor_meta([3, 3], False),
            "sin_cache": _tensor_meta([3, 3], False),
            "cos_cache": _tensor_meta([3, 3], False),
            "updated_x": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": False,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("hidden_states")}},
            {"user_input": {"arg": _as_tensor("sin_cache")}},
            {"user_input": {"arg": _as_tensor("cos_cache")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("updated_x")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_auto_functionalized.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=0)
    print("Generated: annotated_auto_functionalized.epair")


def gen_annotated_bad_fallback_input() -> None:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("bad_out")],
        "nodes": [
            {
                "target": "torch.ops.aten.add.Tensor",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "other", "arg": {"as_int": 1}, "kind": 1},
                ],
                "outputs": [_as_tensor("bad_out")],
                "metadata": {},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {"source": "", "ops": [{"attr": {}}]},
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([3, 3], False),
            "bad_out": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("bad_out")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_bad_fallback_input.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=0)
    print("Generated: annotated_bad_fallback_input.epair")


def gen_annotated_missing_weight_record() -> None:
    out = os.path.join(FIXTURES_DIR, "annotated_missing_weight_record.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string(
            "models/model.json",
            json.dumps({"graph_module": {"graph": {}, "signature": {}}}),
        )
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json",
            json.dumps(
                {
                    "graph_module": {
                        "graph": {
                            "inputs": [_as_tensor("p_weight")],
                            "outputs": [_as_tensor("p_weight")],
                            "nodes": [],
                            "tensor_values": {"p_weight": _tensor_meta([3, 3], True)},
                            "sym_int_values": {},
                            "sym_bool_values": {},
                            "sym_float_values": {},
                            "custom_obj_values": {},
                            "is_single_tensor_return": True,
                        },
                        "signature": {
                            "input_specs": [
                                {
                                    "parameter": {
                                        "arg": {"name": "p_weight"},
                                        "parameter_name": "weight",
                                    }
                                }
                            ],
                            "output_specs": [
                                {"user_output": {"arg": _as_tensor("p_weight")}}
                            ],
                        },
                        "module_call_graph": [],
                        "metadata": {},
                        "treespec_namedtuple_fields": {},
                    },
                    "opset_version": {"aten": 10},
                    "range_constraints": {},
                    "schema_version": {"major": 8, "minor": 7},
                    "verifiers": ["ATEN"],
                    "torch_version": "2.7.1+cpu",
                }
            ),
        )
        writer.write_string(
            "data/weights/model_weights_config.json",
            json.dumps(
                {
                    "config": {
                        "weight": {
                            "path_name": "weight_missing",
                            "is_param": True,
                            "use_pickle": False,
                            "tensor_meta": _tensor_meta([3, 3], True),
                        },
                    }
                }
            ),
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=0, converted=0)
    print("Generated: annotated_missing_weight_record.epair")


def gen_annotated_bad_weight_size() -> None:
    out = os.path.join(FIXTURES_DIR, "annotated_bad_weight_size.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string(
            "models/model.json",
            json.dumps({"graph_module": {"graph": {}, "signature": {}}}),
        )
        writer.write_string(
            "extra/ep_json_with_ascend_ir.json",
            json.dumps(
                {
                    "graph_module": {
                        "graph": {
                            "inputs": [_as_tensor("p_weight")],
                            "outputs": [_as_tensor("p_weight")],
                            "nodes": [],
                            "tensor_values": {"p_weight": _tensor_meta([3, 3], True)},
                            "sym_int_values": {},
                            "sym_bool_values": {},
                            "sym_float_values": {},
                            "custom_obj_values": {},
                            "is_single_tensor_return": True,
                        },
                        "signature": {
                            "input_specs": [
                                {
                                    "parameter": {
                                        "arg": {"name": "p_weight"},
                                        "parameter_name": "weight",
                                    }
                                }
                            ],
                            "output_specs": [
                                {"user_output": {"arg": _as_tensor("p_weight")}}
                            ],
                        },
                        "module_call_graph": [],
                        "metadata": {},
                        "treespec_namedtuple_fields": {},
                    },
                    "opset_version": {"aten": 10},
                    "range_constraints": {},
                    "schema_version": {"major": 8, "minor": 7},
                    "verifiers": ["ATEN"],
                    "torch_version": "2.7.1+cpu",
                }
            ),
        )
        writer.write_string(
            "data/weights/model_weights_config.json",
            json.dumps(
                {
                    "config": {
                        "weight": {
                            "path_name": "weight_0",
                            "is_param": True,
                            "use_pickle": False,
                            "tensor_meta": _tensor_meta([3, 3], True),
                        },
                    }
                }
            ),
        )
        writer.write_bytes("data/weights/weight_0", _float_bytes([1.0, 2.0, 3.0]))
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=0, converted=0)
    print("Generated: annotated_bad_weight_size.epair")


def gen_missing_graph_source() -> None:
    out = os.path.join(FIXTURES_DIR, "missing_graph_source.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(_ep_json(["x"], ["x"], [])))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer)
    print("Generated: missing_graph_source.epair")


def _inline_const_model_json() -> dict:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("expand")],
        "nodes": [
            {
                "target": "torch.ops.aten.expand.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "size", "arg": {"as_ints": [2, 3]}, "kind": 1},
                ],
                "outputs": [_as_tensor("expand")],
                "metadata": {"torch_fn": "expand_1;builtin_function_or_method.expand"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "ConstShape",
                            "type": "Const",
                            "attr": _const_tensor_attr_int64([2, 3]),
                            "outputDesc": [
                                {
                                    "dtype": "DT_INT64",
                                    "shape": {"dim": ["2"]},
                                    "layout": "ND",
                                },
                            ],
                        },
                        {
                            "name": "BroadcastToMain",
                            "type": "BroadcastTo",
                            "input": ["x", "ConstShape:0"],
                            "inputDesc": [
                                {
                                    "name": "x",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["1", "3"]},
                                    "layout": "ND",
                                },
                                {
                                    "name": "shape",
                                    "dtype": "DT_INT64",
                                    "shape": {"dim": ["2"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        },
                    ],
                    "mapping": {
                        "inputs": {},
                        "outputs": {"expand": "BroadcastToMain:0"},
                    },
                },
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([1, 3], False),
            "expand": _tensor_meta([2, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    return {
        "graph_module": {
            "graph": graph,
            "signature": {
                "input_specs": [{"user_input": {"arg": _as_tensor("x")}}],
                "output_specs": [{"user_output": {"arg": _as_tensor("expand")}}],
            },
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }


def gen_annotated_inline_const() -> None:
    model_json = _inline_const_model_json()
    out = os.path.join(FIXTURES_DIR, "annotated_inline_const.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_inline_const.epair")


def _inline_scalar_const_model_json() -> dict:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("add")],
        "nodes": [
            {
                "target": "torch.ops.aten.add.Tensor",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {
                        "name": "other",
                        "arg": {"as_float": 0.08838834764831843},
                        "kind": 1,
                    },
                ],
                "outputs": [_as_tensor("add")],
                "metadata": {"torch_fn": "add_1;builtin_function_or_method.add"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "ConstScalar",
                            "type": "Const",
                            "attr": _const_tensor_attr_float_scalar(
                                0.08838834764831843
                            ),
                            "outputDesc": [
                                {"dtype": "DT_FLOAT", "shape": {}, "layout": "ND"},
                            ],
                        },
                        {
                            "name": "AddMain",
                            "type": "Add",
                            "input": ["x", "ConstScalar:0"],
                            "inputDesc": [
                                {
                                    "name": "x1",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["1"]},
                                    "layout": "ND",
                                },
                                {
                                    "name": "x2",
                                    "dtype": "DT_FLOAT",
                                    "shape": {},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["1"]},
                                    "layout": "ND",
                                },
                            ],
                        },
                    ],
                    "mapping": {"inputs": {}, "outputs": {"add": "AddMain:0"}},
                },
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([1], False),
            "add": _tensor_meta([1], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    return {
        "graph_module": {
            "graph": graph,
            "signature": {
                "input_specs": [{"user_input": {"arg": _as_tensor("x")}}],
                "output_specs": [{"user_output": {"arg": _as_tensor("add")}}],
            },
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }


def gen_annotated_inline_scalar_const() -> None:
    model_json = _inline_scalar_const_model_json()
    out = os.path.join(FIXTURES_DIR, "annotated_inline_scalar_const.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_inline_scalar_const.epair")


def gen_annotated_inline_const_bad_value() -> None:
    model_json = _inline_const_model_json()
    model_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0]["attr"] = {
        "value": {"i": "1"}
    }
    out = os.path.join(FIXTURES_DIR, "annotated_inline_const_bad_value.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_inline_const_bad_value.epair")


def gen_annotated_inline_const_desc_mismatch() -> None:
    model_json = _inline_const_model_json()
    model_json["graph_module"]["graph"]["nodes"][0]["ascend_ir"]["ops"][0][
        "outputDesc"
    ] = [
        {"dtype": "DT_INT32", "shape": {"dim": ["1"]}, "layout": "ND"},
    ]
    out = os.path.join(FIXTURES_DIR, "annotated_inline_const_desc_mismatch.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_inline_const_desc_mismatch.epair")


def gen_annotated_sym_int_input() -> None:
    graph = {
        "inputs": [_as_tensor("x"), {"as_sym_int": {"as_name": "batch_size"}}],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "Relu",
                            "type": "Relu",
                            "input": ["x"],
                            "attr": {},
                            "inputDesc": [
                                {
                                    "name": "x",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["4", "3"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["4", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        }
                    ],
                    "mapping": {"inputs": {}, "outputs": {"relu": "Relu:0"}},
                },
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([4, 3], False),
            "relu": _tensor_meta([4, 3], False),
        },
        "sym_int_values": {"batch_size": {"as_int": 4, "type": 7}},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
            {
                "user_input": {
                    "arg": {"as_sym_int": {"as_name": "batch_size"}},
                    "value_type": "sym_int",
                }
            },
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {"batch_size": {"min_val": 1, "max_val": 8}},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_sym_int_input.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_sym_int_input.epair")


def gen_annotated_sym_float_bool_input() -> None:
    graph = {
        "inputs": [
            _as_tensor("x"),
            {"as_sym_float": {"as_name": "threshold"}},
            {"as_sym_bool": {"as_name": "flag"}},
        ],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "Relu",
                            "type": "Relu",
                            "input": ["x"],
                            "attr": {},
                            "inputDesc": [
                                {
                                    "name": "x",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        }
                    ],
                    "mapping": {"inputs": {}, "outputs": {"relu": "Relu:0"}},
                },
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([2, 3], False),
            "relu": _tensor_meta([2, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {"flag": {"as_bool": True, "type": 11}},
        "sym_float_values": {"threshold": {"as_float": 0.5, "type": 6}},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
            {
                "user_input": {
                    "arg": {"as_sym_float": {"as_name": "threshold"}},
                    "value_type": "sym_float",
                }
            },
            {
                "user_input": {
                    "arg": {"as_sym_bool": {"as_name": "flag"}},
                    "value_type": "sym_bool",
                }
            },
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_sym_float_bool_input.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_sym_float_bool_input.epair")


def gen_annotated_constant_input() -> None:
    graph = {
        "inputs": [_as_tensor("x"), {"as_int": 2}],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "Relu",
                            "type": "Relu",
                            "input": ["x"],
                            "attr": {},
                            "inputDesc": [
                                {
                                    "name": "x",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        }
                    ],
                    "mapping": {"inputs": {}, "outputs": {"relu": "Relu:0"}},
                },
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([2, 3], False),
            "relu": _tensor_meta([2, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
            {"constant_input": {"name": "scale_factor", "value": {"as_int": 2}}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_constant_input.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_constant_input.epair")


def gen_annotated_dynamic_dim_symbols() -> None:
    tensor_meta_with_expr = {
        "dtype": 7,
        "sizes": [{"as_expr": {"expr_str": "Symbol('s0')", "hint": {"as_int": 4}}}, 3],
        "strides": [3, 1],
        "storage_offset": 0,
        "layout": 7,
        "requires_grad": False,
        "device": {"type": "cpu"},
    }
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "Relu",
                            "type": "Relu",
                            "input": ["x"],
                            "attr": {},
                            "inputDesc": [
                                {
                                    "name": "x",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["4", "3"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["4", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        }
                    ],
                    "mapping": {"inputs": {}, "outputs": {"relu": "Relu:0"}},
                },
            },
        ],
        "tensor_values": {
            "x": tensor_meta_with_expr,
            "relu": _tensor_meta([4, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {"s0": {"min_val": 1, "max_val": 8}},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_dynamic_dim_symbols.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_dynamic_dim_symbols.epair")


def gen_annotated_scalar_value_maps() -> None:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("relu")],
        "nodes": [
            {
                "target": "torch.ops.aten.relu.default",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                ],
                "outputs": [_as_tensor("relu")],
                "metadata": {"torch_fn": "relu_1;builtin_function_or_method.relu"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "Relu",
                            "type": "Relu",
                            "input": ["x"],
                            "attr": {},
                            "inputDesc": [
                                {
                                    "name": "x",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["2", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        }
                    ],
                    "mapping": {"inputs": {}, "outputs": {"relu": "Relu:0"}},
                },
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([2, 3], False),
            "relu": _tensor_meta([2, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "int_values": {"count": {"as_int": 10}},
        "float_values": {"scale": {"as_float": 0.5}},
        "bool_values": {"enabled": {"as_bool": True}},
        "string_values": {"mode": {"as_string": "train"}},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("relu")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_scalar_value_maps.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_scalar_value_maps.epair")


def gen_annotated_tensor_constant() -> None:
    graph = {
        "inputs": [_as_tensor("c_bias"), _as_tensor("x")],
        "outputs": [_as_tensor("add")],
        "nodes": [
            {
                "target": "torch.ops.aten.add.Tensor",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "other", "arg": _as_tensor("c_bias"), "kind": 1},
                ],
                "outputs": [_as_tensor("add")],
                "metadata": {"torch_fn": "add_1;method_descriptor.add"},
                "is_hop_single_tensor_return": None,
                "ascend_ir": {
                    "source": "converter",
                    "ops": [
                        {
                            "name": "Add",
                            "type": "Add",
                            "input": ["x", "c_bias"],
                            "attr": {},
                            "inputDesc": [
                                {
                                    "name": "x1",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["3", "3"]},
                                    "layout": "ND",
                                },
                                {
                                    "name": "x2",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["3"]},
                                    "layout": "ND",
                                },
                            ],
                            "outputDesc": [
                                {
                                    "name": "y",
                                    "dtype": "DT_FLOAT",
                                    "shape": {"dim": ["3", "3"]},
                                    "layout": "ND",
                                },
                            ],
                        }
                    ],
                    "mapping": {"inputs": {}, "outputs": {"add": "Add:0"}},
                },
            },
        ],
        "tensor_values": {
            "c_bias": _tensor_meta([3], False),
            "x": _tensor_meta([3, 3], False),
            "add": _tensor_meta([3, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {
                "tensor_constant": {
                    "arg": {"name": "c_bias"},
                    "tensor_constant_name": "bias",
                }
            },
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("add")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    constants_config = {
        "config": {
            "bias": {
                "path_name": "tensor_0",
                "use_pickle": False,
                "tensor_meta": _tensor_meta([3], False),
            },
        }
    }
    out = os.path.join(FIXTURES_DIR, "annotated_tensor_constant.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps(constants_config)
        )
        writer.write_bytes("data/constants/tensor_0", _float_bytes([1.0, 2.0, 3.0]))
        _write_manifest(writer, total_nodes=1, converted=1)
    print("Generated: annotated_tensor_constant.epair")


def gen_annotated_scalar_fallback() -> None:
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("add")],
        "nodes": [
            {
                "target": "torch.ops.aten.add.Scalar",
                "inputs": [
                    {"name": "self", "arg": _as_tensor("x"), "kind": 1},
                    {"name": "other", "arg": {"as_int": 5}, "kind": 1},
                ],
                "outputs": [_as_tensor("add")],
                "metadata": {"torch_fn": "add_1;builtin_function_or_method.add"},
                "is_hop_single_tensor_return": None,
            },
        ],
        "tensor_values": {
            "x": _tensor_meta([2, 3], False),
            "add": _tensor_meta([2, 3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("add")}},
        ],
    }
    model_json = {
        "graph_module": {
            "graph": graph,
            "signature": signature,
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 7},
        "verifiers": ["ATEN"],
        "torch_version": "2.7.1+cpu",
    }
    out = os.path.join(FIXTURES_DIR, "annotated_scalar_fallback.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string(
            "data/weights/model_weights_config.json", json.dumps({"config": {}})
        )
        writer.write_string(
            "data/constants/model_constants_config.json", json.dumps({"config": {}})
        )
        _write_manifest(writer, total_nodes=1, converted=0)
    print("Generated: annotated_scalar_fallback.epair")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(output_dir: str = FIXTURES_DIR) -> None:
    global FIXTURES_DIR
    FIXTURES_DIR = output_dir
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    gen_torch_export_fixtures()
    gen_empty_compute()
    gen_npu_cpu_npu()
    gen_shared_input()
    gen_non_adjacent()
    gen_no_model_json()
    gen_missing_op_type()
    gen_annotated_mm_add()
    gen_annotated_mixed()
    gen_annotated_host_cpu()
    gen_annotated_host_cpu_non4byte_inputs()
    gen_annotated_host_cpu_default_args()
    gen_annotated_assert_tensor_metadata()
    gen_annotated_converter_no_ops()
    gen_annotated_converter_noop_copy()
    gen_annotated_attr_types()
    gen_annotated_fallback_with_attrs()
    gen_annotated_dynamic_concat()
    gen_annotated_dynamic_concat_bad_prefix()
    gen_annotated_auto_functionalized()
    gen_annotated_bad_fallback_input()
    gen_annotated_bad_converter_noop_meta()
    gen_annotated_missing_weight_record()
    gen_annotated_bad_weight_size()
    gen_missing_graph_source()
    gen_annotated_inline_const()
    gen_annotated_inline_scalar_const()
    gen_annotated_inline_const_bad_value()
    gen_annotated_inline_const_desc_mismatch()
    gen_annotated_sym_int_input()
    gen_annotated_sym_float_bool_input()
    gen_annotated_constant_input()
    gen_annotated_dynamic_dim_symbols()
    gen_annotated_scalar_value_maps()
    gen_annotated_tensor_constant()
    gen_annotated_scalar_fallback()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .epair test fixtures")
    parser.add_argument(
        "--output-dir", default=FIXTURES_DIR, help="Directory to write fixtures into"
    )
    args = parser.parse_args()
    generate(args.output_dir)
