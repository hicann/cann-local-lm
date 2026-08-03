#!/usr/bin/env python3
"""Generate directory-based test fixtures for consolidated weights C++ tests."""

import argparse
import base64
import json
import os
import struct

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _as_tensor(name):
    return {"as_tensor": {"name": name}}


def _tensor_meta(shape, requires_grad):
    strides = []
    running = 1
    for size in reversed(shape):
        strides.append(running)
        running *= size
    strides.reverse()
    return {
        "dtype": 7,
        "sizes": [{"as_int": v} for v in shape],
        "requires_grad": requires_grad,
        "device": {"type": "cpu", "index": None},
        "strides": [{"as_int": v} for v in strides],
        "storage_offset": {"as_int": 0},
        "layout": 7,
    }


def _float_bytes(values):
    return struct.pack("<" + "f" * len(values), *values)


def _common_model_json(graph, signature):
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


# ---------------------------------------------------------------------------
# Helper: write a full fixture directory
# ---------------------------------------------------------------------------
def _write_fixture(fixture_name: str, model_json: dict, bin_data: bytes = b"",
                   index_data: list | None = None) -> str:
    """Write a fixture directory.

    Args:
        fixture_name: directory name (without .epair)
        model_json: the full model JSON to write to extra/ep_json_with_ascend_ir.json
        bin_data: consolidated weights.bin content
        index_data: consolidated weights_index.json list
    """
    out = os.path.join(FIXTURES_DIR, fixture_name)
    _make_dir(out)

    # Model JSON
    _write_json(os.path.join(out, "extra", "ep_json_with_ascend_ir.json"), model_json)

    # Consolidated weights
    if bin_data:
        _write_bytes(os.path.join(out, "data", "weights.bin"), bin_data)
    if index_data is not None:
        _write_json(os.path.join(out, "data", "weights_index.json"), index_data)

    print(f"Generated: {fixture_name}/")
    return out


# ---------------------------------------------------------------------------
# Fixture: annotated_mm_add  (consolidated format)
# ---------------------------------------------------------------------------
def gen_annotated_mm_add():
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
            "weight": _tensor_meta([3, 3], True),
            "bias": _tensor_meta([3], False),
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
            {"buffer": {"arg": {"name": "b_bias"}, "buffer_name": "bias", "persistent": True}},
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("add")}},
        ],
    }
    model_json = _common_model_json(graph, signature)
    w_data = _float_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    b_data = _float_bytes([0.5, 1.5, 2.5])
    bin_data = w_data + b_data
    index = [
        {"name": "weight", "offset": 0, "size": len(w_data)},
        {"name": "bias", "offset": len(w_data), "size": len(b_data)},
    ]
    _write_fixture("annotated_mm_add", model_json, bin_data, index)


# ---------------------------------------------------------------------------
# Fixture: annotated_consolidated_mm_add  (consolidated format)
# ---------------------------------------------------------------------------
def gen_annotated_consolidated_mm_add():
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
            "weight": _tensor_meta([3, 3], True),
            "bias": _tensor_meta([3], False),
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
            {"buffer": {"arg": {"name": "b_bias"}, "buffer_name": "bias", "persistent": True}},
            {"user_input": {"arg": _as_tensor("x")}},
        ],
        "output_specs": [
            {"user_output": {"arg": _as_tensor("add")}},
        ],
    }
    model_json = _common_model_json(graph, signature)
    w_data = _float_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    b_data = _float_bytes([0.5, 1.5, 2.5])
    bin_data = w_data + b_data
    index = [
        {"name": "weight", "offset": 0, "size": len(w_data)},
        {"name": "bias", "offset": len(w_data), "size": len(b_data)},
    ]
    _write_fixture("annotated_consolidated_mm_add", model_json, bin_data, index)


# ---------------------------------------------------------------------------
# Fixture: annotated_consolidated_missing_meta
# ---------------------------------------------------------------------------
def gen_annotated_consolidated_missing_meta():
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("x")],
        "nodes": [],
        "tensor_values": {
            "x": _tensor_meta([3], False),
        },
        "sym_int_values": {},
        "sym_bool_values": {},
        "sym_float_values": {},
        "custom_obj_values": {},
        "is_single_tensor_return": True,
    }
    signature = {
        "input_specs": [
            {"parameter": {"arg": {"name": "p_unknown"}, "parameter_name": "unknown_weight"}},
        ],
        "output_specs": [],
    }
    model_json = _common_model_json(graph, signature)
    bin_data = _float_bytes([1.0, 2.0, 3.0])
    index = [{"name": "unknown_weight", "offset": 0, "size": len(bin_data)}]
    _write_fixture("annotated_consolidated_missing_meta", model_json, bin_data, index)


# ---------------------------------------------------------------------------
# Fixture: annotated_consolidated_no_index  (no consolidated index)
# ---------------------------------------------------------------------------
def gen_annotated_consolidated_no_index():
    graph = {
        "inputs": [_as_tensor("x")],
        "outputs": [_as_tensor("x")],
        "nodes": [],
        "tensor_values": {
            "x": _tensor_meta([3], False),
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
        ],
        "output_specs": [],
    }
    model_json = _common_model_json(graph, signature)
    _write_fixture("annotated_consolidated_no_index", model_json,
                   bin_data=b"dummy")


# ---------------------------------------------------------------------------
# Fixture: annotated_missing_weight_record
# ---------------------------------------------------------------------------
def gen_annotated_missing_weight_record():
    graph = {
        "inputs": [_as_tensor("p_weight")],
        "outputs": [_as_tensor("x")],
        "nodes": [{
            "target": "torch.ops.aten.mul.Tensor",
            "inputs": [
                {"name": "self", "arg": _as_tensor("p_weight"), "kind": 1},
                {"name": "other", "arg": _as_tensor("x"), "kind": 1},
            ],
            "outputs": [_as_tensor("out")],
            "metadata": {"torch_fn": "mul_1;method_descriptor.mul"},
            "is_hop_single_tensor_return": None,
        }],
        "tensor_values": {
            "weight": _tensor_meta([2, 2], True),
            "x": _tensor_meta([2, 2], False),
            "out": _tensor_meta([2, 2], False),
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
        "output_specs": [{"user_output": {"arg": _as_tensor("out")}}],
    }
    model_json = _common_model_json(graph, signature)
    _write_fixture("annotated_missing_weight_record", model_json)


# ---------------------------------------------------------------------------
# Fixture: annotated_bad_weight_size
# ---------------------------------------------------------------------------
def gen_annotated_bad_weight_size():
    graph = {
        "inputs": [_as_tensor("p_weight")],
        "outputs": [_as_tensor("x")],
        "nodes": [],
        "tensor_values": {
            "weight": _tensor_meta([3, 3], True),
            "x": _tensor_meta([3, 3], False),
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
        "output_specs": [],
    }
    model_json = _common_model_json(graph, signature)
    bin_data = _float_bytes([42.0])
    index = [{"name": "weight", "offset": 0, "size": len(bin_data)}]
    _write_fixture("annotated_bad_weight_size", model_json, bin_data, index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate(output_dir=None):
    global FIXTURES_DIR
    if output_dir:
        FIXTURES_DIR = output_dir
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    gen_annotated_mm_add()
    gen_annotated_consolidated_mm_add()
    gen_annotated_consolidated_missing_meta()
    gen_annotated_consolidated_no_index()
    gen_annotated_missing_weight_record()
    gen_annotated_bad_weight_size()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate directory-based test fixtures")
    parser.add_argument("--output-dir", default=FIXTURES_DIR)
    args = parser.parse_args()
    generate(args.output_dir)
