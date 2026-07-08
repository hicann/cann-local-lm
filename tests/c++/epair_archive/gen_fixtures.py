#!/usr/bin/env python3
"""Generate .epair test fixtures for consolidated weights C++ tests."""

import argparse
import base64
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "cann_lm", "converter", "export"))

from torch.export.pt2_archive import PT2ArchiveWriter

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
ArchiveWriter = PT2ArchiveWriter


def _manifest_json(total_nodes=0, converted=0):
    return json.dumps({
        "version": 1,
        "backend_name": "ascend_ge",
        "total_nodes": total_nodes,
        "converted_nodes": converted,
    })


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
# Fixture: annotated_mm_add.epair  (legacy format — separate weight files)
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

    out = os.path.join(FIXTURES_DIR, "annotated_mm_add.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string("data/weights/model_weights_config.json", json.dumps({
            "config": {
                "weight": {
                    "path_name": "weight_0", "is_param": True,
                    "use_pickle": False, "tensor_meta": _tensor_meta([3, 3], True),
                },
                "bias": {
                    "path_name": "weight_1", "is_param": False,
                    "use_pickle": False, "tensor_meta": _tensor_meta([3], False),
                },
            }
        }))
        writer.write_bytes("data/weights/weight_0", w_data)
        writer.write_bytes("data/weights/weight_1", b_data)
        writer.write_string("data/constants/model_constants_config.json", json.dumps({"config": {}}))
        writer.write_string("extra/epair_manifest.json", _manifest_json(total_nodes=2, converted=2))
    print("Generated: annotated_mm_add.epair (legacy)")


# ---------------------------------------------------------------------------
# Fixture: annotated_consolidated_mm_add.epair  (consolidated format)
# Same model but weights in data/weights.bin + data/weights_index.json
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

    # Consolidated: single bin + index
    bin_data = w_data + b_data
    index = [
        {"name": "weight", "offset": 0, "size": len(w_data)},
        {"name": "bias", "offset": len(w_data), "size": len(b_data)},
    ]

    out = os.path.join(FIXTURES_DIR, "annotated_consolidated_mm_add.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_bytes("data/weights.bin", bin_data)
        writer.write_string("data/weights_index.json", json.dumps(index))
        writer.write_string("extra/epair_manifest.json", _manifest_json(total_nodes=2, converted=2))
    print("Generated: annotated_consolidated_mm_add.epair (consolidated)")


# ---------------------------------------------------------------------------
# Fixture: annotated_consolidated_missing_meta.epair
# Consolidated index entry "unknown_weight" has no matching tensor_values key
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

    out = os.path.join(FIXTURES_DIR, "annotated_consolidated_missing_meta.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_bytes("data/weights.bin", bin_data)
        writer.write_string("data/weights_index.json", json.dumps(index))
        writer.write_string("extra/epair_manifest.json", _manifest_json(total_nodes=0, converted=0))
    print("Generated: annotated_consolidated_missing_meta.epair")


# ---------------------------------------------------------------------------
# Fixture: annotated_consolidated_no_index.epair  (no consolidated index)
# Should fall back to legacy; no legacy config → MISSING_PLACEHOLDER_RECORD
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

    out = os.path.join(FIXTURES_DIR, "annotated_consolidated_no_index.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_bytes("data/weights.bin", b"dummy")
        writer.write_string("extra/epair_manifest.json", _manifest_json(total_nodes=0, converted=0))
    print("Generated: annotated_consolidated_no_index.epair")


# ---------------------------------------------------------------------------
# Fixture: missing_graph_source.epair  (no model.json / no ep_json_with_ir)
# ---------------------------------------------------------------------------
def gen_missing_graph_source():
    out = os.path.join(FIXTURES_DIR, "missing_graph_source.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("extra/epair_manifest.json", _manifest_json())
        writer.write_bytes("data/dummy", b"x")
    print("Generated: missing_graph_source.epair")


# ---------------------------------------------------------------------------
# Fixture: annotated_missing_weight_record.epair
# Model references "p_weight" but no weight record / config exists
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
    out = os.path.join(FIXTURES_DIR, "annotated_missing_weight_record.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        # No data/weights/ directory → triggers MISSING_PLACEHOLDER_RECORD
        writer.write_string("data/constants/model_constants_config.json", json.dumps({"config": {}}))
        writer.write_string("extra/epair_manifest.json", _manifest_json(total_nodes=1, converted=1))
    print("Generated: annotated_missing_weight_record.epair")


# ---------------------------------------------------------------------------
# Fixture: annotated_bad_weight_size.epair
# Config says shape [3,3] float but payload is only 4 bytes
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
    out = os.path.join(FIXTURES_DIR, "annotated_bad_weight_size.epair")
    with ArchiveWriter(out) as writer:
        writer.write_string("models/model.json", json.dumps(model_json))
        writer.write_string("extra/ep_json_with_ascend_ir.json", json.dumps(model_json))
        writer.write_string("data/weights/model_weights_config.json", json.dumps({
            "config": {
                "weight": {
                    "path_name": "weight_0", "is_param": True,
                    "use_pickle": False, "tensor_meta": _tensor_meta([3, 3], True),
                },
            }
        }))
        # Payload is only 1 float (4 bytes), but meta says 9 floats (36 bytes)
        writer.write_bytes("data/weights/weight_0", _float_bytes([42.0]))
        writer.write_string("data/constants/model_constants_config.json", json.dumps({"config": {}}))
        writer.write_string("extra/epair_manifest.json", _manifest_json(total_nodes=0, converted=0))
    print("Generated: annotated_bad_weight_size.epair")


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
    gen_missing_graph_source()
    gen_annotated_missing_weight_record()
    gen_annotated_bad_weight_size()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .epair test fixtures")
    parser.add_argument("--output-dir", default=FIXTURES_DIR)
    args = parser.parse_args()
    generate(args.output_dir)
