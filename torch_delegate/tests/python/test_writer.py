# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pytest

from torch_delegate.converter.base import NodeIR
from torch_delegate.epair.writer import (
    EpairWriter,
    _extract_names_from_arg,
    _flatten_output,
)


class TestExtractNamesFromArg:
    def test_as_tensor(self):
        arg = {"as_tensor": {"name": "x"}}
        assert _extract_names_from_arg(arg) == ["x"]

    def test_as_tensors(self):
        arg = {"as_tensors": [{"name": "a"}, {"name": "b"}]}
        assert _extract_names_from_arg(arg) == ["a", "b"]

    def test_as_optional_tensor_present(self):
        arg = {"as_optional_tensor": {"as_tensor": {"name": "opt"}}}
        assert _extract_names_from_arg(arg) == ["opt"]

    def test_as_optional_tensor_absent(self):
        arg = {"as_optional_tensor": {}}
        assert _extract_names_from_arg(arg) == []

    def test_as_optional_tensors(self):
        arg = {"as_optional_tensors": [{"as_tensor": {"name": "a"}}, {}]}
        assert _extract_names_from_arg(arg) == ["a"]

    def test_as_sym_int(self):
        arg = {"as_sym_int": {"as_name": "n"}}
        assert _extract_names_from_arg(arg) == ["n"]

    def test_as_sym_ints(self):
        arg = {"as_sym_ints": [{"as_name": "i"}, {"as_name": "j"}]}
        assert _extract_names_from_arg(arg) == ["i", "j"]

    def test_as_sym_float(self):
        arg = {"as_sym_float": {"as_name": "f"}}
        assert _extract_names_from_arg(arg) == ["f"]

    def test_as_sym_floats(self):
        arg = {"as_sym_floats": [{"as_name": "f1"}]}
        assert _extract_names_from_arg(arg) == ["f1"]

    def test_as_sym_bool(self):
        arg = {"as_sym_bool": {"as_name": "b"}}
        assert _extract_names_from_arg(arg) == ["b"]

    def test_as_sym_bools(self):
        arg = {"as_sym_bools": [{"as_name": "b1"}, {"as_name": "b2"}]}
        assert _extract_names_from_arg(arg) == ["b1", "b2"]

    def test_as_custom_obj(self):
        arg = {"as_custom_obj": {"name": "obj"}}
        assert _extract_names_from_arg(arg) == ["obj"]

    def test_as_graph(self):
        arg = {"as_graph": {"name": "g"}}
        assert _extract_names_from_arg(arg) == ["g"]

    def test_unknown_type(self):
        arg = {"as_unknown": {"name": "x"}}
        assert _extract_names_from_arg(arg) == []


class TestFlattenOutput:
    def test_as_none(self):
        assert _flatten_output({"as_none": {}}) == ["None"]

    def test_as_tensor(self):
        assert _flatten_output({"as_tensor": {"name": "out"}}) == ["out"]

    def test_as_tensors(self):
        assert _flatten_output({"as_tensors": [{"name": "a"}, {"name": "b"}]}) == [
            "a",
            "b",
        ]

    def test_as_optional_tensor_present(self):
        out = {"as_optional_tensor": {"as_tensor": {"name": "opt"}}}
        assert _flatten_output(out) == ["opt"]

    def test_as_optional_tensor_none(self):
        out = {"as_optional_tensor": {"as_none": {}}}
        assert _flatten_output(out) == ["None"]

    def test_as_optional_tensors(self):
        out = {"as_optional_tensors": [{"as_tensor": {"name": "a"}}, {"as_none": {}}]}
        assert _flatten_output(out) == ["a", "None"]

    def test_as_sym_int_name(self):
        assert _flatten_output({"as_sym_int": {"as_name": "n"}}) == ["n"]

    def test_as_sym_int_const(self):
        assert _flatten_output({"as_sym_int": {"as_int": 5}}) == ["CONST_int"]

    def test_as_sym_ints(self):
        out = {"as_sym_ints": [{"as_name": "i"}, {"as_int": 1}]}
        assert _flatten_output(out) == ["i", "CONST_int"]

    def test_as_sym_float_name(self):
        assert _flatten_output({"as_sym_float": {"as_name": "f"}}) == ["f"]

    def test_as_sym_float_const(self):
        assert _flatten_output({"as_sym_float": {"as_float": 1.0}}) == ["CONST_float"]

    def test_as_sym_floats(self):
        out = {"as_sym_floats": [{"as_name": "f1"}]}
        assert _flatten_output(out) == ["f1"]

    def test_as_sym_bool_name(self):
        assert _flatten_output({"as_sym_bool": {"as_name": "b"}}) == ["b"]

    def test_as_sym_bool_const(self):
        assert _flatten_output({"as_sym_bool": {"as_bool": True}}) == ["CONST_bool"]

    def test_as_sym_bools(self):
        out = {"as_sym_bools": [{"as_name": "b1"}, {"as_bool": False}]}
        assert _flatten_output(out) == ["b1", "CONST_bool"]

    def test_as_int(self):
        assert _flatten_output({"as_int": 42}) == ["CONST_int"]

    def test_as_ints(self):
        assert _flatten_output({"as_ints": [1, 2]}) == ["CONST_int", "CONST_int"]

    def test_as_float(self):
        assert _flatten_output({"as_float": 3.14}) == ["CONST_float"]

    def test_as_floats(self):
        assert _flatten_output({"as_floats": [1.0, 2.0]}) == [
            "CONST_float",
            "CONST_float",
        ]

    def test_as_bool(self):
        assert _flatten_output({"as_bool": True}) == ["CONST_bool"]

    def test_as_bools(self):
        assert _flatten_output({"as_bools": [True, False]}) == [
            "CONST_bool",
            "CONST_bool",
        ]

    def test_as_string(self):
        assert _flatten_output({"as_string": "hello"}) == ["CONST_string"]

    def test_as_strings(self):
        assert _flatten_output({"as_strings": ["a", "b"]}) == [
            "CONST_string",
            "CONST_string",
        ]

    def test_as_scalar_type(self):
        assert _flatten_output({"as_scalar_type": {}}) == ["CONST_scalar_type"]

    def test_as_memory_format(self):
        assert _flatten_output({"as_memory_format": {}}) == ["CONST_memory_format"]

    def test_as_layout(self):
        assert _flatten_output({"as_layout": {}}) == ["CONST_layout"]

    def test_as_device(self):
        assert _flatten_output({"as_device": {}}) == ["CONST_device"]

    def test_as_complex(self):
        assert _flatten_output({"as_complex": {}}) == ["CONST_complex"]

    def test_as_operator(self):
        assert _flatten_output({"as_operator": {}}) == ["CONST_operator"]

    def test_as_custom_obj(self):
        assert _flatten_output({"as_custom_obj": {"name": "obj"}}) == ["obj"]

    def test_as_graph(self):
        assert _flatten_output({"as_graph": {"name": "g"}}) == ["g"]

    def test_unknown_type(self):
        assert _flatten_output({"as_unknown": {}}) == []


class TestEpairWriterGetOpsList:
    def _make_writer(self):
        return EpairWriter.__new__(EpairWriter)

    def test_source_empty_match(self):
        writer = self._make_writer()
        node_ir = NodeIR(
            source="",
            ops=[{"type": "Add", "attr": {"alpha": 1}}],
        )
        result = writer._get_ops_list(node_ir, "Add")
        assert result == [{"attr": {"alpha": 1}}]

    def test_source_empty_no_match(self):
        writer = self._make_writer()
        node_ir = NodeIR(
            source="",
            ops=[{"type": "Mul", "attr": {}}],
        )
        result = writer._get_ops_list(node_ir, "Add")
        assert result is None

    def test_source_nonempty(self):
        writer = self._make_writer()
        ops = [{"name": "MatMul", "attr": {}, "input": ["a", "b"]}]
        node_ir = NodeIR(source="converter", ops=ops)
        result = writer._get_ops_list(node_ir, "mm")
        assert result is ops


class TestEpairWriterResolveInputNames:
    def _make_writer(self):
        return EpairWriter.__new__(EpairWriter)

    def test_basic(self):
        writer = self._make_writer()
        ops_list = [{"input": ["ge_a", "ge_b"]}]
        geo_2_fxo = {"ge_a": "fx_a", "ge_b": "fx_b"}
        fx_input_names = {"fx_a", "fx_b"}
        mapping = writer._resolve_input_names(ops_list, geo_2_fxo, fx_input_names)
        assert mapping == {"fx_a": "ge_a", "fx_b": "ge_b"}
        assert ops_list[0]["input"] == ["fx_a", "fx_b"]

    def test_no_input_key(self):
        writer = self._make_writer()
        ops_list = [{"name": "Add"}]
        mapping = writer._resolve_input_names(ops_list, {}, set())
        assert mapping == {}

    def test_empty_ge_input(self):
        writer = self._make_writer()
        ops_list = [{"input": ["", "ge_b"]}]
        geo_2_fxo = {"ge_b": "fx_b"}
        fx_input_names = {"fx_b"}
        mapping = writer._resolve_input_names(ops_list, geo_2_fxo, fx_input_names)
        assert mapping == {"fx_b": "ge_b"}

    def test_not_in_geo_2_fxo(self):
        writer = self._make_writer()
        ops_list = [{"input": ["ge_x"]}]
        mapping = writer._resolve_input_names(ops_list, {}, {"fx_x"})
        assert mapping == {}

    def test_not_in_fx_input_names(self):
        writer = self._make_writer()
        ops_list = [{"input": ["ge_a"]}]
        geo_2_fxo = {"ge_a": "fx_a"}
        mapping = writer._resolve_input_names(ops_list, geo_2_fxo, set())
        assert mapping == {}


class TestEpairWriterMapOutputNames:
    def _make_writer(self):
        return EpairWriter.__new__(EpairWriter)

    def test_basic(self):
        writer = self._make_writer()
        fx_outputs = [{"as_tensor": {"name": "out"}}]
        node_ir_mapping = {"outputs": ["ge_out"]}
        geo_2_fxo = {}
        mapping = writer._map_output_names(fx_outputs, node_ir_mapping, geo_2_fxo)
        assert mapping == {"out": "ge_out"}
        assert geo_2_fxo == {"ge_out": "out"}

    def test_none_mapping(self):
        writer = self._make_writer()
        mapping = writer._map_output_names([], None, {})
        assert mapping == {}

    def test_no_outputs_key(self):
        writer = self._make_writer()
        mapping = writer._map_output_names([], {}, {})
        assert mapping == {}

    def test_const_placeholder_skipped(self):
        writer = self._make_writer()
        fx_outputs = [{"as_int": 42}]
        node_ir_mapping = {"outputs": ["ge_const"]}
        geo_2_fxo = {}
        mapping = writer._map_output_names(fx_outputs, node_ir_mapping, geo_2_fxo)
        assert mapping == {}

    def test_none_pair_skipped(self):
        writer = self._make_writer()
        fx_outputs = [{"as_none": {}}]
        node_ir_mapping = {"outputs": ["None"]}
        geo_2_fxo = {}
        mapping = writer._map_output_names(fx_outputs, node_ir_mapping, geo_2_fxo)
        assert mapping == {}

    def test_count_mismatch_raises(self):
        writer = self._make_writer()
        fx_outputs = [{"as_tensor": {"name": "a"}}, {"as_tensor": {"name": "b"}}]
        node_ir_mapping = {"outputs": ["ge_a"]}
        with pytest.raises(AssertionError, match="not matched"):
            writer._map_output_names(fx_outputs, node_ir_mapping, {})


class TestEpairWriterWrite:
    def test_write_returns_tuple(self):
        import torch
        from torch.export import export
        from torch_delegate import export_to_epair

        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        ep = export(M(), (torch.randn(2, 3), torch.randn(2, 3)))
        contents = export_to_epair(ep)
        assert contents.manifest.total_nodes > 0
        assert contents.manifest.converted_nodes >= 0

    def test_write_injects_ascend_ir(self):
        import torch
        from torch.export import export
        from torch_delegate import export_to_epair

        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        ep = export(M(), (torch.randn(2, 3), torch.randn(3, 4)))
        contents = export_to_epair(ep)
        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        has_ascend_ir = any("ascend_ir" in n for n in nodes)
        assert has_ascend_ir
