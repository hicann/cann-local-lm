# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from typing import Any, Callable

from ..core.utils import logger
from ..converter.base import NodeIR, NodeIRConverter

_CONST_PLACEHOLDER_PREFIX = "CONST_"


def _names_direct(v: dict) -> list[str]:
    return [v["name"]]


def _names_list(v: list) -> list[str]:
    return [ts["name"] for ts in v]


def _names_opt_single(v: dict) -> list[str]:
    return [v["as_tensor"]["name"]] if "as_tensor" in v else []


def _names_opt_list(v: list) -> list[str]:
    return [opt["as_tensor"]["name"] for opt in v if "as_tensor" in opt]


def _names_sym_single(v: dict) -> list[str]:
    return [v["as_name"]] if "as_name" in v else []


def _names_sym_list(v: list) -> list[str]:
    return [sym["as_name"] for sym in v if "as_name" in sym]


_ARG_NAME_EXTRACTORS: dict[str, Callable[[Any], list[str]]] = {
    "as_tensor": _names_direct,
    "as_tensors": _names_list,
    "as_optional_tensor": _names_opt_single,
    "as_optional_tensors": _names_opt_list,
    "as_sym_int": _names_sym_single,
    "as_sym_ints": _names_sym_list,
    "as_sym_float": _names_sym_single,
    "as_sym_floats": _names_sym_list,
    "as_sym_bool": _names_sym_single,
    "as_sym_bools": _names_sym_list,
    "as_custom_obj": _names_direct,
    "as_graph": _names_direct,
}


def _extract_names_from_arg(arg: dict) -> list[str]:
    for key, extractor in _ARG_NAME_EXTRACTORS.items():
        if key in arg:
            return extractor(arg[key])
    return []


def _flat_none(v: Any) -> list[str]:
    return ["None"]


def _flat_named(v: dict) -> list[str]:
    return [v["name"]]


def _flat_named_list(v: list) -> list[str]:
    return [ts["name"] for ts in v]


def _flat_opt_single(v: dict) -> list[str]:
    if "as_tensor" in v:
        return [v["as_tensor"]["name"]]
    if "as_none" in v:
        return ["None"]
    return []


def _flat_opt_list(v: list) -> list[str]:
    items: list[str] = []
    for opt in v:
        if "as_tensor" in opt:
            items.append(opt["as_tensor"]["name"])
        elif "as_none" in opt:
            items.append("None")
    return items


def _make_sym_single(const_type: str) -> Callable[[dict], list[str]]:
    const_key = f"as_{const_type}"

    def handler(v: dict) -> list[str]:
        if "as_name" in v:
            return [v["as_name"]]
        if const_key in v:
            return [f"{_CONST_PLACEHOLDER_PREFIX}{const_type}"]
        return []

    return handler


def _make_sym_list(const_type: str) -> Callable[[list], list[str]]:
    const_key = f"as_{const_type}"

    def handler(v: list) -> list[str]:
        items: list[str] = []
        for sym in v:
            if "as_name" in sym:
                items.append(sym["as_name"])
            elif const_key in sym:
                items.append(f"{_CONST_PLACEHOLDER_PREFIX}{const_type}")
        return items

    return handler


def _make_const_single(const_type: str) -> Callable[[Any], list[str]]:
    def handler(v: Any) -> list[str]:
        return [f"{_CONST_PLACEHOLDER_PREFIX}{const_type}"]

    return handler


def _make_const_list(const_type: str) -> Callable[[list], list[str]]:
    def handler(v: list) -> list[str]:
        return [f"{_CONST_PLACEHOLDER_PREFIX}{const_type}" for _ in v]

    return handler


def _make_const_singleton(const_type: str) -> Callable[[Any], list[str]]:
    def handler(v: Any) -> list[str]:
        return [f"{_CONST_PLACEHOLDER_PREFIX}{const_type}"]

    return handler


_OUTPUT_FLATTENERS: dict[str, Callable[[Any], list[str]]] = {
    "as_none": _flat_none,
    "as_tensor": _flat_named,
    "as_tensors": _flat_named_list,
    "as_optional_tensor": _flat_opt_single,
    "as_optional_tensors": _flat_opt_list,
    "as_sym_int": _make_sym_single("int"),
    "as_sym_ints": _make_sym_list("int"),
    "as_sym_float": _make_sym_single("float"),
    "as_sym_floats": _make_sym_list("float"),
    "as_sym_bool": _make_sym_single("bool"),
    "as_sym_bools": _make_sym_list("bool"),
    "as_int": _make_const_single("int"),
    "as_ints": _make_const_list("int"),
    "as_float": _make_const_single("float"),
    "as_floats": _make_const_list("float"),
    "as_bool": _make_const_single("bool"),
    "as_bools": _make_const_list("bool"),
    "as_string": _make_const_single("string"),
    "as_strings": _make_const_list("string"),
    "as_scalar_type": _make_const_singleton("scalar_type"),
    "as_memory_format": _make_const_singleton("memory_format"),
    "as_layout": _make_const_singleton("layout"),
    "as_device": _make_const_singleton("device"),
    "as_complex": _make_const_singleton("complex"),
    "as_operator": _make_const_singleton("operator"),
    "as_custom_obj": _flat_named,
    "as_graph": _flat_named,
}


def _flatten_output(output: dict) -> list[str]:
    for key, handler in _OUTPUT_FLATTENERS.items():
        if key in output:
            return handler(output[key])
    return []


class EpairWriter:
    def __init__(self, converter: NodeIRConverter) -> None:
        self._converter = converter

    def write(self, ep_json: dict) -> tuple[dict, int, int]:
        return self._inject_ascend_ir(ep_json)

    def _inject_ascend_ir(self, ep_json: dict) -> tuple[dict, int, int]:
        ascend_ir_map = self._converter.get_ascend_ir_map()
        graph_nodes: list[dict] = (
            ep_json.get("graph_module", {}).get("graph", {}).get("nodes", [])
        )
        total_nodes = len(graph_nodes)
        converted_nodes = 0
        geo_2_fxo: dict[str, str] = {}

        for serial_node in graph_nodes:
            if self._process_node(serial_node, ascend_ir_map, geo_2_fxo):
                converted_nodes += 1

        logger.debug(f"Total graph nodes: {total_nodes}, converted: {converted_nodes}")
        return ep_json, total_nodes, converted_nodes

    def _process_node(
        self,
        serial_node: dict,
        ascend_ir_map: dict[str, NodeIR],
        geo_2_fxo: dict[str, str],
    ) -> bool:
        torch_fn = serial_node.get("metadata").get("torch_fn")
        target = serial_node.get("target")
        if target == "torch.ops.higher_order.auto_functionalized_v2":
            inputs = serial_node.get("inputs")
            target = inputs[0].get("arg").get("as_operator")
        target = target.removeprefix("torch.ops.")

        parts = torch_fn.rsplit(";", 1)
        if len(parts) > 1:
            torch_fn = parts[0]
            aten_ir = parts[1]
        else:
            torch_fn = None
            aten_ir = parts[0]
        serial_node.get("metadata")["torch_fn"] = torch_fn
        logger.debug(f"Try to get key: {aten_ir}")

        fx_input_names = self._extract_fx_input_names(serial_node)

        if aten_ir not in ascend_ir_map:
            return False

        logger.debug(f"Got key: {aten_ir} in map")
        node_ir = ascend_ir_map[aten_ir]

        ops_list = self._get_ops_list(node_ir, target)
        if ops_list is None:
            return False

        input_mapping = self._resolve_input_names(ops_list, geo_2_fxo, fx_input_names)
        output_mapping = self._map_output_names(
            serial_node.get("outputs", []), node_ir.mapping, geo_2_fxo
        )

        serial_node["ascend_ir"] = {
            "source": node_ir.source,
            "ops": ops_list,
            "mapping": {"inputs": input_mapping, "outputs": output_mapping},
        }
        return True

    def _extract_fx_input_names(self, serial_node: dict) -> set[str]:
        fx_input_names: set[str] = set()
        for inp in serial_node.get("inputs", []):
            arg = inp.get("arg", {})
            for name in _extract_names_from_arg(arg):
                fx_input_names.add(name)
        return fx_input_names

    def _get_ops_list(self, node_ir: NodeIR, target: str) -> list[dict] | None:
        if len(node_ir.source) == 0:
            for ir in node_ir.ops:
                if target == ir.get("type"):
                    return [{"attr": ir.get("attr", {})}]
            return None
        return node_ir.ops

    def _resolve_input_names(
        self, ops_list: list[dict], geo_2_fxo: dict[str, str], fx_input_names: set[str]
    ) -> dict[str, str]:
        input_mapping: dict[str, str] = {}
        for op in ops_list:
            if "input" not in op:
                continue
            for idx, ge_input in enumerate(op["input"]):
                if not ge_input:
                    continue

                if ge_input in geo_2_fxo:
                    fx_name = geo_2_fxo[ge_input]
                    if fx_name not in fx_input_names:
                        continue
                    input_mapping[fx_name] = ge_input
                    op["input"][idx] = fx_name

        return input_mapping

    def _map_output_names(
        self,
        fx_outputs: list[dict],
        node_ir_mapping: dict | None,
        geo_2_fxo: dict[str, str],
    ) -> dict[str, str]:
        output_mapping: dict[str, str] = {}
        if node_ir_mapping is None or "outputs" not in node_ir_mapping:
            return output_mapping

        ge_outputs = node_ir_mapping["outputs"]
        fx_flat: list[str] = []
        for fx_output in fx_outputs:
            fx_flat.extend(_flatten_output(fx_output))

        if len(fx_flat) != len(ge_outputs):
            raise AssertionError(
                f"fx_outputs({len(fx_flat)}) and ge_outputs({len(ge_outputs)}) not matched."
            )
        for fx_out, ge_out in zip(fx_flat, ge_outputs):
            if fx_out.startswith(_CONST_PLACEHOLDER_PREFIX):
                continue
            if fx_out == "None" and ge_out == "None":
                continue
            output_mapping[fx_out] = ge_out
            geo_2_fxo[ge_out] = fx_out

        return output_mapping

    def _get_cann_version(self) -> str:
        try:
            import torch_npu

            return torch_npu.npu.utils.get_cann_version()
        except Exception as e:
            logger.warning(f"CANN version unavailable: {e}")
            return "unknown"
