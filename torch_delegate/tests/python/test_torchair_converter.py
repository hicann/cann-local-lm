# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import tempfile

import torch
import pytest

from torch.export import export
from torch_delegate import export_to_epair, epair_save, epair_load
from torch_delegate.epair.schema import EpairContents, EpairManifest
from torch_delegate.converter.torchair_converter import TorchAirGEConverter


class SupportedModel(torch.nn.Module):
    def forward(self, x, y):
        z = torch.mm(x, y)
        return torch.abs(z)


class MixedModel(torch.nn.Module):
    def forward(self, x, y):
        z = torch.mm(x, y)
        return torch.relu(z)


@pytest.fixture(scope="module")
def epair_contents():
    model = SupportedModel()
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    ep = export(model, (x, y))
    contents = export_to_epair(ep)
    with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
        path = f.name
    try:
        epair_save(contents, path)
        yield epair_load(path)
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def mixed_epair_contents():
    model = MixedModel()
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    ep = export(model, (x, y))
    contents = export_to_epair(ep)
    with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
        path = f.name
    try:
        epair_save(contents, path)
        yield epair_load(path)
    finally:
        os.unlink(path)


def test_manifest_fields(epair_contents: EpairContents):
    m = epair_contents.manifest
    assert m.version == 1
    assert m.backend_name == "ascend_ge"
    assert m.total_nodes > 0
    assert m.converted_nodes <= m.total_nodes


def test_ascend_ir_injected(epair_contents: EpairContents):
    nodes = epair_contents.ep_json["graph_module"]["graph"]["nodes"]
    assert any("ascend_ir" in node for node in nodes)


def test_ascend_ir_new_schema(epair_contents: EpairContents):
    """测试新 ascend_ir 格式：source + ops + mapping"""
    nodes = epair_contents.ep_json["graph_module"]["graph"]["nodes"]
    for node in nodes:
        if "ascend_ir" in node:
            ir = node["ascend_ir"]
            assert "ops" in ir
            assert isinstance(ir["ops"], list)


def test_roundtrip_via_api():
    model = SupportedModel()
    x = torch.randn(2, 4)
    y = torch.randn(4, 2)
    ep = export(model, (x, y))
    contents = export_to_epair(ep)
    with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
        path = f.name
    try:
        epair_save(contents, path)
        loaded = epair_load(path)
        assert isinstance(loaded, EpairContents)
        assert isinstance(loaded.manifest, EpairManifest)
        assert loaded.manifest.version == 1
        assert "graph_module" in loaded.ep_json
    finally:
        os.unlink(path)


def test_torchair_converter_convert_graph_basic():
    """测试 convert_graph 基本功能"""

    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    model = SimpleModel()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    ep = export(model, args=(x, y))

    converter = TorchAirGEConverter()
    converter.convert_graph(ep)
    ascend_ir_map = converter.get_ascend_ir_map()

    assert isinstance(ascend_ir_map, dict)
    assert len(ascend_ir_map) > 0

    add_node = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.add.Tensor)
    assert add_node.name in ascend_ir_map

    node_ir = ascend_ir_map[add_node.name]
    assert node_ir.source == "converter"
    assert isinstance(node_ir.ops, list)


def test_torchair_converter_convert_graph_multiple_ops():
    """测试 convert_graph 处理多个算子"""

    class Model(torch.nn.Module):
        def forward(self, x, y):
            z = torch.mm(x, y)
            return torch.abs(z)

    model = Model()
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    ep = export(model, args=(x, y))

    converter = TorchAirGEConverter()
    converter.convert_graph(ep)
    ascend_ir_map = converter.get_ascend_ir_map()

    assert len(ascend_ir_map) >= 2

    mm_node = next(
        (n for n in ep.graph.nodes if n.target == torch.ops.aten.mm.default), None
    )
    abs_node = next(
        (n for n in ep.graph.nodes if n.target == torch.ops.aten.abs.default), None
    )

    if mm_node:
        assert mm_node.name in ascend_ir_map
        assert ascend_ir_map[mm_node.name].source == "converter"

    if abs_node:
        assert abs_node.name in ascend_ir_map
        assert ascend_ir_map[abs_node.name].source == "converter"


def test_torchair_converter_is_node_supported():
    """测试 is_node_supported 方法"""

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    model = Model()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    ep = export(model, args=(x, y))

    converter = TorchAirGEConverter()

    add_node = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.add.Tensor)
    assert converter.is_node_supported(add_node) is True

    placeholder_node = next(n for n in ep.graph.nodes if n.op == "placeholder")
    assert converter.is_node_supported(placeholder_node) is False


def test_torchair_converter_nodeir_structure():
    """测试 NodeIR 结构包含 source/ops/mapping"""

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    model = Model()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    ep = export(model, args=(x, y))

    converter = TorchAirGEConverter()
    converter.convert_graph(ep)
    ascend_ir_map = converter.get_ascend_ir_map()

    for node_name, node_ir in ascend_ir_map.items():
        assert hasattr(node_ir, "source")
        assert hasattr(node_ir, "ops")
        assert hasattr(node_ir, "mapping")
        assert node_ir.source in ["converter", ""]
        assert isinstance(node_ir.ops, list)


def test_assert_tensor_metadata_stripped():
    """测试 _assert_tensor_metadata 节点在转换前被剥离。

    torch.export 会自动插入 _assert_tensor_metadata 断言节点（无计算语义），
    convert_graph 应在调用 torchair 前将其移除，使其不出现在 epair JSON 中。
    """

    class DtypeCastModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.float32)

    model = DtypeCastModel().eval()
    x = torch.randn(2, 3, dtype=torch.float16)
    ep = export(model, (x,))

    assert_target = torch.ops.aten._assert_tensor_metadata.default
    raw_nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
    assert any(n.target is assert_target for n in raw_nodes), (
        "导出图中应含 _assert_tensor_metadata 节点"
    )

    contents = export_to_epair(ep)

    nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
    targets = [n.get("target", "") for n in nodes]
    assert not any("_assert_tensor_metadata" in t for t in targets), (
        f"epair JSON 中不应含 _assert_tensor_metadata 节点，实际: {targets}"
    )
