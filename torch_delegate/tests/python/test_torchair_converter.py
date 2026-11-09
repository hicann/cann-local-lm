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
from torch_delegate.converter.base import NodeIR
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
    assert m.version == 2
    assert m.cpu_nodes >= 0
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
        assert loaded.manifest.version == 2
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


# ===========================================================================
# Device 归属自动判定（monkeypatch dispatcher 探测，不依赖 torch_npu）
# ===========================================================================


class TestResolveDevice:
    @pytest.fixture
    def converter(self):
        from torch_delegate.converter.torchair_converter import TorchAirGEConverter

        return TorchAirGEConverter()

    def _patch_dispatch(
        self,
        monkeypatch,
        has_privateuse1: bool,
        has_cpu: bool,
        has_composite: bool = False,
    ):
        import torch_delegate.converter.torchair_converter as tc

        def fake_has_kernel(target, key):
            if key == "PrivateUse1":
                return has_privateuse1
            if key == "CPU":
                return has_cpu
            if key == "CompositeImplicitAutograd":
                return has_composite
            raise AssertionError(f"unexpected key {key}")

        monkeypatch.setattr(tc, "_dispatch_has_kernel", fake_has_kernel)

    def test_has_privateuse1_resolves_npu(self, converter, monkeypatch):
        self._patch_dispatch(monkeypatch, has_privateuse1=True, has_cpu=True)
        device, has_pu1, has_cpu = converter._resolve_device(
            torch.ops.aten.relu.default
        )
        assert device == "npu"
        assert has_pu1 is True and has_cpu is True

    def test_cpu_only_resolves_cpu(self, converter, monkeypatch):
        self._patch_dispatch(monkeypatch, has_privateuse1=False, has_cpu=True)
        device, has_pu1, has_cpu = converter._resolve_device(
            torch.ops.aten.relu.default
        )
        assert device == "cpu"
        assert has_pu1 is False and has_cpu is True

    def test_composite_only_resolves_cpu(self, converter, monkeypatch):
        """仅注册 CompositeImplicitAutograd 的算子：回调时传 CPU 张量，判定为 CPU。"""
        self._patch_dispatch(
            monkeypatch, has_privateuse1=False, has_cpu=False, has_composite=True
        )
        device, has_pu1, has_cpu = converter._resolve_device(
            torch.ops.aten.relu.default
        )
        assert device == "cpu"
        assert has_pu1 is False and has_cpu is False

    def test_neither_kernel_raises(self, converter, monkeypatch):
        self._patch_dispatch(monkeypatch, has_privateuse1=False, has_cpu=False)
        with pytest.raises(
            RuntimeError,
            match="has neither PrivateUse1, CPU nor CompositeImplicitAutograd kernel",
        ):
            converter._resolve_device(torch.ops.aten.relu.default)

    def test_decision_record_fields(self, converter, monkeypatch):
        self._patch_dispatch(monkeypatch, has_privateuse1=False, has_cpu=True)
        device, has_pu1, has_cpu = converter._resolve_device(
            torch.ops.aten.relu.default
        )
        node_ir = converter._ascend_ir_map
        decision = converter._make_decision(
            _FakeFxNode("relu_1", torch.ops.aten.relu.default),
            torch.ops.aten.relu.default,
            device,
            has_pu1,
            has_cpu,
        )
        assert decision.node_name == "relu_1"
        assert decision.device == "cpu"
        assert decision.rule == "cpu_only"
        assert decision.namespace == "aten"
        assert decision.op_name == "aten::relu"
        assert decision.overload == ""
        assert node_ir == {}


class _FakeFxNode:
    """最小 FX 节点替身：仅提供 _make_decision / _resolve_mock_node_devices 需要的属性。"""

    def __init__(self, name, target, args=(), op="call_function"):
        self.name = name
        self.target = target
        self.args = args
        self.op = op


class _FakeOptimizedGraph:
    """最小 torchair 优化图替身：提供 _resolve_mock_node_devices 需要的
    optimized.graph.nodes 访问路径。"""

    class _Graph:
        def __init__(self, nodes):
            self.nodes = nodes

    class _Optimized:
        def __init__(self, nodes):
            self.graph = _FakeOptimizedGraph._Graph(nodes)

    def __init__(self, nodes):
        self.optimized = _FakeOptimizedGraph._Optimized(nodes)


class TestResolveMockNodeDevices:
    """fallback 节点归属判定：HOP 解包与 Dispatcher 探测（不依赖 torchair）。"""

    def _run(self, monkeypatch, nodes):
        import torch_delegate.converter.torchair_converter as tc

        converter = TorchAirGEConverter()
        converter._ascend_ir_map = {n.name: NodeIR(source="", ops=[]) for n in nodes}
        converter._ge_graph_ascend = _FakeOptimizedGraph(nodes)
        monkeypatch.setattr(
            tc,
            "_dispatch_has_kernel",
            lambda target, key: key == "CPU",
        )
        converter._resolve_mock_node_devices()
        return converter

    def test_auto_functionalized_v2_unwraps_inner_op(self, monkeypatch):
        """auto_functionalized_v2 的 target 不是 OpOverload，须解包后再判定。

        torch >= 2.9 导出图中节点 args[0] 直接是被包裹的 OpOverload
        （auto_functionalized_v2(_mutable_op, **kwargs) 的第一个位置参数）。
        """
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

        inner_op = torch.ops.aten.relu_.default
        node = _FakeFxNode(
            "relu_mutated_1",
            auto_functionalized_v2,
            args=(inner_op,),
        )
        converter = self._run(monkeypatch, [node])
        decisions = converter.get_device_decisions()
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.device == "cpu"
        assert decision.target == "torch.ops.aten.relu_.default"
        assert decision.namespace == "aten"
        assert converter._ascend_ir_map[node.name].device == "cpu"

    def test_auto_functionalized_v1_unwraps_inner_op(self, monkeypatch):
        """v1 HOP（args[0] 同样是被包裹算子）也能解包判定。"""
        from torch._higher_order_ops.auto_functionalize import auto_functionalized

        inner_op = torch.ops.aten.relu_.default
        node = _FakeFxNode("relu_mutated_v1", auto_functionalized, args=(inner_op,))
        converter = self._run(monkeypatch, [node])
        decisions = converter.get_device_decisions()
        assert len(decisions) == 1
        assert decisions[0].device == "cpu"
        assert decisions[0].target == "torch.ops.aten.relu_.default"

    def test_auto_functionalized_v2_wrapper_with_func_attr(self, monkeypatch):
        """兼容 AutoFunctionalizedV2(func=...) 包装形态：取 .func 解包。"""
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

        inner_op = torch.ops.aten.relu_.default

        class _FuncWrapper:
            func = inner_op

        node = _FakeFxNode(
            "relu_mutated_wrap", auto_functionalized_v2, args=(_FuncWrapper(),)
        )
        converter = self._run(monkeypatch, [node])
        decisions = converter.get_device_decisions()
        assert len(decisions) == 1
        assert decisions[0].device == "cpu"
        assert decisions[0].target == "torch.ops.aten.relu_.default"

    def test_unsupported_hop_target_still_raises(self, monkeypatch):
        """非 auto_functionalized 的 HOP（target 非 OpOverload）保持严格报错。"""

        def unknown_hop(x):
            return x

        node = _FakeFxNode("hop_1", unknown_hop, args=(1,))
        with pytest.raises(RuntimeError, match="unsupported target type"):
            self._run(monkeypatch, [node])
