# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
测试 torch_delegate 数据结构：NodeIR, EpairManifest, EpairContents
"""

import torch
from torch.export import export

from torch_delegate.epair.schema import EpairManifest
from torch_delegate.converter.base import NodeIR
from torch_delegate import export_to_epair


class TestNodeIR:
    def test_nodeir_source_converter(self):
        node_ir = NodeIR(
            source="converter",
            ops=[{"name": "Add", "attr": []}],
            mapping={"outputs": ["add_1:0"]},
        )
        assert node_ir.source == "converter"

    def test_nodeir_source_empty_for_mock(self):
        node_ir = NodeIR(source="", ops=[{"attr": []}], mapping=None)
        assert node_ir.source == ""

    def test_nodeir_ops_is_list(self):
        node_ir = NodeIR(
            source="converter",
            ops=[
                {"name": "Add", "attr": [], "input": [], "output": []},
                {"name": "Cast", "attr": [], "input": [], "output": []},
            ],
            mapping={},
        )
        assert isinstance(node_ir.ops, list)
        assert len(node_ir.ops) == 2

    def test_nodeir_ops_dict_structure(self):
        node_ir = NodeIR(
            source="converter",
            ops=[{"name": "Add", "attr": [], "input": [], "output": []}],
            mapping={},
        )
        op = node_ir.ops[0]
        assert isinstance(op, dict)

    def test_nodeir_mapping_optional(self):
        node_ir_with_mapping = NodeIR(
            source="converter", ops=[], mapping={"outputs": ["node:0"]}
        )
        node_ir_without_mapping = NodeIR(source="converter", ops=[], mapping=None)
        assert node_ir_with_mapping.mapping is not None
        assert node_ir_without_mapping.mapping is None

    def test_nodeir_mapping_outputs_format(self):
        node_ir = NodeIR(
            source="converter", ops=[], mapping={"outputs": ["add_1:0", "add_1:1"]}
        )
        assert isinstance(node_ir.mapping["outputs"], list)
        for item in node_ir.mapping["outputs"]:
            assert ":" in item

    def test_nodeir_equality(self):
        ir1 = NodeIR(source="converter", ops=[{"name": "Add"}], mapping={})
        ir2 = NodeIR(source="converter", ops=[{"name": "Add"}], mapping={})
        ir3 = NodeIR(source="", ops=[{"name": "Add"}], mapping={})

        assert ir1 == ir2
        assert ir1 != ir3


class TestEpairManifest:
    def test_manifest_all_fields(self):
        manifest = EpairManifest(
            version=1,
            backend_name="ascend_ge",
            cann_version="8.0.RC3",
            total_nodes=10,
            converted_nodes=7,
        )
        assert manifest.version == 1
        assert manifest.backend_name == "ascend_ge"
        assert manifest.cann_version == "8.0.RC3"
        assert manifest.total_nodes == 10
        assert manifest.converted_nodes == 7

    def test_manifest_cann_version_unknown(self):
        manifest = EpairManifest(
            version=1,
            backend_name="ascend_ge",
            cann_version="unknown",
            total_nodes=5,
            converted_nodes=3,
        )
        assert manifest.cann_version == "unknown"

    def test_manifest_converted_not_exceed_total(self):
        manifest = EpairManifest(
            version=1,
            backend_name="ascend_ge",
            cann_version="8.0.RC3",
            total_nodes=10,
            converted_nodes=10,
        )
        assert manifest.converted_nodes <= manifest.total_nodes

    def test_manifest_from_real_conversion(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        manifest = contents.manifest
        assert isinstance(manifest, EpairManifest)
        assert manifest.version == 2
        assert manifest.backend_name == "ascend_ge"
        assert isinstance(manifest.cann_version, str)
        assert manifest.total_nodes > 0
        assert manifest.cpu_nodes >= 0


class TestEpairContents:
    def test_contents_structure(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        assert hasattr(contents, "manifest")
        assert hasattr(contents, "ep_json")
        assert hasattr(contents, "ep")

    def test_contents_manifest_type(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        assert isinstance(contents.manifest, EpairManifest)

    def test_contents_ep_json_type(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        assert isinstance(contents.ep_json, dict)
        assert "graph_module" in contents.ep_json

    def test_contents_ep_type(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        from torch.export import ExportedProgram

        assert isinstance(contents.ep, ExportedProgram)

    def test_contents_ep_executable(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        output = contents.ep.module()(x)
        assert output.shape == (2, 4)


class TestAscendIRFormat:
    def test_ascend_ir_has_source(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        for node in nodes:
            if "ascend_ir" in node:
                ir = node["ascend_ir"]
                if "source" in ir:
                    assert ir["source"] in ["converter", ""]

    def test_ascend_ir_has_ops(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        for node in nodes:
            if "ascend_ir" in node:
                ir = node["ascend_ir"]
                assert "ops" in ir
                assert isinstance(ir["ops"], list)

    def test_ascend_ir_ops_elements(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        for node in nodes:
            if "ascend_ir" in node:
                ir = node["ascend_ir"]
                for op in ir["ops"]:
                    assert isinstance(op, dict)

    def test_ascend_ir_mapping_optional(self):
        model = torch.nn.Linear(3, 4)
        x = torch.randn(2, 3)
        ep = export(model, (x,))
        contents = export_to_epair(ep)

        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        for node in nodes:
            if "ascend_ir" in node:
                ir = node["ascend_ir"]
                if "mapping" in ir:
                    assert isinstance(ir["mapping"], dict)


class TestNodeIRDeviceField:
    """NodeIR.device 字段。"""

    def test_device_defaults_to_npu(self):
        node_ir = NodeIR(source="converter", ops=[])
        assert node_ir.device == "npu"

    def test_device_cpu(self):
        node_ir = NodeIR(source="", ops=[], device="cpu")
        assert node_ir.device == "cpu"

    def test_device_roundtrip(self):
        import dataclasses

        node_ir = NodeIR(source="", ops=[], device="cpu")
        dumped = dataclasses.asdict(node_ir)
        assert dumped["device"] == "cpu"
        restored = NodeIR(**dumped)
        assert restored == node_ir


class TestEpairManifestV2:
    """manifest v2：version=2 与 cpu_nodes 计数。"""

    def test_manifest_v2_fields(self):
        manifest = EpairManifest(
            version=2,
            backend_name="ascend_ge",
            cann_version="9.2.0",
            total_nodes=10,
            converted_nodes=7,
            cpu_nodes=3,
        )
        assert manifest.version == 2
        assert manifest.cpu_nodes == 3

    def test_manifest_cpu_nodes_default_zero(self):
        manifest = EpairManifest(
            version=2,
            backend_name="ascend_ge",
            cann_version="unknown",
            total_nodes=5,
            converted_nodes=5,
        )
        assert manifest.cpu_nodes == 0

    def test_manifest_v1_semantics_unchanged(self):
        manifest = EpairManifest(
            version=1,
            backend_name="ascend_ge",
            cann_version="9.1.0",
            total_nodes=5,
            converted_nodes=5,
        )
        assert manifest.version == 1

    def test_manifest_v2_serialization_roundtrip(self):
        import dataclasses
        import json as json_lib

        manifest = EpairManifest(
            version=2,
            backend_name="ascend_ge",
            cann_version="9.2.0",
            total_nodes=8,
            converted_nodes=8,
            cpu_nodes=2,
        )
        payload = json_lib.dumps(dataclasses.asdict(manifest))
        loaded = json_lib.loads(payload)
        assert loaded["version"] == 2
        assert loaded["cpu_nodes"] == 2
        assert loaded["cann_version"] == "9.2.0"


class TestDeviceDecision:
    """DeviceDecision 判定记录结构。"""

    def test_decision_fields(self):
        from torch_delegate.epair.schema import DeviceDecision

        decision = DeviceDecision(
            node_name="relu_1",
            target="torch.ops.aten.relu.default",
            namespace="aten",
            op_name="aten::relu",
            overload="",
            device="cpu",
            rule="cpu_only",
            has_privateuse1=False,
            has_cpu=True,
        )
        assert decision.device == "cpu"
        assert decision.rule == "cpu_only"
        assert decision.has_cpu is True
        assert decision.has_privateuse1 is False

    def test_decision_rules(self):
        from torch_delegate.epair.schema import DeviceDecision

        for rule in ("converter", "has_privateuse1", "cpu_only"):
            decision = DeviceDecision(
                node_name="n",
                target="t",
                namespace="aten",
                op_name="aten::op",
                overload="",
                device="npu",
                rule=rule,
                has_privateuse1=True,
                has_cpu=True,
            )
            assert decision.rule == rule
