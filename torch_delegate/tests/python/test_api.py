# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
测试 torch_delegate API：export_to_epair, epair_save, epair_load
"""

import os
import tempfile
import pytest

import torch
from torch.export import ExportedProgram, export

from torch_delegate import export_to_epair, epair_save, epair_load
from torch_delegate.epair.schema import EpairContents, EpairManifest
from torch_delegate.epair.config import EpairConfig


class SimpleAddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)


class MatMulAbsModel(torch.nn.Module):
    def forward(self, x, y):
        z = torch.mm(x, y)
        return torch.abs(z)


class MixedModel(torch.nn.Module):
    def forward(self, x, y):
        z = torch.mm(x, y)
        return torch.relu(z)


@pytest.fixture
def simple_ep():
    model = SimpleAddModel()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    return export(model, (x, y))


@pytest.fixture
def matmul_ep():
    model = MatMulAbsModel()
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    return export(model, (x, y))


class TestExportToEpair:
    def test_returns_epair_contents(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert isinstance(contents, EpairContents)

    def test_contents_has_manifest(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert isinstance(contents.manifest, EpairManifest)

    def test_contents_has_ep_json(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert isinstance(contents.ep_json, dict)
        assert "graph_module" in contents.ep_json

    def test_contents_has_ep(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert isinstance(contents.ep, ExportedProgram)

    def test_manifest_version_is_2(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert contents.manifest.version == 2

    def test_manifest_backend_name(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert contents.manifest.backend_name == "ascend_ge"

    def test_manifest_has_cann_version(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert hasattr(contents.manifest, "cann_version")
        assert isinstance(contents.manifest.cann_version, str)

    def test_manifest_total_nodes_positive(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert contents.manifest.total_nodes > 0

    def test_manifest_converted_nodes_valid(self, simple_ep):
        contents = export_to_epair(simple_ep)
        assert contents.manifest.converted_nodes <= contents.manifest.total_nodes

    def test_ep_json_has_ascend_ir(self, matmul_ep):
        contents = export_to_epair(matmul_ep)
        nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        has_ascend_ir = any("ascend_ir" in node for node in nodes)
        assert has_ascend_ir

    def test_with_default_config(self, simple_ep):
        contents = export_to_epair(simple_ep, EpairConfig())
        assert isinstance(contents, EpairContents)

    def test_with_custom_config(self, simple_ep):
        config = EpairConfig(remove_noop_ops=False, pattern_fusion_pass=False)
        contents = export_to_epair(simple_ep, config)
        assert isinstance(contents, EpairContents)


class TestEpairSaveLoad:
    def test_save_creates_file(self, simple_ep):
        contents = export_to_epair(simple_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_returns_epair_contents(self, simple_ep):
        contents = export_to_epair(simple_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)
            loaded = epair_load(path)
            assert isinstance(loaded, EpairContents)
        finally:
            os.unlink(path)

    def test_load_manifest_matches(self, simple_ep):
        contents = export_to_epair(simple_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)
            loaded = epair_load(path)

            assert loaded.manifest.version == contents.manifest.version
            assert loaded.manifest.backend_name == contents.manifest.backend_name
            assert loaded.manifest.total_nodes == contents.manifest.total_nodes
            assert loaded.manifest.converted_nodes == contents.manifest.converted_nodes
        finally:
            os.unlink(path)

    def test_load_ep_json_matches(self, simple_ep):
        contents = export_to_epair(simple_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)
            loaded = epair_load(path)

            assert "graph_module" in loaded.ep_json
        finally:
            os.unlink(path)

    def test_load_ep_is_executable(self, simple_ep):
        contents = export_to_epair(simple_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)
            loaded = epair_load(path)

            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            output = loaded.ep.module()(x, y)
            assert isinstance(output, torch.Tensor)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_data(self, matmul_ep):
        contents = export_to_epair(matmul_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)
            loaded = epair_load(path)

            x = torch.randn(2, 3)
            y = torch.randn(3, 4)
            output1 = contents.ep.module()(x, y)
            output2 = loaded.ep.module()(x, y)
            assert torch.allclose(output1, output2)
        finally:
            os.unlink(path)


class TestEpairContents:
    def test_contents_execution(self, matmul_ep):
        contents = export_to_epair(matmul_ep)
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        output = contents.ep.module()(x, y)
        assert output.shape == (2, 4)

    def test_contents_json_access(self, matmul_ep):
        contents = export_to_epair(matmul_ep)
        graph_nodes = contents.ep_json["graph_module"]["graph"]["nodes"]
        assert len(graph_nodes) > 0

    def test_contents_manifest_access(self, matmul_ep):
        contents = export_to_epair(matmul_ep)
        assert contents.manifest.version == 2
        assert contents.manifest.backend_name == "ascend_ge"


class TestEpairConfig:
    def test_default_config(self):
        config = EpairConfig()
        assert config.remove_noop_ops
        assert config.pattern_fusion_pass

    def test_custom_config(self):
        config = EpairConfig(remove_noop_ops=False, pattern_fusion_pass=False)
        assert not config.remove_noop_ops
        assert not config.pattern_fusion_pass

    def test_config_affects_conversion(self, simple_ep):
        config1 = EpairConfig(remove_noop_ops=True)
        config2 = EpairConfig(remove_noop_ops=False)

        contents1 = export_to_epair(simple_ep, config1)
        contents2 = export_to_epair(simple_ep, config2)

        assert isinstance(contents1, EpairContents)
        assert isinstance(contents2, EpairContents)

    def test_default_whitelist_decompositions(self):
        config = EpairConfig()
        assert torch.ops.aten.isfinite.default in config.whitelist_decompositions
        for op in [
            torch.ops.aten.negative.default,
            torch.ops.aten.absolute.default,
            torch.ops.aten.relu6.default,
            torch.ops.aten.arccos.default,
            torch.ops.aten.arcsin.default,
            torch.ops.aten.arctan.default,
            torch.ops.aten.arccosh.default,
            torch.ops.aten.arcsinh.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.square.default,
        ]:
            assert op in config.whitelist_decompositions
        assert len(config.whitelist_decompositions) == 18

    @pytest.mark.parametrize(
        "op",
        sorted(EpairConfig().whitelist_decompositions, key=str),
        ids=lambda op: str(op),
    )
    def test_default_whitelist_ops_all_in_decomp_table(self, op):
        """默认白名单中的每个算子都必须存在于 torch.export.default_decompositions() 表中，
        否则该白名单项不会生效（torchair_converter 会与该表取交集）。"""
        from torch.export import default_decompositions

        assert op in default_decompositions()


class TestWhitelistDecompositionE2E:
    """端到端验证默认白名单算子在 export_to_epair 后被正确分解。"""

    class AliasOpsModel(torch.nn.Module):
        def forward(self, x):
            y = torch.ops.aten.negative.default(x)
            z = torch.ops.aten.absolute.default(y)
            a = torch.ops.aten.arccos.default(z)
            b = torch.ops.aten.arcsin.default(a)
            c = torch.ops.aten.arctan.default(b)
            d = torch.ops.aten.arccosh.default(c)
            e = torch.ops.aten.arcsinh.default(d)
            r = torch.ops.aten.relu6.default(e)
            return torch.ops.aten.square.default(r)

    class Conv2dModel(torch.nn.Module):
        def forward(self, x):
            w = torch.randn(2, 3, 3, 3)
            return torch.ops.aten.conv2d.default(x, w, None, [1, 1], [0, 0], [1, 1], 1)

    @staticmethod
    def _graph_ops(contents):
        return [
            str(n.target) for n in contents.ep.graph.nodes if n.op == "call_function"
        ]

    def test_alias_ops_decomposed(self):
        ep = export(self.AliasOpsModel(), (torch.randn(2, 2),))
        contents = export_to_epair(ep)
        ops = self._graph_ops(contents)

        for alias in [
            "aten.negative.default",
            "aten.absolute.default",
            "aten.arccos.default",
            "aten.arcsin.default",
            "aten.arctan.default",
            "aten.arccosh.default",
            "aten.arcsinh.default",
            "aten.relu6.default",
            "aten.square.default",
        ]:
            assert alias not in ops, f"{alias} 应被分解"
        for canonical in [
            "aten.neg.default",
            "aten.abs.default",
            "aten.acos.default",
            "aten.asin.default",
            "aten.atan.default",
            "aten.acosh.default",
            "aten.asinh.default",
            "aten.hardtanh.default",
            "aten.pow.Tensor_Scalar",
        ]:
            assert canonical in ops, f"{canonical} 应出现在分解后的图中"

    def test_conv2d_decomposed(self):
        ep = export(self.Conv2dModel(), (torch.randn(1, 3, 8, 8),))
        contents = export_to_epair(ep)
        ops = self._graph_ops(contents)

        assert "aten.conv2d.default" not in ops
        assert "aten.convolution.default" in ops


class TestMixedModel:
    def test_mixed_model_conversion(self):
        model = MixedModel()
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        ep = export(model, (x, y))

        contents = export_to_epair(ep)
        assert isinstance(contents, EpairContents)

    def test_mixed_model_some_nodes_converted(self):
        model = MixedModel()
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        ep = export(model, (x, y))

        contents = export_to_epair(ep)
        assert contents.manifest.converted_nodes > 0
        assert contents.manifest.converted_nodes <= contents.manifest.total_nodes

    def test_epair_file_structure(self, matmul_ep):
        import zipfile

        contents = export_to_epair(matmul_ep)
        with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
            path = f.name

        try:
            epair_save(contents, path)

            with zipfile.ZipFile(path, "r") as zf:
                filenames = zf.namelist()
                assert any(f.endswith("epair_manifest.json") for f in filenames)
                assert any(f.endswith("ep_json_with_ascend_ir.json") for f in filenames)
        finally:
            os.unlink(path)


# ===========================================================================
# Device 归属报告与严格校验（不依赖 torchair）
# ===========================================================================


class TestDeviceReport:
    def _decision(self, **overrides):
        from torch_delegate.epair.schema import DeviceDecision

        fields = dict(
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
        fields.update(overrides)
        return DeviceDecision(**fields)

    def test_report_contains_both_sides(self):
        from torch_delegate.converter.device_decision import format_device_report

        decisions = [
            self._decision(
                node_name="mm_1",
                op_name="aten::mm",
                device="npu",
                rule="converter",
                has_cpu=True,
            ),
            self._decision(
                node_name="relu_1", op_name="aten::relu", device="cpu", rule="cpu_only"
            ),
            self._decision(
                node_name="merge_1",
                target="torch.ops.my_ops.merge.default",
                namespace="my_ops",
                op_name="my_ops::merge",
                device="cpu",
                rule="cpu_only",
            ),
        ]
        text = format_device_report(decisions)
        assert "Device ownership report" in text
        assert "NPU (1)" in text
        assert "CPU (2)" in text
        assert "suspicious" in text  # aten→CPU 属 WARNING 级可疑
        assert "expected" in text  # 自定义 namespace→CPU 属预期
        # 双清单带 overload 与探测依据，机器可读
        assert "has_privateuse1=False" in text
        assert "aten::relu.default" in text

    def test_aten_cpu_warning_logged(self, caplog):
        from torch_delegate.converter.device_decision import log_device_report

        with caplog.at_level("WARNING", logger="torch_delegate"):
            log_device_report([self._decision()])
        assert any("no PrivateUse1 kernel" in r.message for r in caplog.records)

    def test_custom_namespace_cpu_is_not_suspicious(self, caplog):
        from torch_delegate.converter.device_decision import log_device_report

        decision = self._decision(
            node_name="merge_1",
            target="torch.ops.my_ops.merge.default",
            namespace="my_ops",
            op_name="my_ops::merge",
        )
        with caplog.at_level("WARNING", logger="torch_delegate"):
            log_device_report([decision])
        assert not any("no PrivateUse1 kernel" in r.message for r in caplog.records)

    def test_strict_validation_raises_with_full_list(self):
        from torch_delegate.converter.device_decision import validate_device_decisions

        decisions = [
            self._decision(node_name="relu_1", op_name="aten::relu"),
            self._decision(node_name="unique_1", op_name="aten::_unique2"),
        ]
        with pytest.raises(RuntimeError) as exc_info:
            validate_device_decisions(decisions, require_aten_on_npu=True)
        message = str(exc_info.value)
        assert "2 aten op(s)" in message
        assert "aten::relu" in message
        assert "aten::_unique2" in message
        assert "relu_1" in message

    def test_strict_validation_ignores_custom_namespace(self):
        from torch_delegate.converter.device_decision import validate_device_decisions

        decisions = [
            self._decision(
                node_name="merge_1",
                target="torch.ops.my_ops.merge.default",
                namespace="my_ops",
                op_name="my_ops::merge",
            ),
        ]
        validate_device_decisions(decisions, require_aten_on_npu=True)  # 不抛异常

    def test_strict_validation_disabled_by_default(self):
        from torch_delegate.converter.device_decision import validate_device_decisions

        decisions = [self._decision()]
        validate_device_decisions(decisions, require_aten_on_npu=False)  # 仅告警不报错

    def test_epair_config_has_strict_switch(self):
        config = EpairConfig()
        assert config.require_aten_on_npu is False
        strict = EpairConfig(require_aten_on_npu=True)
        assert strict.require_aten_on_npu is True
