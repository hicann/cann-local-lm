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

from torch_delegate import export_to_epair, epair_save, epair_load
from torch_delegate.epair.schema import EpairContents, EpairManifest


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
    ep = torch.export.export(model, (x, y))
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
    ep = torch.export.export(model, (x, y))
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


def test_ascend_ir_source_field(epair_contents: EpairContents):
    """测试 source 字段：converter 或 空"""
    nodes = epair_contents.ep_json["graph_module"]["graph"]["nodes"]
    for node in nodes:
        if "ascend_ir" in node:
            ir = node["ascend_ir"]
            if "source" in ir:
                assert ir["source"] in ["converter", ""]


def test_ascend_ir_ops_is_list(epair_contents: EpairContents):
    """测试 ops 是 list[dict]"""
    nodes = epair_contents.ep_json["graph_module"]["graph"]["nodes"]
    for node in nodes:
        if "ascend_ir" in node:
            ir = node["ascend_ir"]
            assert isinstance(ir["ops"], list)
            for op in ir["ops"]:
                assert isinstance(op, dict)


def test_mock_node_source_empty(mixed_epair_contents: EpairContents):
    """测试 mock 节点 source 为空"""
    nodes = mixed_epair_contents.ep_json["graph_module"]["graph"]["nodes"]
    for node in nodes:
        if "ascend_ir" in node:
            ir = node["ascend_ir"]
            if "source" in ir and ir["source"] == "":
                assert "ops" in ir


def test_roundtrip_via_api():
    model = SupportedModel()
    x = torch.randn(2, 4)
    y = torch.randn(4, 2)
    ep = torch.export.export(model, (x, y))
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


def test_device_decisions_export_time_only():
    """DeviceDecision 判定记录仅在导出期持有，不随 epair 落盘。

    归属结论经 ascend_ir.device 固化进 epair；加载侧 C++ 按 device 与
    target namespace 复述归属，不再序列化逐节点判定记录（避免扩展
    extra/ 目录的格式演进负担）。
    """
    model = SupportedModel()
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    ep = torch.export.export(model, (x, y))
    contents = export_to_epair(ep)

    assert len(contents.decisions) > 0
    assert all(d.device in ("npu", "cpu") for d in contents.decisions)

    with tempfile.NamedTemporaryFile(suffix=".epair", delete=False) as f:
        path = f.name
    try:
        epair_save(contents, path)
        loaded = epair_load(path)
    finally:
        os.unlink(path)

    assert loaded.decisions == []
    # 归属结论仍可从 ep_json 的 ascend_ir.device 读取。
    devices = [
        node.get("ascend_ir", {}).get("device", "npu")
        for node in loaded.ep_json["graph_module"]["graph"]["nodes"]
        if "ascend_ir" in node
    ]
    assert devices, "fallback nodes must carry ascend_ir"
    assert all(d in ("npu", "cpu") for d in devices)
