# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch
from executorch.exir._serialize._named_data_store import NamedDataStore
from test_backend import (
    _fake_torch_delegate,
    _FakeOfflineCompiler,
    _mock_ge_initialization,
    Add,
)

from executorch_npu.compiler import delegate_adapter, pipeline
from executorch_npu.compiler.compile_spec import AscendCompileOptions


def test_multimethod_drops_consumed_files_but_keeps_packaged_bytes(
    monkeypatch, tmp_path
):
    _mock_ge_initialization(monkeypatch)
    monkeypatch.setattr(delegate_adapter, "_load_torch_delegate", _fake_torch_delegate)
    compiled = []

    class Compiler(_FakeOfflineCompiler):
        def compile(self, graph, output, **kwargs):
            # Input archive is consumed before GE starts writing large weights.
            assert not output.parent.joinpath("model.epair").exists()
            # A previous method must not contribute to this method's disk peak.
            for previous in compiled:
                assert not previous.with_suffix(".om").exists()
                assert not previous.parent.joinpath("weight").exists()
            super().compile(graph, output, **kwargs)
            compiled.append(output)

    monkeypatch.setattr(pipeline, "offline_compiler", Compiler())
    store = NamedDataStore()
    for method in ("prefill", "decode"):
        ep = torch.export.export(Add(), (torch.ones(1, 4), torch.ones(1, 4)))
        specs = AscendCompileOptions(
            method_name=method, delegate_index=0, acl_soc_name="Ascend910_9362"
        )
        manifest = pipeline.compile_partition(ep, specs, store, tmp_path)
        assert manifest.method == method
    assert not list(tmp_path.rglob("model.epair"))
    assert not list(tmp_path.rglob("*.om"))
    assert not list(tmp_path.rglob("weight_*"))
    payloads = store.get_named_data_store_output().buffers
    assert b"fake-om" in payloads
    assert b"fake-weight" in payloads
