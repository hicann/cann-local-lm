# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pytest
import torch
import torch_npu  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode

from examples.qwen3_5 import Qwen3_5Config
from examples.qwen3_5.export.adapter import create_cache
from examples.qwen3_5.model.modeling import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)


def test_recurrent_state_uses_native_layout_and_bf16():
    cfg = Qwen3_5Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention"],
        }
    ).text_config
    cache = create_cache(
        cfg, max_seq_len=5120, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert cache.layers[0].recurrent_state_0.shape == (1, 4, 16, 8)
    assert cache.layers[0].recurrent_state_0.is_contiguous()
    assert cache.layers[0].recurrent_state_0.dtype == torch.bfloat16
    assert cache.layers[0].conv_state_0.dtype == torch.bfloat16


class GDN(torch.nn.Module):
    def forward(self, q, k, v, g, beta):
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            max_seq_len=5120,
        )


def test_export_has_one_fused_op_and_preserves_dynamic_length():
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    g = torch.zeros(1, 65, 2)
    beta = g.to(torch.bfloat16)
    seq = torch.export.Dim("tokens", min=4, max=5119)
    ep = torch.export.export(
        GDN(),
        (q, q.clone(), q.clone(), g, beta),
        dynamic_shapes=tuple({1: seq} for _ in range(5)),
        strict=False,
    )
    calls = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    assert calls.count("npu.npu_chunk_gated_delta_rule.default") == 1
    assert not any("slice_scatter" in x or "constant_pad" in x for x in calls)
    with FakeTensorMode(allow_non_fake_inputs=True):
        for n in [4, 64, 65, 128, 512, 2048, 5119]:
            q = torch.zeros(1, n, 2, 128, dtype=torch.bfloat16)
            g = torch.zeros(1, n, 2)
            beta = g.to(torch.bfloat16)
            out, state = ep.module()(q, q.clone(), q.clone(), g, beta)
            assert out.shape == (1, n, 2, 128)
            assert state.shape == (1, 2, 128, 128)
            assert state.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_state_orientation_and_normalization_match_reference():
    torch.manual_seed(23)
    q, k, v = [torch.randn(1, 65, 2, 128, dtype=torch.bfloat16) * 0.1 for _ in range(3)]
    g = -torch.rand(1, 65, 2) * 0.05
    beta = torch.rand(1, 65, 2, dtype=torch.bfloat16)
    state = torch.randn(1, 2, 128, 128, dtype=torch.bfloat16) * 0.1
    kw = {"output_final_state": True, "use_qk_l2norm_in_kernel": True}
    from examples.qwen3_5.model.modeling import l2norm

    with torch.inference_mode():
        actual = chunk_gated_delta_rule(
            *(x.npu() for x in (q, k, v, g, beta)), initial_state=state.npu(), **kw
        )
        native_out, native_state = torch_npu.npu_chunk_gated_delta_rule(
            l2norm(q.npu())[0].contiguous(),
            l2norm(k.npu())[0].contiguous(),
            v.npu()[0].contiguous(),
            beta=beta.npu()[0].contiguous(),
            g=g.npu()[0].contiguous(),
            initial_state=state.npu().contiguous(),
            actual_seq_lengths=torch.tensor([65], device="npu", dtype=torch.int32),
            scale=128**-0.5,
        )
    assert torch.equal(actual[0], native_out.unsqueeze(0))
    assert torch.equal(actual[1], native_state)


@pytest.mark.parametrize(
    ("entrypoint", "operator", "seq_len"),
    [
        (chunk_gated_delta_rule, "npu_chunk_gated_delta_rule", 4),
        (recurrent_gated_delta_rule, "npu_recurrent_gated_delta_rule_functional", 1),
    ],
)
def test_gdn_passes_native_state_layout_through(
    monkeypatch, entrypoint, operator, seq_len
):
    # Equal K/V dimensions hide layout errors unless the values are asymmetric.
    state = (
        torch.arange(2 * 128 * 128, dtype=torch.float32)
        .reshape(1, 2, 128, 128)
        .to(torch.bfloat16)
    )
    native_state = state.flip(-1).contiguous()
    observed = []

    def native(query, key, value, *args, **kwargs):
        observed.append(
            kwargs["initial_state"] if "initial_state" in kwargs else args[0]
        )
        return torch.zeros_like(value), native_state

    monkeypatch.setattr(torch_npu, operator, native)
    q = torch.zeros(1, seq_len, 2, 128, dtype=torch.bfloat16)
    g = torch.zeros(1, seq_len, 2)
    _, result = entrypoint(
        q,
        q,
        q,
        g,
        g.to(torch.bfloat16),
        initial_state=state,
        output_final_state=True,
    )
    assert torch.equal(observed[0], state)
    assert torch.equal(result, native_state)


def test_converter_survives_builtin_custom_loading(tmp_path):
    from unittest.mock import patch

    from torchair._ge_concrete_graph.fx2ge_converter import _get_converter

    from examples.qwen3_5.export.pte import register_chunk_gdn_converter

    library = tmp_path / "registration.so"
    library.touch()
    with patch("ctypes.CDLL", return_value=object()):
        register_chunk_gdn_converter(library)
    _get_converter(torch.ops.npu.scatter_update_.default)
    converter = _get_converter(torch.ops.npu.npu_chunk_gated_delta_rule.default)
    assert converter.__module__ == "examples.qwen3_5.export.pte"


def test_exported_state_io_keeps_bf16_when_compute_is_float32():
    from examples.common.cache import _Qwen3_5LinearAttentionLayer
    from executorch_npu.artifacts.io_contract import build_io_contract

    class Update(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _Qwen3_5LinearAttentionLayer(
                torch.zeros(1, 8, 4, dtype=torch.bfloat16),
                torch.zeros(1, 2, 8, 8, dtype=torch.bfloat16),
                4,
            )

        def forward(self, state):
            return self.layer.update_recurrent_state(state)

    ep = torch.export.export(Update(), (torch.randn(1, 2, 8, 8),), strict=False)
    ep = ep.run_decompositions({})
    io, state = build_io_contract(ep)
    assert len(state.bindings) == 1
    assert state.bindings[0].dtype == "bfloat16"
    assert all(
        x.dtype == "bfloat16"
        for x in io.inputs + io.outputs
        if x.semantic_role != "data"
    )
