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

from examples.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
from examples.qwen3_5.export.adapter import (
    build_causal_mask,
    create_cache,
    ExportWrapper,
)
from examples.qwen3_5.model.modeling import _merge_linears
from executorch_npu.artifacts.io_contract import build_io_contract


def config():
    cfg = Qwen3_5Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        )
    ).text_config
    cfg._attn_implementation = "eager"
    return cfg


def test_projection_merge_preserves_order_and_bias():
    torch.manual_seed(2)
    linears = [torch.nn.Linear(7, n, bias=True, dtype=torch.float64) for n in [5, 3, 9]]
    x = torch.randn(2, 4, 7, dtype=torch.float64)
    expected = torch.cat([m(x) for m in linears], -1)
    torch.testing.assert_close(
        _merge_linears(linears)(x), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_fused_model_checkpoint_and_dynamic_export(tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = (
        Qwen3_5ForCausalLM(config())
        .to(dtype=torch.bfloat16, device="npu")
        .eval()
        .requires_grad_(False)
    )
    # Native checkpoint names are accepted before one-time projection merging.
    checkpoint = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    clone = (
        Qwen3_5ForCausalLM(config())
        .to(dtype=torch.bfloat16, device="npu")
        .eval()
        .requires_grad_(False)
    )
    clone.load_state_dict(checkpoint, strict=True)
    outputs = []
    for m in (model, clone):
        cache = create_cache(
            m.config, max_seq_len=128, device=torch.device("npu"), dtype=torch.bfloat16
        )
        wrapper = ExportWrapper(m, cache)
        values = []
        with torch.inference_mode():
            for n, start in ((8, 0), (1, 8), (1, 9)):
                ids = torch.arange(n, device="npu").reshape(1, -1) + 3
                pos = torch.arange(start, start + n, device="npu")
                y = (
                    wrapper(
                        ids,
                        pos,
                        build_causal_mask(
                            n, 128, device=torch.device("npu"), dtype=torch.bfloat16
                        ),
                    )
                    if not start
                    else wrapper(ids, pos)
                )
                values.append(y.clone())
        outputs.append(values)
    for a, b in zip(*outputs):
        assert torch.equal(a, b)
    assert not hasattr(model.model.layers[0].linear_attn, "in_proj_qkv")
    assert not hasattr(model.model.layers[1].self_attn, "q_proj")
    from examples.qwen3_5.export.pt2 import export_split

    export_split(
        model,
        prefill_path=tmp_path / "prefill.pt2",
        decode_path=tmp_path / "decode.pt2",
        trace_len=8,
        max_seq_len=128,
        device=torch.device("npu"),
        dynamic_prefill=True,
    )
    for method in ("prefill", "decode"):
        ep = torch.export.load(tmp_path / f"{method}.pt2")
        build_io_contract(ep.run_decompositions({}))
        calls = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        for target in (
            "npu_fused_infer_attention_score_v2",
            "npu_add_rms_norm",
            "npu_rms_norm",
            "npu_swiglu",
        ):
            assert any(target in x for x in calls)
        if method == "decode":
            assert any("npu_recurrent_gated_delta_rule_functional" in x for x in calls)
