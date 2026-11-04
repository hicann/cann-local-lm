# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pytest
import torch

from examples.auto import AutoModelForCausalLM
from examples.gemma4.model.configuration import Gemma4TextConfig
from examples.gemma4.model.modeling import fused_rope, Gemma4ForCausalLM, Gemma4RMSNorm


def config():
    return Gemma4TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 2,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
        sliding_window=512,
        max_position_embeddings=640,
        enable_moe_block=False,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )


def test_strict_checkpoint_loading_initializes_unscaled_norm_buffers(tmp_path):
    model = Gemma4ForCausalLM(config()).to(dtype=torch.bfloat16)
    model.save_pretrained(tmp_path)
    loaded, info = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        strict=True,
        output_loading_info=True,
    )
    assert not info.get("missing_keys") and not info.get("unexpected_keys")
    norms = [
        m for m in loaded.modules() if isinstance(m, Gemma4RMSNorm) and not m.with_scale
    ]
    assert norms
    for norm in norms:
        assert norm.ones.device.type != "meta"
        assert torch.equal(norm.ones, torch.ones_like(norm.ones))


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_partial_rope_preserves_paired_dimensions():
    model = Gemma4ForCausalLM(config()).to(device="npu", dtype=torch.bfloat16).eval()
    rope = model.model.rotary_emb
    rope.prepare_cache(640, torch.device("npu"), torch.bfloat16)
    positions = torch.tensor([[0, 511, 512, 513, 639]], device="npu")
    cos, sin = rope(next(model.parameters()), positions, "full_attention")
    x = torch.randn(1, 5, 2, 512, device="npu", dtype=torch.bfloat16)
    actual = fused_rope(x, cos, sin, 128)
    rotate_half = torch.cat((-x[..., 256:], x[..., :256]), dim=-1).float()
    expected = (
        x.float() * cos.unsqueeze(2).float() + rotate_half * sin.unsqueeze(2).float()
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.99999
    assert torch.equal(actual[..., 64:256], x[..., 64:256])
    assert torch.equal(actual[..., 320:], x[..., 320:])
