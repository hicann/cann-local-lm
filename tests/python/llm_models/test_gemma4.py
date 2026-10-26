#!/usr/bin/env python3
"""gemma4 模型定义单元测试 (从 llm_models_run_on_cpu/debug_gemma4/verify_gemma4.py 抽出).

pytest 风格: 顶层 test_* 函数, check() 失败即 raise (pytest 捕获).
用 converter venv (无 torch_npu) 跑: pytest tests/llm_models/test_gemma4.py
"""

import torch
import pytest

from cann_lm.models.gemma4.configuration_gemma4 import (
    Gemma4TextConfig,
    Gemma4Config,
    Gemma4VisionConfig,
)
from cann_lm.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextModel,
    Gemma4TextDecoderLayer,
    Gemma4TextAttention,
    Gemma4TextMLP,
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
    eager_attention_forward,
)
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper


def check(name, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    assert condition, f"{name} failed"


def info(msg):
    print(f"  [INFO] {msg}")


def make_small_config() -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        vocab_size=1000,
        num_kv_shared_layers=0,
    )
    config.__post_init__()
    return config


def make_wrapper_config() -> Gemma4TextConfig:
    """Small config for export-wrapper tests: tiny PLE + small sliding_window."""
    config = Gemma4TextConfig(
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        vocab_size=1000,
        num_kv_shared_layers=0,
        vocab_size_per_layer_input=1000,
        hidden_size_per_layer_input=16,
        sliding_window=4,
    )
    config.__post_init__()
    return config


@pytest.fixture
def config():
    return make_small_config()


def make_causal_mask(seq_len, dtype):
    """4D causal mask [1, 1, seq_len, seq_len] (cat 模式, 单元测试用)."""
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def test_config():
    print("\n=== Configuration ===")
    config = Gemma4TextConfig(num_hidden_layers=30)
    config.__post_init__()
    check("layer_types length", len(config.layer_types) == 30)
    check("last layer is full_attention", config.layer_types[-1] == "full_attention")
    check(
        "sliding window pattern (5:1)",
        config.layer_types.count("sliding_attention") == 25,
    )
    check(
        "rope_parameters has both types",
        "full_attention" in config.rope_parameters
        and "sliding_attention" in config.rope_parameters,
    )
    check("global_head_dim default", config.global_head_dim == 512)
    check("head_dim default", config.head_dim == 256)

    small = make_small_config()
    check("small config layers", len(small.layer_types) == 4)

    tiny = Gemma4TextConfig(num_hidden_layers=2)
    tiny.__post_init__()
    check("tiny 2 layers", tiny.layer_types == ["sliding_attention", "full_attention"])
    info(f"  tiny layer_types: {tiny.layer_types}")


def test_components(config: Gemma4TextConfig):
    print("\n=== Basic Components ===")
    b, s, d = 2, 8, config.hidden_size

    norm = Gemma4RMSNorm(d)
    x = torch.randn(b, s, d)
    out = norm(x)
    check("RMSNorm shape", out.shape == x.shape)
    check("RMSNorm not all zeros", out.abs().sum() > 0)

    rope = Gemma4TextRotaryEmbedding(config)
    pos_ids = torch.arange(0, s).unsqueeze(0)
    cos, sin = rope(x, pos_ids, "full_attention")
    check(
        "RoPE full cos shape", cos.shape[-1] == rope.cos_cached_full_attention.shape[-1]
    )
    cos_sl, sin_sl = rope(x, pos_ids, "sliding_attention")
    check("RoPE sliding cos shape", cos_sl.shape[-1] == config.head_dim // 2 * 2)

    n_heads = config.num_attention_heads
    hd = config.global_head_dim
    q = torch.randn(b, s, n_heads, hd)
    k = torch.randn(b, s, n_heads, hd)
    q_out, _ = apply_rotary_pos_emb(q, k, cos, sin, "full_attention")
    check("apply_rotary_pos_emb shape", q_out.shape == q.shape)

    kv = torch.randn(2, 2, 4, hd)
    expanded = repeat_kv(kv, n_rep=2)
    check("repeat_kv shape", expanded.shape == (2, 4, 4, hd))

    rh = rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    check("rotate_half", torch.equal(rh, torch.tensor([-3.0, -4.0, 1.0, 2.0])))


def test_attention(config: Gemma4TextConfig):
    print("\n=== Attention ===")
    b, s = 2, 8
    d = config.hidden_size

    for layer_idx in range(config.num_hidden_layers):
        layer_type = config.layer_types[layer_idx]
        attn = Gemma4TextAttention(config, layer_idx)
        x = torch.randn(b, s, d)

        rope = Gemma4TextRotaryEmbedding(config)
        pos_ids = torch.arange(0, s).unsqueeze(0)
        pos_emb = rope(x, pos_ids, layer_type)

        out = attn(x, position_embeddings=pos_emb)
        check(f"attn layer {layer_idx} ({layer_type}) shape", out.shape == x.shape)

        pv = [None] * config.num_hidden_layers
        _ = attn(x, position_embeddings=pos_emb, past_key_values=pv)
        check(f"attn layer {layer_idx} KV cache store", pv[layer_idx] is not None)

        x1 = torch.randn(b, 1, d)
        pos_ids_1 = torch.tensor([[s]])
        pos_emb_1 = rope(x1, pos_ids_1, layer_type)
        out1 = attn(x1, position_embeddings=pos_emb_1, past_key_values=pv)
        check(f"attn layer {layer_idx} cached shape", out1.shape == x1.shape)
        break


def test_mlp(config: Gemma4TextConfig):
    print("\n=== MLP ===")
    b, s = 2, 8
    d = config.hidden_size
    x = torch.randn(b, s, d)
    for layer_idx in range(min(2, config.num_hidden_layers)):
        mlp = Gemma4TextMLP(config, layer_idx)
        out = mlp(x)
        check(f"MLP layer {layer_idx} shape", out.shape == x.shape)


def test_decoder_layer(config: Gemma4TextConfig):
    print("\n=== DecoderLayer ===")
    b, s = 2, 8
    d = config.hidden_size
    x = torch.randn(b, s, d)

    rope = Gemma4TextRotaryEmbedding(config)
    pos_ids = torch.arange(0, s).unsqueeze(0)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = config.layer_types[layer_idx]
        pos_emb = rope(x, pos_ids, layer_type)
        layer = Gemma4TextDecoderLayer(config, layer_idx)
        out = layer(x, position_embeddings=pos_emb)
        check(f"decoder layer {layer_idx} ({layer_type}) shape", out.shape == x.shape)
        break


def test_text_model(config: Gemma4TextConfig):
    print("\n=== TextModel ===")
    b, s = 2, 8

    model = Gemma4TextModel(config)
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    hs, pv, skv = model(input_ids=input_ids)
    check("TextModel output shape", hs.shape == (b, s, config.hidden_size))
    check("past_key_values correct", len(pv) == config.num_hidden_layers)

    hs2, _, _ = model(input_ids=input_ids, past_key_values=None)
    check("TextModel no-cache output shape", hs2.shape == (b, s, config.hidden_size))

    mask = torch.ones(b, s)
    mask[0, -2:] = 0
    hs3, _, _ = model(input_ids=input_ids, attention_mask=mask)
    check("TextModel masked output shape", hs3.shape == (b, s, config.hidden_size))

    n_params = sum(p.numel() for p in model.parameters())
    info(f"TextModel parameters: {n_params / 1e6:.2f}M")


def test_causal_lm(config: Gemma4TextConfig):
    print("\n=== ForCausalLM ===")
    b, s = 2, 8

    lm = Gemma4ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    out = lm(input_ids=input_ids)
    check("ForCausalLM logits shape", out["logits"].shape == (b, s, config.vocab_size))
    check("no loss without labels", out["loss"] is None)

    labels = input_ids.clone()
    labels[0, 0] = -100
    out = lm(input_ids=input_ids, labels=labels)
    check("loss is computed", out["loss"] is not None)
    check("loss is scalar", out["loss"].ndim == 0)

    config2 = Gemma4TextConfig(**{**config.__dict__, "final_logit_softcapping": 30.0})
    config2.__post_init__()
    lm2 = Gemma4ForCausalLM(config2)
    out2 = lm2(input_ids=input_ids)
    check("softcapped logits shape", out2["logits"].shape == (b, s, config2.vocab_size))

    gen = lm.generate(input_ids, max_new_tokens=4)
    check("generate returns longer seq", gen.shape[1] > s)

    gen2 = lm.generate(
        input_ids, max_new_tokens=4, do_sample=True, temperature=0.7, top_k=50
    )
    check("sampling generate", gen2.shape[1] > s)

    n_params = sum(p.numel() for p in lm.parameters())
    info(f"ForCausalLM parameters: {n_params / 1e6:.2f}M")


def test_kv_sharing():
    print("\n=== KV Sharing ===")
    config = Gemma4TextConfig(
        num_hidden_layers=6,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        vocab_size=1000,
        num_kv_shared_layers=3,
    )
    config.__post_init__()

    b, s = 2, 8
    model = Gemma4TextModel(config)
    input_ids = torch.randint(0, 1000, (b, s))
    hs, pv, skv = model(input_ids=input_ids)

    check("KV shared model output shape", hs.shape == (b, s, config.hidden_size))
    check("shared_kv_states not empty", len(skv) > 0)

    for i in range(3, 6):
        layer = model.layers[i]
        attn = layer.self_attn
        check(f"layer {i} is_kv_shared", attn.is_kv_shared_layer)
        check(f"layer {i} no k_proj", not hasattr(attn, "k_proj"))


def test_truncation():
    print("\n=== Layer Truncation ===")
    for n_layers in [2, 4, 8]:
        config = Gemma4TextConfig(
            num_hidden_layers=n_layers,
            hidden_size=128,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            global_head_dim=32,
            vocab_size=1000,
        )
        config.__post_init__()

        lm = Gemma4ForCausalLM(config)
        input_ids = torch.randint(0, 1000, (1, 4))
        out = lm(input_ids=input_ids)
        check(f"truncated {n_layers} layers output", out["logits"].shape[-1] == 1000)
        info(
            f"  {n_layers} layers: {sum(p.numel() for p in lm.parameters()) / 1e6:.2f}M params"
        )


def test_wrapper_vs_model(config: Gemma4TextConfig):
    """验证 Gemma4ExportWrapper 单步 forward 与原始 model 一致 (max_diff=0.0)."""
    print("\n=== Wrapper vs Model (单步) ===")
    b, s = 1, 8
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    dtype = next(model.parameters()).dtype

    input_ids = torch.randint(0, config.vocab_size, (b, s))
    mask = make_causal_mask(s, dtype)
    pos_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out_orig = model(input_ids=input_ids, attention_mask=mask, shared_kv_states={})
        logits_orig = out_orig["logits"]
        kv_orig = out_orig["past_key_values"]
        new_kv_pos = torch.arange(s, dtype=torch.int64)
        mask_full = wrapper._build_mask_full(s, s, dtype, new_kv_pos)
        mask_sliding = wrapper._build_mask_sliding(s, s, dtype, new_kv_pos)
        logits_wrap, *kv_flat = wrapper(
            input_ids, mask_full, mask_sliding, pos_ids, None, new_kv_pos, None, None
        )

    diff = (
        (logits_wrap[:, -1:, :].float() - logits_orig[:, -1:, :].float())
        .abs()
        .max()
        .item()
    )
    match = torch.allclose(logits_wrap[:, -1:, :], logits_orig[:, -1:, :], atol=1e-3)
    check(f"wrapper logits match (max_diff={diff:.6e})", match)

    kv_pairs = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, len(kv_flat), 2)]
    kv_match = True
    max_kv_diff = 0.0
    for i in range(min(len(kv_orig), len(kv_pairs))):
        if kv_orig[i] is None:
            continue
        k_orig, v_orig = kv_orig[i]
        k_wrap, v_wrap = kv_pairs[i]
        if k_orig.shape != k_wrap.shape:
            kv_match = False
            continue
        k_diff = (k_wrap.float() - k_orig.float()).abs().max().item()
        v_diff = (v_wrap.float() - v_orig.float()).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)
        if not torch.allclose(k_wrap, k_orig, atol=1e-3) or not torch.allclose(
            v_wrap, v_orig, atol=1e-3
        ):
            kv_match = False
    check(f"wrapper KV cache match (max_diff={max_kv_diff:.6e})", kv_match)
    info(f"  wrapper vs model: logits diff={diff:.6e}, KV diff={max_kv_diff:.6e}")


# ── export wrapper helpers (pure logic: mask builders / sample inputs / names) ──


def test_wrapper_build_mask_sliding_true_branch():
    """kv_max > sliding_window -> real window-mask construction (lines 220-226)."""
    config = make_wrapper_config()  # sliding_window=4
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    dtype = next(model.parameters()).dtype
    seq, kv_max = 2, 8  # kv_max(8) > sliding_window(4) -> true branch
    new_pos = torch.arange(seq, dtype=torch.int64)

    mask = wrapper._build_mask_sliding(seq, kv_max, dtype, new_pos)

    assert mask.shape == (
        1,
        1,
        seq,
        kv_max,
    )  # 矩形 [b,1,seq_len,kv_max] (arch31 tiling)
    assert mask.dtype == torch.bool
    # first seq row (q_abs=0): attend only kv[0] (diff=0 <= w-1=3, diff>=0)
    row0 = mask[0, 0, 0]
    assert row0[0].item() is False  # attend (False=attend)
    assert row0[1].item() is True  # masked out


def test_wrapper_build_mask_sliding_degenerate_to_full():
    """kv_max <= sliding_window -> degenerates to _build_mask_full (causal)."""
    config = make_wrapper_config()  # sliding_window=4
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    dtype = next(model.parameters()).dtype
    seq, kv_max = 4, 4  # kv_max(4) <= sliding_window(4) -> degenerate
    new_pos = torch.arange(seq, dtype=torch.int64)

    full = wrapper._build_mask_full(seq, kv_max, dtype, new_pos)
    sliding = wrapper._build_mask_sliding(seq, kv_max, dtype, new_pos)

    assert torch.equal(sliding, full)


def test_wrapper_get_sample_inputs_input_ids_mode():
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    seq = 4

    inputs = wrapper.get_sample_inputs(seq_len=seq, use_embeddings=False)
    # (input_ids, mask_full, mask_sliding, pos_ids, past_kv, new_kv_pos, emb, out_pos, ple, e_scale, ple_scale)
    assert inputs[0] is not None and inputs[0].shape == (1, seq)
    assert inputs[1].shape == (1, 1, seq, 16)  # mask_full [b,1,seq_len,kv_max] (矩形)
    assert inputs[2].shape == (1, 1, seq, 16)  # mask_sliding
    assert inputs[5].shape == (seq,)  # new_kv_cache_pos
    assert len(inputs[4]) == config.num_hidden_layers  # past_kv list
    # per-layer head dim: full layers use global_head_dim(32), sliding use head_dim(32)
    hd = wrapper._get_layer_head_dim(0)
    assert inputs[4][0][0].shape[-1] == hd


def test_wrapper_get_sample_inputs_embeddings_mode():
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    seq = 4

    inputs = wrapper.get_sample_inputs(
        seq_len=seq, use_embeddings=True, quant_embeddings=False
    )
    # input_ids is None placeholder; embeddings provided; ple provided (PLE enabled)
    assert inputs[0] is None
    assert inputs[6] is not None and inputs[6].shape == (1, seq, config.hidden_size)
    assert inputs[8] is not None  # per_layer_token_identity


def test_wrapper_get_sample_inputs_quant_embeddings_mode():
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    wrapper.set_quant_embeddings(
        {
            "model.model.embed_tokens.weight": (
                torch.zeros(config.vocab_size, config.hidden_size, dtype=torch.int8),
                torch.ones(config.vocab_size, dtype=torch.float32),
            )
        }
    )
    seq = 4

    inputs = wrapper.get_sample_inputs(
        seq_len=seq, use_embeddings=True, quant_embeddings=True
    )
    assert inputs[0] is None
    assert inputs[6].dtype == torch.int8 and inputs[6].shape == (
        1,
        seq,
        config.hidden_size,
    )
    assert inputs[9].shape == (1, seq)  # embeddings_scale


def test_wrapper_onnx_helpers():
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()

    si = wrapper.make_sample_inputs(seq_len=4)
    # (input_ids, mask_full, mask_sliding, pos_ids, past_kv, new_kv_pos, None, None, None, None)
    assert si[0].shape == (1, 4)
    assert si[1].shape == (1, 1, 4, 16)  # mask_full [b,1,seq_len,kv_max] (矩形)
    assert si[2].shape == (1, 1, 4, 16)  # mask_sliding
    assert si[7] is None and si[8] is None and si[9] is None

    in_names, out_names = wrapper.make_input_output_names()
    n = config.num_hidden_layers
    assert in_names[0] == "input_ids"
    assert in_names[1] == "mask_full" and in_names[2] == "mask_sliding"
    assert len(in_names) == 5 + 2 * n  # 5 scalars + 2 per layer
    assert out_names[0] == "logits"
    assert len(out_names) == 1 + 2 * n


def test_wrapper_host_names_and_detach():
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()

    names = wrapper.host_side_weight_names()
    assert "model.model.embed_tokens.weight" in names
    assert "model.model.embed_tokens_per_layer.weight" in names

    tm = model.model
    assert hasattr(tm, "embed_tokens") and hasattr(tm, "embed_tokens_per_layer")
    wrapper.detach_externalized_lookups()
    assert not hasattr(tm, "embed_tokens")
    assert not hasattr(tm, "embed_tokens_per_layer")


def test_wrapper_forward_with_output_pos_and_embeddings():
    """forward output_pos index_select + embeddings dequant (lines 66-67, 86)."""
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config).eval()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    dtype = next(model.parameters()).dtype
    b, s = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    new_kv_pos = torch.arange(s, dtype=torch.int64)
    mask_full = wrapper._build_mask_full(s, s, dtype, new_kv_pos)
    mask_sliding = wrapper._build_mask_sliding(s, s, dtype, new_kv_pos)
    pos_ids = torch.arange(s, dtype=torch.long).unsqueeze(0)
    output_pos = torch.tensor([s - 1], dtype=torch.int32)
    embeddings = model.model.embed_tokens(input_ids).detach()
    embeddings_scale = torch.ones(b, s, dtype=dtype)

    with torch.no_grad():
        logits, *_ = wrapper(
            input_ids,
            mask_full,
            mask_sliding,
            pos_ids,
            None,
            new_kv_pos,
            embeddings,
            output_pos,
            None,
            embeddings_scale,
            None,
        )
    assert logits.shape == (b, 1, config.vocab_size)


# ── export: torch.export 可导出 + FX 图无 masked_fill (NPU 兼容) ──


def _count_masked_fill(ep) -> int:
    """Count call_function nodes whose target contains 'masked_fill'."""
    return sum(
        1
        for n in ep.graph.nodes
        if n.op == "call_function" and "masked_fill" in str(n.target)
    )


def test_wrapper_export_no_masked_fill():
    """torch.export 能 trace wrapper + FX 图无 masked_fill 节点 (NPU 不支持 ge.MaskedFill).

    bypass 重写核心目标: mask 改 host 预构 (图输入), 图内不再构造 mask (无 masked_fill)。
    两种模式都测: use_embeddings=False (图内查表) / True (host 查表 + detach 外置查表子模块,
    模拟 saver)。从 debug_gemma4/verify_gemma4_wrapper_export.py 抽出。
    """
    config = make_wrapper_config()
    model = Gemma4ForCausalLM(config).eval()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()

    # 1. 图内查表模式 (input_ids + PLE 内查)
    si = wrapper.get_sample_inputs(seq_len=4, use_embeddings=False)
    ep = torch.export.export(wrapper, si, strict=False)
    n_mf = _count_masked_fill(ep)
    check(
        f"[use_embeddings=False] torch.export OK; masked_fill={n_mf} (期望 0)",
        n_mf == 0,
    )

    # 2. use_embeddings=True (host 查表 + detach 外置 embed_tokens/embed_tokens_per_layer)
    si2 = wrapper.get_sample_inputs(seq_len=4, use_embeddings=True)
    wrapper.detach_externalized_lookups()
    ep2 = torch.export.export(wrapper, si2, strict=False)
    n_mf2 = _count_masked_fill(ep2)
    check(
        f"[use_embeddings=True] torch.export OK (detached); masked_fill={n_mf2} (期望 0)",
        n_mf2 == 0,
    )


# ── bypass decode: wrapper 单 token decode == model.forward (maxdiff=0) ──


def _cob_mask(q_positions, kv_len, dtype):
    """caller 4D causal-over-buffer [1,1,q,kv_len] (k<=q → 0, else MIN)."""
    min_val = torch.finfo(dtype).min
    q = torch.tensor(q_positions, dtype=dtype).unsqueeze(1)
    k = torch.arange(kv_len, dtype=dtype).unsqueeze(0)
    return (
        torch.where(k <= q, torch.tensor(0.0, dtype=dtype), min_val)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def test_wrapper_vs_model_decode():
    """wrapper 单 token decode (q@pos S) == model.forward scatter-write (logits+KV maxdiff=0).

    扩展 test_wrapper_vs_model (仅 prefill) 到 decode 步: 验证 scatter-write decode 路径
    (host 预构 mask + 跳过 pad 槽位 new_kv_cache_pos) 与 model.forward (4D cob +
    内部 _prepare_causal_mask) 数值逐位一致 (maxdiff=0, KV torch.equal)。
    从 debug_gemma4/verify_gemma4_wrapper_bypass.py 抽出 (mask 逐位相等那条已失效:
    host 改 BOOL 后 torch.equal 假阴性, 但被 logits maxdiff=0 逻辑蕴含, 故不移植)。
    """
    config = make_wrapper_config()  # sliding_window=4, PLE on, 4 层
    model = Gemma4ForCausalLM(config).eval()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=8, batch=1).eval()
    dtype = torch.float32
    b, s, kv_max = 1, 4, 8
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    next_tok = torch.randint(0, config.vocab_size, (b, 1))
    nkv = config.num_key_value_heads

    def zero_bufs():
        return [
            (
                torch.zeros(
                    b, nkv, kv_max, wrapper._get_layer_head_dim(i), dtype=dtype
                ),
                torch.zeros(
                    b, nkv, kv_max, wrapper._get_layer_head_dim(i), dtype=dtype
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

    # --- prefill ---
    nkp_pf = torch.arange(s, dtype=torch.int64)
    cob_pf = _cob_mask(list(range(s)), kv_max, dtype)
    pos_pf = torch.arange(s, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out_old = model(
            input_ids=input_ids,
            attention_mask=cob_pf,
            position_ids=pos_pf,
            past_key_values=zero_bufs(),
            new_kv_cache_pos=nkp_pf,
            shared_kv_states=None,
        )
        old_kv = out_old["past_key_values"]
        h_full_pf = wrapper._build_mask_full(s, kv_max, dtype, nkp_pf)
        h_slide_pf = wrapper._build_mask_sliding(s, kv_max, dtype, nkp_pf)
        _, *new_flat0 = wrapper(
            input_ids, h_full_pf, h_slide_pf, pos_pf, zero_bufs(), nkp_pf, None, None
        )

    # --- decode (q@pos s, sliding window=[s-w+1..s]) ---
    nkp_dec = torch.tensor([s], dtype=torch.int64)
    cob_dec = _cob_mask([s], kv_max, dtype)
    pos_dec = torch.tensor([[s]], dtype=torch.long)
    kv_new = [(new_flat0[i], new_flat0[i + 1]) for i in range(0, len(new_flat0), 2)]
    with torch.no_grad():
        out_old_d = model(
            input_ids=next_tok,
            attention_mask=cob_dec,
            position_ids=pos_dec,
            past_key_values=old_kv,
            new_kv_cache_pos=nkp_dec,
            shared_kv_states=None,
        )
        old_l1, old_kv_d = out_old_d["logits"], out_old_d["past_key_values"]
        h_full_d = wrapper._build_mask_full(1, kv_max, dtype, nkp_dec)
        h_slide_d = wrapper._build_mask_sliding(1, kv_max, dtype, nkp_dec)
        new_l1, *new_flat1 = wrapper(
            next_tok, h_full_d, h_slide_d, pos_dec, kv_new, nkp_dec, None, None
        )

    diff = (new_l1.float() - old_l1.float()).abs().max().item()
    check(f"decode logits max_diff={diff:.2e} (期望 0)", diff == 0.0)
    for i in range(len(old_kv_d)):
        if old_kv_d[i] is None:
            continue
        k_eq = torch.equal(old_kv_d[i][0], new_flat1[i * 2])
        v_eq = torch.equal(old_kv_d[i][1], new_flat1[i * 2 + 1])
        check(f"decode layer{i} KV equal (k={k_eq}, v={v_eq})", k_eq and v_eq)


# ── modeling edge branches (eager_attention_forward softcap/scaling; KV scatter; mask) ──


class _AttnStub:
    """Minimal module for eager_attention_forward (head_dim/num_key_value_groups/training)."""

    head_dim = 32
    num_key_value_groups = 2
    training = False


def test_eager_attention_forward_scaling_none_and_softcap():
    """scaling=None -> head_dim**-0.5 (190-191); softcap tanh path (196-201)."""
    b, h, s, d = 1, 4, 4, 32
    q = torch.randn(b, h, s, d, dtype=torch.float32)
    k = torch.randn(b, 2, s, d, dtype=torch.float32)  # 2 kv heads -> repeat 2x
    v = torch.randn(b, 2, s, d, dtype=torch.float32)
    stub = _AttnStub()

    # scaling=None -> uses head_dim ** -0.5
    out_none, _ = eager_attention_forward(stub, q, k, v, None, scaling=None)
    assert out_none.shape == (b, s, h, d)
    assert torch.isfinite(out_none).all()

    # softcap path: tanh scaling
    out_sc, aw = eager_attention_forward(
        stub, q, k, v, None, scaling=0.25, softcap=30.0
    )
    assert out_sc.shape == (b, s, h, d)
    assert torch.isfinite(out_sc).all()
    # attn_weights bounded by softcap (tanh range)
    assert aw.abs().max().item() <= 30.0 + 1e-3


def test_text_model_scatter_write_cache(config: Gemma4TextConfig):
    """new_kv_cache_pos + pre-allocated buffer -> scatter-write branch (558-571).

    Uses the export wrapper (pre-builds [b,1,kv_max,kv_max] bool masks compatible with
    the fixed kv_max buffer); eager TextModel.forward computes past_len from cache.shape[2]
    which mismatches a pre-allocated buffer, so the wrapper path is the right harness.
    """
    model = Gemma4ForCausalLM(config).eval()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    b, s = 1, 4
    # get pre-allocated KV buffers (right per-layer head_dim) + host-built masks
    inputs = wrapper.get_sample_inputs(seq_len=s, use_embeddings=False)
    input_ids, mask_full, mask_sliding, pos_ids, past_kv, new_kv_pos, *_ = inputs

    with torch.no_grad():
        logits, *kv_flat = wrapper(
            input_ids,
            mask_full,
            mask_sliding,
            pos_ids,
            past_kv,
            new_kv_pos,
            None,
            None,
        )
    check("scatter-write logits shape", logits.shape == (b, s, config.vocab_size))
    # scatter-write keeps kv_max fixed (buffer not grown via torch.cat)
    check("kv buffer stays kv_max", kv_flat[0].shape[2] == 16)


def test_prepare_causal_mask_4d_sliding_window():
    """4D mask + sliding_attention + kv_len > sliding_window -> window restriction (797-805)."""
    config = make_wrapper_config()  # sliding_window=4
    model = Gemma4TextModel(config)
    b, s, d = 1, 4, config.hidden_size
    inputs_embeds = torch.randn(b, s, d, dtype=torch.float32)
    kv_len = 8  # > sliding_window(4)
    min_val = torch.finfo(torch.float32).min
    mask4d = torch.zeros(b, 1, s, kv_len, dtype=torch.float32)
    new_pos = torch.arange(s, dtype=torch.int64)

    out = model._prepare_causal_mask(
        inputs_embeds,
        mask4d,
        "sliding_attention",
        new_kv_cache_pos=new_pos,
        past_len=0,
    )
    assert out.shape == (b, 1, s, kv_len)
    # window restriction masks out far kv positions -> some -inf present
    assert (out == min_val).any()


def test_prepare_causal_mask_2d_padding():
    """2D attention_mask -> bool key_valid + causal (gemma4-style, 765-785)."""
    config = make_small_config()
    model = Gemma4TextModel(config)
    b, s, d = 1, 6, config.hidden_size
    inputs_embeds = torch.randn(b, s, d, dtype=torch.float32)
    mask2d = torch.ones(b, s)
    mask2d[0, -2:] = 0  # last 2 tokens padded
    min_val = torch.finfo(torch.float32).min

    out = model._prepare_causal_mask(
        inputs_embeds, mask2d, "full_attention", past_len=0
    )
    assert out.shape == (b, 1, s, s)
    # padded key positions get -inf
    assert (out[0, 0, :, -2] == min_val).all()


def test_config_to_dict_and_gemma4_config():
    """to_dict (109) + Gemma4Config __post_init__ dict/None (171-184) + to_dict (187)."""
    config = make_small_config()
    d = config.to_dict()
    assert d["model_type"] == "gemma4_text"
    assert not any(k.startswith("_") for k in d)

    # Gemma4Config: all None -> defaults created
    gc = Gemma4Config()
    assert isinstance(gc.text_config, Gemma4TextConfig)
    assert isinstance(gc.vision_config, Gemma4VisionConfig)

    # dict text_config -> converted to dataclass
    gc2 = Gemma4Config(text_config={"vocab_size": 100})
    assert isinstance(gc2.text_config, Gemma4TextConfig)
    assert gc2.text_config.vocab_size == 100

    gd = gc.to_dict()
    assert gd["model_type"] == "gemma4"
