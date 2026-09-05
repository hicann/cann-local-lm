#!/usr/bin/env python3
"""gemma4 模型定义单元测试 (从 llm_models_run_on_cpu/debug_gemma4/verify_gemma4.py 抽出).

pytest 风格: 顶层 test_* 函数, check() 失败即 raise (pytest 捕获).
用 converter venv (无 torch_npu) 跑: pytest tests/llm_models/test_gemma4.py
"""
import torch
import pytest

from cann_lm.models.gemma4.configuration_gemma4 import Gemma4TextConfig
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
)
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper


def check(name, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    assert condition, f"{name} failed"


def info(msg):
    print(f"  [INFO] {msg}")


def make_small_config() -> Gemma4TextConfig:
    config = Gemma4TextConfig(num_hidden_layers=4, hidden_size=128, intermediate_size=512,
                             num_attention_heads=4, num_key_value_heads=2,
                             head_dim=32, global_head_dim=32,
                             vocab_size=1000, num_kv_shared_layers=0)
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
    check("sliding window pattern (5:1)", config.layer_types.count("sliding_attention") == 25)
    check("rope_parameters has both types", "full_attention" in config.rope_parameters
          and "sliding_attention" in config.rope_parameters)
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
    check("RoPE full cos shape", cos.shape[-1] == rope.cos_cached_full_attention.shape[-1])
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

    gen2 = lm.generate(input_ids, max_new_tokens=4, do_sample=True, temperature=0.7, top_k=50)
    check("sampling generate", gen2.shape[1] > s)

    n_params = sum(p.numel() for p in lm.parameters())
    info(f"ForCausalLM parameters: {n_params / 1e6:.2f}M")


def test_kv_sharing():
    print("\n=== KV Sharing ===")
    config = Gemma4TextConfig(
        num_hidden_layers=6, hidden_size=128, intermediate_size=512,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, global_head_dim=32,
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
            num_hidden_layers=n_layers, hidden_size=128, intermediate_size=512,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, global_head_dim=32,
            vocab_size=1000,
        )
        config.__post_init__()

        lm = Gemma4ForCausalLM(config)
        input_ids = torch.randint(0, 1000, (1, 4))
        out = lm(input_ids=input_ids)
        check(f"truncated {n_layers} layers output", out["logits"].shape[-1] == 1000)
        info(f"  {n_layers} layers: {sum(p.numel() for p in lm.parameters()) / 1e6:.2f}M params")


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
        logits_wrap, *kv_flat = wrapper(input_ids, mask_full, mask_sliding, pos_ids, None,
                                        new_kv_pos, None, None)

    diff = (logits_wrap[:, -1:, :].float() - logits_orig[:, -1:, :].float()).abs().max().item()
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
        if not torch.allclose(k_wrap, k_orig, atol=1e-3) or not torch.allclose(v_wrap, v_orig, atol=1e-3):
            kv_match = False
    check(f"wrapper KV cache match (max_diff={max_kv_diff:.6e})", kv_match)
    info(f"  wrapper vs model: logits diff={diff:.6e}, KV diff={max_kv_diff:.6e}")
