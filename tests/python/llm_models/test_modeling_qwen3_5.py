from __future__ import annotations

import sys
import json
import torch

from cann_local_lm.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from cann_local_lm.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    apply_mask_to_padding_states,
    l2norm,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
    SimpleKVCache,
    SimpleDeltaNetCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Attention,
    Qwen3_5MLP,
    Qwen3_5DecoderLayer,
    Qwen3_5TextModel,
    Qwen3_5ForCausalLM,
    scatter_kv_to_cache,
    _create_causal_mask,
)
from cann_local_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper


def make_small_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        full_attention_interval=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 2, 2],
        },
    )


passed = 0
failed = 0


def report(name: str, ok: bool):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


def test_rms_norm():
    print("=== Qwen3_5RMSNorm ===")
    dim = 32
    norm = Qwen3_5RMSNorm(dim)
    x = torch.randn(2, 8, dim)
    out = norm(x)
    report("output_shape", out.shape == (2, 8, dim))
    report("output_dtype_preserved", out.dtype == x.dtype)
    report("not_nan", not torch.isnan(out).any())
    x2 = torch.randn(2, 8, dim) * 0 + 5.0
    out2 = norm(x2)
    report("constant_input_normalized", out2.std().item() < 0.1)


def test_rms_norm_gated():
    print("=== Qwen3_5RMSNormGated ===")
    hidden_size = 32
    norm = Qwen3_5RMSNormGated(hidden_size)
    x = torch.randn(2, 8, hidden_size)
    gate = torch.randn(2, 8, hidden_size)
    out = norm(x, gate)
    report("output_shape", out.shape == (2, 8, hidden_size))
    report("not_nan", not torch.isnan(out).any())


def test_rotary_embedding():
    print("=== Qwen3_5TextRotaryEmbedding ===")
    config = make_small_config()
    rope = Qwen3_5TextRotaryEmbedding(config)
    x = torch.randn(2, 4, 8, config.head_dim)
    position_ids = torch.arange(0, 8).unsqueeze(0).expand(2, -1)
    position_ids_3d = position_ids[None, ...].expand(3, 2, 8)
    cos, sin = rope(x, position_ids_3d)
    rotary_dim = int(
        config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)
    )
    report("cos_shape", cos.shape[-1] == rotary_dim)
    report("sin_shape", sin.shape[-1] == rotary_dim)
    report("cos_not_nan", not torch.isnan(cos).any())
    report("sin_not_nan", not torch.isnan(sin).any())


def test_rotate_half():
    print("=== rotate_half ===")
    x = torch.randn(2, 4, 8, 32)
    out = rotate_half(x)
    report("output_shape", out.shape == x.shape)
    report("not_nan", not torch.isnan(out).any())


def test_apply_rotary_pos_emb():
    print("=== apply_rotary_pos_emb ===")
    config = make_small_config()
    q = torch.randn(2, 4, 8, config.head_dim)
    k = torch.randn(2, 2, 8, config.head_dim)
    # cos/sin 3D [batch, seq, rotary_dim] (apply_rotary_pos_emb 内部 unsqueeze 到 4D)
    rotary_dim = int(
        config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)
    )
    cos = torch.randn(1, 8, rotary_dim)
    sin = torch.randn(1, 8, rotary_dim)
    q_emb, k_emb = apply_rotary_pos_emb(q, k, cos, sin)
    report("q_shape", q_emb.shape == q.shape)
    report("k_shape", k_emb.shape == k.shape)
    report("not_nan", not torch.isnan(q_emb).any() and not torch.isnan(k_emb).any())


def test_repeat_kv():
    print("=== repeat_kv ===")
    x = torch.randn(2, 2, 8, 32)
    n_rep = 2
    out = repeat_kv(x, n_rep)
    report("output_shape", out.shape == (2, 4, 8, 32))
    out_1 = repeat_kv(x, 1)
    report("n_rep_1_same", out_1.shape == x.shape)


def test_eager_attention_forward():
    print("=== eager_attention_forward ===")
    config = make_small_config()
    attn = Qwen3_5Attention(config, layer_idx=0)
    attn.num_key_value_groups = 2
    q = torch.randn(2, 4, 8, config.head_dim)
    k = torch.randn(2, 2, 8, config.head_dim)
    v = torch.randn(2, 2, 8, config.head_dim)
    out, weights = eager_attention_forward(
        attn, q, k, v, None, scaling=config.head_dim**-0.5
    )
    report("output_shape", out.shape == (2, 8, 4, config.head_dim))
    report("weights_shape", weights.shape == (2, 4, 8, 8))
    report("not_nan", not torch.isnan(out).any())


def test_apply_mask_to_padding_states():
    print("=== apply_mask_to_padding_states ===")
    x = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8)
    out = apply_mask_to_padding_states(x, mask)
    report("no_mask_no_change", torch.allclose(out, x))
    mask_zero = torch.zeros(2, 8)
    out2 = apply_mask_to_padding_states(x, mask_zero)
    report("zero_mask_zeros", (out2 == 0).all())


def test_l2norm():
    print("=== l2norm ===")
    x = torch.randn(2, 8, 32)
    out = l2norm(x)
    report("output_shape", out.shape == x.shape)
    norms = (out * out).sum(dim=-1)
    report("unit_norm", torch.allclose(norms, torch.ones_like(norms), atol=1e-4))


def test_causal_conv1d_update():
    print("=== torch_causal_conv1d_update ===")
    batch, dim, seq = 2, 16, 4
    kernel = 4
    hidden = torch.randn(batch, dim, seq)
    conv_state = torch.zeros(batch, dim, kernel)
    weight = torch.randn(dim, 1, kernel)
    bias = torch.randn(dim)
    out, new_conv_state = torch_causal_conv1d_update(hidden, conv_state, weight, bias)
    report("output_shape", out.shape == (batch, dim, seq))
    report("not_nan", not torch.isnan(out).any())
    report("conv_state_updated", new_conv_state.shape == (batch, dim, kernel))


def test_chunk_gated_delta_rule():
    print("=== torch_chunk_gated_delta_rule ===")
    batch, heads, seq, k_dim, v_dim = 2, 4, 64, 16, 16
    query = torch.randn(batch, seq, heads, k_dim)
    key = torch.randn(batch, seq, heads, k_dim)
    value = torch.randn(batch, seq, heads, v_dim)
    g = torch.randn(batch, seq, heads)
    beta = torch.randn(batch, seq, heads)
    out, final_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=64,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    report("output_shape", out.shape == (batch, seq, heads, v_dim))
    report("final_state_shape", final_state.shape == (batch, heads, k_dim, v_dim))
    report("not_nan", not torch.isnan(out).any())


def test_recurrent_gated_delta_rule():
    print("=== torch_recurrent_gated_delta_rule ===")
    batch, heads, seq, k_dim, v_dim = 2, 4, 8, 16, 16
    query = torch.randn(batch, seq, heads, k_dim)
    key = torch.randn(batch, seq, heads, k_dim)
    value = torch.randn(batch, seq, heads, v_dim)
    g = torch.randn(batch, seq, heads)
    beta = torch.randn(batch, seq, heads)
    out, final_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    report("output_shape", out.shape == (batch, seq, heads, v_dim))
    report("final_state_shape", final_state.shape == (batch, heads, k_dim, v_dim))
    report("not_nan", not torch.isnan(out).any())


def test_simple_kv_cache():
    print("=== SimpleKVCache ===")
    cache = SimpleKVCache()
    k = torch.randn(2, 2, 8, 32)
    v = torch.randn(2, 2, 8, 32)
    k_out, v_out = cache.update(k, v, 0)
    report("first_update_same", torch.equal(k_out, k))
    report("seq_length_8", cache.get_seq_length() == 8)
    k2 = torch.randn(2, 2, 4, 32)
    v2 = torch.randn(2, 2, 4, 32)
    k_out2, v_out2 = cache.update(k2, v2, 0)
    report("second_update_concatenated", k_out2.shape[2] == 12)


def test_simple_delta_net_cache():
    print("=== SimpleDeltaNetCache ===")
    cache = SimpleDeltaNetCache(
        num_layers=4,
        conv_dim=64,
        conv_kernel_size=4,
        k_head_dim=16,
        v_head_dim=16,
        num_v_heads=4,
        batch_size=2,
        dtype=torch.float32,
    )
    report("initial_no_state", not cache.has_previous_state(0))
    new_conv = torch.randn(2, 64, 4)
    cache.update_conv_state(new_conv, 0)
    report("has_state_after_update", cache.has_previous_state(0))
    report("conv_state_updated", torch.equal(cache.layers[0]["conv_states"], new_conv))


def test_gated_delta_net():
    print("=== Qwen3_5GatedDeltaNet ===")
    config = make_small_config()
    gdn = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    x = torch.randn(2, 8, config.hidden_size)
    out, _, _ = gdn(x)
    report("output_shape", out.shape == (2, 8, config.hidden_size))
    report("not_nan", not torch.isnan(out).any())
    delta_cache = SimpleDeltaNetCache(
        num_layers=config.num_hidden_layers,
        conv_dim=gdn.conv_dim,
        conv_kernel_size=gdn.conv_kernel_size,
        k_head_dim=gdn.head_k_dim,
        v_head_dim=gdn.head_v_dim,
        num_v_heads=gdn.num_v_heads,
        batch_size=2,
        dtype=torch.float32,
    )
    out_cached, _, _ = gdn(x, cache_params=delta_cache)
    report("output_with_cache", out_cached.shape == (2, 8, config.hidden_size))


def test_attention():
    print("=== Qwen3_5Attention ===")
    config = make_small_config()
    attn = Qwen3_5Attention(config, layer_idx=1)
    x = torch.randn(2, 8, config.hidden_size)
    rope = Qwen3_5TextRotaryEmbedding(config)
    position_ids = torch.arange(0, 8).unsqueeze(0).expand(2, -1)
    position_ids_3d = position_ids[None, ...].expand(3, 2, 8)
    cos, sin = rope(x, position_ids_3d)
    out, weights = attn(x, position_embeddings=(cos, sin))
    report("output_shape", out.shape == (2, 8, config.hidden_size))
    report("not_nan", not torch.isnan(out).any())


def test_mlp():
    print("=== Qwen3_5MLP ===")
    config = make_small_config()
    mlp = Qwen3_5MLP(config)
    x = torch.randn(2, 8, config.hidden_size)
    out = mlp(x)
    report("output_shape", out.shape == (2, 8, config.hidden_size))
    report("not_nan", not torch.isnan(out).any())


def test_decoder_layer():
    print("=== Qwen3_5DecoderLayer ===")
    config = make_small_config()
    rope = Qwen3_5TextRotaryEmbedding(config)

    linear_layer = Qwen3_5DecoderLayer(config, layer_idx=0)
    x = torch.randn(2, 8, config.hidden_size)
    position_ids = torch.arange(0, 8).unsqueeze(0).expand(2, -1)
    position_ids_3d = position_ids[None, ...].expand(3, 2, 8)
    cos, sin = rope(x, position_ids_3d)
    out_linear, _, _ = linear_layer(x, position_embeddings=(cos, sin))
    report("linear_layer_output_shape", out_linear.shape == (2, 8, config.hidden_size))
    report("linear_layer_not_nan", not torch.isnan(out_linear).any())

    full_layer = Qwen3_5DecoderLayer(config, layer_idx=1)
    out_full, _, _ = full_layer(x, position_embeddings=(cos, sin))
    report(
        "full_attention_layer_output_shape",
        out_full.shape == (2, 8, config.hidden_size),
    )
    report("full_attention_layer_not_nan", not torch.isnan(out_full).any())


def test_text_model():
    print("=== Qwen3_5TextModel ===")
    config = make_small_config()
    model = Qwen3_5TextModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    hidden, kv_cache, delta_cache = model(input_ids=input_ids, use_cache=True)
    report("hidden_shape", hidden.shape == (2, 8, config.hidden_size))
    report("not_nan", not torch.isnan(hidden).any())
    report("kv_cache_created", kv_cache is not None)
    report("delta_cache_created", delta_cache is not None)


def test_causal_lm():
    print("=== Qwen3_5ForCausalLM ===")
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids=input_ids)
    report("logits_shape", out["logits"].shape == (2, 8, config.vocab_size))
    report("not_nan", not torch.isnan(out["logits"]).any())
    report("loss_none_by_default", out["loss"] is None)

    labels = input_ids.clone()
    out_with_loss = model(input_ids=input_ids, labels=labels)
    report(
        "loss_with_labels",
        out_with_loss["loss"] is not None and not torch.isnan(out_with_loss["loss"]),
    )
    report("loss_is_scalar", out_with_loss["loss"].dim() == 0)


def test_generate():
    print("=== Qwen3_5ForCausalLM.generate ===")
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    generated = model.generate(input_ids, max_new_tokens=4, do_sample=False)
    report("generated_length", generated.shape[1] == 8)
    report("generated_not_nan", not torch.isnan(generated).any())


def test_generate_with_eos():
    print("=== Qwen3_5ForCausalLM.generate with eos ===")
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    eos_id = 0
    generated = model.generate(
        input_ids, max_new_tokens=20, do_sample=False, eos_token_id=eos_id
    )
    report("generated_with_eos", generated.shape[1] >= 4)


def test_model_with_inputs_embeds():
    print("=== Qwen3_5ForCausalLM with inputs_embeds ===")
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config)
    embeds = torch.randn(2, 8, config.hidden_size)
    out = model(inputs_embeds=embeds)
    report("logits_shape_embeds", out["logits"].shape == (2, 8, config.vocab_size))
    report("not_nan_embeds", not torch.isnan(out["logits"]).any())


def test_incremental_decode():
    print("=== Incremental decode (KV cache reuse) ===")
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    out1 = model(input_ids=input_ids, use_cache=True)
    kv = out1["past_key_values"]
    dc = out1["delta_cache"]
    next_token = out1["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
    out2 = model(
        input_ids=next_token, past_key_values=kv, delta_states=dc, use_cache=True
    )
    report(
        "incremental_logits_shape", out2["logits"].shape == (1, 1, config.vocab_size)
    )
    report("incremental_not_nan", not torch.isnan(out2["logits"]).any())


def test_error_conditions():
    print("=== Error conditions ===")
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config)
    try:
        model()
        report("no_input_error", False)
    except ValueError:
        report("no_input_error", True)

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    embeds = torch.randn(1, 4, config.hidden_size)
    try:
        model(input_ids=input_ids, inputs_embeds=embeds)
        report("both_inputs_error", False)
    except ValueError:
        report("both_inputs_error", True)


# ── pure-logic edge branches (scatter / cache / mask / export-delta) ──


def test_scatter_kv_to_cache_b1_and_b2():
    """b=1 -> npu_scatter_nd_update path (412-423); b!=1 -> index_copy (411)."""
    # b=1: scatter_nd_update
    b, n, kv_max, d = 1, 2, 8, 4
    cache = torch.zeros(b, n, kv_max, d)
    pos = torch.tensor([0, 3], dtype=torch.int64)
    kv_new = torch.randn(b, n, 2, d)
    out = scatter_kv_to_cache(cache, pos, kv_new)
    assert out.shape == cache.shape
    assert torch.equal(out[0, 0, 0], kv_new[0, 0, 0])
    assert torch.equal(out[0, 0, 3], kv_new[0, 0, 1])

    # b=2: index_copy fallback
    b2 = 2
    cache2 = torch.zeros(b2, n, kv_max, d)
    kv_new2 = torch.randn(b2, n, 2, d)
    out2 = scatter_kv_to_cache(cache2, pos, kv_new2)
    expected = cache2.clone().index_copy(2, pos, kv_new2)
    assert torch.equal(out2, expected)


def test_simple_kv_cache_scatter_write():
    """new_kv_cache_pos + existing buffer -> scatter-write (447-450)."""
    cache = SimpleKVCache()
    b, n, kv_max, d = 1, 2, 8, 4
    # pre-populate a kv_max buffer (so keys[layer_idx] is not None)
    cache.keys = [torch.zeros(b, n, kv_max, d)]
    cache.values = [torch.zeros(b, n, kv_max, d)]
    k_new = torch.randn(b, n, 2, d)
    v_new = torch.randn(b, n, 2, d)
    pos = torch.tensor([0, 5], dtype=torch.int64)

    k_out, v_out = cache.update(k_new, v_new, 0, new_kv_cache_pos=pos)
    assert k_out.shape[2] == kv_max  # buffer stays kv_max (scatter, not cat)
    assert torch.equal(k_out[0, 0, 0], k_new[0, 0, 0])
    assert torch.equal(k_out[0, 0, 5], k_new[0, 0, 1])
    assert cache.get_seq_length() == kv_max


def test_simple_kv_cache_cat_fallback():
    """existing buffer + new_kv_cache_pos=None -> torch.cat (455-457)."""
    cache = SimpleKVCache()
    b, n, d = 1, 2, 4
    cache.keys = [torch.randn(b, n, 3, d)]
    cache.values = [torch.randn(b, n, 3, d)]
    k_new = torch.randn(b, n, 2, d)
    v_new = torch.randn(b, n, 2, d)
    k_out, v_out = cache.update(k_new, v_new, 0, new_kv_cache_pos=None)
    assert k_out.shape[2] == 5  # 3 + 2 (cat grew)


def test_create_causal_mask_2d_padding():
    """2D attention_mask -> extended_mask + causal (816-818)."""
    config = make_small_config()
    inputs_embeds = torch.randn(1, 4, config.hidden_size)
    mask2d = torch.ones(1, 4)
    mask2d[0, -1] = 0  # last token padded
    out = _create_causal_mask(mask2d, inputs_embeds, past_key_values=None)
    assert out.shape == (1, 1, 4, 4)
    # padded key position is masked for all queries (causal-masked -> -inf,
    # diagonal -> min_val from padding; none attend/0)
    assert (out[0, 0, :, -1] != 0).all()
    # unpadded key 0 is attended by causal queries (0.0) for q>=0
    assert out[0, 0, 0, 0] == 0


def test_create_causal_mask_4d_passthrough():
    """4D attention_mask returned as-is (797-798)."""
    config = make_small_config()
    inputs_embeds = torch.randn(1, 4, config.hidden_size)
    mask4d = torch.zeros(1, 1, 4, 4)
    out = _create_causal_mask(mask4d, inputs_embeds, past_key_values=None)
    assert out is mask4d  # returned as-is


def test_text_model_export_delta_states():
    """delta_states as tuple (conv_list, rec_list) -> export is_export_delta path (917-921)."""
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    # first linear layer's GDN for state shapes
    lin_idx = next(
        i for i, lt in enumerate(config.layer_types) if lt == "linear_attention"
    )
    gdn = model.model.layers[lin_idx].linear_attn
    n_linear = sum(1 for lt in config.layer_types if lt == "linear_attention")
    conv_list = [
        torch.zeros(1, gdn.conv_dim, gdn.conv_kernel_size) for _ in range(n_linear)
    ]
    rec_list = [
        torch.zeros(1, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim)
        for _ in range(n_linear)
    ]
    delta_states = (conv_list, rec_list)

    with torch.no_grad():
        hidden, kv, delta_out = model.model(
            input_ids=input_ids,
            delta_states=delta_states,
            use_cache=False,
        )
    assert hidden.shape == (1, 4, config.hidden_size)
    # export path returns (new_conv_list, new_rec_list) tuple
    assert isinstance(delta_out, tuple)
    assert len(delta_out[0]) == n_linear  # conv_list


def test_text_model_2d_mask_forward():
    """2D attention_mask through full TextModel.forward (covers mask plumbing)."""
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    mask2d = torch.ones(1, 6)
    mask2d[0, -2:] = 0
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask2d)
    assert out["logits"].shape == (1, 6, config.vocab_size)
    assert torch.isfinite(out["logits"]).all()


# ── qwen3_5 export wrapper (helper methods + forward) ──


def test_qwen3_5_wrapper_layer_index_helpers():
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()

    full_idx = wrapper._get_full_attn_layer_indices()
    lin_idx = wrapper._linear_layer_indices()
    assert all(config.layer_types[i] == "full_attention" for i in full_idx)
    assert all(config.layer_types[i] == "linear_attention" for i in lin_idx)
    assert len(full_idx) + len(lin_idx) == config.num_hidden_layers
    # full + linear partition all layers, no overlap
    assert set(full_idx).isdisjoint(set(lin_idx))


def test_qwen3_5_wrapper_build_zero_delta_states():
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()

    delta = wrapper._build_zero_delta_states()
    assert isinstance(delta, tuple)
    n_linear = len(wrapper._linear_layer_indices())
    assert len(delta[0]) == n_linear  # conv_flat
    assert len(delta[1]) == n_linear  # rec_flat
    # conv_state shape [batch, conv_dim, conv_kernel_size]
    lin = next(
        layer.linear_attn
        for layer in model.model.layers
        if layer.layer_type == "linear_attention"
    )
    assert delta[0][0].shape == (1, lin.conv_dim, lin.conv_kernel_size)
    assert delta[1][0].shape == (1, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim)


def test_qwen3_5_wrapper_get_sample_inputs():
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()

    inputs = wrapper.get_sample_inputs(seq_len=4)
    full_idx = wrapper._get_full_attn_layer_indices()
    # past_kv reduced to full_attn layers only (not all layers)
    assert len(inputs[3]) == len(full_idx)
    # delta_states (zero tuple) appended as last element
    assert inputs[-1] is not None and isinstance(inputs[-1], tuple)


def test_qwen3_5_wrapper_forward_export_path():
    """wrapper forward with pre-built sample inputs -> _drive_layers export path (95-140)."""
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    b, s = 1, 4
    inputs = wrapper.get_sample_inputs(seq_len=s)
    input_ids, attn_mask, pos_ids, past_kv, new_kv_pos, _, out_pos, _, _, delta = inputs

    with torch.no_grad():
        logits, *rest = wrapper(
            input_ids,
            attn_mask,
            pos_ids,
            past_kv,
            new_kv_pos,
            None,
            out_pos,
            None,
            None,
            delta,
        )
    # output_pos selects last position -> [b, 1, vocab]
    assert logits.shape == (b, 1, config.vocab_size)
    assert torch.isfinite(logits).all()
    # delta_out_flat appended after kv_flat (conv, rec pairs per linear layer)
    n_linear = len(wrapper._linear_layer_indices())
    full_idx = wrapper._get_full_attn_layer_indices()
    delta_out_flat = rest[2 * len(full_idx) :]  # kv_flat is 2 per full_attn layer
    assert len(delta_out_flat) == 2 * n_linear


def test_qwen3_5_wrapper_forward_eager_delta():
    """delta_states=None -> _delta_cache_box persists across calls (123-124)."""
    config = make_small_config()
    model = Qwen3_5ForCausalLM(config).eval()
    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=16, batch=1).eval()
    b, s = 1, 4
    inputs = wrapper.get_sample_inputs(seq_len=s)
    input_ids, attn_mask, pos_ids, past_kv, new_kv_pos = (
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
    )
    out_pos = inputs[6]

    with torch.no_grad():
        logits1, *_ = wrapper(
            input_ids,
            attn_mask,
            pos_ids,
            past_kv,
            new_kv_pos,
            None,
            out_pos,
            None,
            None,
            None,
        )
    assert logits1.shape == (b, 1, config.vocab_size)
    # _delta_cache_box persisted (eager path creates SimpleDeltaNetCache)
    assert wrapper._delta_cache_box[0] is not None


def test_qwen3_5_config_branches(tmp_path):
    """from_dict (71-74), from_json_file (78-82), VisionConfig.from_dict (112-114),
    Qwen3_5Config.__post_init__ dict/None (129-137), from_json_file (141-143)."""
    # Qwen3_5TextConfig.from_dict filters unknown keys
    cfg = Qwen3_5TextConfig.from_dict({"vocab_size": 99, "hidden_size": 16, "bad": 1})
    assert cfg.vocab_size == 99 and cfg.hidden_size == 16

    # Qwen3_5TextConfig.from_json_file with text_config wrapper
    p = tmp_path / "text.json"
    p.write_text(json.dumps({"text_config": {"vocab_size": 99}}))
    cfg2 = Qwen3_5TextConfig.from_json_file(str(p))
    assert cfg2.vocab_size == 99

    # Qwen3_5VisionConfig.from_dict
    vc = Qwen3_5VisionConfig.from_dict({"depth": 5, "bad": 1})
    assert vc.depth == 5

    # Qwen3_5Config.__post_init__: None -> defaults; dict -> converted
    c = Qwen3_5Config()
    assert isinstance(c.text_config, Qwen3_5TextConfig)
    assert isinstance(c.vision_config, Qwen3_5VisionConfig)
    c2 = Qwen3_5Config(text_config={"vocab_size": 50})
    assert (
        isinstance(c2.text_config, Qwen3_5TextConfig)
        and c2.text_config.vocab_size == 50
    )
    c3 = Qwen3_5Config(vision_config={"depth": 7})
    assert (
        isinstance(c3.vision_config, Qwen3_5VisionConfig)
        and c3.vision_config.depth == 7
    )

    # Qwen3_5Config.from_json_file
    p2 = tmp_path / "top.json"
    p2.write_text(
        json.dumps({"text_config": {"vocab_size": 50}, "tie_word_embeddings": True})
    )
    c4 = Qwen3_5Config.from_json_file(str(p2))
    assert c4.tie_word_embeddings is True
    assert isinstance(c4.text_config, Qwen3_5TextConfig)


def main():
    torch.manual_seed(42)
    test_rms_norm()
    test_rms_norm_gated()
    test_rotary_embedding()
    test_rotate_half()
    test_apply_rotary_pos_emb()
    test_l2norm()
    test_repeat_kv()
    test_apply_mask_to_padding_states()
    test_causal_conv1d_update()
    test_chunk_gated_delta_rule()
    test_recurrent_gated_delta_rule()
    test_simple_kv_cache()
    test_simple_delta_net_cache()
    test_eager_attention_forward()
    test_gated_delta_net()
    test_attention()
    test_mlp()
    test_decoder_layer()
    test_text_model()
    test_causal_lm()
    test_generate()
    test_generate_with_eos()
    test_model_with_inputs_embeds()
    test_incremental_decode()
    test_error_conditions()

    print(f"\n{'=' * 40}")
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed!")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
