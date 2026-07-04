from __future__ import annotations

import sys
import os
import torch
import torch.nn.functional as F

from cann_lm.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from cann_lm.models.qwen3_5.modeling_qwen3_5 import (
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
)


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
    out_no_gate = norm(x)
    report("no_gate_output", out_no_gate.shape == (2, 8, hidden_size))


def test_rotary_embedding():
    print("=== Qwen3_5TextRotaryEmbedding ===")
    config = make_small_config()
    rope = Qwen3_5TextRotaryEmbedding(config)
    x = torch.randn(2, 4, 8, config.head_dim)
    position_ids = torch.arange(0, 8).unsqueeze(0).expand(2, -1)
    position_ids_3d = position_ids[None, ...].expand(3, 2, 8)
    cos, sin = rope(x, position_ids_3d)
    report("cos_shape", cos.shape[-1] == config.head_dim)
    report("sin_shape", sin.shape[-1] == config.head_dim)
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
    cos = torch.randn(1, 1, 8, config.head_dim)
    sin = torch.randn(1, 1, 8, config.head_dim)
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
    out, weights = eager_attention_forward(attn, q, k, v, None, scaling=config.head_dim ** -0.5)
    report("output_shape", out.shape == (2, 8, 4 * config.head_dim))
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
    weight = torch.randn(dim, kernel)
    bias = torch.randn(dim)
    out = torch_causal_conv1d_update(hidden, conv_state, weight, bias)
    report("output_shape", out.shape == (batch, dim, seq))
    report("not_nan", not torch.isnan(out).any())
    report("conv_state_updated", conv_state.shape == (batch, dim, kernel))


def test_chunk_gated_delta_rule():
    print("=== torch_chunk_gated_delta_rule ===")
    batch, heads, seq, k_dim, v_dim = 2, 4, 64, 16, 16
    query = torch.randn(batch, seq, heads, k_dim)
    key = torch.randn(batch, seq, heads, k_dim)
    value = torch.randn(batch, seq, heads, v_dim)
    g = torch.randn(batch, seq, heads)
    beta = torch.randn(batch, seq, heads)
    out, final_state = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        chunk_size=64, output_final_state=True,
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
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=True,
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
        num_layers=4, conv_dim=64, conv_kernel_size=4,
        k_head_dim=16, v_head_dim=16, num_v_heads=4,
        batch_size=2, dtype=torch.float32,
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
    out = gdn(x)
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
    out_cached = gdn(x, cache_params=delta_cache)
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
    out_linear = linear_layer(x, position_embeddings=(cos, sin))
    report("linear_layer_output_shape", out_linear.shape == (2, 8, config.hidden_size))
    report("linear_layer_not_nan", not torch.isnan(out_linear).any())

    full_layer = Qwen3_5DecoderLayer(config, layer_idx=1)
    out_full = full_layer(x, position_embeddings=(cos, sin))
    report("full_attention_layer_output_shape", out_full.shape == (2, 8, config.hidden_size))
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
    report("loss_with_labels", out_with_loss["loss"] is not None and not torch.isnan(out_with_loss["loss"]))
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
    generated = model.generate(input_ids, max_new_tokens=20, do_sample=False, eos_token_id=eos_id)
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
    out2 = model(input_ids=next_token, past_key_values=kv, delta_cache=dc, use_cache=True)
    report("incremental_logits_shape", out2["logits"].shape == (1, 1, config.vocab_size))
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

    print(f"\n{'='*40}")
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed!")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
