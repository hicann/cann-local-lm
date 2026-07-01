"""Test script for Gemma-4 pure PyTorch implementation.

Usage:
    python test_gemma4.py              # unit tests only
    python test_gemma4.py --model PATH # also test weight loading from safetensors
"""

import sys
import os
import time

# Ensure local models are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from models.gemma4.configuration_gemma4 import (
    Gemma4TextConfig, Gemma4VisionConfig, Gemma4AudioConfig, Gemma4Config,
)
from models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    Gemma4TextMLP,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4ForCausalLM,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_qk,
    repeat_kv,
    eager_attention_forward,
    rotate_half,
    load_gemma4_weights,
)

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

PASS = 0
FAIL = 0
SKIP = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def info(msg):
    print(f"  [INFO] {msg}")

# ──────────────────────────────────────────────
# test config
# ──────────────────────────────────────────────

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

    # Small config for fast tests
    small = Gemma4TextConfig(num_hidden_layers=4, hidden_size=128, intermediate_size=512,
                             num_attention_heads=4, num_key_value_heads=2,
                             head_dim=32, global_head_dim=32,
                             vocab_size=1000, num_kv_shared_layers=0)
    small.__post_init__()
    check("small config layers", len(small.layer_types) == 4)

    # Test layer truncation behavior
    tiny = Gemma4TextConfig(num_hidden_layers=2)
    tiny.__post_init__()
    check("tiny 2 layers", tiny.layer_types == ["sliding_attention", "full_attention"])
    info(f"  tiny layer_types: {tiny.layer_types}")

    return small


# ──────────────────────────────────────────────
# test components
# ──────────────────────────────────────────────

def test_components(config: Gemma4TextConfig):
    print("\n=== Basic Components ===")
    b, s, d = 2, 8, config.hidden_size

    # RMSNorm
    norm = Gemma4RMSNorm(d)
    x = torch.randn(b, s, d)
    out = norm(x)
    check("RMSNorm shape", out.shape == x.shape)
    check("RMSNorm not all zeros", out.abs().sum() > 0)

    # RoPE
    rope = Gemma4TextRotaryEmbedding(config)
    pos_ids = torch.arange(0, s).unsqueeze(0)
    cos, sin = rope(x, pos_ids, "full_attention")
    check("RoPE full cos shape", cos.shape[-1] == config.global_head_dim // 2 * 2)
    cos_sl, sin_sl = rope(x, pos_ids, "sliding_attention")
    check("RoPE sliding cos shape", cos_sl.shape[-1] == config.head_dim // 2 * 2)

    # apply_rotary_pos_emb_qk
    n_heads = config.num_attention_heads
    hd = config.global_head_dim
    q = torch.randn(b, s, n_heads, hd)
    k = torch.randn(b, s, n_heads // 2, hd)  # GQA: kv heads = num_heads/2
    q_out, k_out = apply_rotary_pos_emb_qk(q, k, cos, sin)
    check("rope_qk q shape", q_out.shape == q.shape)
    check("rope_qk k shape", k_out.shape == k.shape)

    # repeat_kv
    kv = torch.randn(2, 2, 4, hd)  # (b, kv_heads, s, d)
    expanded = repeat_kv(kv, n_rep=2)
    check("repeat_kv shape", expanded.shape == (2, 4, 4, hd))

    # rotate_half
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

        # With KV cache
        pv = [None] * config.num_hidden_layers
        # First forward to populate cache for this layer
        _ = attn(x, position_embeddings=pos_emb, past_key_values=pv)
        check(f"attn layer {layer_idx} KV cache store", pv[layer_idx] is not None)

        # Second forward with cache (single token)
        x1 = torch.randn(b, 1, d)
        pos_ids_1 = torch.tensor([[s]])  # position = s
        pos_emb_1 = rope(x1, pos_ids_1, layer_type)
        out1 = attn(x1, position_embeddings=pos_emb_1, past_key_values=pv)
        check(f"attn layer {layer_idx} cached shape", out1.shape == x1.shape)
        break  # only test first layer for speed


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
        break  # only test first layer


# ──────────────────────────────────────────────
# test full model
# ──────────────────────────────────────────────

def test_text_model(config: Gemma4TextConfig):
    print("\n=== TextModel ===")
    b, s = 2, 8

    model = Gemma4TextModel(config)
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    hs, pv, skv = model(input_ids=input_ids)
    check("TextModel output shape", hs.shape == (b, s, config.hidden_size))
    check("past_key_values correct", len(pv) == config.num_hidden_layers)

    # Without KV cache
    hs2, _, _ = model(input_ids=input_ids, past_key_values=None)
    check("TextModel no-cache output shape", hs2.shape == (b, s, config.hidden_size))

    # With attention mask
    mask = torch.ones(b, s)
    mask[0, -2:] = 0  # mask last 2 tokens of first sample
    hs3, _, _ = model(input_ids=input_ids, attention_mask=mask)
    check("TextModel masked output shape", hs3.shape == (b, s, config.hidden_size))

    n_params = sum(p.numel() for p in model.parameters())
    info(f"TextModel parameters: {n_params / 1e6:.2f}M")

    return model


def test_causal_lm(config: Gemma4TextConfig):
    print("\n=== ForCausalLM ===")
    b, s = 2, 8

    lm = Gemma4ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (b, s))
    out = lm(input_ids=input_ids)
    check("ForCausalLM logits shape", out["logits"].shape == (b, s, config.vocab_size))
    check("no loss without labels", out["loss"] is None)

    # With labels
    labels = input_ids.clone()
    labels[0, 0] = -100
    out = lm(input_ids=input_ids, labels=labels)
    check("loss is computed", out["loss"] is not None)
    check("loss is scalar", out["loss"].ndim == 0)

    # final_logit_softcapping
    config2 = Gemma4TextConfig(**{**config.__dict__, "final_logit_softcapping": 30.0})
    config2.__post_init__()
    lm2 = Gemma4ForCausalLM(config2)
    out2 = lm2(input_ids=input_ids)
    check("softcapped logits shape", out2["logits"].shape == (b, s, config2.vocab_size))

    # Generate (greedy)
    gen = lm.generate(input_ids, max_new_tokens=4)
    check("generate returns longer seq", gen.shape[1] > s)

    # Generate (sampling)
    gen2 = lm.generate(input_ids, max_new_tokens=4, do_sample=True, temperature=0.7, top_k=50)
    check("sampling generate", gen2.shape[1] > s)

    n_params = sum(p.numel() for p in lm.parameters())
    info(f"ForCausalLM parameters: {n_params / 1e6:.2f}M")


# ──────────────────────────────────────────────
# test KV sharing
# ──────────────────────────────────────────────

def test_kv_sharing():
    print("\n=== KV Sharing ===")
    config = Gemma4TextConfig(
        num_hidden_layers=6, hidden_size=128, intermediate_size=512,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, global_head_dim=32,
        vocab_size=1000,
        num_kv_shared_layers=3,  # last 3 layers share KV
    )
    config.__post_init__()

    b, s = 2, 8
    model = Gemma4TextModel(config)
    input_ids = torch.randint(0, 1000, (b, s))
    hs, pv, skv = model(input_ids=input_ids)

    check("KV shared model output shape", hs.shape == (b, s, config.hidden_size))
    check("shared_kv_states not empty", len(skv) > 0)

    # Check that shared layers don't have their own KV projections
    for i in range(3, 6):
        layer = model.layers[i]
        attn = layer.self_attn
        check(f"layer {i} is_kv_shared", attn.is_kv_shared_layer)
        check(f"layer {i} no k_proj", not hasattr(attn, "k_proj"))


# ──────────────────────────────────────────────
# test truncation (fewer layers for speed)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# test weight loading (requires safetensors checkpoint)
# ──────────────────────────────────────────────

def test_weight_loading(model_path: str):
    global SKIP
    print(f"\n=== Weight Loading ({model_path}) ===")
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("  [SKIP] safetensors not installed (pip install safetensors)")
        SKIP += 1
        return

    # Try to infer config from checkpoint
    import json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        print(f"  [SKIP] config.json not found in {model_path}")
        SKIP += 1
        return

    with open(config_path) as f:
        cfg = json.load(f)

    # Determine if text-only or multimodal
    text_cfg = cfg.get("text_config", cfg)
    if "vocab_size" not in text_cfg:
        print("  [SKIP] Cannot determine text config from config.json")
        global SKIP
        SKIP += 1
        return

    config = Gemma4TextConfig(
        vocab_size=text_cfg.get("vocab_size", 262144),
        hidden_size=text_cfg.get("hidden_size", 2304),
        intermediate_size=text_cfg.get("intermediate_size", 9216),
        num_hidden_layers=text_cfg.get("num_hidden_layers", 30),
        num_attention_heads=text_cfg.get("num_attention_heads", 8),
        num_key_value_heads=text_cfg.get("num_key_value_heads", 4),
        head_dim=text_cfg.get("head_dim", 256),
        global_head_dim=text_cfg.get("global_head_dim", 512),
        sliding_window=text_cfg.get("sliding_window", 512),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        num_kv_shared_layers=text_cfg.get("num_kv_shared_layers", 0),
        use_double_wide_mlp=text_cfg.get("use_double_wide_mlp", False),
        attention_k_eq_v=text_cfg.get("attention_k_eq_v", False),
        num_global_key_value_heads=text_cfg.get("num_global_key_value_heads"),
        rope_parameters=text_cfg.get("rope_parameters"),
        layer_types=text_cfg.get("layer_types"),
        final_logit_softcapping=text_cfg.get("final_logit_softcapping"),
        pad_token_id=cfg.get("pad_token_id", 0),
        eos_token_id=cfg.get("eos_token_id", 1),
        bos_token_id=cfg.get("bos_token_id", 2),
    )
    config.__post_init__()

    info(f"Config from checkpoint: {config.hidden_size}d x {config.num_hidden_layers} layers")
    info(f"  head_dim={config.head_dim}, global_head_dim={config.global_head_dim}")

    model = Gemma4ForCausalLM(config)
    n_total = sum(p.numel() for p in model.parameters())
    info(f"Model parameters (random init): {n_total / 1e9:.2f}B")

    start = time.time()
    model = load_gemma4_weights(model, model_path)
    elapsed = time.time() - start
    info(f"Loaded weights in {elapsed:.2f}s")

    # Quick forward pass
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    input_ids = torch.tensor([[config.bos_token_id]], device=device)
    with torch.no_grad():
        out = model(input_ids)
    check("loaded model forward yields logits", out["logits"].shape[-1] == config.vocab_size)
    info(f"  Forward pass OK. Logits shape: {out['logits'].shape}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    global PASS, FAIL, SKIP

    import argparse
    parser = argparse.ArgumentParser(description="Test Gemma-4 pure pytorch implementation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to safetensors checkpoint for weight loading test")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device (cpu, cuda, auto)")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch: {torch.__version__}  Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    # Basic config
    config = test_config()

    # Components
    test_components(config)
    test_attention(config)
    test_mlp(config)
    test_decoder_layer(config)

    # Full model
    test_text_model(config)
    test_causal_lm(config)

    # Advanced features
    test_kv_sharing()
    test_truncation()

    # Weight loading (if checkpoint provided)
    if args.model:
        test_weight_loading(args.model)

    # Summary
    print(f"\n{'='*50}")
    total = PASS + FAIL + SKIP
    print(f"Results: {PASS} PASS, {FAIL} FAIL, {SKIP} SKIP ({total} total)")
    print(f"{'='*50}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
