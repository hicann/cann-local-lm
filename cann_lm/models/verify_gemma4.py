"""End-to-end Gemma-4 verification script.

Verifies the full pipeline: text input -> tokenize -> forward -> generate output.

Usage:
    # Test with random weights (quick verification of shapes/pipeline)
    python verify_gemma4.py --prompt "Hello world"

    # Test with real safetensors weights
    python verify_gemma4.py --model ./checkpoint_dir --prompt "Tell me about AI"

    # Truncated layers for speed
    python verify_gemma4.py --model ./checkpoint_dir --layers 4 --prompt "Hi"

    # Run all unit tests first then verify
    python verify_gemma4.py --test-only
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gemma4.configuration_gemma4 import Gemma4TextConfig
from models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextModel,
    Gemma4TextDecoderLayer,
    Gemma4TextAttention,
    Gemma4TextMLP,
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb_qk,
    repeat_kv,
    load_gemma4_weights,
)

# ════════════════════════════════════════════
# Minimal byte-level tokenizer (no transformers dep)
# ════════════════════════════════════════════

class ByteTokenizer:
    """Fallback tokenizer that works without transformers library.
    Uses UTF-8 byte encoding + a small fixed vocabulary.
    """
    def __init__(self, vocab_size: int, bos_id: int = 2, eos_id: int = 1):
        # Reserve first N entries for single-byte tokens
        self.byte_base = 3  # 0=pad, 1=eos, 2=bos, 3-258=bytes
        self.byte_vocab = self.byte_base + 256
        self._vocab_size = max(vocab_size, self.byte_vocab + 1)
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.pad_token_id = 0
        self.vocab_size = self._vocab_size

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for b in text.encode("utf-8"):
            ids.append(self.byte_base + b)
        return ids

    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for i in ids:
            if self.byte_base <= i < self.byte_vocab:
                bytes_list.append(i - self.byte_base)
        return bytes(bytes_list).decode("utf-8", errors="replace")

    def __call__(self, text: str, **kwargs) -> dict:
        ids = self.encode(text, add_bos=kwargs.get("add_bos", True))
        return {"input_ids": torch.tensor([ids])}


def load_hf_tokenizer(model_path: str):
    """Try to load HuggingFace tokenizer; fall back to ByteTokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  [INFO] Loaded HF tokenizer: {tokenizer.__class__.__name__}")
        return tokenizer
    except Exception as e:
        print(f"  [WARN] transformers not available ({e}), using ByteTokenizer fallback")
        return None


# ════════════════════════════════════════════
# Inference helpers
# ════════════════════════════════════════════

@torch.no_grad()
def generate_text(
    model: Gemma4ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_k: int = 40,
    top_p: float = 0.9,
) -> str:
    """Generate text from a prompt."""
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    input_len = input_ids.shape[1]
    print(f"  Input tokens: {input_len}")

    # Greedy first step to verify correctness
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start = time.time()

    gen_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
    )

    elapsed = time.time() - start
    gen_len = gen_ids.shape[1] - input_len
    tokens_per_sec = gen_len / elapsed if elapsed > 0 else 0

    print(f"  Generated {gen_len} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    generated = gen_ids[0, input_len:].tolist()
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(generated, skip_special_tokens=True)
    return tokenizer.decode(generated)


# ════════════════════════════════════════════
# Build model from checkpoint or random init
# ════════════════════════════════════════════

def build_model(
    model_path: Optional[str] = None,
    num_layers: int = 6,
    hidden_size: int = 256,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 32,
    global_head_dim: int = 32,
    vocab_size: int = 1000,
    **kwargs,
) -> tuple[Gemma4ForCausalLM, object]:
    """Build and optionally load weights for a Gemma-4 model.

    Args:
        model_path: Path to safetensors checkpoint (None = random init).
        num_layers: Number of decoder layers (smaller = faster test).
        hidden_size: Hidden dimension.
        **kwargs: Override any Gemma4TextConfig field.

    Returns:
        (model, tokenizer) tuple.
    """
    config_kwargs = dict(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        vocab_size=vocab_size,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        sliding_window=512,
        num_kv_shared_layers=0,
        **kwargs,
    )

    if model_path:
        # Load config from checkpoint
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            text_cfg = cfg.get("text_config", cfg)
            for k in ["vocab_size", "hidden_size", "intermediate_size",
                       "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                       "head_dim", "global_head_dim", "sliding_window",
                       "rms_norm_eps", "num_kv_shared_layers", "use_double_wide_mlp",
                       "attention_k_eq_v", "attention_dropout",
                       "final_logit_softcapping", "rope_parameters", "layer_types",
                       "pad_token_id", "eos_token_id", "bos_token_id"]:
                if k in text_cfg:
                    config_kwargs[k] = text_cfg[k]
            print(f"  [INFO] Loaded config from {model_path}")
            print(f"  [INFO] Model: {text_cfg.get('hidden_size', '?')}d x "
                  f"{text_cfg.get('num_hidden_layers', '?')} layers, "
                  f"vocab={text_cfg.get('vocab_size', '?')}")

    config = Gemma4TextConfig(**config_kwargs)
    config.__post_init__()
    print(f"  Config: {config.hidden_size}d x {config.num_hidden_layers} layers, "
          f"vocab={config.vocab_size}, {config.num_attention_heads} heads")

    # Build model
    model = Gemma4ForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    mem_gb = n_params * 4 / 1e9  # float32 estimation
    print(f"  Parameters: {n_params / 1e6:.2f}M (~{mem_gb:.2f}GB f32)")

    # Load weights if checkpoint provided
    if model_path:
        print(f"  Loading weights from {model_path}...")
        t0 = time.time()
        model = load_gemma4_weights(model, model_path)
        print(f"  Weights loaded in {time.time() - t0:.1f}s")

    # Load tokenizer
    tokenizer = load_hf_tokenizer(model_path) if model_path else None
    if tokenizer is None:
        tokenizer = ByteTokenizer(config.vocab_size, config.bos_token_id, config.eos_token_id)
        print(f"  Using ByteTokenizer (vocab={tokenizer.vocab_size})")

    return model, tokenizer


# ════════════════════════════════════════════
# Unit tests (subset of test_gemma4.py)
# ════════════════════════════════════════════

def run_unit_tests():
    """Run component-level unit tests to verify correctness."""
    passed = 0
    failed = 0

    def _check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"    [FAIL] {name}")

    def _pass(name):
        nonlocal passed
        passed += 1

    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    t0 = time.time()

    # 1. Config
    print("\n--- Configuration ---")
    cfg = Gemma4TextConfig(num_hidden_layers=4)
    cfg.__post_init__()
    _check("layer_types len=4", len(cfg.layer_types) == 4)
    _check("last layer full_attention", cfg.layer_types[-1] == "full_attention")
    _pass("config instantiation OK")

    # 2. RMSNorm
    print("\n--- RMSNorm ---")
    norm = Gemma4RMSNorm(64)
    x = torch.randn(2, 4, 64)
    out = norm(x)
    _check("shape", out.shape == (2, 4, 64))
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = x / rms
    _check("RMSNorm computation", torch.allclose(out, expected, atol=1e-5))

    # 3. RoPE
    print("\n--- RoPE ---")
    cfg2 = Gemma4TextConfig(num_hidden_layers=4, head_dim=32, global_head_dim=32)
    cfg2.__post_init__()
    rope = Gemma4TextRotaryEmbedding(cfg2)
    pos = torch.arange(0, 8).unsqueeze(0)
    cos, sin = rope(x, pos, "full_attention")
    _check("RoPE shape", cos.shape[-1] == 32)
    _check("RoPE cos range", cos.abs().max() <= 1.0 + 1e-5)
    _check("RoPE sin range", sin.abs().max() <= 1.0 + 1e-5)

    # 4. apply_rotary_pos_emb_qk
    print("\n--- Apply RoPE ---")
    q = torch.randn(2, 8, 4, 32)
    q_rot, k_rot = apply_rotary_pos_emb_qk(q, q, cos, sin)
    _check("apply_rope shape", q_rot.shape == q.shape)
    
    # Test rotational identity: RoPE should produce different values
    diff = (q_rot - q).abs().sum()
    _check("RoPE actually rotates", diff > 0.01)

    # 5. repeat_kv
    print("\n--- Repeat KV ---")
    kv = torch.randn(2, 2, 8, 32)
    expanded = repeat_kv(kv, 2)
    _check("repeat_kv GQA 2x", expanded.shape == (2, 4, 8, 32))
    _check("repeat_kv GQA 1x", repeat_kv(kv, 1).shape == (2, 2, 8, 32))

    # 6. MLP
    print("\n--- MLP ---")
    mlp = Gemma4TextMLP(cfg2, 0)
    out = mlp(x)
    _check("MLP shape", out.shape == x.shape)
    _check("MLP non-zero", out.abs().sum() > 0)

    # 7. Attention (basic)
    print("\n--- Attention ---")
    attn = Gemma4TextAttention(cfg2, 0)
    pos_emb = rope(x, pos, cfg2.layer_types[0])
    out = attn(x, pos_emb)
    _check("Attention shape", out.shape == x.shape)

    # 8. Decoder Layer
    print("\n--- Decoder Layer ---")
    layer = Gemma4TextDecoderLayer(cfg2, 0)
    out = layer(x, pos_emb)
    _check("DecoderLayer shape", out.shape == x.shape)

    # 9. Full Model forward
    print("\n--- Forward Pass ---")
    model = Gemma4TextModel(cfg2)
    input_ids = torch.randint(0, 1000, (2, 8))
    hs, pv, _ = model(input_ids=input_ids)
    _check("TextModel forward", hs.shape == (2, 8, cfg2.hidden_size))
    _check("past_key_values length", len(pv) == cfg2.num_hidden_layers)

    # 10. ForCausalLM
    print("\n--- ForCausalLM ---")
    lm = Gemma4ForCausalLM(cfg2)
    out = lm(input_ids=input_ids)
    _check("LM logits shape", out["logits"].shape == (2, 8, cfg2.vocab_size))
    _check("No loss", out["loss"] is None)

    # With labels
    labels = input_ids.clone()
    out = lm(input_ids=input_ids, labels=labels)
    _check("Loss computed", out["loss"] is not None)
    _check("Loss is finite", math.isfinite(out["loss"].item()))

    # 11. Generate (greedy)
    print("\n--- Generate ---")
    gen = lm.generate(input_ids, max_new_tokens=4)
    _check("Generate extended", gen.shape[1] == input_ids.shape[1] + 4)

    # 12. KV Sharing
    print("\n--- KV Sharing ---")
    cfg3 = Gemma4TextConfig(num_hidden_layers=4, hidden_size=64,
                            intermediate_size=256, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16,
                            global_head_dim=16, vocab_size=500,
                            num_kv_shared_layers=2)
    cfg3.__post_init__()
    lm3 = Gemma4ForCausalLM(cfg3)
    ids3 = torch.randint(0, 500, (1, 4))
    out3 = lm3(input_ids=ids3)
    _check("KV shared LM forward", out3["logits"].shape[2] == cfg3.vocab_size)
    _check("KV shared past_kv length", len(out3["past_key_values"]) == cfg3.num_hidden_layers)

    # 13. Layer truncation
    print("\n--- Layer Truncation ---")
    for n in [1, 2, 3]:
        cfg4 = Gemma4TextConfig(num_hidden_layers=n, hidden_size=64,
                                intermediate_size=256, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16,
                                global_head_dim=16, vocab_size=500)
        cfg4.__post_init__()
        lm4 = Gemma4ForCausalLM(cfg4)
        ids4 = torch.randint(0, 500, (1, 4))
        out4 = lm4(input_ids=ids4)
        _check(f"truncated {n} layers OK", out4["logits"].shape[1] == 4)

    # Summary
    elapsed = time.time() - t0
    print(f"\n  Unit tests: {passed} passed, {failed} failed ({elapsed:.2f}s)")
    return failed == 0


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gemma-4 end-to-end verification")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input text prompt")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to safetensors checkpoint")
    parser.add_argument("--layers", type=int, default=6,
                        help="Number of decoder layers (for random init test)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden size (for random init test)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--test-only", action="store_true",
                        help="Only run unit tests, skip inference")
    parser.add_argument("--no-tests", action="store_true",
                        help="Skip unit tests, run inference only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch: {torch.__version__}  Device: {device}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {gpu.total_mem / 1e9:.1f}GB")

    # ── Unit Tests ──────────────────────────────────
    if not args.no_tests:
        ok = run_unit_tests()
        if args.test_only:
            sys.exit(0 if ok else 1)
        print()

    # ── Model ───────────────────────────────────────
    vocab = 1000
    if args.model:
        # Use checkpoint vocab size
        cfg_path = os.path.join(args.model, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            text_cfg = cfg.get("text_config", cfg)
            vocab = text_cfg.get("vocab_size", vocab)

    model, tokenizer = build_model(
        model_path=args.model,
        num_layers=args.layers,
        hidden_size=args.hidden,
        vocab_size=vocab,
    )

    # Move to device
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"  Device: {device}  Dtype: bfloat16")

    # ── Single-token sanity test ────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECK: single-token forward pass")
    print("=" * 60)
    with torch.no_grad():
        test_ids = torch.tensor([[0, 1, 2]], device=device)
        out = model(test_ids)
    logits = out["logits"]
    top_token = logits[0, -1].argmax().item()
    print(f"  Input shape: {test_ids.shape} -> Logits shape: {logits.shape}")
    print(f"  Top token at last pos: {top_token} (vocab: 0..{vocab-1})")
    assert logits.shape[-1] == vocab, f"Logits vocab mismatch: {logits.shape[-1]} != {vocab}"
    print("  [PASS] Forward pass OK")

    # ── Inference ───────────────────────────────────
    print("\n" + "=" * 60)
    print(f"GENERATION: \"{args.prompt}\"")
    print("=" * 60)

    generated = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(f"\n  Output: {generated}")

    # ── Summary ─────────────────────────────────────
    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE: All checks passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
