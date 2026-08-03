#!/usr/bin/env python3
"""End-to-end Gemma-4 verification: unit tests + text generation.

合并自 test_gemma4.py (单元测试) + verify_gemma4.py (端到端推理), 去除重复.

Usage:
    # unit tests only (random weights, quick shape/pipeline check)
    python verify_gemma4.py --test-only

    # unit tests + inference with real safetensors weights
    python verify_gemma4.py --model /path/to/gemma-4-E2B-it --prompt "Tell me about AI"

    # truncated layers for speed (random init)
    python verify_gemma4.py --layers 4 --prompt "Hi"

环境变量 (运行前 source, 详见 compare_hf_vs_ep.py docstring):
    source /home/developer/Ascend/cann/set_env.sh
    export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    load_gemma4_weights,
)
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper

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


# ════════════════════════════════════════════
# Unit tests (from test_gemma4.py)
# ════════════════════════════════════════════

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

    # apply_rotary_pos_emb (single tensor, fused via cann_lm::rotary_mul)
    n_heads = config.num_attention_heads
    hd = config.global_head_dim
    q = torch.randn(b, s, n_heads, hd)
    q_out = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
    check("apply_rotary_pos_emb shape", q_out.shape == q.shape)

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

    # With attention mask (2D)
    mask = torch.ones(b, s)
    mask[0, -2:] = 0  # mask last 2 tokens of first sample
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


def make_causal_mask(seq_len, dtype):
    """4D causal mask [1, 1, seq_len, seq_len] (cat 模式, 单元测试用)."""
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def test_wrapper_vs_model(config: Gemma4TextConfig):
    """验证 Gemma4ExportWrapper 单步 forward 与原始 model 一致 (max_diff=0.0).

    wrapper 是 model 的薄封装 (I/O 翻译), 单步 forward 应与 model 完全一致.
    合并自 verify_gemma4_all.py.
    """
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
        logits_wrap, *kv_flat = wrapper(input_ids, mask, pos_ids, None,
                                        torch.arange(s, dtype=torch.long), None, None)

    # logits
    diff = (logits_wrap[:, -1:, :].float() - logits_orig[:, -1:, :].float()).abs().max().item()
    match = torch.allclose(logits_wrap[:, -1:, :], logits_orig[:, -1:, :], atol=1e-3)
    check(f"wrapper logits match (max_diff={diff:.6e})", match)

    # KV cache
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


def run_unit_tests() -> bool:
    print("\n" + "=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    t0 = time.time()
    config = test_config()
    test_components(config)
    test_attention(config)
    test_mlp(config)
    test_decoder_layer(config)
    test_text_model(config)
    test_causal_lm(config)
    test_kv_sharing()
    test_truncation()
    test_wrapper_vs_model(config)
    elapsed = time.time() - t0
    print(f"\n  Unit tests: {PASS} passed, {FAIL} failed, {SKIP} skipped ({elapsed:.2f}s)")
    return FAIL == 0


# ════════════════════════════════════════════
# Tokenizer (ByteTokenizer fallback + HF loader)
# ════════════════════════════════════════════

class ByteTokenizer:
    """Fallback tokenizer that works without transformers library.
    Uses UTF-8 byte encoding + a small fixed vocabulary.
    """
    def __init__(self, vocab_size: int, bos_id: int = 2, eos_id: int = 1):
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
    """Try to load HuggingFace tokenizer; fall back to None."""
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
    """Generate text from a prompt (cann_lm eager model.generate, cat 模式)."""
    model.eval()
    device = next(model.parameters()).device

    # Gemma tokenizer 不自动加 BOS, 必须手动加 (否则输出退化重复, 见 skill #1)
    bos_id = getattr(tokenizer, "bos_token_id", 2)
    ids = tokenizer.encode(prompt, add_special_tokens=False) if hasattr(tokenizer, "encode") \
        else tokenizer(prompt)["input_ids"][0].tolist()
    input_ids = torch.tensor([[bos_id] + ids], dtype=torch.long, device=device)
    input_len = input_ids.shape[1]
    print(f"  Input tokens: {input_len} (BOS + {len(ids)})")

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
# Build model (random init or from checkpoint)
# ════════════════════════════════════════════

_CONFIG_KEYS = [
    "vocab_size", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "global_head_dim", "sliding_window",
    "rms_norm_eps", "num_kv_shared_layers", "use_double_wide_mlp",
    "attention_k_eq_v", "attention_dropout",
    "final_logit_softcapping", "rope_parameters", "layer_types",
    "num_global_key_value_heads",
    "pad_token_id", "eos_token_id", "bos_token_id",
]


def _config_from_checkpoint(model_path: str) -> Optional[Gemma4TextConfig]:
    """Load config from checkpoint's config.json (handles multimodal text_config nesting)."""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    if "vocab_size" not in text_cfg:
        return None
    config_kwargs = {k: v for k, v in text_cfg.items() if k in _CONFIG_KEYS}
    config = Gemma4TextConfig(**config_kwargs)
    config.__post_init__()
    print(f"  [INFO] Config from {model_path}: {config.hidden_size}d x "
          f"{config.num_hidden_layers} layers, vocab={config.vocab_size}")
    return config


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
    """Build model from checkpoint or random init.

    Args:
        model_path: Path to safetensors checkpoint (None = random init).
        num_layers/hidden_size/...: random init defaults (smaller = faster test).

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

    # Load config from checkpoint if provided (覆盖默认)
    if model_path:
        ck_config = _config_from_checkpoint(model_path)
        if ck_config is not None:
            for k in _CONFIG_KEYS:
                v = getattr(ck_config, k, None)
                if v is not None:
                    config_kwargs[k] = v

    config = Gemma4TextConfig(**config_kwargs)
    config.__post_init__()
    print(f"  Config: {config.hidden_size}d x {config.num_hidden_layers} layers, "
          f"vocab={config.vocab_size}, {config.num_attention_heads} heads")

    model = Gemma4ForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    mem_gb = n_params * 4 / 1e9
    print(f"  Parameters: {n_params / 1e6:.2f}M (~{mem_gb:.2f}GB f32)")

    if model_path:
        print(f"  Loading weights from {model_path}...")
        t0 = time.time()
        model = load_gemma4_weights(model, model_path)
        print(f"  Weights loaded in {time.time() - t0:.1f}s")

    tokenizer = load_hf_tokenizer(model_path) if model_path else None
    if tokenizer is None:
        tokenizer = ByteTokenizer(config.vocab_size, config.bos_token_id, config.eos_token_id)
        print(f"  Using ByteTokenizer (vocab={tokenizer.vocab_size})")

    return model, tokenizer


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gemma-4 verification: unit tests + inference")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input text prompt")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to safetensors checkpoint (None = random init)")
    parser.add_argument("--layers", type=int, default=6,
                        help="Number of decoder layers (for random init test)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden size (for random init test)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device (cpu, cuda, auto)")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run unit tests, skip inference")
    parser.add_argument("--no-tests", action="store_true",
                        help="Skip unit tests, run inference only")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch: {torch.__version__}  Device: {device}")

    # ── Unit Tests ──────────────────────────────────
    if not args.no_tests:
        ok = run_unit_tests()
        if args.test_only:
            sys.exit(0 if ok else 1)
        print()

    # ── Model ───────────────────────────────────────
    vocab = 1000
    if args.model:
        ck_config = _config_from_checkpoint(args.model)
        if ck_config is not None:
            vocab = ck_config.vocab_size

    model, tokenizer = build_model(
        model_path=args.model,
        num_layers=args.layers,
        hidden_size=args.hidden,
        vocab_size=vocab,
    )
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
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(f"\n  Output: {generated}")

    # ── Summary ─────────────────────────────────────
    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
