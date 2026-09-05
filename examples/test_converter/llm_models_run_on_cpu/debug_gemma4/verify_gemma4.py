#!/usr/bin/env python3
"""End-to-end Gemma-4 verification: 真实权重加载 + chat_template 出字 (CPU eager).

单元测试已抽到 tests/llm_models/test_gemma4.py (pytest 风格).

Usage:
    # 单元测试 (随机权重, 快):
    pytest tests/llm_models/test_gemma4.py

    # e2e 出字 (真实 safetensors 权重):
    python verify_gemma4.py --model /path/to/gemma-4-E2B-it --prompt "Tell me about AI"

    # 截断层数 (随机初始化, 快):
    python verify_gemma4.py --layers 4 --prompt "Hi"

环境 (运行前 source 基底, 详见 compare_hf_vs_ep.py docstring):
    source ../../../../scripts/set_env_export.sh   # 或 source ../check_env.sh
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, load_gemma4_weights

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

    # 套 chat template (gemma4 裸 prompt 会退化成 EOS/重复, 见 om_npu_verify/run_gemma4_gen.py).
    # apply_chat_template 已含 BOS + 控制 token (<|turn>user ... <turn|> <|turn>model); 无 template 时回退手动 BOS.
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(dtype=torch.long, device=device)
    else:
        # 无 chat_template (如 ByteTokenizer 随机初始化 fallback): 手动加 BOS.
        # HF tokenizer 用 add_special_tokens; ByteTokenizer 用 add_bos; 兼容两者.
        bos_id = getattr(tokenizer, "bos_token_id", 2)
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            try:
                ids = tokenizer.encode(prompt, add_bos=False)
            except TypeError:
                ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([[bos_id] + ids], dtype=torch.long, device=device)
    input_len = input_ids.shape[1]
    print(f"  Input tokens: {input_len}")

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
        try:
            return tokenizer.decode(generated, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(generated)  # ByteTokenizer 等 fallback
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
    parser = argparse.ArgumentParser(description="Gemma-4 e2e 验证: 真实权重加载 + 出字 (单元测试见 tests/llm_models/test_gemma4.py)")
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
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch: {torch.__version__}  Device: {device}")

    # 单元测试已抽到 tests/llm_models/test_gemma4.py: pytest tests/llm_models/test_gemma4.py

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
