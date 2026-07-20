#!/usr/bin/env python3
"""验证 gemma4 分 chunk 推理：实际文本输入 → 文本输出

对比:
  A. 一次性 prefill 整句 → 逐 token decode → 解码为文本
  B. 分 chunk prefill   → 逐 token decode → 解码为文本

用法:
  python verify_gemma4_chunk_text.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
NUM_DECODE_STEPS = 10
CHUNK_SIZE = 8

# 测试文本（中英文混合）
TEST_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


def make_causal_mask(q_len, k_len, dtype):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_decode_mask(k_len, dtype):
    return torch.zeros(1, 1, 1, k_len, dtype=dtype)


def load_model():
    with open(f"{HF_PATH}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    import dataclasses
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid})
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    model = Gemma4ForCausalLM(config).to(dtype=torch.float16).eval()
    model = load_gemma4_weights(model, HF_PATH, device="cpu", dtype=torch.float16)
    return model, config


def encode_text(text, tokenizer):
    """用 HF tokenizer 编码文本"""
    ids = tokenizer.encode(text, return_tensors="pt")
    return ids


def decode_tokens(token_ids, tokenizer):
    """解码 token id 列表为文本"""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def generate_non_chunk(model, input_ids, num_steps, dtype):
    """A: 一次性 prefill + 逐 token decode"""
    seq_len = input_ids.shape[1]
    mask = make_causal_mask(seq_len, seq_len, dtype)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask, shared_kv_states={})
        logits = out["logits"]
        kv = out["past_key_values"]
        shared = out.get("shared_kv_states", {})

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = seq_len + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=mask_dec,
                past_key_values=kv,
                shared_kv_states=shared,
            )
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})
    return tokens


def generate_chunked(model, input_ids, chunk_size, num_steps, dtype):
    """B: 分 chunk prefill + 逐 token decode"""
    seq_len = input_ids.shape[1]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    kv = None
    shared = {}

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        chunk_len = end - start
        k_len = start

        if chunk_idx == 0:
            mask = make_causal_mask(chunk_len, chunk_len, dtype)
        else:
            total_k = k_len + chunk_len
            mask = torch.zeros(1, 1, chunk_len, total_k, dtype=dtype)
            causal_part = make_causal_mask(chunk_len, chunk_len, dtype)
            mask[:, :, :, k_len:] = causal_part

        with torch.no_grad():
            out = model(
                input_ids=chunk,
                attention_mask=mask,
                past_key_values=kv,
                shared_kv_states=shared,
            )
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = seq_len + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=mask_dec,
                past_key_values=kv,
                shared_kv_states=shared,
            )
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})
    return tokens


def main():
    dtype = torch.float16

    print("[1/4] Loading model + tokenizer")
    model, config = load_model()
    print(f"      Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"      Layers: {config.num_hidden_layers} (shared: {getattr(config, 'num_kv_shared_layers', 0)})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"      Tokenizer: {tokenizer.__class__.__name__}")

    all_pass = True

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"[2/4] Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        input_ids = encode_text(prompt, tokenizer)
        seq_len = input_ids.shape[1]
        print(f"  tokens: {seq_len}, chunks: {(seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE} (size={CHUNK_SIZE})")

        # A: 非分 chunk
        print(f"\n  A: 一次性 prefill...")
        tokens_a = generate_non_chunk(model, input_ids, NUM_DECODE_STEPS, dtype)
        text_a = decode_tokens(tokens_a, tokenizer)
        print(f"  A output: \"{text_a}\"")

        # B: 分 chunk
        print(f"\n  B: 分 chunk prefill (chunk_size={CHUNK_SIZE})...")
        tokens_b = generate_chunked(model, input_ids, CHUNK_SIZE, NUM_DECODE_STEPS, dtype)
        text_b = decode_tokens(tokens_b, tokenizer)
        print(f"  B output: \"{text_b}\"")

        # 对比
        token_match = tokens_a == tokens_b
        text_match = text_a == text_b
        all_pass = all_pass and token_match and text_match
        print(f"\n  Token match: {token_match}")
        print(f"  Text match:  {text_match}")
        if not token_match:
            print(f"  A tokens: {tokens_a}")
            print(f"  B tokens: {tokens_b}")
            # 找第一个不同的位置
            for i in range(min(len(tokens_a), len(tokens_b))):
                if tokens_a[i] != tokens_b[i]:
                    print(f"  First diff at step {i}: A={tokens_a[i]} B={tokens_b[i]}")
                    break

    print(f"\n{'='*60}")
    print(f"Summary: gemma4 chunk vs non-chunk (text-level)")
    print(f"{'='*60}")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"  Decode steps per prompt: {NUM_DECODE_STEPS}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  All match: {all_pass}")
    print()
    if all_pass:
        print("[PASS] Chunk 推理生成文本与非 chunk 完全一致")
    else:
        print("[FAIL] 存在不一致")


if __name__ == "__main__":
    main()
