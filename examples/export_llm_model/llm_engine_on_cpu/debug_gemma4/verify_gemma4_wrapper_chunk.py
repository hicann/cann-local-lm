#!/usr/bin/env python3
"""用 Gemma4ExportWrapper 验证三个 prompt 的 chunk 推理

BOS token + bfloat16 + wrapper I/O 契约
对比: 非分 chunk vs 分 chunk, 生成文本

用法:
  python verify_gemma4_wrapper_chunk.py
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
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from transformers import AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
NUM_DECODE_STEPS = 10
CHUNK_SIZE = 8
BOS_TOKEN_ID = 2

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


def reconstruct_kv(flat_kv):
    """从展平的 [k0,v0,k1,v1,...] 重建 [(k0,v0),(k1,v1),...]"""
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def generate_wrapper(wrapper, input_ids, num_steps, dtype):
    """A: wrapper 一次性 prefill + decode"""
    seq_len = input_ids.shape[1]
    mask = make_causal_mask(seq_len, seq_len, dtype)

    with torch.no_grad():
        logits, *kv_flat = wrapper(
            input_ids, mask, None, None,
            torch.arange(seq_len, dtype=torch.long), None, None,
        )
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = seq_len + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, None, kv,
                torch.tensor([cached_len - 1], dtype=torch.long), None, None,
            )
        kv = reconstruct_kv(kv_flat)
    return tokens


def generate_wrapper_chunked(wrapper, input_ids, chunk_size, num_steps, dtype):
    """B: wrapper 分 chunk prefill + decode"""
    seq_len = input_ids.shape[1]
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    kv = None
    logits = None

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
            mask[:, :, :, k_len:] = make_causal_mask(chunk_len, chunk_len, dtype)

        with torch.no_grad():
            logits, *kv_flat = wrapper(
                chunk, mask, None, kv,
                torch.arange(start, end, dtype=torch.long), None, None,
            )
        kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = seq_len + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, None, kv,
                torch.tensor([cached_len - 1], dtype=torch.long), None, None,
            )
        kv = reconstruct_kv(kv_flat)
    return tokens


def main():
    dtype = torch.bfloat16

    print("[1/3] Loading model + tokenizer")
    with open(f"{HF_PATH}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    import dataclasses
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid})
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    model = Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    model = load_gemma4_weights(model, HF_PATH, device="cpu", dtype=dtype)
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Wrapper: {wrapper.__class__.__name__}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    all_pass = True

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"[2/3] Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        seq_len = input_ids.shape[1]
        num_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  tokens: {seq_len} (含 BOS), chunks: {num_chunks} (size={CHUNK_SIZE})")

        # A: wrapper 一次性
        print(f"\n  A: wrapper 一次性 prefill...")
        tokens_a = generate_wrapper(wrapper, input_ids, NUM_DECODE_STEPS, dtype)
        text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
        print(f"  A output: \"{text_a}\"")

        # B: wrapper 分 chunk
        print(f"\n  B: wrapper 分 chunk prefill (chunk_size={CHUNK_SIZE})...")
        tokens_b = generate_wrapper_chunked(wrapper, input_ids, CHUNK_SIZE, NUM_DECODE_STEPS, dtype)
        text_b = tokenizer.decode(tokens_b, skip_special_tokens=True)
        print(f"  B output: \"{text_b}\"")

        # 对比
        token_match = tokens_a == tokens_b
        text_match = text_a == text_b
        all_pass = all_pass and token_match
        print(f"\n  Token match: {token_match}")
        print(f"  Text match:  {text_match}")
        if not token_match:
            print(f"  A tokens: {tokens_a[:5]}...")
            print(f"  B tokens: {tokens_b[:5]}...")

    print(f"\n{'='*60}")
    print(f"[3/3] Summary: Gemma4ExportWrapper chunk vs non-chunk")
    print(f"{'='*60}")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"  Decode: {NUM_DECODE_STEPS} steps")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  BOS: yes (id={BOS_TOKEN_ID})")
    print(f"  dtype: bfloat16")
    print(f"  All match: {all_pass}")
    print()
    if all_pass:
        print("[PASS] ExportWrapper chunk 推理与非 chunk 生成完全一致")
    else:
        print("[FAIL] 存在不一致")


if __name__ == "__main__":
    main()
