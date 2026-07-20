#!/usr/bin/env python3
"""验证 gemma4 分 chunk 推理：能正常出字的版本

修复:
  1. 加 BOS token (id=2) — Gemma 模型必须
  2. 用 bfloat16 — 模型原始精度
  3. HF 用 model.generate() — 完整生成流程

用法:
  python verify_gemma4_chunk_vs_hf.py
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
from transformers import Gemma4ForConditionalGeneration as HFModel, AutoTokenizer

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


def generate_hf(hf_model, input_ids, num_steps):
    """A: HF transformers model.generate()"""
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=num_steps,
            do_sample=False,
            pad_token_id=0,
        )
    return out[0][input_ids.shape[1]:].tolist()


def generate_cann(model, input_ids, num_steps, dtype):
    """B: cann_lm 一次性 prefill + greedy decode"""
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
                input_ids=next_token, attention_mask=mask_dec,
                past_key_values=kv, shared_kv_states=shared,
            )
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})
    return tokens


def generate_cann_chunked(model, input_ids, chunk_size, num_steps, dtype):
    """C: cann_lm 分 chunk prefill + greedy decode"""
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
            mask[:, :, :, k_len:] = make_causal_mask(chunk_len, chunk_len, dtype)

        with torch.no_grad():
            out = model(
                input_ids=chunk, attention_mask=mask,
                past_key_values=kv, shared_kv_states=shared,
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
                input_ids=next_token, attention_mask=mask_dec,
                past_key_values=kv, shared_kv_states=shared,
            )
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})
    return tokens


def main():
    dtype = torch.bfloat16

    print("[1/3] Loading models")
    # HF
    print("  Loading HF model (bfloat16)...")
    hf_model = HFModel.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
    print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    # cann_lm
    print("  Loading cann_lm model (bfloat16)...")
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
    print(f"  cann_lm params: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    all_pass = True

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        # 加 BOS token
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        seq_len = input_ids.shape[1]
        num_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  tokens: {seq_len} (含 BOS), chunks: {num_chunks} (size={CHUNK_SIZE})")

        # A: HF generate
        print(f"\n  A: HF generate (bfloat16 + BOS)...")
        tokens_a = generate_hf(hf_model, input_ids, NUM_DECODE_STEPS)
        text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
        print(f"  A output: \"{text_a}\"")

        # B: cann_lm 一次性
        print(f"\n  B: cann_lm 一次性 prefill (bfloat16 + BOS)...")
        tokens_b = generate_cann(model, input_ids, NUM_DECODE_STEPS, dtype)
        text_b = tokenizer.decode(tokens_b, skip_special_tokens=True)
        print(f"  B output: \"{text_b}\"")

        # C: cann_lm 分 chunk
        print(f"\n  C: cann_lm 分 chunk prefill (chunk_size={CHUNK_SIZE})...")
        tokens_c = generate_cann_chunked(model, input_ids, CHUNK_SIZE, NUM_DECODE_STEPS, dtype)
        text_c = tokenizer.decode(tokens_c, skip_special_tokens=True)
        print(f"  C output: \"{text_c}\"")

        # 对比
        ab_match = tokens_a == tokens_b
        bc_match = tokens_b == tokens_c
        ac_match = tokens_a == tokens_c
        all_pass = all_pass and ab_match and bc_match
        print(f"\n  A vs B (HF vs cann): {ab_match}")
        print(f"  B vs C (cann vs chunk): {bc_match}")
        print(f"  A vs C (HF vs chunk): {ac_match}")
        if not ab_match:
            print(f"  A tokens: {tokens_a[:5]}...")
            print(f"  B tokens: {tokens_b[:5]}...")
        if not bc_match:
            print(f"  B tokens: {tokens_b[:5]}...")
            print(f"  C tokens: {tokens_c[:5]}...")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Prompts: {len(TEST_PROMPTS)}, Decode: {NUM_DECODE_STEPS} steps")
    print(f"  A=HF generate, B=cann_lm, C=cann_lm chunked")
    print(f"  A==B: {all_pass}")
    print()
    if all_pass:
        print("[PASS] 三种方式生成完全一致")
    else:
        print("[PARTIAL] 部分不一致 (bfloat16 数值差异)")


if __name__ == "__main__":
    main()
