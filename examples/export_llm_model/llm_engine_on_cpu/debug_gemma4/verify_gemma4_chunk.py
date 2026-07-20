#!/usr/bin/env python3
"""验证 gemma4 分 chunk 推理：多轮 decode 对比 chunk vs 非_chunk

流程:
  A. 非分 chunk: 一次性处理完整序列 → 逐 token decode
  B. 分 chunk:   将 prefill 分成 2 个 chunk → 逐 token decode
  对比: 生成的 token 序列是否一致

用法:
  python verify_gemma4_chunk.py
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
NUM_DECODE_STEPS = 3
SEQ_LEN = 8
CHUNK_SIZE = 4  # prefill 分成 2 个 chunk


def make_causal_mask(q_len, k_len, dtype):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_decode_mask(k_len, dtype):
    return torch.zeros(1, 1, 1, k_len, dtype=dtype)


def main():
    dtype = torch.float16

    print(f"[1/5] Loading gemma4 model")
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
    print(f"      Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"      Layers: {config.num_hidden_layers} (shared: {getattr(config, 'num_kv_shared_layers', 0)})")

    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    print(f"      input_ids: {input_ids.tolist()}")
    print(f"      seq_len={SEQ_LEN}, chunk_size={CHUNK_SIZE} → 2 chunks")

    # === A. 非分 chunk: 一次性 prefill ===
    print(f"\n[2/5] A: 一次性 prefill (seq_len={SEQ_LEN})")
    mask_full = make_causal_mask(SEQ_LEN, SEQ_LEN, dtype)
    with torch.no_grad():
        out_a = model(
            input_ids=input_ids,
            attention_mask=mask_full,
            shared_kv_states={},
        )
        logits_a = out_a["logits"]
        kv_a = out_a["past_key_values"]
        shared_a = out_a.get("shared_kv_states", {})

    token_a = torch.argmax(logits_a[:, -1, :], dim=-1).item()
    print(f"      first token: {token_a}")

    # === B. 分 chunk prefill ===
    print(f"\n[3/5] B: 分 chunk prefill (chunk_size={CHUNK_SIZE})")
    chunk1 = input_ids[:, :CHUNK_SIZE]
    chunk2 = input_ids[:, CHUNK_SIZE:]

    # Chunk 1
    mask1 = make_causal_mask(CHUNK_SIZE, CHUNK_SIZE, dtype)
    with torch.no_grad():
        out1 = model(
            input_ids=chunk1,
            attention_mask=mask1,
            shared_kv_states={},
        )
        kv1 = out1["past_key_values"]
        shared1 = out1.get("shared_kv_states", {})

    print(f"      chunk1 done, kv layers: {sum(1 for x in kv1 if x is not None)}")

    # Chunk 2: 用 chunk1 的 KV cache
    k_len = CHUNK_SIZE
    total_len = CHUNK_SIZE * 2
    # mask: chunk2 的 query 可以看到 chunk1 + chunk2 的所有 key
    mask2 = torch.zeros(1, 1, CHUNK_SIZE, total_len, dtype=dtype)
    # causal: chunk2 的位置不能看到 chunk2 后面的
    causal_part = make_causal_mask(CHUNK_SIZE, CHUNK_SIZE, dtype)
    mask2[:, :, :, CHUNK_SIZE:] = causal_part

    with torch.no_grad():
        out2 = model(
            input_ids=chunk2,
            attention_mask=mask2,
            past_key_values=kv1,
            shared_kv_states=shared1,
        )
        logits_b = out2["logits"]
        kv_b = out2["past_key_values"]
        shared_b = out2.get("shared_kv_states", {})

    token_b = torch.argmax(logits_b[:, -1, :], dim=-1).item()
    print(f"      chunk2 done, first token: {token_b}")

    # === 对比 prefill ===
    print(f"\n[4/5] 对比 prefill 结果")
    diff_prefill = (logits_b[:, -1:, :] - logits_a[:, -1:, :]).abs().max().item()
    match_prefill = torch.allclose(logits_b[:, -1:, :], logits_a[:, -1:, :], atol=1e-3)
    print(f"  logits max_diff: {diff_prefill:.6e}")
    print(f"  allclose(atol=1e-3): {match_prefill}")
    print(f"  token match: {token_a == token_b} (A={token_a}, B={token_b})")

    # KV cache 对比
    kv_match = True
    max_kv_diff = 0.0
    for i in range(len(kv_a)):
        if kv_a[i] is None and kv_b[i] is None:
            continue
        if kv_a[i] is None or kv_b[i] is None:
            print(f"  layer {i}: one is None (a={kv_a[i] is not None}, b={kv_b[i] is not None})")
            kv_match = False
            continue
        k_a, v_a = kv_a[i]
        k_b, v_b = kv_b[i]
        if k_a.shape != k_b.shape:
            print(f"  layer {i}: shape mismatch a={list(k_a.shape)} b={list(k_b.shape)}")
            kv_match = False
            continue
        k_diff = (k_b - k_a).abs().max().item()
        v_diff = (v_b - v_a).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)
        if not torch.allclose(k_b, k_a, atol=1e-3) or not torch.allclose(v_b, v_a, atol=1e-3):
            kv_match = False
    print(f"  KV cache all_match: {kv_match} (max_diff={max_kv_diff:.6e})")

    # === 多轮 decode 对比 ===
    print(f"\n[5/5] 多轮 decode ({NUM_DECODE_STEPS} steps)")
    tokens_a = [token_a]
    tokens_b = [token_b]
    current_kv_a = kv_a
    current_shared_a = shared_a
    current_kv_b = kv_b
    current_shared_b = shared_b
    all_decode_match = match_prefill and kv_match

    for step in range(NUM_DECODE_STEPS):
        next_a = torch.tensor([[tokens_a[-1]]], dtype=torch.long)
        next_b = torch.tensor([[tokens_b[-1]]], dtype=torch.long)
        cached_len_a = SEQ_LEN + step
        cached_len_b = SEQ_LEN + step
        mask_dec = make_decode_mask(cached_len_a + 1, dtype)

        with torch.no_grad():
            out_a = model(
                input_ids=next_a,
                attention_mask=mask_dec,
                past_key_values=current_kv_a,
                shared_kv_states=current_shared_a,
            )
            logits_a = out_a["logits"]
            current_kv_a = out_a["past_key_values"]
            current_shared_a = out_a.get("shared_kv_states", {})

            out_b = model(
                input_ids=next_b,
                attention_mask=mask_dec,
                past_key_values=current_kv_b,
                shared_kv_states=current_shared_b,
            )
            logits_b = out_b["logits"]
            current_kv_b = out_b["past_key_values"]
            current_shared_b = out_b.get("shared_kv_states", {})

        token_a = torch.argmax(logits_a[:, -1, :], dim=-1).item()
        token_b = torch.argmax(logits_b[:, -1, :], dim=-1).item()
        tokens_a.append(token_a)
        tokens_b.append(token_b)
        step_diff = (logits_b - logits_a).abs().max().item()
        step_match = torch.allclose(logits_b, logits_a, atol=1e-3)
        all_decode_match = all_decode_match and step_match and (token_a == token_b)
        print(f"  step {step}: A={token_a} B={token_b} diff={step_diff:.6e} match={step_match and token_a == token_b}")

    # === Summary ===
    print(f"\n{'='*60}")
    print("Summary: gemma4 chunk vs non-chunk (multi-decode)")
    print(f"{'='*60}")
    print(f"  Prefill: diff={diff_prefill:.6e} match={match_prefill} token={token_a == token_b}")
    print(f"  KV cache: match={kv_match} max_diff={max_kv_diff:.6e}")
    print(f"  Generated (non-chunk): {tokens_a}")
    print(f"  Generated (chunk):     {tokens_b}")
    print(f"  Tokens match: {tokens_a == tokens_b}")
    print()
    if tokens_a == tokens_b:
        print("[PASS] Chunk 推理生成结果与非 chunk 一致")
    else:
        print("[FAIL] Token 序列不一致")
    if all_decode_match:
        print("[PASS] Logits 全部在容差范围内")


if __name__ == "__main__":
    main()
