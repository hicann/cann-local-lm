#!/usr/bin/env python3
"""用 ExportWrapper 导出 epair → 回环推理，验证三个 prompt

流程:
  1. 加载 gemma4 模型 + Gemma4ExportWrapper
  2. torch.export → ExportedProgram
  3. ep.module() 回环推理 (非分 chunk + 分 chunk)
  4. 对比文本输出

用法:
  python verify_gemma4_epair_chunk.py
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
from cann_lm.models.fake_ops import register_ge_converters
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
    """4D causal mask (no padding)"""
    return make_causal_mask_with_padding(q_len, k_len, q_len, k_len, dtype)


def make_causal_mask_with_padding(q_len, k_len, real_q_len, real_k_len, dtype):
    """4D causal mask + padding mask

    位置 0~real_q_len-1: 正常 causal (下三角 attend, 上三角 mask)
    位置 real_q_len~q_len-1: padding → 全部 mask 为 -inf
    key 位置 real_k_len~k_len-1: padding → 全部 mask 为 -inf
    """
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    # 屏蔽 padding query 位置
    if real_q_len < q_len:
        mask[real_q_len:, :] = min_val
    # 屏蔽 padding key 位置
    if real_k_len < k_len:
        mask[:, real_k_len:] = min_val
    return mask.unsqueeze(0).unsqueeze(0)


def make_decode_mask(k_len, dtype):
    return torch.zeros(1, 1, 1, k_len, dtype=dtype)


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def export_wrapper(model, seq_len, dtype):
    """导出 wrapper 为 ExportedProgram"""
    register_ge_converters()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=seq_len)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    return ep, wrapper


def generate_ep(ep, wrapper, sample_inputs, input_ids_padded, real_seq_len, num_steps, dtype):
    """A: ep.module() prefill (padded to traced seq_len) + wrapper decode

    ep.module() 的 guard 要求 input shape 和追踪时一致 (seq_len=8)。
    所以 input_ids 必须 pad 到 export_len。
    output_pos 设为 real_seq_len - 1 (最后一个有效 token 的位置)。
    """
    module = ep.module()
    traced_inputs = list(sample_inputs)
    traced_inputs[0] = input_ids_padded  # 替换 input_ids (padded)
    traced_inputs[1] = make_causal_mask_with_padding(
        input_ids_padded.shape[1], input_ids_padded.shape[1],
        real_seq_len, real_seq_len, dtype)  # 带 padding 屏蔽的 4D mask
    traced_inputs[6] = torch.tensor([real_seq_len - 1], dtype=torch.int32)  # output_pos = 最后有效位置

    with torch.no_grad():
        out = module(*traced_inputs)
        logits = out[0]
        kv_flat = list(out[1:])
    kv = reconstruct_kv(kv_flat)

    # Decode: 用 wrapper 直接调用 (ep.module() 无法处理非 None 的 past_key_values)
    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = input_ids_padded.shape[1] + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, None, kv,
                torch.tensor([cached_len - 1], dtype=torch.long), None, None,
            )
        kv = reconstruct_kv(kv_flat)
    return tokens


def generate_wrapper_direct(wrapper, input_ids, num_steps, dtype):
    """B: wrapper 直接调用 (对比基准)"""
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


def main():
    dtype = torch.bfloat16

    print("[1/4] Loading model + tokenizer")
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
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    # 找最长的 prompt 序列长度用于导出
    max_len = max(len([BOS_TOKEN_ID] + tokenizer.encode(p, add_special_tokens=False)) for p in TEST_PROMPTS)
    export_len = ((max_len + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE  # 对齐到 chunk_size
    print(f"  Export seq_len: {export_len} (max prompt={max_len})")

    print(f"\n[2/4] Exporting wrapper → ExportedProgram (seq_len={export_len})")
    register_ge_converters()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=export_len)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    print(f"  Graph nodes: {len(ep.graph.nodes)}")
    print(f"  Export OK")

    all_pass = True

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"[3/4] Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Pad 到 export_len
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        seq_len = input_ids.shape[1]
        print(f"  tokens: {seq_len} (含 BOS), padded to {export_len}")

        # Pad input_ids 到 export_len (用 pad_token_id=0)
        if seq_len < export_len:
            pad_len = export_len - seq_len
            input_ids_padded = torch.cat([
                input_ids,
                torch.zeros(1, pad_len, dtype=torch.long),
            ], dim=1)
        else:
            input_ids_padded = input_ids

        # A: ep.module() prefill + wrapper decode
        print(f"\n  A: ep.module() prefill + wrapper decode...")
        try:
            tokens_a = generate_ep(ep, wrapper, sample_inputs, input_ids_padded, seq_len, NUM_DECODE_STEPS, dtype)
            text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
            print(f"  A output: \"{text_a}\"")
        except Exception as e:
            print(f"  A FAILED: {e}")
            tokens_a = None
            text_a = f"ERROR: {e}"

        # B: wrapper 直接调用 (基准, 不 padding)
        print(f"\n  B: wrapper 直接调用 (seq_len={seq_len})...")
        tokens_b = generate_wrapper_direct(wrapper, input_ids, NUM_DECODE_STEPS, dtype)
        text_b = tokenizer.decode(tokens_b, skip_special_tokens=True)
        print(f"  B output: \"{text_b}\"")

        # 对比
        if tokens_a is not None:
            match = tokens_a == tokens_b
            all_pass = all_pass and match
            print(f"\n  A vs B match: {match}")
            if not match:
                print(f"  A tokens: {tokens_a[:5]}...")
                print(f"  B tokens: {tokens_b[:5]}...")
        else:
            all_pass = False

    print(f"\n{'='*60}")
    print(f"[4/4] Summary: epair ep.module() vs wrapper direct")
    print(f"{'='*60}")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"  Decode: {NUM_DECODE_STEPS} steps")
    print(f"  dtype: bfloat16")
    print(f"  All match: {all_pass}")
    print()
    if all_pass:
        print("[PASS] epair 回环推理与 wrapper 直接调用一致")
    else:
        print("[PARTIAL] 部分不一致")


if __name__ == "__main__":
    main()
