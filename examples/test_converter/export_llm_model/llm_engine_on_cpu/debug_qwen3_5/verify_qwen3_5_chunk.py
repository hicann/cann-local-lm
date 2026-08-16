#!/usr/bin/env python3
"""验证 qwen3.5 分 chunk 推理: chunk vs 非 chunk (+ 可选 vs HF)

参考 gemma4 verify_gemma4_chunk.py, 适配 qwen3.5 (full_attn SimpleKVCache + linear_attn delta_cache).

qwen3.5 model.forward 的 _create_causal_mask 自动用 past_len 算 total_len,
故 attention_mask=None 即可 (不需手动构造 chunk mask).

模式:
  --mode cann (默认): cann_lm model.forward (cat KV + delta_cache 累积)
  --vs-hf: 加 HF transformers 对比

固定: chat_template + bfloat16

用法:
  python verify_qwen3_5_chunk.py                       # cann, chunk vs 非 chunk, 三 prompt
  python verify_qwen3_5_chunk.py --vs-hf               # 加 HF 对比
  python verify_qwen3_5_chunk.py --chunk-size 4 --num-steps 8
  python verify_qwen3_5_chunk.py --prompt "你好"

环境变量 (详见 verify_qwen3_5_all.py docstring):
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper
from cann_lm.models.qwen3_5.modeling_qwen3_5 import SimpleKVCache, SimpleDeltaNetCache

HF_PATH = "/mnt/workspace/data/models/Qwen3.5-0.8B"
DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


# ──────────────────────────────────────────────
# cann_lm model.forward (cat KV + delta_cache)
# ──────────────────────────────────────────────

def generate_cann(model, input_ids, num_steps, dtype, chunked=False, chunk_size=8):
    """cann_lm 一次性/分 chunk prefill + greedy decode.

    attention_mask=None: model _create_causal_mask 自动构造 causal (用 past_len 算 total_len).
    delta_cache 跨 chunk 累积 (linear attention recurrent state).
    """
    seq_len = input_ids.shape[1]
    kv = None
    delta = None
    logits = None

    if chunked:
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            with torch.no_grad():
                out = model(
                    input_ids=chunk,
                    attention_mask=None,
                    past_key_values=kv,
                    delta_cache=delta,
                    use_cache=True,
                )
            logits = out["logits"]
            kv = out["past_key_values"]
            delta = out["delta_cache"]
    else:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=kv,
                delta_cache=delta,
                use_cache=True,
            )
        logits = out["logits"]
        kv = out["past_key_values"]
        delta = out["delta_cache"]

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=None,
                past_key_values=kv,
                delta_cache=delta,
                use_cache=True,
            )
        logits = out["logits"]
        kv = out["past_key_values"]
        delta = out["delta_cache"]
    return tokens


# ──────────────────────────────────────────────
# wrapper (Qwen3_5ExportWrapper.forward, cat KV + delta_cache box 持久)
# ──────────────────────────────────────────────

def reconstruct_kv(flat_kv):
    """展平 [k0,v0,...] → [(k0,v0),...] (wrapper 用)."""
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def generate_wrapper(wrapper, input_ids, num_steps, dtype, chunked=False, chunk_size=8):
    """wrapper 一次性/分 chunk prefill + greedy decode.

    attention_mask=None + position_ids=None: model 自动构造 causal (用 past_len).
    delta_cache 由 wrapper _delta_cache_box 持久 (跨 chunk/decode 累积).
    past_key_values=list[(k,v)] (full_attn 6 对, reconstruct_kv 重建).
    """
    seq_len = input_ids.shape[1]
    kv = None
    logits = None
    wrapper._delta_cache_box[0] = None  # 重置 delta (跨调用)

    if chunked:
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            with torch.no_grad():
                logits, *kv_flat = wrapper(chunk, None, None, kv, None, None, None)
            kv = reconstruct_kv(kv_flat)
    else:
        with torch.no_grad():
            logits, *kv_flat = wrapper(input_ids, None, None, None, None, None, None)
        kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


# ──────────────────────────────────────────────
# HF transformers
# ──────────────────────────────────────────────

def generate_hf(hf_model, input_ids, num_steps):
    """HF model.generate greedy."""
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=input_ids, max_new_tokens=num_steps,
            do_sample=False, pad_token_id=0)
    return out[0][input_ids.shape[1]:].tolist()


# ──────────────────────────────────────────────
# model loading
# ──────────────────────────────────────────────

def load_cann_model(hf_path, dtype):
    config = Qwen3_5TextConfig.from_pretrained(hf_path)
    model = Qwen3_5ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_5_weights(model, hf_path, device="cpu", dtype=dtype)
    return model, config


def encode_prompt(tokenizer, prompt):
    """用 chat_template 编码 (对齐 run_qwen3_5)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return input_ids


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="qwen3.5 chunk 推理验证 (chunk vs 非 chunk + 可选 HF)")
    parser.add_argument("--mode", choices=["cann", "wrapper"], default="cann",
                        help="cann_lm model.forward (cat KV + delta_cache) 或 Qwen3_5ExportWrapper")
    parser.add_argument("--vs-hf", action="store_true", help="加 HF transformers 对比")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None, help="单 prompt (默认三 prompt)")
    parser.add_argument("--token-level", action="store_true", help="打印 token 级详细对比")
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print("[1/3] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_lm params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Layers: {config.num_hidden_layers} (layer_types: {config.layer_types[:8]}...)")

    wrapper = None
    if args.mode == "wrapper":
        wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        print(f"  Wrapper: {wrapper.__class__.__name__}")

    hf_model = None
    if args.vs_hf:
        from transformers import AutoModelForCausalLM
        print("  Loading HF model (bfloat16)...")
        hf_model = AutoModelForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    all_pass = True
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"[2/3] Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        input_ids = encode_prompt(tokenizer, prompt)
        seq_len = input_ids.shape[1]
        num_chunks = (seq_len + args.chunk_size - 1) // args.chunk_size
        print(f"  tokens: {seq_len}, chunks: {num_chunks} (size={args.chunk_size})")

        # A: 非 chunk
        if args.mode == "cann":
            tokens_a = generate_cann(model, input_ids, args.num_steps, dtype)
        else:
            tokens_a = generate_wrapper(wrapper, input_ids, args.num_steps, dtype)
        text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
        print(f"\n  A (非 chunk): \"{text_a}\"")

        # B: chunk
        if args.mode == "cann":
            tokens_b = generate_cann(model, input_ids, args.num_steps, dtype,
                                     chunked=True, chunk_size=args.chunk_size)
        else:
            tokens_b = generate_wrapper(wrapper, input_ids, args.num_steps, dtype,
                                        chunked=True, chunk_size=args.chunk_size)
        text_b = tokenizer.decode(tokens_b, skip_special_tokens=True)
        print(f"  B (chunk):    \"{text_b}\"")

        ab_match = tokens_a == tokens_b
        all_pass = all_pass and ab_match
        print(f"\n  A==B (chunk vs 非 chunk): {ab_match}")

        if args.vs_hf:
            tokens_hf = generate_hf(hf_model, input_ids, args.num_steps)
            text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
            print(f"  HF:           \"{text_hf}\"")
            hf_a = tokens_hf == tokens_a
            hf_b = tokens_hf == tokens_b
            all_pass = all_pass and hf_a
            print(f"  HF==A: {hf_a}   HF==B: {hf_b}")

        if args.token_level:
            print(f"  A tokens: {tokens_a}")
            print(f"  B tokens: {tokens_b}")
            if args.vs_hf:
                print(f"  HF tokens: {tokens_hf}")

    print(f"\n{'='*60}")
    print(f"[3/3] Summary: chunk_size={args.chunk_size}, steps={args.num_steps}, vs_hf={args.vs_hf}")
    print(f"{'='*60}")
    if all_pass:
        print("[PASS] chunk 推理与非 chunk 一致" + (" + HF 一致" if args.vs_hf else ""))
    else:
        print("[FAIL/PARTIAL] 存在不一致 (bfloat16 数值差异或 bug)")


if __name__ == "__main__":
    main()
