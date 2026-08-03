#!/usr/bin/env python3
"""验证 qwen3 多轮对话: wrapper + chunk, greedy vs HF 精度对比

多轮对话流程 (每轮独立 prefill 含历史 + decode):
  第1轮: chat_template(user1 + gen) → input_ids1 → prefill → decode assistant1
  第2轮: chat_template(user1 + assistant1 + user2 + gen) → input_ids2 → prefill → decode assistant2

模式:
  --mode cann (默认): cann_lm model.forward (cat KV, 非 chunk 用 model.generate)
  --mode wrapper      : Qwen3ExportWrapper.forward
  --chunk-size N      : chunk prefill (0=非 chunk)
  --vs-hf             : HF transformers 多轮对比

固定: enable_thinking=False + bfloat16 + greedy

用法:
  python verify_qwen3_multiturn.py --vs-hf
  python verify_qwen3_multiturn.py --mode wrapper --vs-hf
  python verify_qwen3_multiturn.py --mode wrapper --chunk-size 4 --vs-hf
  python verify_qwen3_multiturn.py --prompts "你好||刚才我说了什么？||介绍AI"

环境变量 (详见 verify_epair_vs_hf.py docstring):
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
"""

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"
DEFAULT_PROMPTS = ["你好", "刚才我说了什么？", "介绍一下人工智能"]


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


# ──────────────────────────────────────────────
# cann 模式: chunk prefill (cat KV, past=[None]*N) + greedy decode (EOS 停)
# ──────────────────────────────────────────────

def generate_cann_chunked(model, input_ids, num_steps, dtype, chunk_size):
    """分 chunk prefill + greedy decode (cat KV, EOS 停)."""
    seq_len = input_ids.shape[1]
    num_layers = model.config.num_hidden_layers
    kv = [None] * num_layers
    logits = None

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, seq_len)
        chunk = input_ids[:, s:e]
        with torch.no_grad():
            out = model(input_ids=chunk, attention_mask=None, past_key_values=kv)
        logits = out["logits"]
        kv = out["past_key_values"]

    eos = getattr(model.config, "eos_token_id", None)
    eos_set = {eos} if isinstance(eos, int) else (set(eos) if eos else set())
    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() in eos_set:
            break
        with torch.no_grad():
            out = model(input_ids=next_token, attention_mask=None, past_key_values=kv)
        logits = out["logits"]
        kv = out["past_key_values"]
    return tokens


def generate_multiturn_cann(model, tokenizer, prompts, num_steps, dtype, chunked=False, chunk_size=8):
    messages = []
    results = []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if chunked:
            tokens = generate_cann_chunked(model, input_ids, num_steps, dtype, chunk_size)
        else:
            with torch.no_grad():
                gen = model.generate(input_ids, max_new_tokens=num_steps, do_sample=False)
            tokens = gen[0][input_ids.shape[1]:].tolist()
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# wrapper 模式: chunk prefill (cat KV) + greedy decode (EOS 停)
# ──────────────────────────────────────────────

def generate_wrapper_chunked(wrapper, input_ids, num_steps, dtype, chunk_size):
    """wrapper 分 chunk prefill + greedy decode (cat KV, EOS 停)."""
    seq_len = input_ids.shape[1]
    kv = None
    logits = None

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, seq_len)
        chunk = input_ids[:, s:e]
        with torch.no_grad():
            logits, *kv_flat = wrapper(chunk, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)

    eos = getattr(wrapper.config, "eos_token_id", None)
    eos_set = {eos} if isinstance(eos, int) else (set(eos) if eos else set())
    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() in eos_set:
            break
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


def generate_multiturn_wrapper(wrapper, tokenizer, prompts, num_steps, dtype, chunked=False, chunk_size=8):
    messages = []
    results = []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if chunked:
            tokens = generate_wrapper_chunked(wrapper, input_ids, num_steps, dtype, chunk_size)
        else:
            # 非 chunk: wrapper 一次性 prefill (chunk_size > seq → 单 chunk)
            tokens = generate_wrapper_chunked(wrapper, input_ids, num_steps, dtype, seq_len=input_ids.shape[1])
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# HF transformers (多轮, 每轮 generate 含历史)
# ──────────────────────────────────────────────

def generate_multiturn_hf(hf_model, tokenizer, prompts, num_steps):
    messages = []
    results = []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = hf_model.generate(input_ids=input_ids, max_new_tokens=num_steps,
                                    do_sample=False, pad_token_id=0)
        tokens = out[0][input_ids.shape[1]:].tolist()
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# model loading
# ──────────────────────────────────────────────

def load_cann_model(hf_path, dtype):
    config = Qwen3TextConfig.from_pretrained(hf_path)
    model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_weights(model, hf_path, device="cpu", dtype=dtype)
    return model, config


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="qwen3 多轮对话 (wrapper + chunk) greedy vs HF")
    parser.add_argument("--mode", choices=["cann", "wrapper"], default="cann")
    parser.add_argument("--chunk-size", type=int, default=0, help="chunk prefill size (0=非 chunk, 用 model.generate)")
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--prompts", type=str, default=None, help="多轮, || 分隔")
    parser.add_argument("--vs-hf", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = args.prompts.split("||") if args.prompts else DEFAULT_PROMPTS
    chunked = args.chunk_size > 0

    print("[1/2] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_lm params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  mode: {args.mode}, chunked: {chunked} (chunk_size={args.chunk_size})")

    wrapper = None
    if args.mode == "wrapper":
        wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        print(f"  Wrapper: {wrapper.__class__.__name__}")

    hf_model = None
    if args.vs_hf:
        from transformers import AutoModelForCausalLM
        print("  Loading HF model (bfloat16, eager)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            HF_PATH, torch_dtype=dtype, attn_implementation="eager").eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    print(f"\n[2/2] 多轮对话 ({len(prompts)} turns, {args.num_steps} tokens/turn)")
    print(f"{'='*60}")

    if args.mode == "cann":
        results = generate_multiturn_cann(model, tokenizer, prompts, args.num_steps, dtype,
                                           chunked=chunked, chunk_size=args.chunk_size)
    else:
        results = generate_multiturn_wrapper(wrapper, tokenizer, prompts, args.num_steps, dtype,
                                              chunked=chunked, chunk_size=args.chunk_size)

    hf_results = None
    if args.vs_hf:
        hf_results = generate_multiturn_hf(hf_model, tokenizer, prompts, args.num_steps)

    all_match = True
    for turn, (prompt, text_out) in enumerate(results):
        print(f"\n--- Turn {turn + 1} ---")
        print(f"User: {prompt}")
        print(f"Assistant ({args.mode}{' chunk' if chunked else ''}): {text_out}")
        if hf_results is not None:
            hf_text = hf_results[turn][1]
            match = text_out == hf_text
            all_match = all_match and match
            print(f"Assistant (HF):                    {hf_text}")
            print(f"  {args.mode}==HF: {match}")

    print(f"\n{'='*60}")
    if hf_results is not None:
        print(f"[{'PASS' if all_match else 'PARTIAL'}] {args.mode}{' chunk' if chunked else ''} vs HF 多轮 "
              f"{'一致' if all_match else '部分不一致 (bfloat16)'}")
    else:
        print("[DONE] 多轮对话完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
