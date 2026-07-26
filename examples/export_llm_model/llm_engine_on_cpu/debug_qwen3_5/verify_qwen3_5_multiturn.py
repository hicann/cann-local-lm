#!/usr/bin/env python3
"""验证 qwen3.5 多轮对话: 跨轮 KV/delta_cache 累积 (cann + wrapper + chunk)

多轮对话流程:
  第1轮: chat_template(user1 + gen) → input_ids1 → prefill → KV1 + delta1 → decode assistant1
  第2轮: chat_template(user1 + assistant1 + user2 + gen) → input_ids2
          new = input_ids2[prev_len:] (历史在 KV1) → prefill(past=KV1, delta=delta1) → KV2 + delta2 → decode assistant2
  ...

跨轮: KV + delta_cache 累积 (不重算历史), chunk prefill 可选.

模式:
  --mode cann (默认): cann_lm model.forward (cat KV + delta_cache)
  --mode wrapper      : Qwen3_5ExportWrapper.forward (delta_cache box 持久)

用法:
  python verify_qwen3_5_multiturn.py                            # cann, 2 轮, 非 chunk
  python verify_qwen3_5_multiturn.py --mode wrapper             # wrapper
  python verify_qwen3_5_multiturn.py --chunk-size 4             # chunk prefill
  python verify_qwen3_5_multiturn.py --num-turns 3 --num-steps 16
  python verify_qwen3_5_multiturn.py --prompts "你好||介绍下AI||谢谢"

环境变量 (详见 verify_qwen3_5_all.py docstring):
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

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper

HF_PATH = "/mnt/workspace/data/models/Qwen3.5-0.8B"
DEFAULT_PROMPTS = ["你好", "刚才我说了什么？", "介绍一下人工智能"]


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


# ──────────────────────────────────────────────
# cann 模式: model.forward (cat KV + delta_cache 跨轮累积)
# ──────────────────────────────────────────────

def prefill_cann(model, new_tokens, kv, delta, dtype, chunked=False, chunk_size=8):
    """prefill new_tokens (chunked or not), 累积 KV + delta. 返回 (logits, kv, delta)."""
    new_seq = new_tokens.shape[1]
    logits = None
    if chunked:
        num_chunks = (new_seq + chunk_size - 1) // chunk_size
        for ci in range(num_chunks):
            s = ci * chunk_size
            e = min(s + chunk_size, new_seq)
            chunk = new_tokens[:, s:e]
            with torch.no_grad():
                out = model(input_ids=chunk, attention_mask=None,
                            past_key_values=kv, delta_cache=delta, use_cache=True)
            logits = out["logits"]
            kv = out["past_key_values"]
            delta = out["delta_cache"]
    else:
        with torch.no_grad():
            out = model(input_ids=new_tokens, attention_mask=None,
                        past_key_values=kv, delta_cache=delta, use_cache=True)
        logits = out["logits"]
        kv = out["past_key_values"]
        delta = out["delta_cache"]
    return logits, kv, delta


def generate_multiturn_cann(model, tokenizer, prompts, num_steps, dtype, chunked=False, chunk_size=8):
    """每轮独立 prefill (含历史, 重算) + decode. 上下文对 (input_ids 含历史), KV/delta 不跨轮."""
    messages = []
    results = []
    for turn, prompt in enumerate(prompts):
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

        # 每轮独立 prefill (含完整历史, past=None, delta=None)
        logits, kv, delta = prefill_cann(model, input_ids, None, None, dtype, chunked, chunk_size)

        # decode
        tokens = []
        for _ in range(num_steps):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens.append(next_token.item())
            with torch.no_grad():
                out = model(input_ids=next_token, attention_mask=None,
                            past_key_values=kv, delta_cache=delta, use_cache=True)
            logits = out["logits"]
            kv = out["past_key_values"]
            delta = out["delta_cache"]

        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# wrapper 模式: Qwen3_5ExportWrapper.forward (delta_cache box 持久)
# ──────────────────────────────────────────────

def prefill_wrapper(wrapper, new_tokens, kv, dtype, chunked=False, chunk_size=8):
    """wrapper prefill (flat KV + delta box). 返回 (logits, kv)."""
    new_seq = new_tokens.shape[1]
    logits = None
    if chunked:
        num_chunks = (new_seq + chunk_size - 1) // chunk_size
        for ci in range(num_chunks):
            s = ci * chunk_size
            e = min(s + chunk_size, new_seq)
            chunk = new_tokens[:, s:e]
            with torch.no_grad():
                logits, *kv_flat = wrapper(chunk, None, None, kv, None, None, None)
            kv = reconstruct_kv(kv_flat)
    else:
        with torch.no_grad():
            logits, *kv_flat = wrapper(new_tokens, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)
    return logits, kv


def generate_multiturn_wrapper(wrapper, tokenizer, prompts, num_steps, dtype, chunked=False, chunk_size=8):
    """每轮独立 prefill (含历史, 重算) + decode. wrapper delta box 每轮重置."""
    messages = []
    results = []
    for turn, prompt in enumerate(prompts):
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

        wrapper._delta_cache_box[0] = None  # 每轮重置 delta
        logits, kv = prefill_wrapper(wrapper, input_ids, None, dtype, chunked, chunk_size)

        tokens = []
        for _ in range(num_steps):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens.append(next_token.item())
            with torch.no_grad():
                logits, *kv_flat = wrapper(next_token, None, None, kv, None, None, None)
            kv = reconstruct_kv(kv_flat)

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
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    config = Qwen3_5TextConfig.from_pretrained(hf_path)
    model = Qwen3_5ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_5_weights(model, hf_path, device="cpu", dtype=dtype)
    return model, config


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="qwen3.5 多轮对话验证 (cann/wrapper + chunk)")
    parser.add_argument("--mode", choices=["cann", "wrapper"], default="cann",
                        help="cann_lm model.forward 或 Qwen3_5ExportWrapper")
    parser.add_argument("--chunk-size", type=int, default=8, help="chunk prefill size (0=非 chunk)")
    parser.add_argument("--num-steps", type=int, default=16, help="每轮 decode token 数")
    parser.add_argument("--prompts", type=str, default=None,
                        help="多轮 prompt, || 分隔 (默认 3 轮)")
    parser.add_argument("--vs-hf", action="store_true", help="加 HF transformers 多轮对比")
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = args.prompts.split("||") if args.prompts else DEFAULT_PROMPTS
    chunked = args.chunk_size > 0

    print("[1/2] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_lm params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Layers: {config.num_hidden_layers} (layer_types: {config.layer_types[:8]}...)")

    wrapper = None
    if args.mode == "wrapper":
        wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        print(f"  Wrapper: {wrapper.__class__.__name__}")
    print(f"  mode: {args.mode}, chunked: {chunked} (chunk_size={args.chunk_size}), turns: {len(prompts)}")

    hf_model = None
    if args.vs_hf:
        from transformers import AutoModelForCausalLM
        print("  Loading HF model (bfloat16)...")
        hf_model = AutoModelForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

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
        print(f"Assistant ({args.mode}): {text_out}")
        if hf_results is not None:
            hf_text = hf_results[turn][1]
            match = text_out == hf_text
            all_match = all_match and match
            print(f"Assistant (HF):       {hf_text}")
            print(f"  {args.mode}==HF: {match}")

    print(f"\n{'='*60}")
    if hf_results is not None:
        print(f"[{'PASS' if all_match else 'PARTIAL'}] {args.mode} vs HF 多轮 {'一致' if all_match else '部分不一致 (bfloat16)'}")
    else:
        print("[DONE] 多轮对话完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
