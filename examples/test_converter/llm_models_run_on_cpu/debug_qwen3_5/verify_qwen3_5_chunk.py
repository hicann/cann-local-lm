#!/usr/bin/env python3
"""验证 qwen3.5 chunk 推理一致性: chunk vs 非 chunk (+ 可选 vs HF)

单轮 chunk invariant: A=非 chunk prefill + decode vs B=chunk prefill + decode, 验证 A==B.
可选 --vs-hf 加 HF model.generate 三方对比.

qwen3.5 混合架构: full_attention SimpleKVCache + linear_attention delta_cache.
attention_mask=None: model._create_causal_mask 自动用 past_len 算 total_len.

模式:
  --mode cann (默认): cann_lm model.forward (cat KV + delta_cache)
  --mode wrapper      : Qwen3_5ExportWrapper.forward (delta box 持久)

固定: chat_template + bfloat16

用法:
  python verify_qwen3_5_chunk.py                       # cann, chunk vs 非 chunk, 三 prompt
  python verify_qwen3_5_chunk.py --vs-hf               # 加 HF 三方对比
  python verify_qwen3_5_chunk.py --chunk-size 4 --num-steps 8   # 真分 chunk
  python verify_qwen3_5_chunk.py --prompt "你好" --token-level

环境变量 (先 source 基底):
  source ../../../../scripts/cann_lm_compile/set_env_export.sh   # 或 source ../check_env.sh
  export QWEN3_5_HF_PATH=<qwen3.5 model dir>    # 必填
"""

import argparse
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    load_qwen3_5_weights,
)
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper

HF_PATH = os.environ["QWEN3_5_HF_PATH"]
DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def load_cann_model(hf_path, dtype):
    config = Qwen3_5TextConfig.from_pretrained(hf_path)
    model = Qwen3_5ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_5_weights(model, hf_path, device="cpu", dtype=dtype)
    return model, config


def encode_prompt(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


# ──────────────────────────────────────────────
# cann: model.forward (cat KV + delta_cache)
# ──────────────────────────────────────────────


def prefill_cann(model, input_ids, kv, delta, chunked=False, chunk_size=8):
    """cann prefill (chunked or not), 累积 KV + delta. 返回 (logits, kv, delta)."""
    if chunked:
        seq_len = input_ids.shape[1]
        for ci in range((seq_len + chunk_size - 1) // chunk_size):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, seq_len)
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[:, s:e],
                    attention_mask=None,
                    past_key_values=kv,
                    delta_states=delta,
                    use_cache=True,
                )
            logits, kv, delta = (
                out["logits"],
                out["past_key_values"],
                out["delta_cache"],
            )
    else:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=kv,
                delta_states=delta,
                use_cache=True,
            )
        logits, kv, delta = out["logits"], out["past_key_values"], out["delta_cache"]
    return logits, kv, delta


def decode_cann(model, logits, kv, delta, num_steps):
    """greedy decode, 累积 kv/delta. 返回 tokens."""
    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=None,
                past_key_values=kv,
                delta_states=delta,
                use_cache=True,
            )
        logits, kv, delta = out["logits"], out["past_key_values"], out["delta_cache"]
    return tokens


def run_cann(model, input_ids, num_steps, chunked=False, chunk_size=8):
    logits, kv, delta = prefill_cann(model, input_ids, None, None, chunked, chunk_size)
    return decode_cann(model, logits, kv, delta, num_steps)


# ──────────────────────────────────────────────
# wrapper: Qwen3_5ExportWrapper.forward (flat KV + delta box 持久)
# ──────────────────────────────────────────────


def prefill_wrapper(wrapper, input_ids, kv, chunked=False, chunk_size=8):
    if chunked:
        seq_len = input_ids.shape[1]
        for ci in range((seq_len + chunk_size - 1) // chunk_size):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, seq_len)
            with torch.no_grad():
                logits, *kv_flat = wrapper(
                    input_ids[:, s:e], None, None, kv, None, None, None
                )
            kv = reconstruct_kv(kv_flat)
    else:
        with torch.no_grad():
            logits, *kv_flat = wrapper(input_ids, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)
    return logits, kv


def decode_wrapper(wrapper, logits, kv, num_steps):
    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


def run_wrapper(wrapper, input_ids, num_steps, chunked=False, chunk_size=8):
    wrapper._delta_cache_box[0] = None
    logits, kv = prefill_wrapper(wrapper, input_ids, None, chunked, chunk_size)
    return decode_wrapper(wrapper, logits, kv, num_steps)


# ──────────────────────────────────────────────
# HF transformers
# ──────────────────────────────────────────────


def generate_hf(hf_model, input_ids, num_steps):
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=num_steps,
            do_sample=False,
            pad_token_id=0,
        )
    return out[0][input_ids.shape[1] :].tolist()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="qwen3.5 chunk 推理验证 (chunk vs 非 chunk + 可选 HF)"
    )
    parser.add_argument(
        "--mode",
        choices=["cann", "wrapper"],
        default="cann",
        help="cann_lm model.forward 或 Qwen3_5ExportWrapper",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="chunk prefill size (需 < seq_len 才真分 chunk)",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="decode token 数")
    parser.add_argument("--vs-hf", action="store_true", help="加 HF transformers 对比")
    parser.add_argument(
        "--prompt", type=str, default=None, help="单 prompt (默认三 prompt)"
    )
    parser.add_argument(
        "--token-level", action="store_true", help="打印 token 级详细对比"
    )
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print("[1/3] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_lm params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Layers: {config.num_hidden_layers} (layer_types: {config.layer_types[:8]}...)"
    )

    wrapper = None
    if args.mode == "wrapper":
        wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()

    hf_model = None
    if args.vs_hf:
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            HF_PATH, torch_dtype=dtype
        ).eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    all_pass = True
    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f'[2/3] Prompt: "{prompt}"')
        print(f"{'=' * 60}")
        input_ids = encode_prompt(tokenizer, [{"role": "user", "content": prompt}])
        print(f"  tokens: {input_ids.shape[1]}")

        # A: 非 chunk
        tokens_a = (
            run_cann(model, input_ids, args.num_steps)
            if args.mode == "cann"
            else run_wrapper(wrapper, input_ids, args.num_steps, chunked=False)
        )
        print(
            f'  A (非 chunk): "{tokenizer.decode(tokens_a, skip_special_tokens=True)}"'
        )

        # B: chunk
        tokens_b = (
            run_cann(
                model,
                input_ids,
                args.num_steps,
                chunked=True,
                chunk_size=args.chunk_size,
            )
            if args.mode == "cann"
            else run_wrapper(
                wrapper,
                input_ids,
                args.num_steps,
                chunked=True,
                chunk_size=args.chunk_size,
            )
        )
        print(
            f'  B (chunk):    "{tokenizer.decode(tokens_b, skip_special_tokens=True)}"'
        )

        ab = tokens_a == tokens_b
        all_pass = all_pass and ab
        print(f"\n  A==B (chunk vs 非 chunk): {ab}")

        if args.vs_hf:
            tokens_hf = generate_hf(hf_model, input_ids, args.num_steps)
            print(
                f'  HF:           "{tokenizer.decode(tokens_hf, skip_special_tokens=True)}"'
            )
            print(f"  HF==A: {tokens_hf == tokens_a}   HF==B: {tokens_hf == tokens_b}")

        if args.token_level:
            print(f"  A tokens: {tokens_a}")
            print(f"  B tokens: {tokens_b}")
            if args.vs_hf:
                print(f"  HF tokens: {tokens_hf}")

    print(f"\n{'=' * 60}")
    tag = "chunk vs 非 chunk 一致" + (" + HF 一致" if args.vs_hf else "")
    print(f"[{'PASS' if all_pass else 'PARTIAL/FAIL'}] {tag}")


if __name__ == "__main__":
    main()
