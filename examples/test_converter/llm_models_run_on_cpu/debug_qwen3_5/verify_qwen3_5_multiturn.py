#!/usr/bin/env python3
"""验证 qwen3.5 多轮对话: 跨轮上下文 (cann/wrapper + chunk + vs HF)

多轮对话: 每轮独立 prefill(含历史, 重算) + decode. 上下文对(input_ids 含历史), KV/delta 不跨轮.
qwen3.5 混合架构: full_attention SimpleKVCache + linear_attention delta_cache.
attention_mask=None: model._create_causal_mask 自动用 past_len 算 total_len.

模式:
  --mode cann (默认): cann_local_lm model.forward (cat KV + delta_cache)
  --mode wrapper      : Qwen3_5ExportWrapper.forward (delta box 持久)

固定: chat_template + bfloat16

用法:
  python verify_qwen3_5_multiturn.py                            # cann, 3 轮, 非 chunk
  python verify_qwen3_5_multiturn.py --mode wrapper             # wrapper
  python verify_qwen3_5_multiturn.py --chunk-size 4             # chunk prefill
  python verify_qwen3_5_multiturn.py --vs-hf --num-steps 16
  python verify_qwen3_5_multiturn.py --prompts "你好||刚才我说了什么？||介绍AI"

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
    p for p in Path(__file__).resolve().parents if (p / "cann_local_lm").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_local_lm.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    load_qwen3_5_weights,
)
from cann_local_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper

HF_PATH = os.environ["QWEN3_5_HF_PATH"]
DEFAULT_TURNS = ["你好", "刚才我说了什么？", "介绍一下人工智能"]


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


def run_multiturn_cann(
    model, tokenizer, prompts, num_steps, chunked=False, chunk_size=8
):
    """每轮独立 prefill(含历史, 重算) + decode. 上下文对(input_ids 含历史), KV/delta 不跨轮."""
    messages, results = [], []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        input_ids = encode_prompt(tokenizer, messages)
        logits, kv, delta = prefill_cann(
            model, input_ids, None, None, chunked, chunk_size
        )
        tokens = decode_cann(model, logits, kv, delta, num_steps)
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


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


def run_multiturn_wrapper(
    wrapper, tokenizer, prompts, num_steps, chunked=False, chunk_size=8
):
    """每轮独立 prefill(含历史, 重算) + decode, 每轮重置 delta box."""
    messages, results = [], []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        input_ids = encode_prompt(tokenizer, messages)
        wrapper._delta_cache_box[0] = None
        logits, kv = prefill_wrapper(wrapper, input_ids, None, chunked, chunk_size)
        tokens = decode_wrapper(wrapper, logits, kv, num_steps)
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# HF transformers (多轮, 每轮 generate 含历史)
# ──────────────────────────────────────────────


def run_multiturn_hf(hf_model, tokenizer, prompts, num_steps):
    messages, results = [], []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        input_ids = encode_prompt(tokenizer, messages)
        with torch.no_grad():
            out = hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=num_steps,
                do_sample=False,
                pad_token_id=0,
            )
        tokens = out[0][input_ids.shape[1] :].tolist()
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="qwen3.5 多轮对话验证 (cann/wrapper + chunk + HF)"
    )
    parser.add_argument(
        "--mode",
        choices=["cann", "wrapper"],
        default="cann",
        help="cann_local_lm model.forward 或 Qwen3_5ExportWrapper",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8, help="chunk prefill size (0=非 chunk)"
    )
    parser.add_argument(
        "--num-steps", type=int, default=16, help="每轮 decode token 数"
    )
    parser.add_argument(
        "--prompts", type=str, default=None, help="多轮 prompt, || 分隔 (默认 3 轮)"
    )
    parser.add_argument(
        "--vs-hf", action="store_true", help="加 HF transformers 多轮对比"
    )
    args = parser.parse_args()

    dtype = torch.bfloat16
    chunked = args.chunk_size > 0
    prompts = args.prompts.split("||") if args.prompts else DEFAULT_TURNS

    print("[1/2] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_local_lm params: {sum(p.numel() for p in model.parameters()):,}")
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

    print(
        f"\n[2/2] 多轮对话: {len(prompts)} turns, {args.num_steps} tokens/turn, "
        f"mode={args.mode}, chunked={chunked} (size={args.chunk_size})"
    )
    print(f"{'=' * 60}")
    if args.mode == "cann":
        results = run_multiturn_cann(
            model,
            tokenizer,
            prompts,
            args.num_steps,
            chunked=chunked,
            chunk_size=args.chunk_size,
        )
    else:
        results = run_multiturn_wrapper(
            wrapper,
            tokenizer,
            prompts,
            args.num_steps,
            chunked=chunked,
            chunk_size=args.chunk_size,
        )
    hf_results = (
        run_multiturn_hf(hf_model, tokenizer, prompts, args.num_steps)
        if args.vs_hf
        else None
    )

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

    print(f"\n{'=' * 60}")
    if hf_results is not None:
        print(
            f"[{'PASS' if all_match else 'PARTIAL'}] {args.mode} vs HF 多轮 "
            f"{'一致' if all_match else '部分不一致 (bfloat16)'}"
        )
    else:
        print("[DONE] 多轮对话完成")


if __name__ == "__main__":
    main()
