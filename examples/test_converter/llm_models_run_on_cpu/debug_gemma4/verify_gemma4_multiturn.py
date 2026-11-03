#!/usr/bin/env python3
"""验证 gemma4 多轮对话: 跨轮上下文 (cann/wrapper + chunk + vs HF)

多轮对话流程:
  第1轮: chat_template(user1 + gen) → input_ids1 → prefill → KV1 → decode assistant1
  第2轮: chat_template(user1 + assistant1 + user2 + gen) → input_ids2
          → prefill(含历史, 重算) → KV2 → decode assistant2
  ...

每轮独立 prefill(含完整历史, 重算) + decode. 上下文对(input_ids 含历史), KV 不跨轮累积
(标准 transformer, 无 delta_cache; 跨轮 KV 复用需精确 token 对齐, 待后续).

gemma4 特有:
  - BOS 必须 (chat_template 自动加 bos_token)
  - cann model.forward 需手动 4D mask (causal/chunk/decode) + shared_kv_states (PLE)
  - wrapper 双 mask BOOL (full/sliding) + new_kv_cache_pos 绝对位 + 零 KV buffer (scatter-write)
  - 无 delta_cache (标准 transformer, 不像 qwen3.5 混合架构)

模式:
  --mode cann (默认): cann_local_lm model.forward (cat KV)
  --mode wrapper      : Gemma4ExportWrapper.forward (scatter-write)

固定: bfloat16 (gemma4 必须)

用法:
  python verify_gemma4_multiturn.py                            # cann, 3 轮, 非 chunk
  python verify_gemma4_multiturn.py --mode wrapper             # wrapper
  python verify_gemma4_multiturn.py --chunk-size 4             # chunk prefill
  python verify_gemma4_multiturn.py --num-turns 3 --num-steps 16
  python verify_gemma4_multiturn.py --vs-hf                    # 加 HF 多轮对比
  python verify_gemma4_multiturn.py --prompts "你好||刚才我说了什么？||介绍AI"

环境变量 (先 source 基底):
  source ../../../../scripts/cann_lm_compile/set_env_export.sh   # 或 source ../check_env.sh
  export GEMMA4_HF_PATH=<gemma4 model dir>      # 必填
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "cann_local_lm").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_local_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_local_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_local_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper

HF_PATH = os.environ["GEMMA4_HF_PATH"]
DEFAULT_TURNS = ["Hello, my name is", "What is my name?", "Tell me about AI"]


# ──────────────────────────────────────────────
# mask helpers (gemma4 手动 4D mask, cat 模式)
# ──────────────────────────────────────────────


def make_causal_mask(q_len, k_len, dtype):
    """4D causal mask [1, 1, q_len, k_len]."""
    min_val = torch.finfo(dtype).min
    return (
        torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def make_decode_mask(k_len, dtype):
    """decode mask [1, 1, 1, k_len] (全可见)."""
    return torch.zeros(1, 1, 1, k_len, dtype=dtype)


def make_chunk_mask(chunk_len, k_len, dtype):
    """分 chunk prefill: chunk query 可看前面所有 key + 本 chunk causal."""
    if k_len == 0:
        return make_causal_mask(chunk_len, chunk_len, dtype)
    total_k = k_len + chunk_len
    mask = torch.zeros(1, 1, chunk_len, total_k, dtype=dtype)
    mask[:, :, :, k_len:] = (
        make_causal_mask(chunk_len, chunk_len, dtype).squeeze(0).squeeze(0)
    )
    return mask


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def make_zero_kv_buffer(wrapper, dtype):
    """零 KV buffer (per-layer head_dim, gemma4 sliding/full 差异)."""
    config = wrapper.config
    num_kv_heads = config.num_key_value_heads
    num_layers = wrapper._num_layers()
    kv_max = wrapper.kv_cache_max_len
    buf = []
    for i in range(num_layers):
        hd = wrapper._get_layer_head_dim(i)
        buf.append(
            (
                torch.zeros(1, num_kv_heads, kv_max, hd, dtype=dtype),
                torch.zeros(1, num_kv_heads, kv_max, hd, dtype=dtype),
            )
        )
    return buf


def build_masks(wrapper, seq_len, new_kv_cache_pos):
    """wrapper 自带 helper 构造 (mask_full, mask_sliding), BOOL scatter-write 绝对位."""
    kv_max = wrapper.kv_cache_max_len
    dtype = wrapper._model_dtype
    return (
        wrapper._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos),
        wrapper._build_mask_sliding(seq_len, kv_max, dtype, new_kv_cache_pos),
    )


# ──────────────────────────────────────────────
# model loading + encode
# ──────────────────────────────────────────────


def load_cann_model(hf_path, dtype):
    with open(f"{hf_path}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid})
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    model = Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    model = load_gemma4_weights(model, hf_path, device="cpu", dtype=dtype)
    return model, config


def encode_prompt(tokenizer, messages):
    """chat_template 编码 (含历史) → input_ids (BOS 由 chat_template 加)."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


# ──────────────────────────────────────────────
# cann: model.forward (cat KV + shared_kv_states PLE)
# ──────────────────────────────────────────────


def prefill_cann(model, input_ids, dtype, chunked=False, chunk_size=8):
    """cann prefill (chunked or not), 累积 KV + shared. 返回 (logits, kv, shared)."""
    seq_len = input_ids.shape[1]
    kv, shared = None, {}
    if chunked:
        for ci in range((seq_len + chunk_size - 1) // chunk_size):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, seq_len)
            mask = make_chunk_mask(e - s, s, dtype)
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[:, s:e],
                    attention_mask=mask,
                    past_key_values=kv,
                    shared_kv_states=shared,
                )
            logits, kv, shared = (
                out["logits"],
                out["past_key_values"],
                out.get("shared_kv_states", {}),
            )
    else:
        mask = make_causal_mask(seq_len, seq_len, dtype)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, shared_kv_states={})
        logits, kv, shared = (
            out["logits"],
            out["past_key_values"],
            out.get("shared_kv_states", {}),
        )
    return logits, kv, shared


def decode_cann(model, logits, kv, shared, num_steps, dtype, seq_len):
    """greedy decode, 累积 kv/shared. 返回 tokens."""
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
        logits, kv, shared = (
            out["logits"],
            out["past_key_values"],
            out.get("shared_kv_states", {}),
        )
    return tokens


def run_multiturn_cann(
    model, tokenizer, prompts, num_steps, dtype, chunked=False, chunk_size=8
):
    """每轮独立 prefill(含历史, 重算) + decode."""
    messages, results = [], []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        input_ids = encode_prompt(tokenizer, messages)
        logits, kv, shared = prefill_cann(model, input_ids, dtype, chunked, chunk_size)
        tokens = decode_cann(
            model, logits, kv, shared, num_steps, dtype, input_ids.shape[1]
        )
        text_out = tokenizer.decode(tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text_out})
        results.append((prompt, text_out))
    return results


# ──────────────────────────────────────────────
# wrapper: Gemma4ExportWrapper.forward (scatter-write, 双 mask BOOL)
# ──────────────────────────────────────────────


def prefill_wrapper(wrapper, input_ids, dtype, chunked=False, chunk_size=8):
    """wrapper prefill (chunked or not), 零 KV buffer scatter-write. 返回 (logits, kv)."""
    seq_len = input_ids.shape[1]
    kv = make_zero_kv_buffer(wrapper, dtype)
    if chunked:
        for ci in range((seq_len + chunk_size - 1) // chunk_size):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, seq_len)
            nkp = torch.arange(s, e, dtype=torch.int32)
            mf, ms = build_masks(wrapper, e - s, nkp)
            pos = torch.arange(s, e, dtype=torch.int32).unsqueeze(0)
            with torch.no_grad():
                logits, *kv_flat = wrapper(
                    input_ids[:, s:e],
                    mf,
                    ms,
                    pos,
                    kv,
                    nkp,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            kv = reconstruct_kv(kv_flat)
    else:
        nkp = torch.arange(seq_len, dtype=torch.int32)
        mf, ms = build_masks(wrapper, seq_len, nkp)
        pos = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                input_ids, mf, ms, pos, kv, nkp, None, output_pos, None, None, None
            )
        kv = reconstruct_kv(kv_flat)
    return logits, kv


def decode_wrapper(wrapper, logits, kv, num_steps, dtype, seq_len):
    """wrapper greedy decode (scatter-write). 返回 tokens."""
    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_pos = seq_len + len(tokens) - 1
        dec_nkp = torch.tensor([cached_pos], dtype=torch.int32)
        mf, ms = build_masks(wrapper, 1, dec_nkp)
        pos = torch.tensor([[cached_pos]], dtype=torch.int32)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mf, ms, pos, kv, dec_nkp, None, None, None, None, None
            )
        kv = reconstruct_kv(kv_flat)
    return tokens


def run_multiturn_wrapper(
    wrapper, tokenizer, prompts, num_steps, dtype, chunked=False, chunk_size=8
):
    """每轮独立 prefill(含历史, 重算) + decode (零 KV buffer 每轮重建)."""
    messages, results = [], []
    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})
        input_ids = encode_prompt(tokenizer, messages)
        logits, kv = prefill_wrapper(wrapper, input_ids, dtype, chunked, chunk_size)
        tokens = decode_wrapper(
            wrapper, logits, kv, num_steps, dtype, input_ids.shape[1]
        )
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
        description="gemma4 多轮对话验证 (cann/wrapper + chunk + HF)"
    )
    parser.add_argument(
        "--mode",
        choices=["cann", "wrapper"],
        default="cann",
        help="cann_local_lm model.forward 或 Gemma4ExportWrapper",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8, help="chunk prefill size (0=非 chunk)"
    )
    parser.add_argument("--num-turns", type=int, default=3, help="对话轮数")
    parser.add_argument(
        "--num-steps", type=int, default=16, help="每轮 decode token 数"
    )
    parser.add_argument(
        "--prompts", type=str, default=None, help="多轮 prompt, || 分隔"
    )
    parser.add_argument(
        "--vs-hf", action="store_true", help="加 HF transformers 多轮对比"
    )
    args = parser.parse_args()

    dtype = torch.bfloat16
    chunked = args.chunk_size > 0
    prompts = (
        args.prompts.split("||") if args.prompts else DEFAULT_TURNS[: args.num_turns]
    )

    print("[1/2] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_local_lm params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Layers: {config.num_hidden_layers} (shared: {getattr(config, 'num_kv_shared_layers', 0)})"
    )

    wrapper = None
    if args.mode == "wrapper":
        wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        print(f"  Wrapper: {wrapper.__class__.__name__}")

    hf_model = None
    if args.vs_hf:
        from transformers import Gemma4ForConditionalGeneration as HFModel

        print("  Loading HF model (bfloat16)...")
        hf_model = HFModel.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    print(
        f"\n[2/2] 多轮对话 ({len(prompts)} turns, {args.num_steps} tokens/turn, "
        f"mode={args.mode}, chunked={chunked})"
    )
    print(f"{'=' * 60}")

    if args.mode == "cann":
        results = run_multiturn_cann(
            model,
            tokenizer,
            prompts,
            args.num_steps,
            dtype,
            chunked=chunked,
            chunk_size=args.chunk_size,
        )
    else:
        results = run_multiturn_wrapper(
            wrapper,
            tokenizer,
            prompts,
            args.num_steps,
            dtype,
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
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
