#!/usr/bin/env python3
"""验证 gemma4 分 chunk 推理: chunk vs 非 chunk (+ 可选 vs HF)

合并自 verify_gemma4_chunk.py / _chunk_text.py / _chunk_vs_hf.py / _wrapper_chunk.py,
去除重复 (make_causal_mask / 模型加载 / chunk 推理核心逻辑).

模式:
  --mode cann      (默认): cann_lm model.forward (cat KV 累积)
  --mode wrapper          : Gemma4ExportWrapper.forward
  --vs-hf                 : 加 HF transformers model.generate 对比

固定: BOS + bfloat16 (gemma4 必须, 见 skill #1/#2)

用法:
  python verify_gemma4_chunk.py                       # cann, chunk vs 非 chunk, 三 prompt
  python verify_gemma4_chunk.py --mode wrapper        # wrapper 模式
  python verify_gemma4_chunk.py --vs-hf               # 加 HF 三方对比
  python verify_gemma4_chunk.py --chunk-size 4 --num-steps 20
  python verify_gemma4_chunk.py --prompt "Hello" --token-level

环境变量 (先 source 基底):
  source ../../../../scripts/set_env_export.sh   # 或 source ../check_env.sh
  export GEMMA4_HF_PATH=<gemma4 model dir>      # 必填
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper

HF_PATH = os.environ["GEMMA4_HF_PATH"]
BOS_TOKEN_ID = 2
DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


# ──────────────────────────────────────────────
# mask helpers (共享)
# ──────────────────────────────────────────────

def make_causal_mask(q_len, k_len, dtype):
    """4D causal mask [1, 1, q_len, k_len] (cat 模式, k_len = 已累积 token 数)."""
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def make_decode_mask(k_len, dtype):
    """decode mask [1, 1, 1, k_len] (全可见, decode 单 token)."""
    return torch.zeros(1, 1, 1, k_len, dtype=dtype)


def make_chunk_mask(chunk_len, k_len, dtype):
    """分 chunk prefill: chunk 的 query 可看前面所有 key (k_len) + 本 chunk causal."""
    if k_len == 0:
        return make_causal_mask(chunk_len, chunk_len, dtype)
    total_k = k_len + chunk_len
    mask = torch.zeros(1, 1, chunk_len, total_k, dtype=dtype)
    mask[:, :, :, k_len:] = make_causal_mask(chunk_len, chunk_len, dtype).squeeze(0).squeeze(0)
    return mask


def reconstruct_kv(flat_kv):
    """展平 [k0,v0,k1,v1,...] → [(k0,v0),(k1,v1),...] (wrapper 用)."""
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


# ──────────────────────────────────────────────
# cann_lm model.forward (cat KV 累积)
# ──────────────────────────────────────────────

def generate_cann(model, input_ids, num_steps, dtype, chunked=False, chunk_size=8,
                  return_logits=False):
    """cann_lm 一次性/分 chunk prefill + greedy decode.

    return_logits=True (仅非 chunk): 返回 (tokens, [(label, logits), ...]).
    """
    seq_len = input_ids.shape[1]
    kv = None
    shared = {}
    logits_list = []

    if chunked:
        if return_logits:
            raise ValueError("return_logits 不支持 chunked 模式 (chunk prefill 多步 logits)")
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            chunk_len = end - start
            mask = make_chunk_mask(chunk_len, start, dtype)
            with torch.no_grad():
                out = model(input_ids=chunk, attention_mask=mask,
                            past_key_values=kv, shared_kv_states=shared)
                logits = out["logits"]
                kv = out["past_key_values"]
                shared = out.get("shared_kv_states", {})
    else:
        mask = make_causal_mask(seq_len, seq_len, dtype)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, shared_kv_states={})
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})
        if return_logits:
            logits_list.append(("prefill", logits.clone()))

    tokens = []
    for i in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = seq_len + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            out = model(input_ids=next_token, attention_mask=mask_dec,
                        past_key_values=kv, shared_kv_states=shared)
            logits = out["logits"]
            kv = out["past_key_values"]
            shared = out.get("shared_kv_states", {})
        if return_logits:
            logits_list.append((f"decode{i+1}", logits.clone()))
    return (tokens, logits_list) if return_logits else tokens


# ──────────────────────────────────────────────
# Gemma4ExportWrapper.forward
# ──────────────────────────────────────────────

def generate_wrapper(wrapper, input_ids, num_steps, dtype, chunked=False, chunk_size=8,
                     return_logits=False):
    """wrapper 一次性/分 chunk prefill + greedy decode (flat KV).

    return_logits=True (仅非 chunk): 返回 (tokens, [(label, logits), ...]).
    """
    seq_len = input_ids.shape[1]
    kv = None
    logits = None
    logits_list = []

    if chunked:
        if return_logits:
            raise ValueError("return_logits 不支持 chunked 模式")
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            chunk_len = end - start
            mask = make_chunk_mask(chunk_len, start, dtype)
            with torch.no_grad():
                logits, *kv_flat = wrapper(
                    chunk, mask, None, kv,
                    torch.arange(start, end, dtype=torch.long), None, None)
            kv = reconstruct_kv(kv_flat)
    else:
        mask = make_causal_mask(seq_len, seq_len, dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                input_ids, mask, None, None,
                torch.arange(seq_len, dtype=torch.long), None, None)
        kv = reconstruct_kv(kv_flat)
        if return_logits:
            logits_list.append(("prefill", logits.clone()))

    tokens = []
    for i in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_len = seq_len + len(tokens)
        mask_dec = make_decode_mask(cached_len, dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, None, kv,
                torch.tensor([cached_len - 1], dtype=torch.long), None, None)
        kv = reconstruct_kv(kv_flat)
        if return_logits:
            logits_list.append((f"decode{i+1}", logits.clone()))
    return (tokens, logits_list) if return_logits else tokens


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


def generate_hf_logits(hf_full, input_ids, num_steps, dtype):
    """HF 逐 step forward (model.language_model + lm_head + softcap), 返回 (tokens, logits_list).

    用于 --logits-diff: 对比 wrapper/cann 与 HF 的逐 step logits 数值精度.
    """
    model_hf = hf_full.model.language_model
    lm_head = hf_full.lm_head
    softcap = (getattr(getattr(hf_full.config, "text_config", None), "final_logit_softcapping", None)
               or getattr(hf_full.config, "final_logit_softcapping", None))
    seq_len = input_ids.shape[1]

    mask = make_causal_mask(seq_len, seq_len, dtype)
    with torch.no_grad():
        out = model_hf(input_ids=input_ids, attention_mask=mask,
                       past_key_values=None, use_cache=True)
        logits = lm_head(out.last_hidden_state)
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        kv = out.past_key_values
    logits_list = [("prefill", logits.clone())]
    tokens = [torch.argmax(logits[:, -1, :], dim=-1).item()]

    for i in range(num_steps):
        next_token = torch.tensor([[tokens[-1]]], dtype=torch.long)
        cached_len = seq_len + i
        mask_dec = torch.zeros(1, 1, 1, cached_len + 1, dtype=dtype)
        with torch.no_grad():
            out = model_hf(input_ids=next_token, attention_mask=mask_dec,
                           past_key_values=kv, use_cache=True)
            logits = lm_head(out.last_hidden_state)
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            kv = out.past_key_values
        logits_list.append((f"decode{i+1}", logits.clone()))
        tokens.append(torch.argmax(logits[:, -1, :], dim=-1).item())
    return tokens, logits_list


# ──────────────────────────────────────────────
# model loading
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


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="gemma4 chunk 推理验证 (chunk vs 非 chunk + 可选 HF)")
    parser.add_argument("--mode", choices=["cann", "wrapper"], default="cann",
                        help="cann_lm model.forward (cat KV) 或 Gemma4ExportWrapper")
    parser.add_argument("--vs-hf", action="store_true", help="加 HF transformers 对比")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None, help="单 prompt (默认三 prompt)")
    parser.add_argument("--token-level", action="store_true", help="打印 token 级详细对比")
    parser.add_argument("--logits-diff", action="store_true",
                        help="逐 step logits 精度对比 vs HF (非 chunk, 隐含 --vs-hf, 用 HF model+lm_head 拆解)")
    args = parser.parse_args()

    if args.logits_diff:
        args.vs_hf = True  # logits-diff 隐含 vs-hf

    dtype = torch.bfloat16
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print("[1/3] Loading model")
    model, config = load_cann_model(HF_PATH, dtype)
    print(f"  cann_lm params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Layers: {config.num_hidden_layers} (shared: {getattr(config, 'num_kv_shared_layers', 0)})")

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

    all_pass = True
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"[2/3] Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        seq_len = input_ids.shape[1]
        num_chunks = (seq_len + args.chunk_size - 1) // args.chunk_size
        print(f"  tokens: {seq_len} (含 BOS), chunks: {num_chunks} (size={args.chunk_size})")

        if args.logits_diff:
            # logits 精度对比 (非 chunk): cann/wrapper 逐 step vs HF (model+lm_head+softcap 拆解)
            if args.mode == "cann":
                tokens_a, logits_a = generate_cann(model, input_ids, args.num_steps, dtype, return_logits=True)
            else:
                tokens_a, logits_a = generate_wrapper(wrapper, input_ids, args.num_steps, dtype, return_logits=True)
            tokens_hf, logits_hf = generate_hf_logits(hf_model, input_ids, args.num_steps, dtype)
            text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
            text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
            print(f"\n  A (非 chunk): \"{text_a}\"")
            print(f"  HF:           \"{text_hf}\"")
            print(f"  logits 逐 step 对比 (atol=1e-2):")
            all_logits_match = True
            for (label_a, la), (_, lh) in zip(logits_a, logits_hf):
                diff = (la.float() - lh.float()).abs().max().item()
                match = torch.allclose(la, lh, atol=1e-2)
                tok_a = torch.argmax(la[:, -1, :], dim=-1).item()
                tok_h = torch.argmax(lh[:, -1, :], dim=-1).item()
                all_logits_match = all_logits_match and match and (tok_a == tok_h)
                print(f"    {label_a}: max_diff={diff:.6e} allclose={match} "
                      f"tok_a={tok_a} tok_hf={tok_h} match={tok_a==tok_h}")
            # generate_hf_logits tokens 含 prefill argmax (长度 N+1), 对齐 generate_cann (长度 N, decode only)
            tok_match = tokens_a == tokens_hf[:len(tokens_a)]
            all_pass = all_pass and tok_match   # token match 即 PASS (logits allclose 是 bfloat16 精度参考, 非判定)
            print(f"\n  A==HF tokens: {tok_match}   logits allclose: {all_logits_match} "
                  f"(bfloat16 数值差异预期, 见 converter_design.md 精度差异分析)")
            continue

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
    print(f"[3/3] Summary: mode={args.mode}, chunk_size={args.chunk_size}, "
          f"steps={args.num_steps}, vs_hf={args.vs_hf}")
    print(f"{'='*60}")
    if all_pass:
        print("[PASS] chunk 推理与非 chunk 一致" + (" + HF 一致" if args.vs_hf else ""))
    else:
        print("[FAIL/PARTIAL] 存在不一致 (bfloat16 数值差异或 bug)")


if __name__ == "__main__":
    main()
