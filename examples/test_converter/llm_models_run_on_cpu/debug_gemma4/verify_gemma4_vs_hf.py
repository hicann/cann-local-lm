#!/usr/bin/env python3
"""Numerical alignment: pure-torch gemma4 vs HuggingFace transformers gemma4.

Loads the SAME checkpoint into cann_lm Gemma4ForCausalLM and HF
Gemma4ForConditionalGeneration (text-only: language_model + lm_head + softcap),
compares per-position logits (max_diff / argmax_match / top5_overlap) + greedy
generation token-id equality.

gemma4 特有:
  - BOS 必须起始 (chat_template 自动加; random 序列手动前置 BOS_TOKEN_ID=2)
  - bfloat16 必须 (float16 输出质量差)
  - cann forward 需手动 4D causal mask + shared_kv_states (PLE)
  - HF logits 需拆解: lm_head(language_model(input_ids).last_hidden_state) + softcap tanh

Usage:
    python verify_gemma4_vs_hf.py --model /path/to/gemma-4-E2B-it
    python verify_gemma4_vs_hf.py --model $GEMMA4_HF_PATH --gen-tokens 8
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

_LLM_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # llm_models_run_on_cpu/ (hf_cpu)
if _LLM_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _LLM_ENGINE_DIR)
import hf_cpu  # noqa: F401  disables torch_npu probe before importing transformers

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights

HF_PATH = os.environ["GEMMA4_HF_PATH"]
BOS_TOKEN_ID = 2


def make_causal_mask(q_len, k_len, dtype):
    """4D causal mask [1, 1, q_len, k_len] (cat 模式)."""
    min_val = torch.finfo(dtype).min
    return torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1).unsqueeze(0).unsqueeze(0)


def load_mine(model_path, dtype):
    with open(f"{model_path}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid})
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    m = Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    m = load_gemma4_weights(m, model_path, device="cpu", dtype=dtype)
    return m


def load_hf(model_path, dtype):
    from transformers import Gemma4ForConditionalGeneration as HFModel
    return HFModel.from_pretrained(model_path, torch_dtype=dtype).eval()


def hf_forward(hf_full, input_ids, dtype):
    """HF text-only forward: language_model + lm_head + softcap 拆解."""
    model_hf = hf_full.model.language_model
    lm_head = hf_full.lm_head
    softcap = (getattr(getattr(hf_full.config, "text_config", None), "final_logit_softcapping", None)
               or getattr(hf_full.config, "final_logit_softcapping", None))
    mask = make_causal_mask(input_ids.shape[1], input_ids.shape[1], dtype)
    with torch.no_grad():
        out = model_hf(input_ids=input_ids, attention_mask=mask, past_key_values=None, use_cache=True)
        logits = lm_head(out.last_hidden_state)
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
    return logits


def mine_forward(mine, input_ids, dtype):
    """cann_lm Gemma4ForCausalLM forward (手动 4D mask + shared_kv_states PLE)."""
    mask = make_causal_mask(input_ids.shape[1], input_ids.shape[1], dtype)
    with torch.no_grad():
        out = mine(input_ids=input_ids, attention_mask=mask,
                   past_key_values=None, shared_kv_states={})
    return out["logits"]


def compare_logits(name, mine_logits, hf_logits):
    diff = (mine_logits - hf_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = hf_logits.abs().mean().item()
    rel = max_abs / max(ref_scale, 1e-8)
    mine_arg = mine_logits.argmax(-1)
    hf_arg = hf_logits.argmax(-1)
    argmatch = (mine_arg == hf_arg).float().mean().item()
    _, mt5 = torch.topk(mine_logits, 5, dim=-1)
    _, ht5 = torch.topk(hf_logits, 5, dim=-1)
    sets = [set(mt5[0, i].tolist()) & set(ht5[0, i].tolist()) for i in range(mt5.shape[1])]
    top5 = sum(len(s) for s in sets) / (5 * len(sets))
    print(f"  [{name}] max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  rel={rel:.2e}  "
          f"argmax_match={argmatch*100:.1f}%  top5_overlap={top5*100:.1f}%")
    return max_abs, argmatch


def main():
    parser = argparse.ArgumentParser(description="gemma4 pure-torch vs HF 数值对齐")
    parser.add_argument("--model", required=True, help="Path to gemma4 checkpoint")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--gen-tokens", type=int, default=8)
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Loading tokenizer + mine + HF (bfloat16) from {args.model}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    mine = load_mine(args.model, dtype)
    hf = load_hf(args.model, dtype)
    print(f"  mine params={sum(p.numel() for p in mine.parameters()):,}  "
          f"hf params={sum(p.numel() for p in hf.parameters()):,}")

    # embed_tokens 对比
    me = mine.model.embed_tokens.weight
    he = hf.model.language_model.embed_tokens.weight
    print(f"  embed_tokens identical: {torch.equal(me, he)}  (max|Δ|={(me-he).abs().max().item():.3e})")

    # chat prompt logits (chat_template, 含 BOS)
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    print(f"\nPrompt tokens: {ids.shape[1]} ids (chat_template + BOS)")
    hf_logits = hf_forward(hf, ids, dtype)
    my_logits = mine_forward(mine, ids, dtype)
    print("Logits comparison (full chat prompt, all positions):")
    compare_logits("chat-prompt", my_logits, hf_logits)

    # random 32-token (前置 BOS, gemma4 必须 BOS 起始)
    torch.manual_seed(42)
    rids = torch.randint(0, tok.vocab_size, (1, 32), dtype=torch.long)
    rids = torch.cat([torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long), rids], dim=1)
    hf_r = hf_forward(hf, rids, dtype)
    my_r = mine_forward(mine, rids, dtype)
    print("\nLogits comparison (random 32-token + BOS):")
    compare_logits("random-32", my_r, hf_r)

    # deterministic re-forward invariant (mine 内部一致性, gemma4 forward 无 use_cache 开关)
    my_logits2 = mine_forward(mine, ids, dtype)
    print(f"\nDeterministic re-forward max|Δ|: {(my_logits2 - my_logits).abs().max().item():.3e}")

    # greedy generation 一致
    n = args.gen_tokens
    with torch.no_grad():
        hf_gen = hf.generate(input_ids=ids, max_new_tokens=n, do_sample=False, pad_token_id=0)
        my_gen = mine.generate(ids, max_new_tokens=n, do_sample=False, eos_token_id=tok.eos_token_id)
    hf_new = hf_gen[0, ids.shape[1]:].tolist()
    my_new = my_gen[0, ids.shape[1]:].tolist()
    match = hf_new == my_new
    print(f"\nGreedy generation ({n} tokens):  identical={match}")
    print(f"  hf : {hf_new}")
    print(f"  mine: {my_new}")
    print(f"  decoded: {tok.decode(my_new, skip_special_tokens=True)!r}")
    if not match:
        for i, (a, b) in enumerate(zip(hf_new, my_new)):
            if a != b:
                print(f"  first divergence at step {i}: hf={a}  mine={b}")
                break

    print("\n" + "=" * 50)
    ok = match and (my_logits.argmax(-1) == hf_logits.argmax(-1)).all().item()
    print("VERDICT:", "PASS — pure-torch matches HF reference" if ok else "MISMATCH — investigate")
    print("=" * 50)


if __name__ == "__main__":
    main()
