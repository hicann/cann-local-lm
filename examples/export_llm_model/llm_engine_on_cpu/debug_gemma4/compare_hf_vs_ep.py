#!/usr/bin/env python3
"""对比 HF model.generate() vs ep 模型推理 — 同样 prompt + 同样 sampling 参数

用法:
  python compare_hf_vs_ep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from cann_lm.models.fake_ops import register_ge_converters
from transformers import Gemma4ForConditionalGeneration as HFModel, AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
MAX_NEW_TOKENS = 30
EXPORT_SEQ_LEN = 8
BOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0
TEMPERATURE = 1.0
TOP_K = 64
TOP_P = 0.95

PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


def make_causal_mask_with_padding(q_len, k_len, real_q_len, real_k_len, dtype):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    if real_q_len < q_len:
        mask[real_q_len:, :] = min_val
    if real_k_len < k_len:
        mask[:, real_k_len:] = min_val
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def sample_next_token(logits, temperature, top_k, top_p):
    if temperature == 0:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_hf(hf_model, input_ids, max_new_tokens):
    """HF model.generate() with sampling"""
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            pad_token_id=PAD_TOKEN_ID,
        )
    return out[0][input_ids.shape[1]:].tolist()


def generate_ep(ep, wrapper, input_ids, real_seq_len, max_new_tokens, dtype):
    """ep.module() prefill + wrapper decode with sampling"""
    module = ep.module()

    if real_seq_len < EXPORT_SEQ_LEN:
        pad_len = EXPORT_SEQ_LEN - real_seq_len
        input_ids_padded = torch.cat([
            input_ids,
            torch.full((1, pad_len), PAD_TOKEN_ID, dtype=torch.long),
        ], dim=1)
    else:
        input_ids_padded = input_ids

    mask = make_causal_mask_with_padding(
        EXPORT_SEQ_LEN, EXPORT_SEQ_LEN, real_seq_len, real_seq_len, dtype)
    output_pos = torch.tensor([real_seq_len - 1], dtype=torch.int32)

    sample = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN)
    traced_inputs = list(sample)
    traced_inputs[0] = input_ids_padded
    traced_inputs[1] = mask
    traced_inputs[6] = output_pos

    with torch.no_grad():
        out = module(*traced_inputs)
        logits = out[0]
        kv_flat = list(out[1:])
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(max_new_tokens):
        next_token = sample_next_token(logits, TEMPERATURE, TOP_K, TOP_P)
        tokens.append(next_token.item())
        if next_token.item() == 1:
            break
        cached_len = EXPORT_SEQ_LEN + len(tokens)
        mask_dec = torch.zeros(1, 1, 1, cached_len, dtype=dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, None, kv,
                torch.tensor([cached_len - 1], dtype=torch.long),
                None, None,
            )
        kv = reconstruct_kv(kv_flat)
    return tokens


def main():
    dtype = torch.bfloat16

    print("[1/3] 加载模型")
    # HF
    print("  HF model...")
    hf_model = HFModel.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
    print(f"    params={sum(p.numel() for p in hf_model.parameters()):,}")

    # cann_lm + ep
    print("  cann_lm + ep...")
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
    register_ge_converters()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    print(f"    params={sum(p.numel() for p in model.parameters()):,}, nodes={len(ep.graph.nodes)}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    print(f"\n[2/3] 推理对比 (max_new_tokens={MAX_NEW_TOKENS}, sampling: temp={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P})")

    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        real_seq_len = input_ids.shape[1]

        print(f"\n{'='*70}")
        print(f"Prompt: \"{prompt}\"")
        print(f"{'='*70}")

        # HF
        torch.manual_seed(42)
        tokens_hf = generate_hf(hf_model, input_ids, MAX_NEW_TOKENS)
        text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
        print(f"\n  HF:     \"{text_hf}\"")

        # ep
        torch.manual_seed(42)
        tokens_ep = generate_ep(ep, wrapper, input_ids, real_seq_len, MAX_NEW_TOKENS, dtype)
        text_ep = tokenizer.decode(tokens_ep, skip_special_tokens=True)
        print(f"  EP:     \"{text_ep}\"")

    print(f"\n{'='*70}")
    print(f"[3/3] 完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
