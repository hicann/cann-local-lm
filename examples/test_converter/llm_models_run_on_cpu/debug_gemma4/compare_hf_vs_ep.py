#!/usr/bin/env python3
"""对比 HF model.generate() vs ep 模型推理 — 同样 prompt + 同样 sampling 参数

运行前必须先 source 基底 (converter venv + torchair stubs, CPU-only 无 NPU):

    source ../../../../scripts/set_env_export.sh   # 或 source ../check_env.sh
    export GEMMA4_HF_PATH=<gemma4 model dir>      # 必填
    $CONVERTER_PYTHON compare_hf_vs_ep.py

环境要点 (详见 AGENTS.md):
- venv: torch 2.9.0+cpu, torchair 源码编译 NO_ASCEND_SDK=1, torch_delegate 0.1.0, transformers 5.14.1
- stubs: 必须用 torchair 编译生成的 build/stubs (ABI 匹配), 不要用真实 CANN lib64 (会 aclrtSetDevice 107001 无设备)
- venv 无 torch_npu 包 → transformers 不 probe, 不需 import hf_cpu
"""

from __future__ import annotations

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
from cann_lm.converter.export.ge_converters import register_ge_converters
from transformers import Gemma4ForConditionalGeneration as HFModel, AutoTokenizer

HF_PATH = os.environ["GEMMA4_HF_PATH"]
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


def make_causal_mask_with_padding(q_len, kv_max, real_q_len, dtype):
    """scatter 模式 prefill mask: [1,1,q_len,kv_max] causal-over-buffer + pad 屏蔽.

    k_pos <= q_pos 可见 (causal over buffer); pad 位置 (real_q_len:) 整行屏蔽.
    未写入的 KV 位置 (k_pos > q_pos) 由 causal 天然屏蔽.
    """
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(q_len, dtype=dtype).unsqueeze(1)      # [q_len, 1]
    k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)    # [1, kv_max]
    mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
    if real_q_len < q_len:
        mask[real_q_len:, :] = min_val                          # pad 行屏蔽
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
        EXPORT_SEQ_LEN, wrapper.kv_cache_max_len, real_seq_len, dtype)
    output_pos = torch.tensor([real_seq_len - 1], dtype=torch.int32)

    sample = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN)
    traced_inputs = list(sample)
    traced_inputs[0] = input_ids_padded
    traced_inputs[1] = mask
    traced_inputs[6] = output_pos

    with torch.no_grad():
        out = module(*traced_inputs)             # ep prefill (exported 图)
        logits = out[0]
        kv_flat = list(out[1:])
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(max_new_tokens):
        next_token = sample_next_token(logits, TEMPERATURE, TOP_K, TOP_P)
        tokens.append(next_token.item())
        if next_token.item() == 1:
            break
        cached_len = EXPORT_SEQ_LEN + len(tokens)      # buffer 已写入数 (含 prefill pad)
        kv_max = wrapper.kv_cache_max_len
        min_val = torch.finfo(dtype).min
        mask_dec = torch.zeros(1, 1, 1, kv_max, dtype=dtype)
        if cached_len < kv_max:                         # 屏蔽未写入位置
            mask_dec[:, :, :, cached_len:] = min_val
        if real_seq_len < EXPORT_SEQ_LEN:               # 屏蔽 prefill pad 位置的 KV
            mask_dec[:, :, :, real_seq_len:EXPORT_SEQ_LEN] = min_val
        with torch.no_grad():
            # position_ids = 序列真实位置 (跳过 pad): prefill 真实 real_seq_len 个 + 第 n 个 decode
            seq_pos = real_seq_len + len(tokens) - 1
            pos_id = torch.tensor([[seq_pos]], dtype=torch.long)
            logits, *kv_flat = wrapper(
                next_token, mask_dec, pos_id, kv,
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
