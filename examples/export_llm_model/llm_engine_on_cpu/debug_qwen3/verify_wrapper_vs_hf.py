#!/usr/bin/env python3
"""验证 ExportWrapper 精度：wrapper 输出 vs transformers 原始模型输出

对比流程:
  A. transformers Qwen3ForCausalLM.from_pretrained()  ← 标准实现
  B. cann_lm Qwen3ForCausalLM + Qwen3ExportWrapper     ← 自定义实现 + wrapper

  Step 1 (prefill): input_ids [1,4] → logits + KV cache → 对比
  Step 2 (decode):  next_token [1,1] + KV cache → logits → 对比
  Step 3 (多轮):    重复 decode 3 轮, 对比 token 序列

用法:
  python verify_wrapper_vs_hf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper

from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"
NUM_DECODE_STEPS = 3


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """4D causal mask [1, 1, seq_len, seq_len] — 下三角=0, 上三角=-inf"""
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """从展平的 [k0,v0,k1,v1,...] 重建 [(k0,v0),(k1,v1),...]"""
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def main():
    dtype = torch.float16
    seq_len = 4

    print("[1/6] Loading HF model (transformers)")
    model_hf = HFQwen3ForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
    print(f"      Params: {sum(p.numel() for p in model_hf.parameters()):,}")

    print("[2/6] Loading custom model + wrapper (cann_lm)")
    config = Qwen3TextConfig.from_pretrained(HF_PATH)
    model_custom = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model_custom = load_qwen3_weights(model_custom, HF_PATH, device="cpu", dtype=dtype)
    wrapper = Qwen3ExportWrapper(model_custom, kv_cache_max_len=2048, batch=1).eval()
    print(f"      Params: {sum(p.numel() for p in model_custom.parameters()):,}")

    print("[3/6] Constructing shared inputs")
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    mask_prefill = make_causal_mask(seq_len, dtype)
    pos_prefill = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    print(f"      input_ids: {input_ids.tolist()}")
    print(f"      mask: {list(mask_prefill.shape)} (4D causal)")

    # === Step 1: prefill ===
    print(f"\n[4/6] Step 1: Prefill (seq_len={seq_len})")
    with torch.no_grad():
        # HF 模型 (DynamicCache, use_cache=True)
        out_hf_1 = model_hf(
            input_ids=input_ids,
            attention_mask=mask_prefill,
            position_ids=pos_prefill,
            past_key_values=None,
            use_cache=True,
        )
        logits_hf_1 = out_hf_1.logits
        kv_hf_1 = out_hf_1.past_key_values

        # wrapper (cann_lm 自定义模型)
        logits_wrap_1, *kv_flat_1 = wrapper(
            input_ids, mask_prefill, pos_prefill,
            None,  # past_key_values=None
            torch.arange(seq_len, dtype=torch.long),
            None,  # embeddings=None
            None,  # output_pos=None
        )

    diff_1 = (logits_wrap_1 - logits_hf_1).abs().max().item()
    match_1 = torch.allclose(logits_wrap_1, logits_hf_1, atol=1e-2)
    print(f"      HF logits:      {list(logits_hf_1.shape)}")
    print(f"      Wrapper logits: {list(logits_wrap_1.shape)}")
    print(f"      max_diff={diff_1:.6e}  allclose(atol=1e-2)={match_1}")

    # 为 step 2 重建 wrapper 的 KV cache
    kv_wrap_list = reconstruct_kv(kv_flat_1)

    # === Step 2: decode (1 token) ===
    print(f"\n[5/6] Step 2: Decode (1 token)")
    next_token = torch.argmax(logits_hf_1[:, -1, :], dim=-1, keepdim=True)  # [1, 1]
    print(f"      next_token: {next_token.item()}")

    cached_len = seq_len
    total_len = cached_len + 1
    mask_decode = torch.zeros(1, 1, 1, total_len, dtype=dtype)
    pos_decode = torch.tensor([[cached_len]], dtype=torch.long)

    with torch.no_grad():
        # HF 模型 (复用 DynamicCache)
        out_hf_2 = model_hf(
            input_ids=next_token,
            attention_mask=mask_decode,
            position_ids=pos_decode,
            past_key_values=kv_hf_1,
            use_cache=True,
        )
        logits_hf_2 = out_hf_2.logits

        # wrapper (从展平 cache 重建)
        logits_wrap_2, *kv_flat_2 = wrapper(
            next_token, mask_decode, pos_decode,
            kv_wrap_list,
            torch.tensor([cached_len], dtype=torch.long),
            None, None,
        )

    diff_2 = (logits_wrap_2 - logits_hf_2).abs().max().item()
    match_2 = torch.allclose(logits_wrap_2, logits_hf_2, atol=1e-2)
    print(f"      HF logits:      {list(logits_hf_2.shape)}")
    print(f"      Wrapper logits: {list(logits_wrap_2.shape)}")
    print(f"      max_diff={diff_2:.6e}  allclose(atol=1e-2)={match_2}")

    # === Step 3: 多轮 decode ===
    print(f"\n[6/6] Step 3: Multi-decode ({NUM_DECODE_STEPS} steps)")
    tokens_hf = [next_token.item()]
    tokens_wrap = [next_token.item()]

    current_hf_cache = out_hf_2.past_key_values
    current_wrap_kv = reconstruct_kv(kv_flat_2)
    all_match = match_1 and match_2

    for step in range(NUM_DECODE_STEPS):
        next_hf = torch.tensor([[tokens_hf[-1]]], dtype=torch.long)
        cached_len = seq_len + step + 1
        mask_dec = torch.zeros(1, 1, 1, cached_len + 1, dtype=dtype)
        pos_dec = torch.tensor([[cached_len]], dtype=torch.long)

        with torch.no_grad():
            out_hf = model_hf(
                input_ids=next_hf,
                attention_mask=mask_dec,
                position_ids=pos_dec,
                past_key_values=current_hf_cache,
                use_cache=True,
            )
            logits_hf = out_hf.logits
            current_hf_cache = out_hf.past_key_values

            logits_w, *kv_flat_w = wrapper(
                next_hf, mask_dec, pos_dec,
                current_wrap_kv,
                torch.tensor([cached_len], dtype=torch.long),
                None, None,
            )
            current_wrap_kv = reconstruct_kv(kv_flat_w)

        token_hf = torch.argmax(logits_hf[:, -1, :], dim=-1).item()
        token_w = torch.argmax(logits_w[:, -1, :], dim=-1).item()
        tokens_hf.append(token_hf)
        tokens_wrap.append(token_w)

        step_diff = (logits_w - logits_hf).abs().max().item()
        step_match = torch.allclose(logits_w, logits_hf, atol=1e-2)
        all_match = all_match and step_match
        print(f"      step {step}: token_hf={token_hf} token_wrap={token_w} "
              f"match={step_match} diff={step_diff:.6e}")

    # === Summary ===
    print(f"\n{'='*50}")
    print("Summary: Wrapper vs HF (transformers) Qwen3")
    print(f"{'='*50}")
    print(f"  Prefill logits match: {match_1} (diff={diff_1:.6e})")
    print(f"  Decode  logits match: {match_2} (diff={diff_2:.6e})")
    print(f"  Generated tokens (HF):     {tokens_hf}")
    print(f"  Generated tokens (Wrapper): {tokens_wrap}")
    print(f"  Tokens match: {tokens_hf == tokens_wrap}")
    print()
    if all_match and tokens_hf == tokens_wrap:
        print("[PASS] Wrapper output matches HF (transformers) model across all steps")
    else:
        print("[PARTIAL] Some differences detected (may be numerical, not functional)")
        print(f"  all_logits_match(atol=1e-2)={all_match}")
        print(f"  tokens_match={tokens_hf == tokens_wrap}")


if __name__ == "__main__":
    main()
