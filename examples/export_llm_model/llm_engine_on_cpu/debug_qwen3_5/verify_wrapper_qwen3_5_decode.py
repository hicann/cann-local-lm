#!/usr/bin/env python3
"""验证 Qwen3_5ExportWrapper 两轮 forward (prefill + decode) 精度

流程:
  Step 1 (prefill): input_ids [1,8] → logits_1 + kv_cache
  Step 2 (decode):  next_token [1,1] + kv_cache → logits_2
  Step 3 (多轮):    重复 decode 5 轮, 对比生成的 token 序列

用法:
  python verify_wrapper_qwen3_5_decode.py
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

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"
NUM_DECODE_STEPS = 3


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """从展平的 [k0,v0,k1,v1,...] 重建 [(k0,v0),(k1,v1),...]"""
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def main():
    print("[1/5] Loading model")
    config = Qwen3TextConfig.from_pretrained(HF_PATH)
    model = Qwen3ForCausalLM(config).to(dtype=torch.float16).eval()
    model = load_qwen3_weights(model, HF_PATH, device="cpu", dtype=torch.float16)
    print(f"      Params: {sum(p.numel() for p in model.parameters()):,}")

    dtype = torch.float16
    seq_len = 4
    vocab_size = config.vocab_size

    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    print(f"      input_ids: {input_ids.tolist()}")

    # === Step 1: prefill ===
    print("\n[2/5] Step 1: Prefill (seq_len=8)")
    mask_1 = make_causal_mask(seq_len, dtype)
    pos_1 = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # 原始模型
        out_orig_1 = model(
            input_ids=input_ids,
            attention_mask=mask_1,
            position_ids=pos_1,
            past_key_values=[None] * config.num_hidden_layers,
        )
        logits_orig_1 = out_orig_1["logits"]
        kv_orig_1 = out_orig_1["past_key_values"]

        # wrapper
        wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        logits_wrap_1, *kv_flat_1 = wrapper(
            input_ids, mask_1, pos_1,
            None,  # past_key_values=None
            torch.arange(seq_len, dtype=torch.long),
            None,  # embeddings=None
            None,  # output_pos=None (返回全部 logits)
        )

    # 对比 prefill logits
    diff_1 = (logits_wrap_1 - logits_orig_1).abs().max().item()
    match_1 = torch.allclose(logits_wrap_1, logits_orig_1, atol=1e-3)
    print(f"      logits: shape_orig={list(logits_orig_1.shape)} shape_wrap={list(logits_wrap_1.shape)}")
    print(f"      logits: max_diff={diff_1:.6e} allclose={match_1}")

    # 保存 KV cache 供 step 2 使用
    kv_wrap_list = reconstruct_kv(kv_flat_1)
    orig_layers = list(range(len(kv_orig_1))) if kv_orig_1 else []
    print(f"      kv_cache: orig={len(orig_layers)} layers, wrap={len(kv_wrap_list)} pairs")

    # === Step 2: decode (1 token) ===
    print("\n[3/5] Step 2: Decode (1 token)")
    next_token = torch.argmax(logits_orig_1[:, -1, :], dim=-1, keepdim=True)  # [1, 1]
    print(f"      next_token: {next_token.item()}")

    # decode mask: 全零 [1, 1, 1, cached_len + 1] (attend to all)
    cached_len = seq_len
    total_len = cached_len + 1
    mask_2 = torch.zeros(1, 1, 1, total_len, dtype=dtype)
    pos_2 = torch.tensor([[cached_len]], dtype=torch.long)

    with torch.no_grad():
        # 原始模型 (复用 kv_orig_1, in-place 更新)
        out_orig_2 = model(
            input_ids=next_token,
            attention_mask=mask_2,
            position_ids=pos_2,
            past_key_values=kv_orig_1,
        )
        logits_orig_2 = out_orig_2["logits"]

        # wrapper (从展平 cache 重建 list of (k,v))
        logits_wrap_2, *kv_flat_2 = wrapper(
            next_token, mask_2, pos_2,
            kv_wrap_list,  # 重建的 (k,v) list
            torch.tensor([cached_len], dtype=torch.long),
            None,
            None,  # output_pos=None
        )

    diff_2 = (logits_wrap_2 - logits_orig_2).abs().max().item()
    match_2 = torch.allclose(logits_wrap_2, logits_orig_2, atol=1e-3)
    print(f"      logits: shape_orig={list(logits_orig_2.shape)} shape_wrap={list(logits_wrap_2.shape)}")
    print(f"      logits: max_diff={diff_2:.6e} allclose={match_2}")

    # === Step 3: 多轮 decode ===
    print(f"\n[4/5] Step 3: Multi-decode ({NUM_DECODE_STEPS} steps)")
    tokens_orig = [next_token.item()]
    tokens_wrap = [next_token.item()]

    # 原始模型: 继续用 kv_orig_1 (已被 step 2 更新)
    # wrapper: 继续用 kv_flat_2 重建
    kv_wrap_decode = reconstruct_kv(kv_flat_2)
    current_orig_cache = kv_orig_1  # in-place updated
    current_wrap_kv = kv_wrap_decode

    all_match = match_1 and match_2
    for step in range(NUM_DECODE_STEPS):
        # 原始模型
        next_orig = torch.tensor([[tokens_orig[-1]]], dtype=torch.long)
        cached_len_orig = seq_len + step + 1
        mask_dec = torch.zeros(1, 1, 1, cached_len_orig + 1, dtype=dtype)
        pos_dec = torch.tensor([[cached_len_orig]], dtype=torch.long)

        with torch.no_grad():
            out_o = model(
                input_ids=next_orig,
                attention_mask=mask_dec,
                position_ids=pos_dec,
                past_key_values=current_orig_cache,
            )
            logits_o = out_o["logits"]
            current_orig_cache = out_o["past_key_values"]

            # wrapper
            logits_w, *kv_flat_w = wrapper(
                next_orig, mask_dec, pos_dec,
                current_wrap_kv,
                torch.tensor([cached_len_orig], dtype=torch.long),
                None, None,
            )
            current_wrap_kv = reconstruct_kv(kv_flat_w)

        token_o = torch.argmax(logits_o[:, -1, :], dim=-1).item()
        token_w = torch.argmax(logits_w[:, -1, :], dim=-1).item()
        tokens_orig.append(token_o)
        tokens_wrap.append(token_w)

        step_diff = (logits_w - logits_o).abs().max().item()
        step_match = torch.allclose(logits_w, logits_o, atol=1e-3)
        all_match = all_match and step_match
        print(f"      step {step}: token_orig={token_o} token_wrap={token_w} "
              f"match={step_match} diff={step_diff:.6e}")

    # === Summary ===
    print("\n[5/5] Summary")
    print(f"  Prefill logits match: {match_1} (diff={diff_1:.6e})")
    print(f"  Decode  logits match: {match_2} (diff={diff_2:.6e})")
    print(f"  Generated tokens (orig): {tokens_orig}")
    print(f"  Generated tokens (wrap): {tokens_wrap}")
    print(f"  Tokens match: {tokens_orig == tokens_wrap}")
    print()
    if all_match and tokens_orig == tokens_wrap:
        print("[PASS] Wrapper output matches original model across all steps")
    else:
        print("[FAIL] Wrapper output differs from original model")
        print(f"  all_logits_match={all_match}, tokens_match={tokens_orig == tokens_wrap}")


if __name__ == "__main__":
    main()
