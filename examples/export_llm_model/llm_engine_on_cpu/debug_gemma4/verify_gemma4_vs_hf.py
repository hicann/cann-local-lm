#!/usr/bin/env python3
"""验证 Gemma4 ExportWrapper 多轮 decode 精度 vs HF transformers

流程:
  Step 1 (prefill): input_ids [1,4] → logits + KV cache
  Step 2 (decode):  next_token + KV cache → logits
  Step 3 (多轮):    重复 decode 3 轮, 对比 token 序列

用法:
  python verify_gemma4_vs_hf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper

from transformers import Gemma4ForConditionalGeneration as HFGemma4ForCausalLM

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
NUM_DECODE_STEPS = 3
SEQ_LEN = 4


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def main():
    dtype = torch.float16

    print("[1/6] Loading HF model (transformers)")
    hf_full = HFGemma4ForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
    model_hf = hf_full.model.language_model
    lm_head_hf = hf_full.lm_head
    softcap = getattr(hf_full.config.text_config, "final_logit_softcapping", None)
    if softcap is None:
        softcap = getattr(hf_full.config, "final_logit_softcapping", None)
    print(f"      Params: {sum(p.numel() for p in hf_full.parameters()):,}")
    print(f"      softcap: {softcap}")

    print("[2/6] Loading custom model + wrapper (cann_lm)")
    with open(f"{HF_PATH}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid_fields})
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    model_custom = Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    model_custom = load_gemma4_weights(model_custom, HF_PATH, device="cpu", dtype=dtype)
    wrapper = Gemma4ExportWrapper(model_custom, kv_cache_max_len=2048, batch=1).eval()
    print(f"      Params: {sum(p.numel() for p in model_custom.parameters()):,}")

    print("[3/6] Constructing shared inputs")
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    mask_prefill = make_causal_mask(SEQ_LEN, dtype)
    print(f"      input_ids: {input_ids.tolist()}")
    print(f"      mask: {list(mask_prefill.shape)} (4D causal)")

    # === Step 1: prefill ===
    print(f"\n[4/6] Step 1: Prefill (seq_len={SEQ_LEN})")
    with torch.no_grad():
        # HF: use language_model + lm_head + softcap
        out_hf_1 = model_hf(input_ids=input_ids, attention_mask=mask_prefill, past_key_values=None, use_cache=True)
        logits_hf_1 = lm_head_hf(out_hf_1.last_hidden_state)
        if softcap is not None:
            logits_hf_1 = torch.tanh(logits_hf_1 / softcap) * softcap
        kv_hf_1 = out_hf_1.past_key_values

        logits_wrap_1, *kv_flat_1 = wrapper(
            input_ids, mask_prefill, None, None,
            torch.arange(SEQ_LEN, dtype=torch.long),
            None, None,
        )
    diff_1 = (logits_wrap_1[:, -1:, :] - logits_hf_1[:, -1:, :]).abs().max().item()
    match_1 = torch.allclose(logits_wrap_1[:, -1:, :], logits_hf_1[:, -1:, :], atol=1e-2)
    token_hf_1 = torch.argmax(logits_hf_1[:, -1, :], dim=-1).item()
    token_wrap_1 = torch.argmax(logits_wrap_1[:, -1, :], dim=-1).item()
    print(f"      HF logits:      {list(logits_hf_1.shape)}")
    print(f"      Wrapper logits: {list(logits_wrap_1.shape)}")
    print(f"      max_diff={diff_1:.6e} allclose(atol=1e-2)={match_1}")
    print(f"      token_hf={token_hf_1} token_wrap={token_wrap_1} match={token_hf_1==token_wrap_1}")

    kv_wrap_list = reconstruct_kv(kv_flat_1)

    # === Step 2: decode (1 token) ===
    print(f"\n[5/6] Step 2: Decode (1 token)")
    next_token = torch.tensor([[token_hf_1]], dtype=torch.long)
    print(f"      next_token: {next_token.item()}")

    cached_len = SEQ_LEN
    total_len = cached_len + 1
    mask_decode = torch.zeros(1, 1, 1, total_len, dtype=dtype)

    with torch.no_grad():
        out_hf_2 = model_hf(input_ids=next_token, attention_mask=mask_decode, past_key_values=kv_hf_1, use_cache=True)
        logits_hf_2 = lm_head_hf(out_hf_2.last_hidden_state)
        if softcap is not None:
            logits_hf_2 = torch.tanh(logits_hf_2 / softcap) * softcap

        logits_wrap_2, *kv_flat_2 = wrapper(
            next_token, mask_decode, None, kv_wrap_list,
            torch.tensor([cached_len], dtype=torch.long),
            None, None,
        )

    diff_2 = (logits_wrap_2 - logits_hf_2).abs().max().item()
    match_2 = torch.allclose(logits_wrap_2, logits_hf_2, atol=1e-2)
    token_hf_2 = torch.argmax(logits_hf_2[:, -1, :], dim=-1).item()
    token_wrap_2 = torch.argmax(logits_wrap_2[:, -1, :], dim=-1).item()
    print(f"      max_diff={diff_2:.6e} allclose(atol=1e-2)={match_2}")
    print(f"      token_hf={token_hf_2} token_wrap={token_wrap_2} match={token_hf_2==token_wrap_2}")

    # === Step 3: multi-decode ===
    print(f"\n[6/6] Step 3: Multi-decode ({NUM_DECODE_STEPS} steps)")
    tokens_hf = [token_hf_1, token_hf_2]
    tokens_wrap = [token_wrap_1, token_wrap_2]

    current_hf_cache = out_hf_2.past_key_values
    current_wrap_kv = reconstruct_kv(kv_flat_2)
    all_match = match_1 and match_2 and (token_hf_1 == token_wrap_1) and (token_hf_2 == token_wrap_2)

    for step in range(NUM_DECODE_STEPS):
        next_hf = torch.tensor([[tokens_hf[-1]]], dtype=torch.long)
        cached_len = SEQ_LEN + step + 1
        mask_dec = torch.zeros(1, 1, 1, cached_len + 1, dtype=dtype)

        with torch.no_grad():
            out_hf = model_hf(input_ids=next_hf, attention_mask=mask_dec, past_key_values=current_hf_cache, use_cache=True)
            logits_hf = lm_head_hf(out_hf.last_hidden_state)
            if softcap is not None:
                logits_hf = torch.tanh(logits_hf / softcap) * softcap
            current_hf_cache = out_hf.past_key_values

            logits_w, *kv_flat_w = wrapper(
                next_hf, mask_dec, None, current_wrap_kv,
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
        all_match = all_match and step_match and (token_hf == token_w)
        print(f"      step {step}: token_hf={token_hf} token_wrap={token_w} "
              f"match={step_match and token_hf == token_w} diff={step_diff:.6e}")

    # === Summary ===
    print(f"\n{'='*60}")
    print("Summary: Gemma4 wrapper vs HF transformers (multi-decode)")
    print(f"{'='*60}")
    print(f"  Prefill: max_diff={diff_1:.6e} allclose={match_1} token_match={token_hf_1==token_wrap_1}")
    print(f"  Decode:  max_diff={diff_2:.6e} allclose={match_2} token_match={token_hf_2==token_wrap_2}")
    print(f"  Generated tokens (HF):     {tokens_hf}")
    print(f"  Generated tokens (Wrapper): {tokens_wrap}")
    print(f"  Tokens match: {tokens_hf == tokens_wrap}")
    print()
    if tokens_hf == tokens_wrap:
        print("[PASS] Wrapper produces same tokens as HF model")
    else:
        print("[FAIL] Token sequences diverge")
    if all_match:
        print("[PASS] Logits within tolerance (atol=1e-2)")


if __name__ == "__main__":
    main()
