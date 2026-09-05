#!/usr/bin/env python3
"""验证 Qwen3.5 ExportWrapper 全流程：单步 + HF 对比 + epair 导出

用法:
  python verify_qwen3_5_all.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper
from cann_lm.models.qwen3_5.modeling_qwen3_5 import SimpleKVCache
from cann_lm.converter.export.ge_converters import register_ge_converters
import torch_delegate
from transformers import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM

HF_PATH = os.environ["QWEN3_5_HF_PATH"]
SEQ_LEN = 4


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def main():
    dtype = torch.float16
    print(f"[Setup] Loading qwen3.5 model from {HF_PATH}")
    config = Qwen3_5TextConfig.from_pretrained(HF_PATH)
    model = Qwen3_5ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_5_weights(model, HF_PATH, device="cpu", dtype=dtype)
    print(f"       Params: {sum(p.numel() for p in model.parameters()):,}")

    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    mask = make_causal_mask(SEQ_LEN, dtype)
    pos = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    output_pos = torch.tensor([SEQ_LEN - 1], dtype=torch.int32)

    wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()

    # === Layer 1: 单步 forward 精度 (wrapper vs 原始模型) ===
    print(f"\n{'='*60}")
    print("Layer 1: 单步 forward 精度 (wrapper vs 原始模型)")
    print(f"{'='*60}")
    with torch.no_grad():
        out_orig = model(
            input_ids=input_ids, attention_mask=mask,
            past_key_values=SimpleKVCache(), delta_states=None, use_cache=True,
        )
        logits_orig = out_orig["logits"]
        kv_orig = out_orig["past_key_values"]
        delta_orig = out_orig["delta_cache"]

        logits_wrap, *kv_flat = wrapper(
            input_ids, mask, pos, None,
            None, None, None,   # cat 模式 (new_kv_cache_pos=None, 不走 scatter index_copy)
        )

    diff_1 = (logits_wrap[:, -1:, :] - logits_orig[:, -1:, :]).abs().max().item()
    match_1 = torch.allclose(logits_wrap[:, -1:, :], logits_orig[:, -1:, :], atol=1e-3)
    print(f"  logits: max_diff={diff_1:.6e} allclose={match_1}")
    print(f"  kv_flat: {len(kv_flat)} tensors (expect 12)")

    orig_layers = []
    if kv_orig is not None:
        for i in range(len(kv_orig.keys)):
            if kv_orig.keys[i] is not None:
                orig_layers.append(i)
    kv_match_1 = True
    for idx, layer_i in enumerate(orig_layers):
        k_orig = kv_orig.keys[layer_i]
        v_orig = kv_orig.values[layer_i]
        k_wrap = kv_flat[idx * 2]
        v_wrap = kv_flat[idx * 2 + 1]
        k_match = torch.allclose(k_wrap, k_orig, atol=1e-3)
        v_match = torch.allclose(v_wrap, v_orig, atol=1e-3)
        if not k_match or not v_match:
            kv_match_1 = False
    print(f"  kv_cache: {len(orig_layers)} layers, all_match={kv_match_1}")
    print(f"  Result: {'PASS' if match_1 and kv_match_1 else 'FAIL'}")

    # === Layer 2: 多轮 decode 精度 (wrapper vs 原始模型) ===
    print(f"\n{'='*60}")
    print(f"Layer 2: 多轮 decode (3 steps, wrapper vs 原始模型)")
    print(f"{'='*60}")
    kv_wrap_list = reconstruct_kv(kv_flat)
    tokens_orig = []
    tokens_wrap = []
    current_orig_kv = kv_orig
    current_orig_delta = delta_orig
    current_wrap_kv = kv_wrap_list
    all_decode_match = True

    for step in range(3):
        if step == 0:
            next_token = torch.argmax(logits_orig[:, -1, :], dim=-1, keepdim=True)
        else:
            next_token = torch.tensor([[tokens_orig[-1]]], dtype=torch.long)

        cached_len = SEQ_LEN + step
        mask_dec = torch.zeros(1, 1, 1, cached_len + 1, dtype=dtype)
        pos_dec = torch.tensor([[cached_len]], dtype=torch.long)

        with torch.no_grad():
            out_o = model(
                input_ids=next_token, attention_mask=mask_dec,
                past_key_values=current_orig_kv, delta_states=current_orig_delta, use_cache=True,
            )
            logits_o = out_o["logits"]
            current_orig_kv = out_o["past_key_values"]
            current_orig_delta = out_o["delta_cache"]

            logits_w, *kv_flat_w = wrapper(
                next_token, mask_dec, pos_dec, current_wrap_kv,
                None, None, None,
            )
            current_wrap_kv = reconstruct_kv(kv_flat_w)

        token_o = torch.argmax(logits_o[:, -1, :], dim=-1).item()
        token_w = torch.argmax(logits_w[:, -1, :], dim=-1).item()
        tokens_orig.append(token_o)
        tokens_wrap.append(token_w)
        step_diff = (logits_w - logits_o).abs().max().item()
        step_match = torch.allclose(logits_w, logits_o, atol=1e-3)
        all_decode_match = all_decode_match and step_match
        print(f"  step {step}: orig={token_o} wrap={token_w} diff={step_diff:.6e} match={step_match}")
    print(f"  tokens_orig={tokens_orig} tokens_wrap={tokens_wrap}")
    print(f"  Result: {'PASS' if all_decode_match and tokens_orig==tokens_wrap else 'FAIL'}")

    # === Layer 3: wrapper vs HF transformers ===
    print(f"\n{'='*60}")
    print("Layer 3: wrapper vs HF transformers")
    print(f"{'='*60}")
    print("  Loading HF model...")
    model_hf = HFQwen3_5ForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()

    with torch.no_grad():
        out_hf = model_hf(
            input_ids=input_ids, attention_mask=mask, position_ids=pos,
            past_key_values=None, use_cache=True,
        )
        logits_hf = out_hf.logits
        kv_hf = out_hf.past_key_values

    diff_3 = (logits_wrap[:, -1:, :] - logits_hf[:, -1:, :]).abs().max().item()
    match_3 = torch.allclose(logits_wrap[:, -1:, :], logits_hf[:, -1:, :], atol=1e-2)
    token_hf = torch.argmax(logits_hf[:, -1, :], dim=-1).item()
    token_wrap = torch.argmax(logits_wrap[:, -1, :], dim=-1).item()
    print(f"  logits: max_diff={diff_3:.6e} allclose(atol=1e-2)={match_3}")
    print(f"  token_hf={token_hf} token_wrap={token_wrap} match={token_hf==token_wrap}")
    print(f"  Result: {'PASS' if token_hf==token_wrap else 'FAIL'}")

    # === Layer 4: epair 导出 vs HF ===
    print(f"\n{'='*60}")
    print("Layer 4: epair 导出 (torch.export) vs HF")
    print(f"{'='*60}")
    register_ge_converters()
    sample_inputs = wrapper.get_sample_inputs(seq_len=SEQ_LEN)
    # 用导出时的 input_ids 替换, 确保 HF 和 ep 用相同输入
    sample_inputs = list(sample_inputs)
    sample_inputs[0] = input_ids  # 用同一个 input_ids
    sample_inputs = tuple(sample_inputs)
    print(f"  sample past pairs={len(sample_inputs[3])} k0={list(sample_inputs[3][0][0].shape)} new_pos={list(sample_inputs[4].shape)}")

    wrapper._delta_cache_box[0] = None  # 重置 delta_cache (避 Layer 1-3 持久影响 ep.module export)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    print(f"  torch.export OK, nodes={len(ep.graph.nodes)}")

    with torch.no_grad():
        ep_out = ep.module()(*sample_inputs)
        logits_ep = ep_out[0]

    diff_4 = (logits_ep - logits_hf[:, -1:, :]).abs().max().item()
    match_4 = torch.allclose(logits_ep, logits_hf[:, -1:, :], atol=1e-2)
    token_ep = torch.argmax(logits_ep[:, -1, :], dim=-1).item()
    print(f"  logits: max_diff={diff_4:.6e} allclose(atol=1e-2)={match_4}")
    print(f"  token_hf={token_hf} token_ep={token_ep} match={token_hf==token_ep}")
    print(f"  Result: {'PASS' if token_hf==token_ep else 'FAIL'}")

    # === Summary ===
    print(f"\n{'='*60}")
    print("Summary: Qwen3.5 全流程验证")
    print(f"{'='*60}")
    print(f"  Layer 1 (单步 vs 原始):    {'PASS' if match_1 and kv_match_1 else 'FAIL'} (diff={diff_1:.6e})")
    print(f"  Layer 2 (多轮 decode):     {'PASS' if all_decode_match and tokens_orig==tokens_wrap else 'FAIL'}")
    print(f"  Layer 3 (wrapper vs HF):   {'PASS' if token_hf==token_wrap else 'FAIL'} (diff={diff_3:.6e})")
    print(f"  Layer 4 (epair vs HF):     {'PASS' if token_hf==token_ep else 'FAIL'} (diff={diff_4:.6e})")
    all_pass = match_1 and kv_match_1 and all_decode_match and tokens_orig==tokens_wrap and token_hf==token_wrap and token_hf==token_ep
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
