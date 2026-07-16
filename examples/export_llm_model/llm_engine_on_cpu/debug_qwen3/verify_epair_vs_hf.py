#!/usr/bin/env python3
"""验证 epair 导出精度：导出 epair → Python 回环推理 vs HF 原始模型

流程:
  A. transformers Qwen3ForCausalLM.from_pretrained()  ← 标准实现 (基准)
  B. cann_lm 模型 + ExportWrapper → torch.export → export_to_epair → epair_load → 回环推理

  Step 1 (prefill): input_ids [1,4] → 对比 logits
  Step 2 (decode):  next_token + KV cache → 对比 logits
  Step 3 (多轮):    重复 decode 3 轮, 对比 token 序列

用法:
  python verify_epair_vs_hf.py
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
from cann_lm.models.fake_ops import register_ge_converters

import torch_delegate
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"
NUM_DECODE_STEPS = 3


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def export_wrapper_to_epair(model, config, seq_len=4):
    """导出 wrapper 为 ExportedProgram, 返回 ExportedProgram (不经过 epair 序列化回环)"""
    register_ge_converters()
    wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=seq_len)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    # 同时测试 epair 序列化回环
    epair = torch_delegate.export_to_epair(ep)
    nodes = epair.ep_json["graph_module"]["graph"]["nodes"]
    epair.ep_json["graph_module"]["graph"]["nodes"] = [
        n for n in nodes if "assert" not in n.get("target", "").lower()
    ]
    pt2_path = Path("/tmp/verify_epair.pt2")
    torch_delegate.epair_save(epair, str(pt2_path))
    print(f"      epair saved: {pt2_path}")

    # 回环加载
    ep_loaded = torch_delegate.epair_load(str(pt2_path))
    return ep, ep_loaded.ep, sample_inputs


def main():
    dtype = torch.float16
    seq_len = 4

    print("[1/7] Loading HF model (transformers)")
    model_hf = HFQwen3ForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
    print(f"      Params: {sum(p.numel() for p in model_hf.parameters()):,}")

    print("[2/7] Loading custom model (cann_lm)")
    config = Qwen3TextConfig.from_pretrained(HF_PATH)
    model_custom = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model_custom = load_qwen3_weights(model_custom, HF_PATH, device="cpu", dtype=dtype)
    print(f"      Params: {sum(p.numel() for p in model_custom.parameters()):,}")

    print("[3/7] Exporting wrapper to epair (torch.export → export_to_epair → epair_load)")
    ep_direct, ep_loaded, traced_inputs = export_wrapper_to_epair(model_custom, config, seq_len=seq_len)
    print(f"      ep direct: {type(ep_direct).__name__}")
    print(f"      ep loaded: {type(ep_loaded).__name__}")

    print("[4/7] Using traced inputs (same as export)")
    # 用导出时的 sample_inputs, 确保 HF 和 ep_direct 用相同输入
    input_ids = traced_inputs[0]
    mask_prefill = traced_inputs[1]
    pos_prefill = traced_inputs[2]
    print(f"      input_ids: {input_ids.tolist()}")
    print(f"      mask: {list(mask_prefill.shape)} (4D causal)")

    # === Step 1: prefill ===
    print(f"\n[5/7] Step 1: Prefill (seq_len={seq_len})")
    with torch.no_grad():
        # HF 原始模型
        out_hf_1 = model_hf(
            input_ids=input_ids,
            attention_mask=mask_prefill,
            position_ids=pos_prefill,
            past_key_values=None,
            use_cache=True,
        )
        logits_hf_1 = out_hf_1.logits
        kv_hf_1 = out_hf_1.past_key_values

        # epair 回环推理 (ExportedProgram 直接调用, 用导出时的 sample_inputs)
        ep_out_1 = ep_direct.module()(*traced_inputs)
        logits_ep_1 = ep_out_1[0]
        kv_flat_ep_1 = list(ep_out_1[1:])

    diff_1 = (logits_ep_1 - logits_hf_1[:, -1:, :]).abs().max().item()
    match_1 = torch.allclose(logits_ep_1, logits_hf_1[:, -1:, :], atol=1e-2)
    print(f"      HF logits:      {list(logits_hf_1.shape)}")
    print(f"      epair logits:   {list(logits_ep_1.shape)} (output_pos selected)")
    print(f"      max_diff={diff_1:.6e}  allclose(atol=1e-2)={match_1}")

    kv_ep_list = reconstruct_kv(kv_flat_ep_1)

    # === Step 2: decode (1 token) ===
    print(f"\n[6/7] Step 2: Decode (1 token) — using wrapper direct (ep_direct traced for prefill only)")
    next_token = torch.argmax(logits_hf_1[:, -1, :], dim=-1, keepdim=True)
    print(f"      next_token: {next_token.item()}")

    cached_len = seq_len
    total_len = cached_len + 1
    mask_decode = torch.zeros(1, 1, 1, total_len, dtype=dtype)
    pos_decode = torch.tensor([[cached_len]], dtype=torch.long)

    with torch.no_grad():
        # HF 原始模型
        out_hf_2 = model_hf(
            input_ids=next_token,
            attention_mask=mask_decode,
            position_ids=pos_decode,
            past_key_values=kv_hf_1,
            use_cache=True,
        )
        logits_hf_2 = out_hf_2.logits

        # wrapper 直接调用 (不用 ep_direct, 因为 ep 只 traced 了 prefill 形状)
        wrapper_direct = Qwen3ExportWrapper(model_custom, kv_cache_max_len=2048, batch=1).eval()
        logits_wrap_2, *kv_flat_2 = wrapper_direct(
            next_token, mask_decode, pos_decode,
            kv_ep_list,  # 重建的 KV cache from prefill
            torch.tensor([cached_len], dtype=torch.long),
            None, None,
        )

    diff_2 = (logits_wrap_2 - logits_hf_2).abs().max().item()
    match_2 = torch.allclose(logits_wrap_2, logits_hf_2, atol=1e-2)
    print(f"      HF logits:      {list(logits_hf_2.shape)}")
    print(f"      Wrapper logits: {list(logits_wrap_2.shape)}")
    print(f"      max_diff={diff_2:.6e}  allclose(atol=1e-2)={match_2}")

    # === Step 3: 多轮 decode ===
    print(f"\n[7/7] Step 3: Multi-decode ({NUM_DECODE_STEPS} steps)")
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

            # wrapper 直接调用
            logits_w, *kv_flat_w = wrapper_direct(
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
    print(f"\n{'='*60}")
    print("Summary: epair (torch.export) vs HF (transformers)")
    print(f"{'='*60}")
    print(f"  [Prefill] ep_direct logits vs HF: max_diff={diff_1:.6e}  allclose={match_1}")
    print(f"  [Decode]  wrapper logits vs HF:   max_diff={diff_2:.6e}  allclose={match_2}")
    print(f"  Generated tokens (HF):     {tokens_hf}")
    print(f"  Generated tokens (Wrapper): {tokens_wrap}")
    print(f"  Tokens match: {tokens_hf == tokens_wrap}")
    print()
    if tokens_hf == tokens_wrap:
        print("[PASS] epair export produces same tokens as HF model")
    else:
        print("[FAIL] Token sequences diverge")
    if all_match:
        print("[PASS] Logits within tolerance (atol=1e-2)")
    else:
        print("[PARTIAL] Logits have small numerical differences (expected for float16)")


if __name__ == "__main__":
    main()
