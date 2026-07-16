#!/usr/bin/env python3
"""验证 use_embeddings=True 场景：图外查表 vs HF transformers

对比流程:
  A. transformers Qwen3ForCausalLM (inputs_embeds=embeddings)
  B. cann_lm ExportWrapper (embeddings 参数, input_ids=None)

  生成 input_ids → 两边各自 embed_tokens 查表 → 传 embeddings 给模型 → 对比 logits

用法:
  python verify_wrapper_vs_hf_embeddings.py
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


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def main():
    dtype = torch.float16
    seq_len = 4

    print("[1/5] Loading HF model (transformers)")
    model_hf = HFQwen3ForCausalLM.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
    print(f"      Params: {sum(p.numel() for p in model_hf.parameters()):,}")

    print("[2/5] Loading custom model + wrapper (cann_lm)")
    config = Qwen3TextConfig.from_pretrained(HF_PATH)
    model_custom = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model_custom = load_qwen3_weights(model_custom, HF_PATH, device="cpu", dtype=dtype)
    wrapper = Qwen3ExportWrapper(model_custom, kv_cache_max_len=2048, batch=1).eval()

    print("[3/5] Generating shared input_ids + computing embeddings (图外查表)")
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)
    print(f"      input_ids: {input_ids.tolist()}")

    # 两边各自做 embedding 查表 (同一份权重, 结果应一致)
    with torch.no_grad():
        embeddings_hf = model_hf.model.embed_tokens(input_ids)
        embeddings_custom = model_custom.model.embed_tokens(input_ids)
    emb_diff = (embeddings_hf - embeddings_custom).abs().max().item()
    print(f"      embeddings shape: {list(embeddings_hf.shape)}")
    print(f"      embedding diff (HF vs custom): {emb_diff:.6e}")

    # 构造 4D causal mask + position_ids
    mask = make_causal_mask(seq_len, dtype)
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    new_kv_cache_pos = torch.arange(seq_len, dtype=torch.long)
    output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    print("[4/5] Running forward with embeddings (inputs_embeds)")
    with torch.no_grad():
        # A. HF 模型: 传 inputs_embeds (不传 input_ids)
        out_hf = model_hf(
            inputs_embeds=embeddings_hf,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=None,
            use_cache=True,
        )
        logits_hf = out_hf.logits  # [1, seq_len, vocab]
        kv_hf = out_hf.past_key_values

        # B. wrapper: 传 embeddings (input_ids=None), 走图外查表路径
        logits_wrap, *kv_flat = wrapper(
            None,           # input_ids=None
            mask,           # attention_mask
            pos,            # position_ids
            None,           # past_key_values=None
            new_kv_cache_pos,
            embeddings_custom,  # embeddings (图外查表)
            None,           # output_pos=None (返回全部 logits)
        )

    print("[5/5] Comparing outputs")
    print()
    print("=== Logits ===")
    print(f"  HF logits:      {list(logits_hf.shape)}")
    print(f"  Wrapper logits: {list(logits_wrap.shape)}")
    max_diff = (logits_wrap - logits_hf).abs().max().item()
    allclose = torch.allclose(logits_wrap, logits_hf, atol=1e-2)
    print(f"  max_diff: {max_diff:.6e}")
    print(f"  allclose(atol=1e-2): {allclose}")

    # 也对比 output_pos 选位后的结果
    logits_wrap_selected = logits_wrap[:, output_pos.item():output_pos.item() + 1, :]
    logits_hf_selected = logits_hf[:, output_pos.item():output_pos.item() + 1, :]
    selected_diff = (logits_wrap_selected - logits_hf_selected).abs().max().item()
    print(f"  selected (output_pos={output_pos.item()}): diff={selected_diff:.6e}")

    # 对比生成的 token
    token_hf = torch.argmax(logits_hf[:, -1, :], dim=-1).item()
    token_wrap = torch.argmax(logits_wrap[:, -1, :], dim=-1).item()
    print(f"  next_token HF: {token_hf}")
    print(f"  next_token Wrap: {token_wrap}")
    print(f"  tokens match: {token_hf == token_wrap}")

    # 对比 KV cache
    print()
    print("=== KV Cache ===")
    num_layers = config.num_hidden_layers
    kv_wrap_list = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, len(kv_flat), 2)]
    print(f"  HF: {num_layers} layers (DynamicCache)")
    print(f"  Wrapper: {len(kv_wrap_list)} pairs ({len(kv_flat)} flat tensors)")

    all_kv_match = True
    max_kv_diff = 0.0
    for i in range(min(num_layers, len(kv_wrap_list))):
        # HF DynamicCache: layers[i].keys, layers[i].values
        k_hf = kv_hf.layers[i].keys
        v_hf = kv_hf.layers[i].values
        k_wrap = kv_wrap_list[i][0]
        v_wrap = kv_wrap_list[i][1]

        if k_hf.shape != k_wrap.shape:
            print(f"  layer {i}: SHAPE MISMATCH k_hf={list(k_hf.shape)} k_wrap={list(k_wrap.shape)}")
            all_kv_match = False
            continue

        k_diff = (k_wrap - k_hf).abs().max().item()
        v_diff = (v_wrap - v_hf).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)
        k_match = torch.allclose(k_wrap, k_hf, atol=1e-2)
        v_match = torch.allclose(v_wrap, v_hf, atol=1e-2)
        if not k_match or not v_match:
            all_kv_match = False
            print(f"  layer {i}: MISMATCH k_diff={k_diff:.6e} v_diff={v_diff:.6e}")

    if all_kv_match:
        print(f"  ALL layers match (max_kv_diff={max_kv_diff:.6e})")

    print()
    print(f"{'='*50}")
    print("Summary: use_embeddings=True (图外查表) vs HF")
    print(f"{'='*50}")
    print(f"  Embedding diff: {emb_diff:.6e}")
    print(f"  Logits max_diff: {max_diff:.6e}  allclose={allclose}")
    print(f"  KV cache max_diff: {max_kv_diff:.6e}  all_match={all_kv_match}")
    print(f"  Tokens match: {token_hf == token_wrap}")
    print()
    if token_hf == token_wrap:
        print("[PASS] 图外查表模式与 HF 模型生成一致")
    else:
        print("[FAIL] Token 不一致")


if __name__ == "__main__":
    main()
