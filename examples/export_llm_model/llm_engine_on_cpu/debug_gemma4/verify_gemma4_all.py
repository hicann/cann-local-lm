#!/usr/bin/env python3
"""验证 Gemma4 ExportWrapper 全流程：单步 forward + HF 对比

用法:
  python verify_gemma4_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
SEQ_LEN = 4


def make_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def main():
    dtype = torch.float16

    print(f"[1/4] Loading gemma4 model from {HF_PATH}")
    with open(f"{HF_PATH}/config.json") as f:
        import json
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    valid_fields = {f.name for f in __import__("dataclasses").fields(Gemma4TextConfig)}
    filtered = {k: v for k, v in text_cfg.items() if k in valid_fields}
    config = Gemma4TextConfig(**filtered)
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    print(f"      hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"head_dim={config.head_dim}, vocab={config.vocab_size}")
    print(f"      layer_types={config.layer_types[:6]}... (len={len(config.layer_types)})")
    print(f"      num_kv_shared_layers={getattr(config, 'num_kv_shared_layers', 0)}")

    model = Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    model = load_gemma4_weights(model, HF_PATH, device="cpu", dtype=dtype)
    print(f"      Params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[2/4] Constructing inputs (seq_len={SEQ_LEN})")
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    mask = make_causal_mask(SEQ_LEN, dtype)
    print(f"      input_ids: {list(input_ids.shape)}")
    print(f"      mask: {list(mask.shape)} (4D causal)")

    print(f"[3/4] Running forward: wrapper vs original model")
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()

    with torch.no_grad():
        # 原始模型
        out_orig = model(
            input_ids=input_ids,
            attention_mask=mask,
            shared_kv_states={},
            output_hidden_states=False,
        )
        logits_orig = out_orig["logits"]
        kv_orig = out_orig["past_key_values"]
        shared_orig = out_orig.get("shared_kv_states")

        # wrapper
        logits_wrap, *kv_flat = wrapper(
            input_ids, mask, None, None,
            torch.arange(SEQ_LEN, dtype=torch.long),
            None, None,
        )

    print(f"[4/4] Comparing outputs")
    print()
    print("=== Logits ===")
    diff = (logits_wrap[:, -1:, :] - logits_orig[:, -1:, :]).abs().max().item()
    match = torch.allclose(logits_wrap[:, -1:, :], logits_orig[:, -1:, :], atol=1e-3)
    print(f"  original: {list(logits_orig.shape)}")
    print(f"  wrapper:  {list(logits_wrap.shape)}")
    print(f"  max_diff: {diff:.6e}")
    print(f"  allclose(atol=1e-3): {match}")

    print()
    print("=== KV Cache ===")
    num_layers = config.num_hidden_layers
    kv_pairs = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, len(kv_flat), 2)]
    print(f"  original: {num_layers} layers")
    print(f"  wrapper:  {len(kv_flat)} flat tensors ({len(kv_pairs)} pairs)")

    kv_match = True
    max_kv_diff = 0.0
    for i in range(min(num_layers, len(kv_pairs))):
        if kv_orig[i] is None:
            if i < len(kv_pairs) and kv_pairs[i][0] is not None:
                print(f"  layer {i}: orig=None but wrap has data")
                kv_match = False
            continue
        k_orig = kv_orig[i][0]
        v_orig = kv_orig[i][1]
        k_wrap = kv_pairs[i][0]
        v_wrap = kv_pairs[i][1]
        if k_orig.shape != k_wrap.shape:
            print(f"  layer {i}: SHAPE MISMATCH k_orig={list(k_orig.shape)} k_wrap={list(k_wrap.shape)}")
            kv_match = False
            continue
        k_diff = (k_wrap - k_orig).abs().max().item()
        v_diff = (v_wrap - v_orig).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)
        if not torch.allclose(k_wrap, k_orig, atol=1e-3) or not torch.allclose(v_wrap, v_orig, atol=1e-3):
            kv_match = False
            print(f"  layer {i}: MISMATCH k_diff={k_diff:.6e} v_diff={v_diff:.6e}")
    if kv_match:
        print(f"  ALL layers match (max_kv_diff={max_kv_diff:.6e})")

    print()
    print("=== shared_kv_states ===")
    if shared_orig is not None:
        print(f"  original has {len(shared_orig)} shared states")
    else:
        print(f"  original shared_kv_states: None")

    print()
    print(f"{'='*50}")
    print("Summary: Gemma4 wrapper verification")
    print(f"{'='*50}")
    print(f"  Logits: max_diff={diff:.6e} allclose={match}")
    print(f"  KV cache: all_match={kv_match} max_diff={max_kv_diff:.6e}")
    if match and kv_match:
        print("[PASS] Wrapper output matches original model")
    else:
        print("[FAIL] Differences detected")


if __name__ == "__main__":
    main()
