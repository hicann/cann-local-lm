#!/usr/bin/env python3
"""验证 Qwen3_5ExportWrapper 精度：wrapper 输出 vs 原始模型输出

用法:
  python verify_wrapper_qwen3_5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper
from cann_lm.models.qwen3_5.modeling_qwen3_5 import SimpleKVCache

HF_PATH = "/mnt/workspace/data/models/qwe3.5-0.8B"


def main():
    print("[1/4] Loading model")
    config = Qwen3_5TextConfig.from_pretrained(HF_PATH)
    model = Qwen3_5ForCausalLM(config).to(dtype=torch.float16).eval()
    model = load_qwen3_5_weights(model, HF_PATH, device="cpu", dtype=torch.float16)
    print(f"      Params: {sum(p.numel() for p in model.parameters()):,}")

    print("[2/4] Constructing shared inputs")
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    vocab_size = config.vocab_size
    seq_len = 8
    dtype = torch.float16

    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    # 4D causal mask [1, 1, seq_len, seq_len]
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    attention_mask = causal.unsqueeze(0).unsqueeze(0)

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    print(f"      input_ids: {list(input_ids.shape)}")
    print(f"      attention_mask: {list(attention_mask.shape)} (4D causal)")
    print(f"      layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    print("[3/4] Running forward: original model vs wrapper")
    with torch.no_grad():
        # 原始模型 forward (SimpleKVCache 空缓存, delta_cache=None, use_cache=True)
        cache_orig = SimpleKVCache()
        out_orig = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache_orig,
            delta_cache=None,
            use_cache=True,
        )
        logits_orig = out_orig["logits"]
        kv_cache_orig = out_orig["past_key_values"]

        # wrapper forward (past_key_values=None, wrapper 内部创建 SimpleKVCache)
        wrapper = Qwen3_5ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        logits_wrap, *kv_flat = wrapper(
            input_ids, attention_mask, position_ids,
            None,  # past_key_values=None (wrapper 内部创建 SimpleKVCache)
            torch.arange(seq_len, dtype=torch.long),
            None,  # embeddings=None
            output_pos,
        )

    print("[4/4] Comparing outputs")
    print()
    print("=== Logits ===")
    print(f"  original shape: {list(logits_orig.shape)}")
    print(f"  wrapper  shape: {list(logits_wrap.shape)}")
    expected_logits = logits_orig[:, output_pos.item():output_pos.item() + 1, :]
    max_diff = (logits_wrap - expected_logits).abs().max().item()
    allclose = torch.allclose(logits_wrap, expected_logits, atol=1e-3)
    print(f"  max_diff: {max_diff:.6e}")
    print(f"  allclose(atol=1e-3): {allclose}")

    print()
    print("=== KV Cache ===")
    # qwen3_5 是混合架构: full_attention 层用 SimpleKVCache, linear_attention 层用 delta_cache
    # 传 delta_cache=None 时, 线性注意力层无 cache, 只有 full_attention 层有 KV cache
    orig_layers = []
    if kv_cache_orig is not None:
        for i in range(len(kv_cache_orig.keys)):
            if kv_cache_orig.keys[i] is not None:
                orig_layers.append(i)
    expected_flat = len(orig_layers) * 2
    print(f"  original: {len(orig_layers)} full-attention layers in SimpleKVCache (layer indices: {orig_layers})")
    print(f"  wrapper:  {len(kv_flat)} flat tensors (expected {expected_flat})")

    all_kv_match = True
    max_kv_diff = 0.0
    for idx, layer_i in enumerate(orig_layers):
        k_orig = kv_cache_orig.keys[layer_i]
        v_orig = kv_cache_orig.values[layer_i]
        k_wrap = kv_flat[idx * 2]
        v_wrap = kv_flat[idx * 2 + 1]

        if k_orig.shape != k_wrap.shape:
            all_kv_match = False
            print(f"  layer {layer_i}: SHAPE MISMATCH k_orig={list(k_orig.shape)} k_wrap={list(k_wrap.shape)}")
            continue

        k_diff = (k_wrap - k_orig).abs().max().item()
        v_diff = (v_wrap - v_orig).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)

        k_match = torch.allclose(k_wrap, k_orig, atol=1e-2)
        v_match = torch.allclose(v_wrap, v_orig, atol=1e-2)
        status = "OK" if (k_match and v_match) else "MISMATCH"
        print(f"  layer {layer_i}: {status} k_diff={k_diff:.6e} v_diff={v_diff:.6e}")
        if not k_match or not v_match:
            all_kv_match = False

    if all_kv_match and orig_layers:
        print(f"  ALL {len(orig_layers)} full-attention layers match (max_kv_diff={max_kv_diff:.6e})")

    print()
    print("=== Summary ===")
    if allclose and all_kv_match:
        print("[PASS] Wrapper output matches original model")
    else:
        print("[FAIL] Wrapper output differs from original model")
        print(f"  logits_match={allclose}, kv_match={all_kv_match}")


if __name__ == "__main__":
    main()
