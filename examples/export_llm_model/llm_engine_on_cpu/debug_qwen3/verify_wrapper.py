#!/usr/bin/env python3
"""验证 ExportWrapper 精度：wrapper 输出 vs 原始模型输出

用法:
  python verify_wrapper.py
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


def main():
    print("[1/4] Loading model")
    config = Qwen3TextConfig.from_pretrained(HF_PATH)
    model = Qwen3ForCausalLM(config).to(dtype=torch.float16).eval()
    model = load_qwen3_weights(model, HF_PATH, device="cpu", dtype=torch.float16)
    print(f"      Params: {sum(p.numel() for p in model.parameters()):,}")

    print("[2/4] Constructing shared inputs")
    seq_len = 8
    dtype = torch.float16

    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)

    # 4D causal mask [1, 1, seq_len, seq_len] — 下三角 = 0 (attend), 上三角 = -inf (mask out)
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    attention_mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    print(f"      input_ids: {list(input_ids.shape)}")
    print(f"      attention_mask: {list(attention_mask.shape)} (4D causal)")
    print(f"      past_key_values: None (prefill, no cache)")

    print("[3/4] Running forward: original model vs wrapper")
    with torch.no_grad():
        # 原始模型 forward ([None]*N 触发 cache 创建, 4D mask 走透传)
        out_orig = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=[None] * config.num_hidden_layers,
        )
        logits_orig = out_orig["logits"]
        kv_orig = out_orig["past_key_values"]

        # wrapper forward (相同输入, past_key_values=None)
        wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
        logits_wrap, *kv_flat = wrapper(
            input_ids, attention_mask, position_ids,
            None,  # past_key_values=None (prefill)
            torch.arange(seq_len, dtype=torch.long),
            None,  # embeddings=None (图内查表)
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
    num_layers = config.num_hidden_layers
    print(f"  original: {num_layers} layers, each (k,v)")
    print(f"  wrapper:  {len(kv_flat)} flat tensors (expected {num_layers * 2})")
    all_kv_match = True
    max_kv_diff = 0.0
    for i in range(min(num_layers, len(kv_flat) // 2)):
        k_orig = kv_orig[i][0]
        v_orig = kv_orig[i][1]
        k_wrap = kv_flat[i * 2]
        v_wrap = kv_flat[i * 2 + 1]

        k_diff = (k_wrap - k_orig).abs().max().item()
        v_diff = (v_wrap - v_orig).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)

        k_match = torch.allclose(k_wrap, k_orig, atol=1e-3)
        v_match = torch.allclose(v_wrap, v_orig, atol=1e-3)
        if not k_match or not v_match:
            all_kv_match = False
            print(f"  layer {i}: MISMATCH k_diff={k_diff:.6e} v_diff={v_diff:.6e}")
            print(f"    k_orig: {list(k_orig.shape)}  k_wrap: {list(k_wrap.shape)}")
            print(f"    v_orig: {list(v_orig.shape)}  v_wrap: {list(v_wrap.shape)}")

    if all_kv_match:
        print(f"  ALL layers match (max_kv_diff={max_kv_diff:.6e})")

    print()
    print("=== Summary ===")
    if allclose and all_kv_match:
        print("[PASS] Wrapper output matches original model")
    else:
        print("[FAIL] Wrapper output differs from original model")
        print(f"  logits_match={allclose}, kv_match={all_kv_match}")


if __name__ == "__main__":
    main()
