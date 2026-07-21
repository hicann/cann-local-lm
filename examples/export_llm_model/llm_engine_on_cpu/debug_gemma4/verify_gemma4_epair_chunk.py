#!/usr/bin/env python3
"""验证 gemma4 scatter-write 模式: epair 导出 + 回环推理 vs wrapper 直接调用

scatter-write 模式特点:
  - KV cache 始终固定 shape [batch, heads, kv_cache_max_len, head_dim]
  - attention_mask [batch, 1, seq_len, kv_cache_max_len] causal over buffer
  - new_kv_cache_pos 指定写入位置
  - 不支持动态 shape → prefill 用 ep.module(), decode 用 wrapper

用法:
  python verify_gemma4_epair_chunk.py

环境变量:
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  _PYTHON_LIB_DIR="/opt/buildtools/python-3.12.9/lib"
  _TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs"
  export LD_LIBRARY_PATH="${_PYTHON_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}:${LD_LIBRARY_PATH:-}"
  export PYTHONUNBUFFERED=1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from cann_lm.models.fake_ops import register_ge_converters
from transformers import AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
NUM_DECODE_STEPS = 10
EXPORT_SEQ_LEN = 8
KV_CACHE_MAX_LEN = 2048
BOS_TOKEN_ID = 2

TEST_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


def make_causal_mask_over_buffer(seq_len, kv_max, dtype):
    """4D causal mask [1, 1, seq_len, kv_max]: k_pos <= q_pos → attend"""
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(seq_len, dtype=dtype).unsqueeze(1)
    k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def make_zero_kv_buffer(wrapper, num_layers, dtype):
    """创建零 KV buffer: list[(k, v), ...]"""
    config = wrapper.config
    num_kv_heads = config.num_key_value_heads
    kv_max = wrapper.kv_cache_max_len
    buf = []
    for i in range(num_layers):
        hd = wrapper._get_layer_head_dim(i)
        k = torch.zeros(1, num_kv_heads, kv_max, hd, dtype=dtype)
        v = torch.zeros(1, num_kv_heads, kv_max, hd, dtype=dtype)
        buf.append((k, v))
    return buf


def generate_wrapper(wrapper, input_ids, num_steps, dtype):
    """B: wrapper 直接调用 (scatter-write, 基准)"""
    seq_len = input_ids.shape[1]
    num_layers = wrapper.config.num_hidden_layers

    # Prefill
    kv = make_zero_kv_buffer(wrapper, num_layers, dtype)
    mask = make_causal_mask_over_buffer(seq_len, KV_CACHE_MAX_LEN, dtype)
    pos_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    new_pos = torch.arange(seq_len, dtype=torch.int64)

    with torch.no_grad():
        logits, *kv_flat = wrapper(input_ids, mask, pos_ids, kv, new_pos, None, None)
    kv = reconstruct_kv(kv_flat)

    # Decode
    tokens = []
    for step in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_pos = seq_len + step  # 下一个写入位置
        mask_dec = make_causal_mask_over_buffer(1, KV_CACHE_MAX_LEN, dtype)
        pos_dec = torch.tensor([[cached_pos]], dtype=torch.int64)
        new_pos_dec = torch.tensor([cached_pos], dtype=torch.int64)
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, mask_dec, pos_dec, kv, new_pos_dec, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


def generate_ep(ep, wrapper, sample_inputs, input_ids, num_steps, dtype):
    """A: ep.module() prefill (padded to EXPORT_SEQ_LEN) + wrapper decode

    ep.module() guard 要求 seq_len=EXPORT_SEQ_LEN, 所以短 prompt 需要 pad。
    用 pad_token_id=0 padding, output_pos 选最后一个有效 token。
    """
    module = ep.module()
    real_seq_len = input_ids.shape[1]
    num_layers = wrapper.config.num_hidden_layers

    # Pad input_ids 到 EXPORT_SEQ_LEN
    if real_seq_len < EXPORT_SEQ_LEN:
        pad_len = EXPORT_SEQ_LEN - real_seq_len
        input_ids_padded = torch.cat([
            input_ids,
            torch.zeros(1, pad_len, dtype=torch.long),
        ], dim=1)
    else:
        input_ids_padded = input_ids

    # 用 traced inputs, 替换可变部分
    traced_inputs = list(sample_inputs)
    traced_inputs[0] = input_ids_padded  # input_ids (padded)
    # mask: causal over buffer, 但 padding 位置不写入有效 KV
    # 注意: padding tokens 会写入 buffer 的位置 real_seq_len..EXPORT_SEQ_LEN-1
    # decode 时需要跳过这些位置
    traced_inputs[1] = make_causal_mask_over_buffer(EXPORT_SEQ_LEN, KV_CACHE_MAX_LEN, dtype)
    traced_inputs[2] = torch.arange(EXPORT_SEQ_LEN, dtype=torch.int64).unsqueeze(0)  # position_ids
    traced_inputs[4] = torch.arange(EXPORT_SEQ_LEN, dtype=torch.int64)  # new_kv_cache_pos
    traced_inputs[6] = torch.tensor([real_seq_len - 1], dtype=torch.int32)  # output_pos

    with torch.no_grad():
        out = module(*traced_inputs)
        logits = out[0]
        kv_flat = list(out[1:])
    kv = reconstruct_kv(kv_flat)

    # Decode: 用 wrapper (ep.module() 无法处理 seq_len != EXPORT_SEQ_LEN)
    tokens = []
    for step in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        cached_pos = EXPORT_SEQ_LEN + step  # 跳过 padding 位置
        mask_dec = make_causal_mask_over_buffer(1, KV_CACHE_MAX_LEN, dtype)
        pos_dec = torch.tensor([[cached_pos]], dtype=torch.int64)
        new_pos_dec = torch.tensor([cached_pos], dtype=torch.int64)
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, mask_dec, pos_dec, kv, new_pos_dec, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


def main():
    dtype = torch.bfloat16

    print("[1/4] Loading model + tokenizer")
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
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    print(f"\n[2/4] Exporting wrapper (scatter-write, seq_len={EXPORT_SEQ_LEN})")
    register_ge_converters()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=KV_CACHE_MAX_LEN, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    print(f"  Graph nodes: {len(ep.graph.nodes)}")
    print(f"  Export OK")

    # 验证 sample inputs 格式
    print(f"  past_key_values: {len(sample_inputs[3])} layers")
    print(f"  k[0] shape: {list(sample_inputs[3][0][0].shape)}")
    print(f"  mask shape: {list(sample_inputs[1].shape)}")

    all_pass = True

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"[3/4] Prompt: \"{prompt}\"")
        print(f"{'='*60}")

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        seq_len = input_ids.shape[1]
        print(f"  tokens: {seq_len} (含 BOS)")

        # A: ep.module() prefill + wrapper decode
        print(f"\n  A: ep prefill (padded to {EXPORT_SEQ_LEN}) + wrapper decode...")
        try:
            tokens_a = generate_ep(ep, wrapper, sample_inputs, input_ids, NUM_DECODE_STEPS, dtype)
            text_a = tokenizer.decode(tokens_a, skip_special_tokens=True)
            print(f"  A output: \"{text_a}\"")
        except Exception as e:
            print(f"  A FAILED: {e}")
            import traceback; traceback.print_exc()
            tokens_a = None
            text_a = f"ERROR: {e}"

        # B: wrapper 直接调用 (基准, 不 padding)
        print(f"\n  B: wrapper direct (seq_len={seq_len})...")
        tokens_b = generate_wrapper(wrapper, input_ids, NUM_DECODE_STEPS, dtype)
        text_b = tokenizer.decode(tokens_b, skip_special_tokens=True)
        print(f"  B output: \"{text_b}\"")

        # 对比
        if tokens_a is not None:
            match = tokens_a == tokens_b
            all_pass = all_pass and match
            print(f"\n  A vs B match: {match}")
            if not match:
                print(f"  A tokens: {tokens_a[:5]}...")
                print(f"  B tokens: {tokens_b[:5]}...")
        else:
            all_pass = False

    print(f"\n{'='*60}")
    print(f"[4/4] Summary: scatter-write ep vs wrapper direct")
    print(f"{'='*60}")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"  Decode: {NUM_DECODE_STEPS} steps")
    print(f"  KV cache: fixed [{KV_CACHE_MAX_LEN}] (scatter-write)")
    print(f"  All match: {all_pass}")
    print()
    if all_pass:
        print("[PASS] ep 回环推理与 wrapper 直接调用一致")
    else:
        print("[PARTIAL] 部分不一致")


if __name__ == "__main__":
    main()
