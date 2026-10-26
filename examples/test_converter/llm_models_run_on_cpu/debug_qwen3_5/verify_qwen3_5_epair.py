#!/usr/bin/env python3
"""验证 qwen3.5 epair/ep 推理: ep.module prefill + wrapper decode 回环 vs wrapper direct (+ vs HF)

qwen3.5 混合架构 (full_attention + linear_attention/GatedDeltaNet) 的 epair 验证.
linear attention 用 cann_lm::chunk_gated_delta_rule 融合算子 (单 GE 节点, 避免
torch.export 展开纯 torch for 循环 -> 19418 节点 -> GE segfault, 遗留任务 3 已解决).

模式:
  --mode ep       (默认): ep.module() prefill (验证 logits) + wrapper decode (ep 回环), 对比 ep vs wrapper
  --mode wrapper         : wrapper direct 推理 (cat KV, past=None)
  --vs-hf                : 加 HF model.generate 对比 (attn_implementation="eager", bfloat16)

qwen3.5 特殊: delta_cache (linear attention recurrent state) 跨调用持久 via wrapper._delta_cache_box.
ep.module() 是 trace 后的图, 不执行 _delta_cache_box[0]=... (Python 副作用), 故 ep 回环:
  1. wrapper prefill (non-ep, cat 模式): 获取 logits_w + KV_w + 更新 _delta_cache_box (delta_cache)
  2. ep.module prefill (scatter 模式, sample_inputs): 获取 logits_ep + KV_ep (验证 logits_ep==logits_w)
  3. wrapper decode: 用 KV_ep + _delta_cache_box (delta_cache 来自 step 1)

固定: bfloat16, chat_template + enable_thinking=False, greedy

用法:
  python verify_qwen3_5_epair.py                        # ep 回环 vs wrapper, greedy, 三 prompt
  python verify_qwen3_5_epair.py --mode wrapper         # wrapper direct 推理
  python verify_qwen3_5_epair.py --vs-hf                 # 加 HF 三方对比
  python verify_qwen3_5_epair.py --prompt "你好"

环境变量 (先 source 基底):
  source ../../../../scripts/cann_lm_compile/set_env_export.sh   # 或 source ../check_env.sh
  export QWEN3_5_HF_PATH=<qwen3.5 model dir>    # 必填
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    load_qwen3_5_weights,
)
from cann_lm.models.qwen3_5.export_wrapper import Qwen3_5ExportWrapper
from cann_lm.converter.export.ge_converters import register_ge_converters
from transformers import AutoTokenizer

HF_PATH = os.environ["QWEN3_5_HF_PATH"]
PAD_TOKEN_ID = 0
KV_CACHE_MAX_LEN = 2048
DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


def encode_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def get_eos_set(config):
    eos = getattr(config, "eos_token_id", None)
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, (list, tuple)):
        return set(eos)
    return set()


def load_model(hf_path, dtype):
    config = Qwen3_5TextConfig.from_pretrained(hf_path)
    model = Qwen3_5ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_5_weights(model, hf_path, device="cpu", dtype=dtype)
    register_ge_converters()
    wrapper = Qwen3_5ExportWrapper(
        model, kv_cache_max_len=KV_CACHE_MAX_LEN, batch=1
    ).eval()
    return wrapper, config


def make_causal_mask(seq_len, dtype):
    min_val = torch.finfo(dtype).min
    return (
        torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def make_decode_mask(cached_len, dtype):
    """decode mask [1,1,1,cached_len+1]: 全 0 (attend all cached)."""
    return torch.zeros(1, 1, 1, cached_len + 1, dtype=dtype)


# ──────────────────────────────────────────────
# wrapper direct (cat 模式, past=None)
# ──────────────────────────────────────────────


def generate_wrapper(wrapper, input_ids, num_steps, dtype, eos_set):
    """wrapper direct prefill (cat, past=None) + decode (cat, past=kv)."""
    seq_len = input_ids.shape[1]
    mask = make_causal_mask(seq_len, dtype)
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    wrapper._delta_cache_box[0] = None  # 重置 delta_cache

    with torch.no_grad():
        logits, *kv_flat = wrapper(input_ids, mask, pos, None, None, None, None)
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() in eos_set:
            break
        cached_len = seq_len + len(tokens) - 1
        mask_dec = make_decode_mask(cached_len, dtype)
        pos_dec = torch.tensor([[cached_len]], dtype=torch.long)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, pos_dec, kv, None, None, None
            )
        kv = reconstruct_kv(kv_flat)
    return tokens


# ──────────────────────────────────────────────
# ep 回环: ep.module prefill + wrapper decode
# ──────────────────────────────────────────────


def export_for_seq_len(wrapper, seq_len):
    """按实际 seq_len 导出 ep (形状特化)."""
    wrapper._delta_cache_box[0] = None
    sample_inputs = wrapper.get_sample_inputs(seq_len=seq_len)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    return ep, sample_inputs


def generate_ep(ep, wrapper, sample_inputs, input_ids, num_steps, dtype, eos_set):
    """ep.module() prefill (验证) + wrapper decode (ep 回环).

    delta_cache: ep.module() 不更新 _delta_cache_box (trace 无副作用), 故先 wrapper prefill
    (non-ep) 获取 delta_cache, 再 ep.module prefill 获取 KV_ep, wrapper decode 用 KV_ep + delta_cache.
    """
    module = ep.module()
    real_seq_len = input_ids.shape[1]

    # 1. wrapper prefill (non-ep, cat): 获取 delta_cache
    wrapper._delta_cache_box[0] = None
    mask_cat = make_causal_mask(real_seq_len, dtype)
    pos_cat = torch.arange(real_seq_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits_w, *kv_flat_w = wrapper(
            input_ids, mask_cat, pos_cat, None, None, None, None
        )
    # delta_cache 现已更新 (wrapper._delta_cache_box[0] 非 None)

    # 2. ep.module prefill (scatter, sample_inputs 形状特化 real_seq_len)
    # ep 输出: logits + 6 对 KV (12) + 36 delta (18 conv + 18 recurrent); 只取 KV, delta 忽略
    # (step 3 decode 用 _delta_cache_box 来自 step 1 wrapper prefill, 非 ep delta 输出)
    traced = list(sample_inputs)
    traced[0] = input_ids
    with torch.no_grad():
        out = module(*traced)
        logits_ep = out[0]
        kv_flat_ep = list(out[1:13])  # 前 12 = 6 对 full_attn KV
    kv_ep = reconstruct_kv(kv_flat_ep)

    # 验证 ep prefill logits vs wrapper prefill logits (最后一 token)
    ep_w_diff = (logits_ep - logits_w[:, -1:, :]).abs().max().item()
    print(f"  [ep prefill] logits_ep vs logits_w max_diff={ep_w_diff:.6e}")

    # 3. wrapper decode: 用 KV_ep + _delta_cache_box (delta_cache 来自 step 1)
    min_val = torch.finfo(dtype).min
    tokens = []
    logits_cur = logits_ep
    for _ in range(num_steps):
        next_token = torch.argmax(logits_cur[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() in eos_set:
            break
        cached_len = real_seq_len + len(tokens) - 1
        dec_nkp = torch.tensor(
            [cached_len], dtype=torch.int32
        )  # scatter-write 位置 (key 固定 kv_max, 对齐 mask kv 维)
        mask_dec = torch.zeros(1, 1, 1, KV_CACHE_MAX_LEN, dtype=dtype)
        if cached_len + 1 < KV_CACHE_MAX_LEN:
            mask_dec[:, :, :, cached_len + 1 :] = min_val
        pos_dec = torch.tensor([[cached_len]], dtype=torch.long)
        with torch.no_grad():
            logits_cur, *kv_flat_ep = wrapper(
                next_token, mask_dec, pos_dec, kv_ep, dec_nkp, None, None
            )
        kv_ep = reconstruct_kv(kv_flat_ep)
    return tokens


# ──────────────────────────────────────────────
# HF
# ──────────────────────────────────────────────


def generate_hf(hf_model, input_ids, num_steps):
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=num_steps,
            pad_token_id=PAD_TOKEN_ID,
            do_sample=False,
        )
    return out[0][input_ids.shape[1] :].tolist()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="qwen3.5 epair/ep 推理验证")
    parser.add_argument(
        "--mode",
        choices=["ep", "wrapper"],
        default="ep",
        help="ep: ep.module prefill+wrapper decode; wrapper: wrapper direct",
    )
    parser.add_argument(
        "--vs-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="加 HF model.generate 对比 (默认开, --no-vs-hf 关闭)",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--prompt", type=str, default=None, help="单 prompt (默认三 prompt)"
    )
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print("[1/3] Loading model")
    wrapper, config = load_model(HF_PATH, dtype)
    print(f"  params: {sum(p.numel() for p in wrapper.model.parameters()):,}")
    print(
        f"  Layers: {config.num_hidden_layers} (full_attn: {sum(1 for lt in getattr(config, 'layer_types', []) if lt == 'full_attention')})"
    )
    print(f"  mode: {args.mode}, vs_hf: {args.vs_hf}")

    eos_set = get_eos_set(config)

    # export ep (按 prompt seq_len 形状特化, 缓存)
    ep_cache = {}
    if args.mode == "ep":
        print("  (ep 按每个 prompt 的 seq_len 形状特化导出, 缓存复用)")

    hf_model = None
    if args.vs_hf:
        from transformers import Qwen3_5ForCausalLM as HFModel

        print("  Loading HF model (bfloat16, eager)...")
        hf_model = HFModel.from_pretrained(
            HF_PATH, torch_dtype=dtype, attn_implementation="eager"
        ).eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    all_pass = True
    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f'[2/3] Prompt: "{prompt}"')
        print(f"{'=' * 60}")
        input_ids = encode_prompt(tokenizer, prompt)
        seq_len = input_ids.shape[1]
        print(f"  tokens: {seq_len} (chat_template + enable_thinking=False)")

        tokens_ep = None
        if args.mode == "ep":
            if seq_len not in ep_cache:
                print(f"  Exporting for seq_len={seq_len}...")
                ep_cache[seq_len] = export_for_seq_len(wrapper, seq_len)
                print(f"  nodes: {len(ep_cache[seq_len][0].graph.nodes)}")
            ep, sample_inputs = ep_cache[seq_len]
            tokens_ep = generate_ep(
                ep, wrapper, sample_inputs, input_ids, args.num_steps, dtype, eos_set
            )
            text_ep = tokenizer.decode(tokens_ep, skip_special_tokens=True)
            print(f'\n  EP (ep.module prefill + wrapper decode): "{text_ep}"')
            tokens_w = generate_wrapper(
                wrapper, input_ids, args.num_steps, dtype, eos_set
            )
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f'  WR (wrapper direct):                   "{text_w}"')
            match = tokens_ep == tokens_w
            all_pass = all_pass and match
            print(f"\n  EP==WR: {match}")
        else:
            tokens_w = generate_wrapper(
                wrapper, input_ids, args.num_steps, dtype, eos_set
            )
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f'\n  Output: "{text_w}"')

        if args.vs_hf:
            tokens_hf = generate_hf(hf_model, input_ids, args.num_steps)
            text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
            print(f'  HF:                                     "{text_hf}"')
            if args.mode == "ep":
                print(
                    f"  HF==EP: {tokens_hf == tokens_ep}   HF==WR: {tokens_hf == tokens_w}"
                )
                all_pass = all_pass and (tokens_hf == tokens_ep)
            else:
                print(f"  HF==WR: {tokens_hf == tokens_w}")
                all_pass = all_pass and (tokens_hf == tokens_w)

    print(f"\n{'=' * 60}")
    print(
        f"[3/3] Summary: mode={args.mode}, steps={args.num_steps}, vs_hf={args.vs_hf}"
    )
    print(f"{'=' * 60}")
    if all_pass:
        tag = "ep 回环 vs wrapper 一致" if args.mode == "ep" else "wrapper 推理完成"
        if args.vs_hf:
            tag += " + HF 一致"
        print(f"[PASS] {tag}")
    else:
        print(
            "[FAIL/PARTIAL] 存在不一致 (bfloat16 数值差异或 ep 回环 delta_cache 差异)"
        )


if __name__ == "__main__":
    main()
