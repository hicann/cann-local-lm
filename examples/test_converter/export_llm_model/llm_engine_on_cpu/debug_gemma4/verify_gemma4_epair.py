#!/usr/bin/env python3
"""验证 gemma4 epair/ep 推理: ep.module 回环 vs wrapper direct

合并自 verify_gemma4_epair_chunk.py (ep 回环对比) + inference_ep.py (wrapper direct 推理),
去除重复 (make_zero_kv_buffer / make_causal_mask_over_buffer / reconstruct_kv / load_and_export).

模式:
  --mode ep       (默认): ep.module() prefill (padded) + wrapper decode, 对比 ep 回环 vs wrapper direct
  --mode wrapper         : wrapper direct 推理 (不 pad, scatter-write, q_start 正确)
  --sampling            : sampling (默认 greedy)
  --vs-hf               : 加 HF 对比

固定: BOS + bfloat16, scatter-write (KV buffer [kv_cache_max_len])

用法:
  python verify_gemma4_epair.py                       # ep 回环 vs wrapper, greedy, 三 prompt
  python verify_gemma4_epair.py --mode wrapper        # wrapper direct 推理
  python verify_gemma4_epair.py --sampling             # sampling 模式
  python verify_gemma4_epair.py --vs-hf                # 加 HF 三方对比
  python verify_gemma4_epair.py --prompt "Hello"
  python verify_gemma4_epair.py --use-embeddings       # 纯 embeddings: ep 外置查表 (含 PLE) vs wrapper/HF

环境变量 (详见 compare_hf_vs_ep.py docstring):
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
"""

import argparse
import dataclasses
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
from cann_lm.converter.export.ge_converters import register_ge_converters
from transformers import AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B-it"
BOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0
EXPORT_SEQ_LEN = 8
KV_CACHE_MAX_LEN = 2048
DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


# ──────────────────────────────────────────────
# mask + KV buffer helpers (共享)
# ──────────────────────────────────────────────

def make_causal_mask_over_buffer(seq_len, kv_max, dtype, q_start=0):
    """4D causal mask [1,1,seq_len,kv_max]: k_pos <= (q_start+q_idx) → attend.

    q_start: query 起始序列位置 (prefill=0, decode=cached_pos). wrapper 模式用.
    """
    min_val = torch.finfo(dtype).min
    q_pos = (torch.arange(seq_len, dtype=dtype) + q_start).unsqueeze(1)
    k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)
    return torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val).unsqueeze(0).unsqueeze(0)


def make_causal_mask_with_padding(q_len, kv_max, real_q_len, dtype):
    """prefill mask (ep 模式, pad 到 EXPORT_SEQ_LEN): causal over buffer + pad 行屏蔽."""
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(q_len, dtype=dtype).unsqueeze(1)
    k_pos = torch.arange(kv_max, dtype=dtype).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype), min_val)
    if real_q_len < q_len:
        mask[real_q_len:, :] = min_val
    return mask.unsqueeze(0).unsqueeze(0)


def make_zero_kv_buffer(wrapper, dtype):
    """零 KV buffer: list[(k,v), ...], per-layer head_dim."""
    config = wrapper.config
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    kv_max = wrapper.kv_cache_max_len
    buf = []
    for i in range(num_layers):
        hd = wrapper._get_layer_head_dim(i)
        k = torch.zeros(1, num_kv_heads, kv_max, hd, dtype=dtype)
        v = torch.zeros(1, num_kv_heads, kv_max, hd, dtype=dtype)
        buf.append((k, v))
    return buf


def reconstruct_kv(flat_kv):
    """展平 [k0,v0,...] → [(k0,v0),...]."""
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def sample_next_token(logits, temperature, top_k, top_p):
    """从 logits 采样 (temperature=0 走 argmax)."""
    if temperature == 0:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ──────────────────────────────────────────────
# model loading + export
# ──────────────────────────────────────────────

def load_and_export(hf_path, dtype, use_embeddings=False):
    with open(f"{hf_path}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid})
    if hasattr(config, "__post_init__"):
        config.__post_init__()
    model = Gemma4ForCausalLM(config).to(dtype=dtype).eval()
    model = load_gemma4_weights(model, hf_path, device="cpu", dtype=dtype)
    register_ge_converters()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=KV_CACHE_MAX_LEN, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN, use_embeddings=use_embeddings)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    return ep, wrapper, config, sample_inputs


# ──────────────────────────────────────────────
# ep 模式: ep.module prefill (padded) + wrapper decode
# ──────────────────────────────────────────────

def _host_ple(wrapper, ids, dtype):
    """host 用 embed_tokens_per_layer 查表 + reshape → per_layer_token_identity [b, s, L, h_pl]."""
    m = wrapper.model.model
    cfg = wrapper.config
    return m.embed_tokens_per_layer(ids).reshape(
        *ids.shape, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input
    ).to(dtype).detach()


def generate_ep(ep, wrapper, sample_inputs, input_ids, num_steps, dtype,
                sampling, temp, top_k, top_p, use_embeddings=False):
    """ep.module() prefill (padded to EXPORT_SEQ_LEN) + wrapper decode.

    decode 跳过 pad: 序列位置 = real_seq_len+n-1; buffer 写位置 = EXPORT_SEQ_LEN+n-1.
    mask 屏蔽 pad 位置 [real_seq_len, EXPORT_SEQ_LEN) + 未写入 [cached_len, kv_max).

    use_embeddings=True 时走纯 embeddings: host 用 embed_tokens / embed_tokens_per_layer
    查表算 embeddings (pos 5) + per_layer_token_identity (pos 7), input_ids (pos 0) 不入图。
    """
    module = ep.module()
    real_seq_len = input_ids.shape[1]
    embed_tokens = wrapper.model.model.embed_tokens

    if real_seq_len < EXPORT_SEQ_LEN:
        pad_len = EXPORT_SEQ_LEN - real_seq_len
        input_ids_padded = torch.cat([
            input_ids,
            torch.full((1, pad_len), PAD_TOKEN_ID, dtype=torch.long),
        ], dim=1)
    else:
        input_ids_padded = input_ids

    traced_inputs = list(sample_inputs)
    if use_embeddings:
        # 纯 embeddings: host 查表喂 embeddings (pos 5) + per_layer_token_identity (pos 7);
        # input_ids (pos 0) 保持 None (不入图)。
        traced_inputs[5] = embed_tokens(input_ids_padded).to(dtype).detach()
        traced_inputs[7] = _host_ple(wrapper, input_ids_padded, dtype)
    else:
        traced_inputs[0] = input_ids_padded
    traced_inputs[1] = make_causal_mask_with_padding(EXPORT_SEQ_LEN, KV_CACHE_MAX_LEN, real_seq_len, dtype)
    traced_inputs[2] = torch.arange(EXPORT_SEQ_LEN, dtype=torch.int64).unsqueeze(0)
    traced_inputs[4] = torch.arange(EXPORT_SEQ_LEN, dtype=torch.int64)
    traced_inputs[6] = torch.tensor([real_seq_len - 1], dtype=torch.int32)

    with torch.no_grad():
        out = module(*traced_inputs)
        logits = out[0]
        kv_flat = list(out[1:])
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        if sampling:
            next_token = sample_next_token(logits, temp, top_k, top_p)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == 1:
            break
        cached_len = EXPORT_SEQ_LEN + len(tokens)
        seq_pos = real_seq_len + len(tokens) - 1
        min_val = torch.finfo(dtype).min
        mask_dec = torch.zeros(1, 1, 1, KV_CACHE_MAX_LEN, dtype=dtype)
        if cached_len < KV_CACHE_MAX_LEN:
            mask_dec[:, :, :, cached_len:] = min_val
        if real_seq_len < EXPORT_SEQ_LEN:
            mask_dec[:, :, :, real_seq_len:EXPORT_SEQ_LEN] = min_val
        pos_id = torch.tensor([[seq_pos]], dtype=torch.long)
        new_pos = torch.tensor([cached_len - 1], dtype=torch.long)
        with torch.no_grad():
            if use_embeddings:
                dec_emb = embed_tokens(next_token).to(dtype).detach()
                dec_ple = _host_ple(wrapper, next_token, dtype)
                logits, *kv_flat = wrapper(None, mask_dec, pos_id, kv, new_pos, dec_emb, None, dec_ple)
            else:
                logits, *kv_flat = wrapper(next_token, mask_dec, pos_id, kv, new_pos, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


# ──────────────────────────────────────────────
# wrapper 模式: wrapper direct (不 pad)
# ──────────────────────────────────────────────

def generate_wrapper(wrapper, input_ids, num_steps, dtype,
                     sampling, temp, top_k, top_p):
    """wrapper direct prefill (不 pad, 实际 seq_len) + wrapper decode (scatter-write)."""
    seq_len = input_ids.shape[1]
    kv = make_zero_kv_buffer(wrapper, dtype)
    mask = make_causal_mask_over_buffer(seq_len, KV_CACHE_MAX_LEN, dtype)
    pos_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    new_pos = torch.arange(seq_len, dtype=torch.int64)

    with torch.no_grad():
        logits, *kv_flat = wrapper(input_ids, mask, pos_ids, kv, new_pos, None, None)
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        if sampling:
            next_token = sample_next_token(logits, temp, top_k, top_p)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == 1:
            break
        cached_pos = seq_len + len(tokens) - 1   # 序列位置 + buffer 写位置 (decode 第 n 个 = seq_len+n-1)
        mask_dec = make_causal_mask_over_buffer(1, KV_CACHE_MAX_LEN, dtype, q_start=cached_pos)
        pos_dec = torch.tensor([[cached_pos]], dtype=torch.int64)
        new_pos_dec = torch.tensor([cached_pos], dtype=torch.int64)
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, mask_dec, pos_dec, kv, new_pos_dec, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


# ──────────────────────────────────────────────
# HF
# ──────────────────────────────────────────────

def generate_hf(hf_model, input_ids, num_steps, sampling, temp, top_k, top_p):
    with torch.no_grad():
        kwargs = dict(input_ids=input_ids, max_new_tokens=num_steps,
                      pad_token_id=PAD_TOKEN_ID)
        if sampling:
            kwargs.update(do_sample=True, temperature=temp, top_k=top_k, top_p=top_p)
        else:
            kwargs["do_sample"] = False
        out = hf_model.generate(**kwargs)
    return out[0][input_ids.shape[1]:].tolist()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="gemma4 epair/ep 推理验证")
    parser.add_argument("--mode", choices=["ep", "wrapper"], default="ep",
                        help="ep: ep.module prefill+wrapper decode (对比回环); "
                             "wrapper: wrapper direct 推理")
    parser.add_argument("--sampling", action="store_true", help="sampling (默认 greedy)")
    parser.add_argument("--vs-hf", action="store_true", help="加 HF 对比")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None, help="单 prompt (默认三 prompt)")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-embeddings", action="store_true",
                        help="纯 embeddings: input_ids 不入图, host 用 embed_tokens / "
                             "embed_tokens_per_layer 查表喂 embeddings + per_layer_token_identity (PLE)。"
                             "对比 ep-emb vs wrapper(in-graph) vs HF, 验证 embedding 外置数值一致。")
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    sampling = args.sampling

    print("[1/3] Loading + export")
    ep, wrapper, config, sample_inputs = load_and_export(HF_PATH, dtype, use_embeddings=args.use_embeddings)
    print(f"  params: {sum(p.numel() for p in wrapper.model.parameters()):,}")
    print(f"  nodes: {len(ep.graph.nodes)}")
    print(f"  Layers: {config.num_hidden_layers} (shared: {getattr(config, 'num_kv_shared_layers', 0)})")
    print(f"  mode: {args.mode}, sampling: {sampling}, use_embeddings: {args.use_embeddings}")

    hf_model = None
    if args.vs_hf:
        from transformers import Gemma4ForConditionalGeneration as HFModel
        print("  Loading HF model (bfloat16)...")
        hf_model = HFModel.from_pretrained(HF_PATH, torch_dtype=dtype).eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    all_pass = True
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"[2/3] Prompt: \"{prompt}\"")
        print(f"{'='*60}")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        real_seq_len = input_ids.shape[1]
        print(f"  tokens: {real_seq_len} (含 BOS)")

        if args.mode == "ep":
            # ep 回环 (use_embeddings 时走纯 embeddings, host 查表喂 embeddings+PLE)
            tokens_ep = generate_ep(ep, wrapper, sample_inputs, input_ids, args.num_steps,
                                    dtype, sampling, args.temp, args.top_k, args.top_p,
                                    use_embeddings=args.use_embeddings)
            text_ep = tokenizer.decode(tokens_ep, skip_special_tokens=True)
            tag = "EP-EMB" if args.use_embeddings else "EP"
            print(f"\n  {tag} (ep.module prefill + wrapper decode): \"{text_ep}\"")
            # wrapper 基准 (in-graph)
            tokens_w = generate_wrapper(wrapper, input_ids, args.num_steps,
                                        dtype, sampling, args.temp, args.top_k, args.top_p)
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f"  WR (wrapper direct, in-graph):        \"{text_w}\"")
            match = tokens_ep == tokens_w
            all_pass = all_pass and match
            print(f"\n  {tag}==WR: {match}")
        else:
            tokens_w = generate_wrapper(wrapper, input_ids, args.num_steps,
                                        dtype, sampling, args.temp, args.top_k, args.top_p)
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f"\n  Output: \"{text_w}\"")

        if args.vs_hf:
            tokens_hf = generate_hf(hf_model, input_ids, args.num_steps,
                                  sampling, args.temp, args.top_k, args.top_p)
            text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
            print(f"  HF:                                     \"{text_hf}\"")
            if args.mode == "ep":
                ep_tag = "EP-EMB" if args.use_embeddings else "EP"
                print(f"  HF=={ep_tag}: {tokens_hf == tokens_ep}   HF==WR: {tokens_hf == tokens_w}")
                all_pass = all_pass and (tokens_hf == tokens_ep)

    print(f"\n{'='*60}")
    print(f"[3/3] Summary: mode={args.mode}, sampling={sampling}, "
          f"steps={args.num_steps}, vs_hf={args.vs_hf}, use_embeddings={args.use_embeddings}")
    print(f"{'='*60}")
    if args.mode == "ep":
        if all_pass:
            tag = "ep-embeddings 回环" if args.use_embeddings else "ep 回环"
            tag += " vs wrapper 一致"
            if args.vs_hf:
                tag += " + HF 一致"
            print(f"[PASS] {tag}")
        else:
            print("[PARTIAL] ep vs wrapper 不一致 (sampling 分叉或 bug)")
    else:
        print("[DONE] wrapper direct 推理完成")


if __name__ == "__main__":
    main()
