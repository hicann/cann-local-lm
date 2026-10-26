#!/usr/bin/env python3
"""验证 gemma4 epair/ep 推理: ep.module 回环 vs wrapper direct

适配当前 Gemma4ExportWrapper API (11 参 + 双 mask + scatter-write 绝对位 BOOL 语义):
  forward(input_ids, mask_full, mask_sliding, position_ids, past_key_values,
          new_kv_cache_pos, embeddings, output_pos, per_layer_token_identity,
          embeddings_scale, per_layer_token_identity_scale)

mask 由 host 预构 (wrapper._build_mask_full/_build_mask_sliding), BOOL
(False=attend, True=mask out); 图内无 masked_fill。KV 为预分配零 buffer,
new_kv_cache_pos 指定绝对写位置 (scatter-write)。

模式:
  --mode ep       (默认): ep.module() prefill (padded to EXPORT_SEQ_LEN) + wrapper decode, 对比 ep 回环 vs wrapper direct
  --mode wrapper        : wrapper direct 推理 (不 pad, scatter-write)
  --sampling            : sampling (默认 greedy)
  --vs-hf               : 加 HF 对比
  --use-embeddings      : 纯 embeddings (host 查表喂 embeddings+PLE), input_ids 不入图

固定: BOS + bfloat16, BOOL mask, scatter-write KV buffer [kv_cache_max_len]

用法:
  python verify_gemma4_epair.py                       # ep 回环 vs wrapper, greedy, 三 prompt
  python verify_gemma4_epair.py --mode wrapper        # wrapper direct 推理
  python verify_gemma4_epair.py --sampling             # sampling 模式
  python verify_gemma4_epair.py --vs-hf                # 加 HF 三方对比
  python verify_gemma4_epair.py --use-embeddings       # 纯 embeddings: ep 外置查表 (含 PLE) vs wrapper/HF

环境变量 (先 source 基底):
  source ../../../../scripts/cann_lm_compile/set_env_export.sh   # 或 source ../check_env.sh
  export GEMMA4_HF_PATH=<gemma4 model dir>      # 必填
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from cann_lm.converter.export.ge_converters import register_ge_converters
from transformers import AutoTokenizer

HF_PATH = os.environ["GEMMA4_HF_PATH"]
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
# mask + KV buffer helpers (scatter-write, BOOL)
# ──────────────────────────────────────────────


def make_zero_kv_buffer(wrapper, dtype):
    """零 KV buffer: list[(k,v), ...], per-layer head_dim (gemma4 sliding/full 差异)."""
    config = wrapper.config
    num_kv_heads = config.num_key_value_heads
    num_layers = wrapper._num_layers()
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


def build_masks(wrapper, seq_len, new_kv_cache_pos):
    """用 wrapper 自带 helper 构造 (mask_full, mask_sliding), BOOL scatter-write 绝对位.

    query @ 绝对位 new_kv_cache_pos[i] attend kv[0:abs+1] (full) 或窗口内 (sliding);
    未写位 [actual:kv_max] 被 mask(True) 屏蔽。
    """
    kv_max = wrapper.kv_cache_max_len
    dtype = wrapper._model_dtype
    mask_full = wrapper._build_mask_full(seq_len, kv_max, dtype, new_kv_cache_pos)
    mask_sliding = wrapper._build_mask_sliding(seq_len, kv_max, dtype, new_kv_cache_pos)
    return mask_full, mask_sliding


def mask_out_pad_rows(mask, real_seq_len):
    """ep padded prefill: pad 行 [real_seq_len:] 全屏蔽 (output 丢弃). 原地改."""
    mask[:, :, real_seq_len:, :] = True
    return mask


def mask_out_pad_slots(mask, real_seq_len, export_seq_len):
    """ep decode: 屏蔽 pad 槽位 [real_seq_len:export_seq_len] (pad KV 不参与注意).

    注: 短 seq (logical_pos < sliding_window) 下 sliding==full, 此屏蔽足够;
    长 seq 跨窗口的 sliding 非连续槽位映射未实现 (本脚本 test prompts 均 < sliding_window)。
    """
    mask[:, :, :, real_seq_len:export_seq_len] = True
    return mask


def sample_next_token(logits, temperature, top_k, top_p):
    """从 logits 采样 (temperature=0 走 argmax)."""
    if temperature == 0:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
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
    wrapper = Gemma4ExportWrapper(
        model, kv_cache_max_len=KV_CACHE_MAX_LEN, batch=1
    ).eval()
    sample_inputs = wrapper.get_sample_inputs(
        seq_len=EXPORT_SEQ_LEN, use_embeddings=use_embeddings
    )
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    return ep, wrapper, config, sample_inputs


# ──────────────────────────────────────────────
# host PLE (use_embeddings 模式)
# ──────────────────────────────────────────────


def _host_ple(wrapper, ids, dtype):
    """host 用 embed_tokens_per_layer 查表 + reshape → per_layer_token_identity [b, s, L, h_pl]."""
    m = wrapper.model.model
    cfg = wrapper.config
    return (
        m.embed_tokens_per_layer(ids)
        .reshape(*ids.shape, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)
        .to(dtype)
        .detach()
    )


# ──────────────────────────────────────────────
# wrapper 模式: wrapper direct (不 pad, scatter-write)
# ──────────────────────────────────────────────


def generate_wrapper(
    wrapper,
    input_ids,
    num_steps,
    dtype,
    sampling,
    temp,
    top_k,
    top_p,
    use_embeddings=False,
):
    """wrapper direct prefill (不 pad, 实际 seq_len) + wrapper decode (scatter-write).

    无 pad: new_kv_cache_pos = arange(seq_len); decode 写位置 = 序列逻辑位置 = seq_len+n-1。
    """
    seq_len = input_ids.shape[1]
    kv = make_zero_kv_buffer(wrapper, dtype)
    new_kv_pos = torch.arange(seq_len, dtype=torch.int32)
    mask_full, mask_sliding = build_masks(wrapper, seq_len, new_kv_pos)
    pos_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    embed_tokens = wrapper.model.model.embed_tokens
    with torch.no_grad():
        if use_embeddings:
            emb = embed_tokens(input_ids).to(dtype).detach()
            ple = _host_ple(wrapper, input_ids, dtype)
            logits, *kv_flat = wrapper(
                None,
                mask_full,
                mask_sliding,
                pos_ids,
                kv,
                new_kv_pos,
                emb,
                output_pos,
                ple,
                None,
                None,
            )
        else:
            logits, *kv_flat = wrapper(
                input_ids,
                mask_full,
                mask_sliding,
                pos_ids,
                kv,
                new_kv_pos,
                None,
                output_pos,
                None,
                None,
                None,
            )
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        if sampling:
            next_token = sample_next_token(logits, temp, top_k, top_p)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == 1:  # EOS
            break
        n = len(tokens)
        cached_pos = seq_len + n - 1  # 序列逻辑位置 = buffer 写位置 (无 pad)
        dec_new_pos = torch.tensor([cached_pos], dtype=torch.int32)
        m_full, m_slide = build_masks(wrapper, 1, dec_new_pos)
        pos_dec = torch.tensor([[cached_pos]], dtype=torch.int32)
        with torch.no_grad():
            if use_embeddings:
                dec_emb = embed_tokens(next_token).to(dtype).detach()
                dec_ple = _host_ple(wrapper, next_token, dtype)
                logits, *kv_flat = wrapper(
                    None,
                    m_full,
                    m_slide,
                    pos_dec,
                    kv,
                    dec_new_pos,
                    dec_emb,
                    None,
                    dec_ple,
                    None,
                    None,
                )
            else:
                logits, *kv_flat = wrapper(
                    next_token,
                    m_full,
                    m_slide,
                    pos_dec,
                    kv,
                    dec_new_pos,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        kv = reconstruct_kv(kv_flat)
    return tokens


# ──────────────────────────────────────────────
# ep 模式: ep.module prefill (padded) + wrapper decode
# ──────────────────────────────────────────────


def generate_ep(
    ep,
    wrapper,
    sample_inputs,
    input_ids,
    num_steps,
    dtype,
    sampling,
    temp,
    top_k,
    top_p,
    use_embeddings=False,
):
    """ep.module() prefill (padded to EXPORT_SEQ_LEN) + wrapper decode (scatter-write).

    prefill: pad 到 EXPORT_SEQ_LEN, mask 屏蔽 pad 行, output_pos 取最后 real token。
    decode: 跳过 pad 槽位 (写位置 = EXPORT_SEQ_LEN+n-1, 序列位置 = real_seq_len+n-1),
            mask 屏蔽 pad 槽 [real_seq_len:EXPORT_SEQ_LEN] + 未写 [cached:kv_max)。

    traced_inputs 结构 (gemma4 11 参):
      [0] input_ids  [1] mask_full  [2] mask_sliding  [3] position_ids
      [4] past_key_values (list[(k,v)], 不可破坏嵌套)  [5] new_kv_cache_pos
      [6] embeddings  [7] output_pos  [8] per_layer_token_identity
      [9] embeddings_scale  [10] per_layer_token_identity_scale

    use_embeddings=True: host 查表喂 embeddings (pos 6) + per_layer_token_identity (pos 8),
                         input_ids (pos 0) 保持 None (不入图)。
    """
    module = ep.module()
    real_seq_len = input_ids.shape[1]
    embed_tokens = wrapper.model.model.embed_tokens

    # pad 到 EXPORT_SEQ_LEN
    if real_seq_len < EXPORT_SEQ_LEN:
        pad_len = EXPORT_SEQ_LEN - real_seq_len
        input_ids_padded = torch.cat(
            [
                input_ids,
                torch.full((1, pad_len), PAD_TOKEN_ID, dtype=torch.long),
            ],
            dim=1,
        )
    else:
        input_ids_padded = input_ids

    # 基于 traced sample_inputs 结构替换 per-prompt 值 (保持 past_key_values [4] 嵌套不变)
    traced_inputs = list(sample_inputs)
    new_kv_pos_prefill = torch.arange(EXPORT_SEQ_LEN, dtype=torch.int32)
    mask_full, mask_sliding = build_masks(wrapper, EXPORT_SEQ_LEN, new_kv_pos_prefill)
    traced_inputs[1] = mask_out_pad_rows(mask_full.clone(), real_seq_len)
    traced_inputs[2] = mask_out_pad_rows(mask_sliding.clone(), real_seq_len)
    traced_inputs[3] = torch.arange(EXPORT_SEQ_LEN, dtype=torch.int32).unsqueeze(0)
    traced_inputs[5] = new_kv_pos_prefill
    traced_inputs[7] = torch.tensor(
        [real_seq_len - 1], dtype=torch.int32
    )  # 取最后 real token

    if use_embeddings:
        traced_inputs[0] = None  # input_ids 不入图
        traced_inputs[6] = embed_tokens(input_ids_padded).to(dtype).detach()
        traced_inputs[8] = _host_ple(wrapper, input_ids_padded, dtype)
    else:
        traced_inputs[0] = input_ids_padded

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
        if next_token.item() == 1:  # EOS
            break
        n = len(tokens)
        write_pos = (
            EXPORT_SEQ_LEN + n - 1
        )  # buffer 写位置 (跳过 pad 槽 real_seq_len..EXPORT_SEQ_LEN-1)
        seq_pos = real_seq_len + n - 1  # 序列逻辑位置 (跳过 pad 位)
        dec_new_pos = torch.tensor([write_pos], dtype=torch.int32)
        m_full, m_slide = build_masks(wrapper, 1, dec_new_pos)
        # 屏蔽 pad 槽 (短 seq 下 sliding==full, 此屏蔽足够)
        m_full = mask_out_pad_slots(m_full, real_seq_len, EXPORT_SEQ_LEN)
        m_slide = mask_out_pad_slots(m_slide, real_seq_len, EXPORT_SEQ_LEN)
        pos_id = torch.tensor([[seq_pos]], dtype=torch.int32)
        with torch.no_grad():
            if use_embeddings:
                dec_emb = embed_tokens(next_token).to(dtype).detach()
                dec_ple = _host_ple(wrapper, next_token, dtype)
                logits, *kv_flat = wrapper(
                    None,
                    m_full,
                    m_slide,
                    pos_id,
                    kv,
                    dec_new_pos,
                    dec_emb,
                    None,
                    dec_ple,
                    None,
                    None,
                )
            else:
                logits, *kv_flat = wrapper(
                    next_token,
                    m_full,
                    m_slide,
                    pos_id,
                    kv,
                    dec_new_pos,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        kv = reconstruct_kv(kv_flat)
    return tokens


# ──────────────────────────────────────────────
# HF
# ──────────────────────────────────────────────


def generate_hf(hf_model, input_ids, num_steps, sampling, temp, top_k, top_p):
    with torch.no_grad():
        kwargs = dict(
            input_ids=input_ids, max_new_tokens=num_steps, pad_token_id=PAD_TOKEN_ID
        )
        if sampling:
            kwargs.update(do_sample=True, temperature=temp, top_k=top_k, top_p=top_p)
        else:
            kwargs["do_sample"] = False
        out = hf_model.generate(**kwargs)
    return out[0][input_ids.shape[1] :].tolist()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="gemma4 epair/ep 推理验证")
    parser.add_argument(
        "--mode",
        choices=["ep", "wrapper"],
        default="ep",
        help="ep: ep.module prefill+wrapper decode (对比回环); "
        "wrapper: wrapper direct 推理",
    )
    parser.add_argument(
        "--sampling", action="store_true", help="sampling (默认 greedy)"
    )
    parser.add_argument("--vs-hf", action="store_true", help="加 HF 对比")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--prompt", type=str, default=None, help="单 prompt (默认三 prompt)"
    )
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="纯 embeddings: input_ids 不入图, host 用 embed_tokens / "
        "embed_tokens_per_layer 查表喂 embeddings + per_layer_token_identity (PLE)。"
        "对比 ep-emb vs wrapper(in-graph) vs HF, 验证 embedding 外置数值一致。",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    sampling = args.sampling

    print("[1/3] Loading + export")
    ep, wrapper, config, sample_inputs = load_and_export(
        HF_PATH, dtype, use_embeddings=args.use_embeddings
    )
    print(f"  params: {sum(p.numel() for p in wrapper.model.parameters()):,}")
    print(f"  nodes: {len(ep.graph.nodes)}")
    print(
        f"  Layers: {config.num_hidden_layers} (shared: {getattr(config, 'num_kv_shared_layers', 0)})"
    )
    print(
        f"  mode: {args.mode}, sampling: {sampling}, use_embeddings: {args.use_embeddings}"
    )

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
        print(f"\n{'=' * 60}")
        print(f'[2/3] Prompt: "{prompt}"')
        print(f"{'=' * 60}")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        real_seq_len = input_ids.shape[1]
        print(f"  tokens: {real_seq_len} (含 BOS)")

        if args.mode == "ep":
            # ep 回环 (use_embeddings 时走纯 embeddings, host 查表喂 embeddings+PLE)
            tokens_ep = generate_ep(
                ep,
                wrapper,
                sample_inputs,
                input_ids,
                args.num_steps,
                dtype,
                sampling,
                args.temp,
                args.top_k,
                args.top_p,
                use_embeddings=args.use_embeddings,
            )
            text_ep = tokenizer.decode(tokens_ep, skip_special_tokens=True)
            tag = "EP-EMB" if args.use_embeddings else "EP"
            print(f'\n  {tag} (ep.module prefill + wrapper decode): "{text_ep}"')
            # wrapper 基准 (in-graph)
            tokens_w = generate_wrapper(
                wrapper,
                input_ids,
                args.num_steps,
                dtype,
                sampling,
                args.temp,
                args.top_k,
                args.top_p,
                use_embeddings=args.use_embeddings,
            )
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            wr_tag = "WR-EMB" if args.use_embeddings else "WR"
            print(f'  {wr_tag} (wrapper direct):                "{text_w}"')
            match = tokens_ep == tokens_w
            all_pass = all_pass and match
            print(f"\n  {tag}=={wr_tag}: {match}")
        else:
            tokens_w = generate_wrapper(
                wrapper,
                input_ids,
                args.num_steps,
                dtype,
                sampling,
                args.temp,
                args.top_k,
                args.top_p,
                use_embeddings=args.use_embeddings,
            )
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f'\n  Output: "{text_w}"')

        if args.vs_hf:
            tokens_hf = generate_hf(
                hf_model,
                input_ids,
                args.num_steps,
                sampling,
                args.temp,
                args.top_k,
                args.top_p,
            )
            text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
            print(f'  HF:                                     "{text_hf}"')
            if args.mode == "ep":
                ep_tag = "EP-EMB" if args.use_embeddings else "EP"
                print(
                    f"  HF=={ep_tag}: {tokens_hf == tokens_ep}   HF==WR: {tokens_hf == tokens_w}"
                )
                all_pass = all_pass and (tokens_hf == tokens_ep)

    print(f"\n{'=' * 60}")
    print(
        f"[3/3] Summary: mode={args.mode}, sampling={sampling}, "
        f"steps={args.num_steps}, vs_hf={args.vs_hf}, use_embeddings={args.use_embeddings}"
    )
    print(f"{'=' * 60}")
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
