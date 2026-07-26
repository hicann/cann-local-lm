#!/usr/bin/env python3
"""验证 qwen3 epair/ep 推理: ep.module 回环 vs wrapper direct (+ 可选 vs HF + 动态导出)

合并自 verify_epair_vs_hf.py (epair 导出 + 回环 vs HF, scatter-write)
+ export_dynamic_shape.py (动态 shape epair 导出)。

模式:
  --mode ep       (默认): ep.module() prefill + wrapper decode (scatter mask), 对比 ep vs wrapper
  --mode wrapper         : wrapper direct 推理 (不 pad, cat KV, past=None)
  --dynamic              : 动态 shape epair 导出 (seq_len+batch symbolic)
  --sampling             : sampling (默认 greedy)
  --vs-hf                : 加 HF model.generate 对比 (attn_implementation="eager", bfloat16)

固定: bfloat16, scatter-write (KV buffer [kv_cache_max_len])
关键修复(已在 cann_lm 中): wrapper decode 用 scatter mask [1,1,1,2048] 屏蔽未写入(cached_len+1:);
_prepare_causal_mask attention_mask=None 用 total_len; prefill past=[None]*N。

用法:
  python verify_qwen3_epair.py                        # ep vs wrapper vs HF, greedy
  python verify_qwen3_epair.py --mode wrapper         # wrapper direct
  python verify_qwen3_epair.py --dynamic              # 动态 shape 导出
  python verify_qwen3_epair.py --sampling --no-vs-hf  # sampling, 不对比 HF
  python verify_qwen3_epair.py --prompt "Hello"

环境变量:
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn
from torch.export import Dim

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper
from cann_lm.converter.export.ge_converter import register_ge_converters
from cann_lm.converter.export.epair_utils import consolidated_weights_save
from transformers import AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"
PAD_TOKEN_ID = 0
KV_CACHE_MAX_LEN = 2048
DYN_OUT_DIR = Path(__file__).resolve().parent / "generate" / "qwen3_dynamic"
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


def sample_next_token(logits, temperature, top_k, top_p):
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


def load_model(hf_path, dtype):
    config = Qwen3TextConfig.from_pretrained(hf_path)
    model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_weights(model, hf_path, device="cpu", dtype=dtype)
    register_ge_converters()
    wrapper = Qwen3ExportWrapper(model, kv_cache_max_len=KV_CACHE_MAX_LEN, batch=1).eval()
    return wrapper, config


def export_for_seq_len(wrapper, seq_len):
    sample_inputs = wrapper.get_sample_inputs(seq_len=seq_len)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)
    return ep, sample_inputs


def generate_ep(ep, wrapper, sample_inputs, input_ids, num_steps, dtype,
                sampling, temp, top_k, top_p, eos_set):
    module = ep.module()
    seq_len = input_ids.shape[1]

    traced_inputs = list(sample_inputs)
    traced_inputs[0] = input_ids

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
        if next_token.item() in eos_set:
            break
        cached_len = seq_len + len(tokens) - 1
        min_val = torch.finfo(dtype).min
        mask_dec = torch.zeros(1, 1, 1, KV_CACHE_MAX_LEN, dtype=dtype)
        if cached_len + 1 < KV_CACHE_MAX_LEN:
            mask_dec[:, :, :, cached_len + 1:] = min_val
        pos_id = torch.tensor([[cached_len]], dtype=torch.long)
        new_pos = torch.tensor([cached_len], dtype=torch.long)
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, mask_dec, pos_id, kv, new_pos, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


def generate_wrapper(wrapper, input_ids, num_steps, dtype,
                     sampling, temp, top_k, top_p, eos_set):
    with torch.no_grad():
        logits, *kv_flat = wrapper(input_ids, None, None, None, None, None, None)
    kv = reconstruct_kv(kv_flat)

    tokens = []
    for _ in range(num_steps):
        if sampling:
            next_token = sample_next_token(logits, temp, top_k, top_p)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() in eos_set:
            break
        with torch.no_grad():
            logits, *kv_flat = wrapper(next_token, None, None, kv, None, None, None)
        kv = reconstruct_kv(kv_flat)
    return tokens


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


class Qwen3PrefillDynWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return out["logits"]


def _reorder_pack_attrs(epair):
    graph = epair.ep_json.get("graph_module", {}).get("graph", {})
    fixed = 0
    for n in graph.get("nodes", []):
        ai = n.get("ascend_ir")
        if not isinstance(ai, dict):
            continue
        for o in ai.get("ops", []):
            if not isinstance(o, dict) or o.get("type") != "Pack":
                continue
            attr = o.get("attr")
            if not isinstance(attr, dict) or "axis" not in attr or "N" not in attr:
                continue
            new = {"axis": attr["axis"], "N": attr["N"]}
            for k, v in attr.items():
                if k not in ("axis", "N"):
                    new[k] = v
            o["attr"] = new
            fixed += 1
    return fixed


def generate_dynamic(hf_path, dtype):
    import torch_delegate

    config = Qwen3TextConfig.from_pretrained(hf_path)
    model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model = load_qwen3_weights(model, hf_path, device="cpu", dtype=dtype)
    wrapper = Qwen3PrefillDynWrapper(model).eval()
    register_ge_converters()

    b, s = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (b, s), dtype=torch.int64)
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((s, s), min_val, dtype=dtype), diagonal=1)
    attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(b, 1, s, s).contiguous()
    position_ids = torch.arange(s, dtype=torch.int64).unsqueeze(0)
    sample_inputs = (input_ids, attention_mask, position_ids)

    seq_dim = Dim("seq_len", min=1, max=config.max_position_embeddings)
    dynamic_shapes = {
        "input_ids": {1: seq_dim},
        "attention_mask": {2: seq_dim, 3: seq_dim},
        "position_ids": {1: seq_dim},
    }

    print(f"[1/3] torch.export (dynamic seq_len, sample s={s})")
    ep = torch.export.export(
        wrapper, sample_inputs, dynamic_shapes=dynamic_shapes, strict=False
    )
    print(f"      export OK; graph nodes: {len(list(ep.graph_module.graph.nodes))}")

    print("[2/3] export_to_epair")
    epair = torch_delegate.export_to_epair(ep)
    fixed = _reorder_pack_attrs(epair)
    if fixed:
        print(f"      reordered Pack attrs (axis before N) in {fixed} nodes")

    DYN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    consolidated_weights_save(
        epair, DYN_OUT_DIR, save_weights=True, max_file_size=512 * 1024 * 1024
    )
    print(f"[3/3] saved dynamic-shape epair -> {DYN_OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="qwen3 epair/ep 推理验证")
    parser.add_argument("--mode", choices=["ep", "wrapper"], default="ep",
                        help="ep: ep.module prefill+wrapper decode; wrapper: wrapper direct")
    parser.add_argument("--dynamic", action="store_true",
                        help="动态 shape epair 导出 (seq_len symbolic)")
    parser.add_argument("--sampling", action="store_true", help="sampling (默认 greedy)")
    parser.add_argument("--vs-hf", action=argparse.BooleanOptionalAction, default=True,
                        help="加 HF model.generate 对比 (默认开, --no-vs-hf 关闭)")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None, help="单 prompt (默认三 prompt)")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    dtype = torch.bfloat16

    if args.dynamic:
        print("[1/1] Dynamic-shape epair export")
        generate_dynamic(HF_PATH, dtype)
        print()
        print("[PASS] dynamic-shape epair 导出完成")
        return

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    sampling = args.sampling

    print("[1/3] Loading model")
    wrapper, config = load_model(HF_PATH, dtype)
    print(f"  params: {sum(p.numel() for p in wrapper.model.parameters()):,}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  mode: {args.mode}, sampling: {sampling}, vs_hf: {args.vs_hf}")

    eos_set = get_eos_set(config)

    hf_model = None
    if args.vs_hf:
        from transformers import Qwen3ForCausalLM as HFModel
        print("  Loading HF model (bfloat16, eager)...")
        hf_model = HFModel.from_pretrained(
            HF_PATH, torch_dtype=dtype, attn_implementation="eager"
        ).eval()
        print(f"  HF params: {sum(p.numel() for p in hf_model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    ep_cache = {}

    all_pass = True
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"[2/3] Prompt: \"{prompt}\"")
        print(f"{'='*60}")
        input_ids = encode_prompt(tokenizer, prompt)
        seq_len = input_ids.shape[1]
        print(f"  tokens: {seq_len} (chat_template + enable_thinking=False)")

        tokens_ep = None
        if args.mode == "ep":
            if seq_len not in ep_cache:
                print(f"  Exporting for seq_len={seq_len}...")
                ep_cache[seq_len] = export_for_seq_len(wrapper, seq_len)
            ep, sample_inputs = ep_cache[seq_len]
            print(f"  nodes: {len(ep.graph.nodes)}")
            tokens_ep = generate_ep(
                ep, wrapper, sample_inputs, input_ids, args.num_steps,
                dtype, sampling, args.temp, args.top_k, args.top_p, eos_set,
            )
            text_ep = tokenizer.decode(tokens_ep, skip_special_tokens=True)
            print(f"\n  EP (ep.module prefill + wrapper decode): \"{text_ep}\"")
            tokens_w = generate_wrapper(
                wrapper, input_ids, args.num_steps,
                dtype, sampling, args.temp, args.top_k, args.top_p, eos_set,
            )
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f"  WR (wrapper direct):                   \"{text_w}\"")
            match = tokens_ep == tokens_w
            all_pass = all_pass and match
            print(f"\n  EP==WR: {match}")
        else:
            tokens_w = generate_wrapper(
                wrapper, input_ids, args.num_steps,
                dtype, sampling, args.temp, args.top_k, args.top_p, eos_set,
            )
            text_w = tokenizer.decode(tokens_w, skip_special_tokens=True)
            print(f"\n  Output: \"{text_w}\"")

        if args.vs_hf:
            tokens_hf = generate_hf(
                hf_model, input_ids, args.num_steps,
                sampling, args.temp, args.top_k, args.top_p,
            )
            text_hf = tokenizer.decode(tokens_hf, skip_special_tokens=True)
            print(f"  HF:                                     \"{text_hf}\"")
            if args.mode == "ep":
                print(f"  HF==EP: {tokens_hf == tokens_ep}   HF==WR: {tokens_hf == tokens_w}")
                all_pass = all_pass and (tokens_hf == tokens_ep)
            else:
                print(f"  HF==WR: {tokens_hf == tokens_w}")
                all_pass = all_pass and (tokens_hf == tokens_w)

    print(f"\n{'='*60}")
    print(f"[3/3] Summary: mode={args.mode}, sampling={sampling}, "
          f"steps={args.num_steps}, vs_hf={args.vs_hf}")
    print(f"{'='*60}")
    if all_pass:
        tag = "ep 回环 vs wrapper 一致" if args.mode == "ep" else "wrapper 推理完成"
        if args.vs_hf:
            tag += " + HF 一致"
        print(f"[PASS] {tag}")
    else:
        print("[FAIL/PARTIAL] 存在不一致 (bfloat16 数值差异或 sampling 分叉)")


if __name__ == "__main__":
    main()
