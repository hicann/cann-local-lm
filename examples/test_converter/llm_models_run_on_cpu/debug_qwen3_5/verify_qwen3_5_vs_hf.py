"""Numerical alignment: pure-torch Qwen3.5 vs HuggingFace transformers Qwen3.5.

Loads the SAME VLM checkpoint into both the standalone text CausalLM and HF's
Qwen3_5ForCausalLM (text-only extraction) and compares, on identical input ids:
  - per-position logits max/mean abs diff (float32, eager attention on both)
  - next-token argmax agreement
  - greedy generation token-id equality

If argmax agrees everywhere and greedy ids match, the pure-torch implementation
is numerically correct relative to the HF reference (incl. the GatedDeltaNet
linear-attention chunk prefill and interleaved MRoPE).

Usage:
    python verify_qwen3_5_vs_hf.py --model /path/to/qwen3.5-0.8B
"""

import argparse
import os
import sys
from pathlib import Path

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "cann_lm").is_dir()
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LLM_ENGINE_DIR = os.path.dirname(
    THIS_DIR
)  # llm_models_run_on_cpu/ holds sibling hf_cpu.py
if _LLM_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _LLM_ENGINE_DIR)
import hf_cpu  # noqa: F401  disables torch_npu probe before importing transformers

from cann_lm.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    load_qwen3_5_weights,
)
from transformers import AutoTokenizer, Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM


def load_mine(model_path, dtype):
    cfg = Qwen3_5TextConfig.from_pretrained(model_path)
    m = Qwen3_5ForCausalLM(cfg).to(dtype=dtype).eval()
    m = load_qwen3_5_weights(m, model_path, device="cpu", dtype=dtype)
    return m


def load_hf(model_path, dtype):
    m = HFQwen3_5ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    m.eval()
    return m


def compare_logits(name, mine_logits, hf_logits):
    diff = (mine_logits - hf_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = hf_logits.abs().mean().item()
    rel = max_abs / max(ref_scale, 1e-8)
    mine_arg = mine_logits.argmax(-1)
    hf_arg = hf_logits.argmax(-1)
    argmatch = (mine_arg == hf_arg).float().mean().item()
    _, mt5 = torch.topk(mine_logits, 5, dim=-1)
    _, ht5 = torch.topk(hf_logits, 5, dim=-1)
    sets = [
        set(mt5[0, i].tolist()) & set(ht5[0, i].tolist()) for i in range(mt5.shape[1])
    ]
    top5 = sum(len(s) for s in sets) / (5 * len(sets))
    print(
        f"  [{name}] max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  rel={rel:.2e}  "
        f"argmax_match={argmatch * 100:.1f}%  top5_overlap={top5 * 100:.1f}%"
    )
    return max_abs, argmatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--gen-tokens", type=int, default=16)
    args = parser.parse_args()

    dtype = torch.float32
    print(f"Loading tokenizer + mine + HF (float32, eager attn) from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    mine = load_mine(args.model, dtype)
    hf = load_hf(args.model, dtype)
    n_mine = sum(p.numel() for p in mine.parameters())
    n_hf = sum(p.numel() for p in hf.parameters())
    print(f"  mine params={n_mine:,}  hf params={n_hf:,}")

    me = mine.model.embed_tokens.weight
    he = hf.model.embed_tokens.weight
    print(
        f"  embed_tokens identical: {torch.equal(me, he)}  (max|Δ|={(me - he).abs().max().item():.3e})"
    )

    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    print(f"\nPrompt tokens: {ids.shape[1]} ids")
    with torch.no_grad():
        hf_logits = hf(ids).logits
        my_logits = mine(input_ids=ids, use_cache=False)["logits"]
    print("Logits comparison (full chat prompt, all positions):")
    compare_logits("chat-prompt", my_logits, hf_logits)

    torch.manual_seed(42)
    rids = torch.randint(0, tok.vocab_size, (1, 32), dtype=torch.long)
    with torch.no_grad():
        hf_r = hf(rids).logits
        my_r = mine(input_ids=rids, use_cache=False)["logits"]
    print("\nLogits comparison (random 32-token sequence):")
    compare_logits("random-32", my_r, hf_r)

    # prefill-with-cache == prefill-without-cache (mine internal invariant)
    from cann_lm.models.qwen3_5 import SimpleKVCache, SimpleDeltaNetCache

    gdn = mine.model.layers[0].linear_attn
    kv = SimpleKVCache()
    dc = SimpleDeltaNetCache(
        num_layers=mine.config.num_hidden_layers,
        conv_dim=gdn.conv_dim,
        conv_kernel_size=gdn.conv_kernel_size,
        k_head_dim=gdn.head_k_dim,
        v_head_dim=gdn.head_v_dim,
        num_v_heads=gdn.num_v_heads,
        batch_size=1,
        device="cpu",
        dtype=dtype,
    )
    with torch.no_grad():
        cached = mine(
            input_ids=ids, past_key_values=kv, delta_states=dc, use_cache=True
        )["logits"]
    print(
        f"\nPrefill cache-vs-nocache max|Δ|: {(cached - my_logits).abs().max().item():.3e}"
    )

    n = args.gen_tokens
    with torch.no_grad():
        hf_gen = hf.generate(ids, max_new_tokens=n, do_sample=False)
        my_gen = mine.generate(
            ids, max_new_tokens=n, do_sample=False, eos_token_id=tok.eos_token_id
        )
    hf_new = hf_gen[0, ids.shape[1] :].tolist()
    my_new = my_gen[0, ids.shape[1] :].tolist()
    match = hf_new == my_new
    print(f"\nGreedy generation ({n} tokens):  identical={match}")
    print(f"  hf : {hf_new}")
    print(f"  mine: {my_new}")
    print(f"  decoded: {tok.decode(my_new, skip_special_tokens=True)!r}")
    if not match:
        for i, (a, b) in enumerate(zip(hf_new, my_new)):
            if a != b:
                print(f"  first divergence at step {i}: hf={a}  mine={b}")
                break

    print("\n" + "=" * 50)
    ok = match and (my_logits.argmax(-1) == hf_logits.argmax(-1)).all().item()
    print(
        "VERDICT:",
        "PASS — pure-torch matches HF reference" if ok else "MISMATCH — investigate",
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
