"""Numerical alignment: pure-torch Qwen3 vs HuggingFace transformers Qwen3.

Loads the SAME checkpoint into both implementations and compares, on identical
input ids:
  - per-position logits max/mean abs diff (float32, eager attention on both)
  - next-token argmax agreement
  - top-5 overlap
  - greedy generation token-id equality

If argmax agrees everywhere and greedy ids match, the pure-torch implementation
is numerically correct relative to the HF reference.

Usage:
    python verify_qwen3_vs_hf.py --model /path/to/qwen3-0.6B
"""

import argparse
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LLM_ENGINE_DIR = os.path.dirname(THIS_DIR)  # llm_engine_on_cpu/ holds sibling hf_cpu.py
if _LLM_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _LLM_ENGINE_DIR)
import hf_cpu  # noqa: F401  disables torch_npu probe before importing transformers

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from transformers import AutoTokenizer, Qwen3ForCausalLM as HFQwen3ForCausalLM


def load_mine(model_path, dtype):
    cfg = Qwen3TextConfig.from_pretrained(model_path)
    m = Qwen3ForCausalLM(cfg).to(dtype=dtype).eval()
    m = load_qwen3_weights(m, model_path, device="cpu", dtype=dtype)
    return m


def load_hf(model_path, dtype):
    m = HFQwen3ForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager",
    )
    m.eval()
    return m


def compare_logits(name, mine_logits, hf_logits):
    diff = (mine_logits - hf_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # reference scale for relative error
    ref_scale = hf_logits.abs().mean().item()
    rel = max_abs / max(ref_scale, 1e-8)
    mine_arg = mine_logits.argmax(-1)
    hf_arg = hf_logits.argmax(-1)
    argmatch = (mine_arg == hf_arg).float().mean().item()
    # top-5 overlap
    _, mine_top5 = torch.topk(mine_logits, 5, dim=-1)
    _, hf_top5 = torch.topk(hf_logits, 5, dim=-1)
    top5_sets = [set(mine_top5[0, i].tolist()) & set(hf_top5[0, i].tolist()) for i in range(mine_top5.shape[1])]
    top5_overlap = sum(len(s) for s in top5_sets) / (5 * len(top5_sets))
    print(f"  [{name}] max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  rel={rel:.2e}  "
          f"argmax_match={argmatch*100:.1f}%  top5_overlap={top5_overlap*100:.1f}%")
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
    print(f"  mine params={n_mine:,}  hf params={n_hf:,}  equal={n_mine == n_hf}")

    # weight-level sanity: embed_tokens should be bit-identical (same checkpoint)
    me = mine.model.embed_tokens.weight
    he = hf.model.embed_tokens.weight
    print(f"  embed_tokens identical: {torch.equal(me, he)}  (max|Δ|={(me-he).abs().max().item():.3e})")

    # --- logits comparison on a real prompt (chat-formatted) ---
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    print(f"\nPrompt tokens: {ids.shape[1]} ids")
    with torch.no_grad():
        hf_logits = hf(ids).logits
        my_logits = mine(input_ids=ids)["logits"]
    print("Logits comparison (full prompt, all positions):")
    compare_logits("chat-prompt", my_logits, hf_logits)

    # --- logits comparison on a random id sequence (breadth) ---
    torch.manual_seed(42)
    rids = torch.randint(0, tok.vocab_size, (1, 32), dtype=torch.long)
    with torch.no_grad():
        hf_r = hf(rids).logits
        my_r = mine(input_ids=rids)["logits"]
    print("\nLogits comparison (random 32-token sequence):")
    compare_logits("random-32", my_r, hf_r)

    # --- greedy generation equality ---
    n = args.gen_tokens
    with torch.no_grad():
        hf_gen = hf.generate(ids, max_new_tokens=n, do_sample=False)
        my_gen = mine.generate(ids, max_new_tokens=n, do_sample=False, eos_token_id=tok.eos_token_id)
    hf_new = hf_gen[0, ids.shape[1]:].tolist()
    my_new = my_gen[0, ids.shape[1]:].tolist()
    match = hf_new == my_new
    print(f"\nGreedy generation ({n} tokens):  identical={match}")
    print(f"  hf : {hf_new}")
    print(f"  mine: {my_new}")
    if not match:
        for i, (a, b) in enumerate(zip(hf_new, my_new)):
            if a != b:
                print(f"  first divergence at step {i}: hf={a} ({tok.decode([a])!r})  mine={b} ({tok.decode([b])!r})")
                break

    # verdict
    print("\n" + "=" * 50)
    ok = match and (mine(rids)["logits"].argmax(-1) == hf_r.argmax(-1)).all().item()
    print("VERDICT:", "PASS — pure-torch matches HF reference" if ok else "MISMATCH — investigate")
    print("=" * 50)


if __name__ == "__main__":
    main()
