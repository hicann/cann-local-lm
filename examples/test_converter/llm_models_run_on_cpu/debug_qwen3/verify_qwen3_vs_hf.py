"""Verify cann_lm Qwen3 vs HuggingFace transformers Qwen3.

Modes (--mode, default=full), all compare against HF (attn_implementation="eager"):
  full       pure-torch model.forward vs HF, float32, max_diff + greedy gen
  wrapper    Qwen3ExportWrapper vs HF, bfloat16, prefill+decode, tokens match
             (wrapper called with new_kv_cache_pos=None → cat mode, avoids
             scatter index out-of-bounds)
  embeddings use_embeddings=True (图外查表) vs HF, bfloat16, tokens match

Usage:
  python verify_qwen3_vs_hf.py --mode full
  python verify_qwen3_vs_hf.py --mode wrapper
  python verify_qwen3_vs_hf.py --mode embeddings
"""

import argparse
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = next(p for p in THIS_DIR.parents if (p / "cann_lm").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LLM_ENGINE_DIR = THIS_DIR.parent
if str(_LLM_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_ENGINE_DIR))
import hf_cpu

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.models.qwen3.export_wrapper import Qwen3ExportWrapper
from transformers import AutoTokenizer, Qwen3ForCausalLM as HFQwen3ForCausalLM

HF_PATH = os.environ["QWEN3_HF_PATH"]
NUM_DECODE_STEPS = 3


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


def make_causal_mask(seq_len, dtype):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def compare_logits(name, mine_logits, hf_logits):
    diff = (mine_logits - hf_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = hf_logits.abs().mean().item()
    rel = max_abs / max(ref_scale, 1e-8)
    mine_arg = mine_logits.argmax(-1)
    hf_arg = hf_logits.argmax(-1)
    argmatch = (mine_arg == hf_arg).float().mean().item()
    _, mine_top5 = torch.topk(mine_logits, 5, dim=-1)
    _, hf_top5 = torch.topk(hf_logits, 5, dim=-1)
    top5_sets = [set(mine_top5[0, i].tolist()) & set(hf_top5[0, i].tolist()) for i in range(mine_top5.shape[1])]
    top5_overlap = sum(len(s) for s in top5_sets) / (5 * len(top5_sets))
    print(f"  [{name}] max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  rel={rel:.2e}  "
          f"argmax_match={argmatch*100:.1f}%  top5_overlap={top5_overlap*100:.1f}%")
    return max_abs, argmatch


def run_full(args):
    dtype = torch.float32
    print(f"Loading tokenizer + mine + HF (float32, eager attn) from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    mine = load_mine(args.model, dtype)
    hf = load_hf(args.model, dtype)
    n_mine = sum(p.numel() for p in mine.parameters())
    n_hf = sum(p.numel() for p in hf.parameters())
    print(f"  mine params={n_mine:,}  hf params={n_hf:,}  equal={n_mine == n_hf}")

    me = mine.model.embed_tokens.weight
    he = hf.model.embed_tokens.weight
    print(f"  embed_tokens identical: {torch.equal(me, he)}  (max|Δ|={(me-he).abs().max().item():.3e})")

    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    print(f"\nPrompt tokens: {ids.shape[1]} ids")
    with torch.no_grad():
        hf_logits = hf(ids).logits
        my_logits = mine(input_ids=ids)["logits"]
    print("Logits comparison (full prompt, all positions):")
    compare_logits("chat-prompt", my_logits, hf_logits)

    torch.manual_seed(42)
    rids = torch.randint(0, tok.vocab_size, (1, 32), dtype=torch.long)
    with torch.no_grad():
        hf_r = hf(rids).logits
        my_r = mine(input_ids=rids)["logits"]
    print("\nLogits comparison (random 32-token sequence):")
    compare_logits("random-32", my_r, hf_r)

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

    print("\n" + "=" * 50)
    ok = match and (mine(rids)["logits"].argmax(-1) == hf_r.argmax(-1)).all().item()
    print("VERDICT:", "PASS — pure-torch matches HF reference" if ok else "MISMATCH — investigate")
    print("=" * 50)


def run_wrapper(args):
    dtype = torch.bfloat16
    seq_len = 4
    model_path = args.model

    print("[1/6] Loading HF model (transformers)")
    model_hf = load_hf(model_path, dtype)
    print(f"      Params: {sum(p.numel() for p in model_hf.parameters()):,}")

    print("[2/6] Loading custom model + wrapper (cann_lm)")
    config = Qwen3TextConfig.from_pretrained(model_path)
    model_custom = load_mine(model_path, dtype)
    wrapper = Qwen3ExportWrapper(model_custom, kv_cache_max_len=2048, batch=1).eval()
    print(f"      Params: {sum(p.numel() for p in model_custom.parameters()):,}")

    print("[3/6] Constructing shared inputs")
    vocab_size = config.vocab_size
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    mask_prefill = make_causal_mask(seq_len, dtype)
    pos_prefill = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    print(f"      input_ids: {input_ids.tolist()}")
    print(f"      mask: {list(mask_prefill.shape)} (4D causal)")

    print(f"\n[4/6] Step 1: Prefill (seq_len={seq_len})")
    with torch.no_grad():
        out_hf_1 = model_hf(
            input_ids=input_ids,
            attention_mask=mask_prefill,
            position_ids=pos_prefill,
            past_key_values=None,
            use_cache=True,
        )
        logits_hf_1 = out_hf_1.logits
        kv_hf_1 = out_hf_1.past_key_values

        logits_wrap_1, *kv_flat_1 = wrapper(
            input_ids, mask_prefill, pos_prefill,
            None,
            None,
            None,
            None,
        )

    diff_1 = (logits_wrap_1 - logits_hf_1).abs().max().item()
    match_1 = torch.allclose(logits_wrap_1, logits_hf_1, atol=1e-2)
    print(f"      HF logits:      {list(logits_hf_1.shape)}")
    print(f"      Wrapper logits: {list(logits_wrap_1.shape)}")
    print(f"      max_diff={diff_1:.6e}  allclose(atol=1e-2)={match_1}")

    kv_wrap_list = reconstruct_kv(kv_flat_1)

    print(f"\n[5/6] Step 2: Decode (1 token)")
    next_token = torch.argmax(logits_hf_1[:, -1, :], dim=-1, keepdim=True)
    print(f"      next_token: {next_token.item()}")

    cached_len = seq_len
    total_len = cached_len + 1
    mask_decode = torch.zeros(1, 1, 1, total_len, dtype=dtype)
    pos_decode = torch.tensor([[cached_len]], dtype=torch.long)

    with torch.no_grad():
        out_hf_2 = model_hf(
            input_ids=next_token,
            attention_mask=mask_decode,
            position_ids=pos_decode,
            past_key_values=kv_hf_1,
            use_cache=True,
        )
        logits_hf_2 = out_hf_2.logits

        logits_wrap_2, *kv_flat_2 = wrapper(
            next_token, mask_decode, pos_decode,
            kv_wrap_list,
            None,
            None, None,
        )

    diff_2 = (logits_wrap_2 - logits_hf_2).abs().max().item()
    match_2 = torch.allclose(logits_wrap_2, logits_hf_2, atol=1e-2)
    print(f"      HF logits:      {list(logits_hf_2.shape)}")
    print(f"      Wrapper logits: {list(logits_wrap_2.shape)}")
    print(f"      max_diff={diff_2:.6e}  allclose(atol=1e-2)={match_2}")

    print(f"\n[6/6] Step 3: Multi-decode ({NUM_DECODE_STEPS} steps)")
    tokens_hf = [next_token.item()]
    tokens_wrap = [next_token.item()]

    current_hf_cache = out_hf_2.past_key_values
    current_wrap_kv = reconstruct_kv(kv_flat_2)
    all_match = match_1 and match_2

    for step in range(NUM_DECODE_STEPS):
        next_hf = torch.tensor([[tokens_hf[-1]]], dtype=torch.long)
        cached_len = seq_len + step + 1
        mask_dec = torch.zeros(1, 1, 1, cached_len + 1, dtype=dtype)
        pos_dec = torch.tensor([[cached_len]], dtype=torch.long)

        with torch.no_grad():
            out_hf = model_hf(
                input_ids=next_hf,
                attention_mask=mask_dec,
                position_ids=pos_dec,
                past_key_values=current_hf_cache,
                use_cache=True,
            )
            logits_hf = out_hf.logits
            current_hf_cache = out_hf.past_key_values

            logits_w, *kv_flat_w = wrapper(
                next_hf, mask_dec, pos_dec,
                current_wrap_kv,
                None,
                None, None,
            )
            current_wrap_kv = reconstruct_kv(kv_flat_w)

        token_hf = torch.argmax(logits_hf[:, -1, :], dim=-1).item()
        token_w = torch.argmax(logits_w[:, -1, :], dim=-1).item()
        tokens_hf.append(token_hf)
        tokens_wrap.append(token_w)

        step_diff = (logits_w - logits_hf).abs().max().item()
        step_match = torch.allclose(logits_w, logits_hf, atol=1e-2)
        all_match = all_match and step_match
        print(f"      step {step}: token_hf={token_hf} token_wrap={token_w} "
              f"match={step_match} diff={step_diff:.6e}")

    print(f"\n{'='*50}")
    print("Summary: Wrapper vs HF (transformers) Qwen3")
    print(f"{'='*50}")
    print(f"  Prefill logits match: {match_1} (diff={diff_1:.6e})")
    print(f"  Decode  logits match: {match_2} (diff={diff_2:.6e})")
    print(f"  Generated tokens (HF):     {tokens_hf}")
    print(f"  Generated tokens (Wrapper): {tokens_wrap}")
    print(f"  Tokens match: {tokens_hf == tokens_wrap}")
    print()
    if tokens_hf == tokens_wrap:
        print("[PASS] Wrapper tokens match HF (transformers) across all steps")
    else:
        print("[FAIL] Token divergence detected")
        print(f"  all_logits_match(atol=1e-2)={all_match}")
        print(f"  tokens_match={tokens_hf == tokens_wrap}")


def run_embeddings(args):
    dtype = torch.bfloat16
    seq_len = 4
    model_path = args.model

    print("[1/5] Loading HF model (transformers)")
    model_hf = load_hf(model_path, dtype)
    print(f"      Params: {sum(p.numel() for p in model_hf.parameters()):,}")

    print("[2/5] Loading custom model + wrapper (cann_lm)")
    config = Qwen3TextConfig.from_pretrained(model_path)
    model_custom = load_mine(model_path, dtype)
    wrapper = Qwen3ExportWrapper(model_custom, kv_cache_max_len=2048, batch=1).eval()

    print("[3/5] Generating shared input_ids + computing embeddings (图外查表)")
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)
    print(f"      input_ids: {input_ids.tolist()}")

    with torch.no_grad():
        embeddings_hf = model_hf.model.embed_tokens(input_ids)
        embeddings_custom = model_custom.model.embed_tokens(input_ids)
    emb_diff = (embeddings_hf - embeddings_custom).abs().max().item()
    print(f"      embeddings shape: {list(embeddings_hf.shape)}")
    print(f"      embedding diff (HF vs custom): {emb_diff:.6e}")

    mask = make_causal_mask(seq_len, dtype)
    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    new_kv_cache_pos = torch.arange(seq_len, dtype=torch.long)
    output_pos = torch.tensor([seq_len - 1], dtype=torch.int32)

    print("[4/5] Running forward with embeddings (inputs_embeds)")
    with torch.no_grad():
        out_hf = model_hf(
            inputs_embeds=embeddings_hf,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=None,
            use_cache=True,
        )
        logits_hf = out_hf.logits
        kv_hf = out_hf.past_key_values

        logits_wrap, *kv_flat = wrapper(
            None,
            mask,
            pos,
            None,
            new_kv_cache_pos,
            embeddings_custom,
            None,
        )

    print("[5/5] Comparing outputs")
    print()
    print("=== Logits ===")
    print(f"  HF logits:      {list(logits_hf.shape)}")
    print(f"  Wrapper logits: {list(logits_wrap.shape)}")
    max_diff = (logits_wrap - logits_hf).abs().max().item()
    allclose = torch.allclose(logits_wrap, logits_hf, atol=1e-2)
    print(f"  max_diff: {max_diff:.6e}")
    print(f"  allclose(atol=1e-2): {allclose}")

    logits_wrap_selected = logits_wrap[:, output_pos.item():output_pos.item() + 1, :]
    logits_hf_selected = logits_hf[:, output_pos.item():output_pos.item() + 1, :]
    selected_diff = (logits_wrap_selected - logits_hf_selected).abs().max().item()
    print(f"  selected (output_pos={output_pos.item()}): diff={selected_diff:.6e}")

    token_hf = torch.argmax(logits_hf[:, -1, :], dim=-1).item()
    token_wrap = torch.argmax(logits_wrap[:, -1, :], dim=-1).item()
    print(f"  next_token HF: {token_hf}")
    print(f"  next_token Wrap: {token_wrap}")
    print(f"  tokens match: {token_hf == token_wrap}")

    print()
    print("=== KV Cache ===")
    num_layers = config.num_hidden_layers
    kv_wrap_list = [(kv_flat[i], kv_flat[i + 1]) for i in range(0, len(kv_flat), 2)]
    print(f"  HF: {num_layers} layers (DynamicCache)")
    print(f"  Wrapper: {len(kv_wrap_list)} pairs ({len(kv_flat)} flat tensors)")

    all_kv_match = True
    max_kv_diff = 0.0
    for i in range(min(num_layers, len(kv_wrap_list))):
        k_hf = kv_hf.layers[i].keys
        v_hf = kv_hf.layers[i].values
        k_wrap = kv_wrap_list[i][0]
        v_wrap = kv_wrap_list[i][1]

        if k_hf.shape != k_wrap.shape:
            print(f"  layer {i}: SHAPE MISMATCH k_hf={list(k_hf.shape)} k_wrap={list(k_wrap.shape)}")
            all_kv_match = False
            continue

        k_diff = (k_wrap - k_hf).abs().max().item()
        v_diff = (v_wrap - v_hf).abs().max().item()
        max_kv_diff = max(max_kv_diff, k_diff, v_diff)
        k_match = torch.allclose(k_wrap, k_hf, atol=1e-2)
        v_match = torch.allclose(v_wrap, v_hf, atol=1e-2)
        if not k_match or not v_match:
            all_kv_match = False
            print(f"  layer {i}: MISMATCH k_diff={k_diff:.6e} v_diff={v_diff:.6e}")

    if all_kv_match:
        print(f"  ALL layers match (max_kv_diff={max_kv_diff:.6e})")

    print()
    print(f"{'='*50}")
    print("Summary: use_embeddings=True (图外查表) vs HF")
    print(f"{'='*50}")
    print(f"  Embedding diff: {emb_diff:.6e}")
    print(f"  Logits max_diff: {max_diff:.6e}  allclose={allclose}")
    print(f"  KV cache max_diff: {max_kv_diff:.6e}  all_match={all_kv_match}")
    print(f"  Tokens match: {token_hf == token_wrap}")
    print()
    if token_hf == token_wrap:
        print("[PASS] 图外查表模式与 HF 模型生成一致")
    else:
        print("[FAIL] Token 不一致")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "wrapper", "embeddings"], default="full")
    parser.add_argument("--model", default=HF_PATH)
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--vs-hf", action="store_true", default=True)
    args = parser.parse_args()

    if args.mode == "full":
        run_full(args)
    elif args.mode == "wrapper":
        run_wrapper(args)
    elif args.mode == "embeddings":
        run_embeddings(args)


if __name__ == "__main__":
    main()
