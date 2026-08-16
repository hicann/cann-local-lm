#!/usr/bin/env python3
"""HF reference: load Gemma-4-E2B via transformers, run prefill analysis + generate."""
import sys
from pathlib import Path

import torch

MP = "/mnt/workspace/data/models/gemma-4-E2B"
DEVICE = "npu:0"
PROMPT = "What is the capital of France?"
PROMPT_FULL = "What is the capital of France? Answer briefly in one sentence."


def main():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    print("[1/3] Loading HF tokenizer + model (bfloat16 -> npu)")
    tok = AutoTokenizer.from_pretrained(MP)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MP, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    print("      model class:", type(model).__name__)
    lm = model.model.language_model
    print("      language_model class:", type(lm).__name__)

    print(f"\n[2/3] Prefill analysis: {PROMPT!r}")
    enc = tok(PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(DEVICE)
    print(f"      input_ids ({input_ids.shape[-1]}):", input_ids[0].tolist())
    print(f"      decoded tokens:", [tok.decode([i]) for i in input_ids[0].tolist()])

    with torch.no_grad():
        lm_out = lm(input_ids=input_ids, output_hidden_states=False)
        hidden = lm_out.last_hidden_state if hasattr(lm_out, "last_hidden_state") else lm_out[0]
        logits = model.lm_head(hidden)
    last = logits[0, -1, :].float().cpu()
    topk = last.topk(10)
    print("      prefill last-token top-10:")
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"        id={i} logit={v:.3f} piece={tok.decode([i])!r}")

    print("\n      prefill argmax-per-position (next-token prediction):")
    preds = logits[0].argmax(-1).tolist()
    for pos, pid in enumerate(preds):
        print(f"        pos {pos}: pred_id={pid} piece={tok.decode([pid])!r}")

    print(f"\n[3/3] HF generate: {PROMPT_FULL!r}")
    enc2 = tok(PROMPT_FULL, return_tensors="pt", add_special_tokens=True)
    ids2 = enc2["input_ids"].to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=ids2,
            max_new_tokens=64,
            do_sample=False,
        )
    gen = tok.decode(out_ids[0, ids2.shape[-1]:], skip_special_tokens=True)
    print(f"      generated: {gen!r}")


if __name__ == "__main__":
    main()
