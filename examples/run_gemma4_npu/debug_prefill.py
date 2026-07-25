#!/usr/bin/env python3
"""Debug: compare checkpoint keys vs model params, and check prefill logits."""
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    load_gemma4_weights,
)
from safetensors.torch import load_file

MP = "/mnt/workspace/data/models/gemma-4-E2B"


def remap(k: str) -> str:
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k


def main():
    cfg = json.load(open(f"{MP}/config.json"))["text_config"]
    cfg = {k: v for k, v in cfg.items() if k not in {"dtype", "expert_intermediate_size"}}
    config = Gemma4TextConfig(**cfg)

    model = Gemma4ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    param_keys = set(model.state_dict().keys())

    ckpt = load_file(f"{MP}/model.safetensors", device="cpu")
    ckpt_remapped = {remap(k): k for k in ckpt.keys()}

    print(f"model params: {len(param_keys)}, checkpoint keys: {len(ckpt)}")
    in_ckpt_not_model = set(ckpt_remapped) - param_keys
    in_model_not_ckpt = param_keys - set(ckpt_remapped)
    print(f"in checkpoint but not model: {len(in_ckpt_not_model)}")
    for k in sorted(in_ckpt_not_model)[:20]:
        print(f"  ckpt-only: {k}  (raw: {ckpt_remapped[k]})")
    print(f"in model but not checkpoint: {len(in_model_not_ckpt)}")
    for k in sorted(in_model_not_ckpt)[:20]:
        print(f"  model-only: {k}")

    # shape check on a few key params
    print("\nshape check (model vs checkpoint):")
    for name in ["model.embed_tokens.weight", "lm_head.weight",
                 "model.layers.0.self_attn.q_proj.weight",
                 "model.layers.4.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.k_proj.weight",
                 "model.layers.4.self_attn.k_proj.weight",
                 "model.per_layer_model_projection.weight",
                 "model.embed_tokens_per_layer.weight"]:
        if name in param_keys and name in ckpt_remapped:
            ms = tuple(model.state_dict()[name].shape)
            cs = tuple(ckpt[ckpt_remapped[name]].shape)
            tag = "OK" if ms == cs else "MISMATCH"
            print(f"  [{tag}] {name}: model={ms} ckpt={cs}")

    # load weights and run prefill
    print("\n--- loading weights ---")
    model = load_gemma4_weights(model, MP, device="cpu", dtype=torch.bfloat16)
    model = model.to("npu:0")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{MP}/tokenizer.json")
    prompt = "What is the capital of France?"
    enc = tok.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([[config.bos_token_id, *enc]], dtype=torch.long, device="npu:0")
    print(f"prefill input_ids ({input_ids.shape[-1]}):", input_ids[0].tolist())

    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out["logits"]  # [1, seq, vocab]
    last = logits[0, -1, :].float().cpu()
    topk = last.topk(10)
    print("prefill last-token top-10:")
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        piece = tok.decode([i])
        print(f"  id={i} logit={v:.3f} piece={piece!r}")

    # also: argmax at every position (sanity: does it copy input?)
    print("\nprefill argmax-per-position (next-token prediction):")
    preds = logits[0].argmax(-1).tolist()
    for pos, pid in enumerate(preds):
        print(f"  pos {pos}: pred_id={pid} piece={tok.decode([pid])!r}")


if __name__ == "__main__":
    main()
