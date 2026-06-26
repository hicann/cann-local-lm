"""Qwen3.5 inference – uses local model definition only (no transformers imports)."""

import json
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

# ── local model files ──────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5Config, Qwen3_5VisionConfig
from modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


# ── weight loader ──────────────────────────────────────────

def load_weights(model, model_path: str, device: str,
                 dtype: torch.dtype = torch.bfloat16):
    from safetensors.torch import load_file

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    single_file = os.path.join(model_path, "model.safetensors")

    if os.path.isfile(index_file):
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif os.path.isfile(single_file):
        shards = ["model.safetensors"]
    else:
        from huggingface_hub import snapshot_download
        cached = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json"])
        return load_weights(model, cached, device, dtype)

    param_dict = dict(model.named_parameters())
    loaded = set()
    for shard in shards:
        shard_path = os.path.join(model_path, shard)
        if not os.path.isfile(shard_path):
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(model_path, shard)
        state = load_file(shard_path, device="cpu")
        filtered = {k: v.to(device=device, dtype=dtype)
                    for k, v in state.items()
                    if k in param_dict}
        model.load_state_dict(filtered, strict=False)
        loaded.update(filtered.keys())
    missing = set(param_dict) - loaded
    if missing:
        print(f"[WARN] {len(missing)} keys missing (e.g. {list(missing)[:3]})")


# ── model loader ───────────────────────────────────────────

def load_model(model_name_or_path: str, device: str = "auto",
               dtype: torch.dtype = torch.bfloat16):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # read config.json to determine model type
    if os.path.isdir(model_name_or_path):
        config_path = os.path.join(model_name_or_path, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_name_or_path, "config.json")

    with open(config_path) as f:
        cfg = json.load(f)

    is_vlm = cfg.get("model_type") == "qwen3_5"

    if is_vlm:
        config = Qwen3_5Config(**cfg)
        model = Qwen3_5ForConditionalGeneration(config)
    else:
        config = Qwen3_5TextConfig(**cfg)
        model = Qwen3_5ForCausalLM(config)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model


# ── tokenizer wrapper (separate, uses transformers) ────────

def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


# ── generation loop ────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             top_p: float = 0.9,
             top_k: int = 50,
             eos_token_id: Optional[int] = None,
             device: Optional[str] = None) -> str:
    if device is None:
        device = next(model.parameters()).device
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id or 151645

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = []
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]

        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[:, -1:]] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_idx = probs.sort(descending=True, dim=-1)
                cumsum = sorted_p.cumsum(dim=-1)
                mask = cumsum - sorted_p > (1 - top_p)
                sorted_p[mask] = 0.0
                probs = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                next_id = torch.multinomial(probs, 1).gather(1, sorted_idx)
            else:
                next_id = torch.multinomial(probs, 1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        past_key_values = outputs.past_key_values
        generated.append(next_id)
        input_ids = next_id

        if next_id.item() == eos_token_id:
            break

    all_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(all_ids[0], skip_special_tokens=True)


# ── CLI ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5 inference (standalone)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--prompt", type=str, default="Hello! Tell me about yourself.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device=args.device)
    print("Loading weights ...")
    load_weights(model, args.model, args.device, torch.bfloat16)
    print("Loading tokenizer ...")
    tokenizer = load_tokenizer(args.model)
    print(f"Generating (max_tokens={args.max_tokens}) ...")
    out = generate(model, tokenizer, args.prompt,
                   max_new_tokens=args.max_tokens,
                   temperature=args.temperature)
    print(f"\n=== Output ===\n{out}")


if __name__ == "__main__":
    main()
