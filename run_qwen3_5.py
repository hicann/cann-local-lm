"""Qwen3.5 inference using rewritten torch-only modeling.

Usage:
    python run_qwen3_5.py --model Qwen/Qwen3.5-27B --prompt "Hello"
    python run_qwen3_5.py --model /path/to/local --prompt "你好" --max-tokens 256

Note: configuration_qwen3_5.py still imports from the `transformers` package,
so this script copies our files into the transformers source tree to resolve
relative imports.  After that, imports go through `transformers.models.qwen3_5`.
"""

import json
import os
import shutil
import sys
from typing import Optional

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── helpers to find/copy our files into the transformers source tree ─────

FILES = [
    "modeling_qwen3_5.py", "modular_qwen3_5.py",
    "configuration_qwen3_5.py", "tokenization_qwen3_5.py", "__init__.py",
]

TRANSFORMERS_CANDIDATES = [
    os.path.normpath(os.path.join(THIS_DIR, "..", "..", "transformers-main",
                                   "transformers-main", "src", "transformers",
                                   "models", "qwen3_5")),
    os.path.join(THIS_DIR, "transformers", "models", "qwen3_5"),
]


def _locate_qwen35_target():
    for cand in TRANSFORMERS_CANDIDATES:
        if os.path.isdir(os.path.dirname(cand)):
            return cand
    # fallback: pip-installed transformers
    import transformers as _tf
    return os.path.join(os.path.dirname(_tf.__file__), "models", "qwen3_5")


def _deploy_files(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for f in FILES:
        src = os.path.join(THIS_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(target_dir, f))
    # make sure our files shadow the installed ones
    pkg_root = os.path.dirname(os.path.dirname(target_dir))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


_deploy_files(_locate_qwen35_target())

# ── imports (now resolve via transformers package) ───────────────────────

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Cache,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5Config,
)
from transformers import AutoTokenizer


# ── weight loading ───────────────────────────────────────────────────────

def load_weights(model, model_name_or_path: str, device: str,
                 dtype: torch.dtype = torch.bfloat16):
    """Load safetensors weights into model.  Handles sharded indices."""
    from safetensors.torch import load_file

    index_file = os.path.join(model_name_or_path, "model.safetensors.index.json")
    single_file = os.path.join(model_name_or_path, "model.safetensors")

    if os.path.isfile(index_file):
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif os.path.isfile(single_file):
        shards = ["model.safetensors"]
    else:
        # download from HuggingFace Hub
        from huggingface_hub import snapshot_download
        cached = snapshot_download(model_name_or_path,
                                   allow_patterns=["*.safetensors", "*.json"])
        return load_weights(model, cached, device, dtype)

    param_dict = dict(model.named_parameters())
    loaded = set()
    for shard in shards:
        shard_path = os.path.join(model_name_or_path, shard)
        if not os.path.isfile(shard_path):
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(model_name_or_path, shard)
        state = load_file(shard_path, device="cpu")
        filtered = {k: v.to(device=device, dtype=dtype)
                    for k, v in state.items()
                    if k in param_dict}
        model.load_state_dict(filtered, strict=False)
        loaded.update(filtered.keys())
    missing = set(param_dict) - loaded
    if missing:
        print(f"[WARN] {len(missing)} keys not loaded (e.g. {list(missing)[:3]})")


# ── model loading ────────────────────────────────────────────────────────

def load_model(model_name_or_path: str, device: str = "auto",
               dtype: torch.dtype = torch.bfloat16):
    """Detect text-only vs VLM and return (model, tokenizer)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # check config for model_type
    config_path = os.path.join(model_name_or_path, "config.json")
    if not os.path.isfile(config_path):
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_name_or_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    is_vlm = cfg.get("model_type") == "qwen3_5"

    if is_vlm:
        config = Qwen3_5Config.from_pretrained(model_name_or_path)
        model = Qwen3_5ForConditionalGeneration(config)
    else:
        config = Qwen3_5TextConfig.from_pretrained(model_name_or_path)
        model = Qwen3_5ForCausalLM(config)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_name_or_path,
                                                trust_remote_code=True)


# ── generation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             top_p: float = 0.9,
             top_k: int = 50,
             eos_token_id: Optional[int] = None,
             device: Optional[str] = None) -> str:
    """Simple autoregressive generation."""
    if device is None:
        device = next(model.parameters()).device
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id or 151645

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    generated = []
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]          # last-token logits

        # ── sampling ──
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
        else:  # greedy
            next_id = logits.argmax(dim=-1, keepdim=True)

        past_key_values = outputs.past_key_values
        generated.append(next_id)
        input_ids = next_id                        # single token for next step

        if next_id.item() == eos_token_id:
            break

    all_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(all_ids[0], skip_special_tokens=True)


# ── chat convenience ─────────────────────────────────────────────────────

def chat(model, tokenizer, messages: list[dict], **gen_kwargs) -> str:
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return generate(model, tokenizer, prompt, **gen_kwargs)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5 inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--prompt", type=str,
                        default="Hello! Tell me about yourself.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device)
    print("Loading weights ...")
    load_weights(model, args.model, args.device, torch.bfloat16)
    print(f"Generating (max_tokens={args.max_tokens}) ...")
    out = generate(model, tokenizer, args.prompt,
                   max_new_tokens=args.max_tokens,
                   temperature=args.temperature)
    print(f"\n=== Output ===\n{out}")


if __name__ == "__main__":
    main()
