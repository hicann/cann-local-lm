"""Qwen3.5 inference using M:\codehub\PC_LLM\models\qwen3_5\ model definition."""

import json
import os
import shutil
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")
TRANSFORMERS_SRC = os.path.normpath(
    r"M:\transformers-main\transformers-main\src"
)

# ── deploy local model files into transformers source tree ─────
FILES_TO_COPY = [
    "__init__.py", "configuration_qwen3_5.py", "modeling_qwen3_5.py",
    "modular_qwen3_5.py", "tokenization_qwen3_5.py",
]


def _deploy():
    target = os.path.join(TRANSFORMERS_SRC, "transformers", "models", "qwen3_5")
    os.makedirs(target, exist_ok=True)
    src_dir = os.path.join(MODELS_DIR, "qwen3_5")
    for f in FILES_TO_COPY:
        src = os.path.join(src_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(target, f))
    next_src = os.path.join(MODELS_DIR, "qwen3_next")
    next_dst = os.path.join(TRANSFORMERS_SRC, "transformers", "models", "qwen3_next")
    if os.path.isdir(next_src):
        os.makedirs(next_dst, exist_ok=True)
        for f in os.listdir(next_src):
            if f.endswith(".py"):
                shutil.copy2(os.path.join(next_src, f), os.path.join(next_dst, f))
    if TRANSFORMERS_SRC not in sys.path:
        sys.path.insert(0, TRANSFORMERS_SRC)


_deploy()

# ── imports (via transformers package) ───────────────────────
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers import AutoTokenizer


# ── weight loading ───────────────────────────────────────────

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


# ── model loading ────────────────────────────────────────────

def load_model(model_name_or_path: str, device: str = "auto",
               dtype: torch.dtype = torch.bfloat16):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(model_name_or_path):
        config_path = os.path.join(model_name_or_path, "config.json")
    else:
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
    return model


# ── generation (uses transformers' GenerationMixin) ──────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, **gen_kwargs) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    generated = outputs[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── CLI ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5 inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--prompt", type=str, default="Hello! Tell me about yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device=args.device)
    print("Loading weights ...")
    load_weights(model, args.model, args.device, torch.bfloat16)
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Generating ...")
    out = generate(model, tokenizer, args.prompt,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature,
                   do_sample=args.do_sample)
    print(f"\n=== Output ===\n{out}")


if __name__ == "__main__":
    main()
