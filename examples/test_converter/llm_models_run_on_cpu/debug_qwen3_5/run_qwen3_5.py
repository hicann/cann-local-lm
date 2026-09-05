"""Qwen3.5 inference using the pure-torch model definition.

Uses the standalone `cann_lm.models.qwen3_5` implementation (no transformers
model code) and only relies on `transformers.AutoTokenizer` for text<->ids.

Usage:
    python run_qwen3_5.py --model /path/to/qwe3.5-0.8B --prompt "你好" --greedy
"""

import argparse
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LLM_ENGINE_DIR = os.path.dirname(THIS_DIR)  # llm_models_run_on_cpu/ holds sibling hf_cpu.py
if _LLM_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _LLM_ENGINE_DIR)
import hf_cpu  # noqa: F401  disables torch_npu probe before importing transformers

from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig, load_qwen3_5_weights
from transformers import AutoTokenizer


def load_model(model_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    cfg = Qwen3_5TextConfig.from_pretrained(model_path)
    model = Qwen3_5ForCausalLM(cfg).to(dtype=dtype).to(device=device).eval()
    model = load_qwen3_5_weights(model, model_path, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, cfg


@torch.no_grad()
def generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 256,
             do_sample: bool = True, temperature: float = 0.6) -> str:
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    out_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 (pure-torch) inference")
    parser.add_argument("--model", required=True, help="Path to Qwen3.5 checkpoint")
    parser.add_argument("--prompt", default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    print(f"Loading model: {args.model}")
    model, tokenizer, cfg = load_model(args.model, dtype=dtype_map[args.dtype])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {type(model).__name__}, {n_params / 1e9:.3f}B params, {args.dtype} on cpu")
    print(f"  layer_types: {cfg.layer_types[:8]}{' ...' if len(cfg.layer_types) > 8 else ''}")

    prompts = [p.strip() for p in args.prompt.split("||")] if "||" in args.prompt else [args.prompt]
    for p in prompts:
        print(f"\n--- prompt ---\n{p}")
        out = generate(
            model, tokenizer, p,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
        )
        print(f"\n=== output ===\n{out}")


if __name__ == "__main__":
    main()
