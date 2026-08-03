#!/usr/bin/env python3
"""Run Gemma-4 (E2B-it / E4B-it) text model on Ascend NPU with torch_npu (raw prompt, no chat template).

Pipeline: config -> model -> load weights -> move to NPU -> tokenize (bos + prompt) -> generate -> decode

Model impl: cann_lm.models.gemma4 (pure PyTorch, no transformers dep).
Tokenizer: `tokenizers` package loads tokenizer.json directly. Raw prompt (no chat
template) works for -it models; base models would loop on raw prompt.

Run env (NPU, system Python 3.11.4 + torch_npu 2.7.1.post4):

    source /home/developer/Ascend/cann/set_env.sh
    export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
    # E2B-it (default):
    pr
    # E4B-it:
    python3 examples/run_gemma4_npu/run_gemma4_npu.py --model-path /mnt/workspace/data/models/gemma-4-E4B-it

Verify NPU available:

    python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"

Note: do NOT set TORCH_DEVICE_BACKEND_AUTOLOAD=0 here -- that flag is only for the
converter venv (CPU + torchair stubs). NPU inference needs the npu backend to autoload.
"""

import argparse
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

DEFAULT_MODEL_PATH = "/mnt/workspace/data/models/gemma-4-E2B-it"
DEVICE = "npu:0"
DTYPE = torch.bfloat16
DEFAULT_PROMPT = "What is the capital of France? Answer briefly in one sentence."
DEFAULT_MAX_NEW_TOKENS = 128

# HF config.json text_config keys absent from Gemma4TextConfig dataclass.
_HF_ONLY_KEYS = {"dtype", "expert_intermediate_size"}


def build_text_config(model_path: str) -> Gemma4TextConfig:
    with open(f"{model_path}/config.json") as f:
        cfg = json.load(f)
    fields = {k: v for k, v in cfg["text_config"].items() if k not in _HF_ONLY_KEYS}
    return Gemma4TextConfig(**fields)


def main(model_path: str, prompt: str, max_new_tokens: int) -> None:
    print(f"[1/5] Building config from {model_path}")
    config = build_text_config(model_path)
    print(
        f"      hidden_size={config.hidden_size} layers={config.num_hidden_layers} "
        f"heads={config.num_attention_heads} kv_heads={config.num_key_value_heads} "
        f"head_dim={config.head_dim} global_head_dim={config.global_head_dim} "
        f"vocab={config.vocab_size}"
    )

    print(f"[2/5] Instantiating model ({DTYPE})")
    model = Gemma4ForCausalLM(config).to(dtype=DTYPE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params:,}")

    print(f"[3/5] Loading weights (CPU) then moving to {DEVICE}")
    model = load_gemma4_weights(model, model_path, device="cpu", dtype=DTYPE)
    model = model.to(DEVICE)

    print(f"[4/5] Tokenizing prompt (raw: bos + prompt tokens)")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{model_path}/tokenizer.json")
    enc_ids = tok.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor(
        [[config.bos_token_id, *enc_ids]], dtype=torch.long, device=DEVICE
    )
    print(f"      prompt token count: {input_ids.shape[-1]}")

    print(f"[5/5] Generating (greedy, max_new_tokens={max_new_tokens})")
    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=config.eos_token_id,
    )
    new_ids = out_ids[0, input_ids.shape[-1]:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    print(f"\n=== Prompt ===\n{prompt}\n")
    print(f"=== Generated ===\n{text}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run Gemma-4-it text model on Ascend NPU.")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                    help=f"model dir (default: {DEFAULT_MODEL_PATH})")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = ap.parse_args()
    main(args.model_path, args.prompt, args.max_new_tokens)
