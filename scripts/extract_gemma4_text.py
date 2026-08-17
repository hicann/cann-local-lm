#!/usr/bin/env python3
"""Extract text-only Gemma4ForCausalLM from a multimodal Gemma4ForConditionalGeneration dir.

dopt GPTQ targets text-only ForCausalLM (it accesses ``model.model.layers`` /
``model.model.embed_tokens`` / ``model.lm_head`` directly). A multimodal
``Gemma4ForConditionalGeneration`` checkpoint has the text model under
``model.language_model`` (plus audio_tower/vision_tower), so dopt's
``AutoModelForCausalLM.from_pretrained`` loads the wrong shape and breaks.

This script loads ``Gemma4ForCausalLM`` directly from the multimodal checkpoint
using the nested ``text_config`` — transformers auto-remaps
``model.language_model.*`` -> ``model.*`` and drops audio/vision keys as
UNEXPECTED — then ``save_pretrained`` a standalone text-only HF dir + copies
tokenizer files. Point dopt / ``scripts/quant_export.sh`` at the result.

Usage:
  python scripts/extract_gemma4_text.py \
      --hf-path /mnt/workspace/data/models/gemma-4-E4B-it/ \
      --output-dir /mnt/workspace/data/models/gemma-4-E4B-it-textonly/
"""
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, Gemma4ForCausalLM

_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-path", required=True, help="multimodal Gemma4ForConditionalGeneration dir")
    ap.add_argument("--output-dir", required=True, help="output text-only Gemma4ForCausalLM dir")
    ap.add_argument("--dtype", default="float16", choices=list(_DTYPE))
    args = ap.parse_args()

    dt = _DTYPE[args.dtype]
    print(f"[1/3] loading text_config from {args.hf_path}")
    full_cfg = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    text_cfg = full_cfg.text_config
    print(f"      text model_type={text_cfg.model_type} layers={text_cfg.num_hidden_layers} "
          f"hidden={text_cfg.hidden_size}")

    print(f"[2/3] loading Gemma4ForCausalLM (text-only) with auto key-remap, dtype={args.dtype}")
    model = Gemma4ForCausalLM.from_pretrained(
        args.hf_path, config=text_cfg, dtype=dt, low_cpu_mem_usage=True,
    )
    model = model.cpu().eval()
    n_layers = len(model.model.layers)
    print(f"      loaded {type(model).__name__}: {n_layers} layers, "
          f"embed_tokens.embed_scale={getattr(model.model.embed_tokens, 'embed_scale', None) is not None}")
    del full_cfg

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] saving text-only model + tokenizer to {out}")
    model.save_pretrained(out, safe_serialization=True)
    del model
    import gc; gc.collect()

    src = Path(args.hf_path)
    for f in _TOKENIZER_FILES:
        s = src / f
        if s.exists():
            shutil.copy2(s, out / f)
            print(f"      copied {f}")
    print(f"\ndone. text-only gemma4 at: {out}")


if __name__ == "__main__":
    main()
