#!/usr/bin/env python3
"""dopt subprocess wrapper: 生成 dopt_config + 跑 GPTQ.

被 auto_quant._run_dopt_subprocess 通过 `python <path>/dopt_wrapper.py ...` 调用,
subprocess 用系统 Python (有 torch_npu + transformers + NPU env), 不走 cann_lm 的 venv.

设计动机见 docs/cann_lm_converter/quant/quant_design.md "为什么需要 _dopt_wrapper_main".
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LINEAR_STRATEGY = "Quant_act_weight_eco"
_LM_HEAD_STRATEGY = "Quant_lm_head"
_EMBED_STRATEGY = "Quant_Embed_MinMax"
_GEMMA4_EMBED_STRATEGY = "Quant_Gemma4_Embed"


def _read_model_type(model_path: str) -> str:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("model_type", "")


def _build_layer_strategy(model, model_type: str) -> dict:
    layer_strategy = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if "lm_head" in name:
                strategy = _LM_HEAD_STRATEGY
            elif "embed_tokens" in name:
                continue
            else:
                strategy = _LINEAR_STRATEGY
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": strategy}
        elif isinstance(m, torch.nn.Embedding):
            strategy = _GEMMA4_EMBED_STRATEGY if model_type.startswith("gemma4") else _EMBED_STRATEGY
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": strategy}
    return layer_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--optimize-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-type-save", action="store_true", default=False)
    parser.add_argument("--quant-bits", default="w4a8", choices=["w4a8"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_type = _read_model_type(args.model_path)
    print(f"[wrapper] model_type={model_type}", flush=True)

    from dopt_lm.opt_funcs.eco_stage.train import model_define, weight_quant_pipline
    print(f"[wrapper] loading transformers model (fp16)...", flush=True)
    model = model_define(args.model_path, torch_dtype=torch.float16)

    layer_strategy = _build_layer_strategy(model, model_type)
    dopt_config_path = os.path.join(args.output_dir, "dopt_config.json")
    with open(dopt_config_path, "w") as f:
        json.dump({"layer_strategy": layer_strategy}, f, indent=2)
    print(f"[wrapper] generated dopt_config: {dopt_config_path} ({len(layer_strategy)} layers)", flush=True)

    del model
    gc.collect()

    print(f"[wrapper] running weight_quant_pipline (GPTQ)...", flush=True)
    weight_quant_pipline(
        model_path=args.model_path,
        config_yaml=args.optimize_config,
        dopt_config=dopt_config_path,
        output_dir=args.output_dir,
        hf_type_save=args.hf_type_save,
    )
    print(f"[wrapper] done, output at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
