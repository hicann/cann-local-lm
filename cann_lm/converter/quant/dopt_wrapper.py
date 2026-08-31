#!/usr/bin/env python3
"""dopt subprocess wrapper: generate dopt_config + run GPTQ.

Invoked by auto_quant._run_dopt_subprocess via ``python <path>/dopt_wrapper.py ...``;
the subprocess uses system Python (torch_npu + transformers + NPU env), not the
cann_lm converter venv. Design rationale: see docs/cann_lm_converter/quant/quant_design.md.
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

_LINEAR_STRATEGY = "Quant_INT8-4_ECO"
_EMBED_STRATEGY = "Quant_Embed_MinMax"


def _read_model_type(model_path: str) -> str:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("model_type", "")


def build_layer_strategy(model, model_type: str) -> dict:
    layer_strategy = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if "embed_tokens" in name:
                continue
            layer_strategy[name] = {
                "type": str(type(m)),
                "quant_strategy": _LINEAR_STRATEGY,
            }
        elif isinstance(m, torch.nn.Embedding):
            strategy = _EMBED_STRATEGY
            layer_strategy[name] = {"type": str(type(m)), "quant_strategy": strategy}
    return layer_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--optimize-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-type-save", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_type = _read_model_type(args.model_path)
    print(f"[wrapper] model_type={model_type}", flush=True)

    from dopt_lm.opt_funcs.eco_stage.train import model_define, weight_quant_pipline
    print("[wrapper] loading transformers model (fp16)...", flush=True)
    model = model_define(args.model_path, torch_dtype=torch.float16)

    layer_strategy = build_layer_strategy(model, model_type)
    dopt_config_path = os.path.join(args.output_dir, "dopt_config.json")
    with open(dopt_config_path, "w") as f:
        json.dump({"layer_strategy": layer_strategy}, f, indent=2)
    print(f"[wrapper] generated dopt_config: {dopt_config_path} ({len(layer_strategy)} layers)", flush=True)

    # free before GPTQ: weight_quant_pipline reloads the model from model_path internally (avoids 2x peak memory)
    del model
    gc.collect()

    print("[wrapper] running weight_quant_pipline (GPTQ)...", flush=True)
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
