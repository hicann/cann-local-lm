#!/usr/bin/env python3
"""Generate golden reference data with per-op intermediate dump.

CPU eager execution via torch.fx.Interpreter — every FX node's output is
saved as a .bin file, mirroring ONNX Runtime's tensor dump capability.

Usage:
  # gemma4 5-layer prefill
  python gen_golden_with_dump.py --model-type gemma4 \
      --hf-path /mnt/workspace/models/unsloth/gemma-4-E2B-it \
      --seq-len 128 --num-layers 5 --dtype float16 \
      --output-dir out/golden_gemma4_5l

  # qwen3.5 5-layer prefill
  python gen_golden_with_dump.py --model-type qwen3_5 \
      --hf-path /mnt/workspace/models/Qwen/Qwen3.5-0.8B \
      --seq-len 8 --num-layers 5 --dtype float16 \
      --output-dir out/golden_qwen35_5l

  # Only dump inputs/outputs (skip intermediates, saves disk)
  python gen_golden_with_dump.py ... --no-intermediates

Output layout (under --output-dir):
  input_<idx>_<name>.bin   — each input tensor
  output_<idx>.bin          — each output tensor
  node_<name>[_<j>].bin     — each FX node's output (if --intermediates)
  golden_meta.json          — metadata (names, shapes, dtypes, targets)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.fx import Interpreter


class DumpInterpreter(Interpreter):
    """FX Interpreter that captures every node's output tensor."""

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.dump: dict[str, Any] = {}

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        self.dump[n.name] = result
        return result


def _dtype_arg(s: str) -> torch.dtype:
    dt = {"float16": torch.float16, "fp16": torch.float16,
          "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float32": torch.float32, "fp32": torch.float32}
    if s not in dt:
        raise ValueError(f"Unsupported dtype: {s}")
    return dt[s]


def _flatten_tensors(obj: Any) -> list[torch.Tensor]:
    """Extract all tensors from a (possibly nested) structure."""
    out: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_flatten_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_tensors(v))
    return out


def _save_tensor(t: torch.Tensor, path: Path) -> dict:
    t = t.cpu().detach().contiguous()
    # numpy 不支持 bfloat16; 无损上转 float32 保存 (bfloat16 → float32 精确, 不丢信息;
    # 比对时 OM 输出也上转 float32 做容差比较即可). meta 顶层 dtype 记模型 dtype (torch.bfloat16).
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    arr = t.numpy()
    arr.tofile(path)
    return {"file": path.name, "shape": list(arr.shape),
            "dtype": str(arr.dtype)}


def _get_sample_inputs(wrapper, seq_len: int, use_embeddings: bool):
    """Get sample inputs from the wrapper, matching the converter's flow."""
    return wrapper.get_sample_inputs(
        seq_len=seq_len, use_embeddings=use_embeddings, quant_embeddings=False
    )


def _get_input_names(wrapper: nn.Module) -> list[str]:
    """wrapper.forward 签名的参数名 (按顺序, 与 get_sample_inputs 返回的 tuple 对齐).

    用 forward 签名而非 FX placeholder 名: strict=False 下 torch.export 给嵌套输入
    (如 past_key_values=list[(K,V)]) 的 placeholder 名是树路径衍生 (past_key_values_0_0),
    且 placeholder 数与 sample_inputs 顶层数不对齐, 致 name 错位。forward 签名是 wrapper
    真实入参契约, 1:1 对齐 sample_inputs 顺序。
    """
    return list(inspect.signature(wrapper.forward).parameters.keys())


def gen_golden_with_dump(
    model_type: str,
    hf_path: str,
    output_dir: str,
    seq_len: int = 128,
    dtype: torch.dtype = torch.float16,
    num_layers: int | None = None,
    use_embeddings: bool = True,
    dump_intermediates: bool = True,
    seed: int = 0,
) -> Path:
    # 全局 seed: 固定 get_sample_inputs 的 input_ids (torch.randint) 等随机源,
    # 使 golden 跨 run 可复现 (load_model 载真实 safetensors 权重不受影响;
    # forward/DumpInterpreter 在 CPU 上对给定输入是确定性的, 无额外 RNG)。
    torch.manual_seed(seed)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from cann_lm.converter.loader import load_model

    artifact = load_model(hf_path, dtype=dtype, device="cpu")

    if num_layers is not None:
        artifact.model.model.layers = artifact.model.model.layers[:num_layers]
        layer_types = getattr(artifact.model.config, "layer_types", None)
        if layer_types:
            artifact.model.config.layer_types = layer_types[:num_layers]

    wrapper = artifact.wrapper_cls(
        artifact.model, kv_cache_max_len=2048, batch=1
    ).eval()

    sample_inputs = _get_sample_inputs(wrapper, seq_len, use_embeddings)

    if use_embeddings:
        wrapper.detach_externalized_lookups()

    print(f"[1/4] torch.export ({model_type}, seq_len={seq_len}, "
          f"layers={num_layers or 'all'})")
    with torch.no_grad():
        ep = torch.export.export(wrapper, sample_inputs, strict=False)
    gm = ep.module()

    input_names = _get_input_names(wrapper)

    print(f"[2/4] DumpInterpreter: {len(list(gm.graph.nodes))} FX nodes")
    with torch.no_grad():
        interp = DumpInterpreter(gm)
        output = interp.run(*sample_inputs)

    meta: dict[str, Any] = {
        "model_type": model_type,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "dtype": str(dtype),
        "use_embeddings": use_embeddings,
        "seed": seed,
        "inputs": [],
        "outputs": [],
        "intermediates": [],
    }

    print("[3/4] Saving inputs")
    for i, val in enumerate(sample_inputs):
        name = input_names[i] if i < len(input_names) else f"arg_{i}"
        tensors = _flatten_tensors(val)
        for j, t in enumerate(tensors):
            suffix = f"_{j}" if len(tensors) > 1 else ""
            fname = f"input_{i}_{name}{suffix}.bin"
            info = _save_tensor(t, output_dir / fname)
            info.update({"index": i, "name": name, "sub_index": j})
            meta["inputs"].append(info)

    print("[3/4] Saving outputs")
    out_tensors = _flatten_tensors(output)
    for i, t in enumerate(out_tensors):
        fname = f"output_{i}.bin"
        info = _save_tensor(t, output_dir / fname)
        info.update({"index": i})
        meta["outputs"].append(info)

    if dump_intermediates:
        print("[4/4] Saving intermediate node outputs")
        node_meta = []
        for node in gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            result = interp.dump.get(node.name)
            if result is None:
                continue
            tensors = _flatten_tensors(result)
            for j, t in enumerate(tensors):
                if t.numel() == 0:
                    continue
                suffix = f"_{j}" if len(tensors) > 1 else ""
                fname = f"node_{node.name}{suffix}.bin"
                info = _save_tensor(t, output_dir / fname)
                info.update({
                    "node_name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "sub_index": j,
                })
                node_meta.append(info)
        meta["intermediates"] = node_meta
        print(f"       {len(node_meta)} intermediate tensors saved")
    else:
        print("[4/4] Skipping intermediates (--no-intermediates)")

    meta_path = output_dir / "golden_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nDone. Golden data -> {output_dir}")
    print(f"  Inputs: {len(meta['inputs'])}, Outputs: {len(meta['outputs'])}, "
          f"Intermediates: {len(meta['intermediates'])}")
    print(f"  Meta: {meta_path}")
    return output_dir


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Generate golden reference data with per-op intermediate dump."
    )
    pa.add_argument("--model-type", required=True,
                    choices=["gemma4", "qwen3_5"],
                    help="Model type")
    pa.add_argument("--hf-path", required=True, help="HF model directory")
    pa.add_argument("--output-dir", required=True, help="Output directory")
    pa.add_argument("--seq-len", type=int, default=128,
                    help="Sequence length (default: 128)")
    pa.add_argument("--dtype", type=_dtype_arg, default=torch.float16,
                    help="Data type (float16/float32, default: float16)")
    pa.add_argument("--num-layers", type=int, default=None,
                    help="Truncate to N layers (default: all)")
    pa.add_argument("--no-embeddings", action="store_true",
                    help="Use input_ids in-graph (default: embeddings out-of-graph)")
    pa.add_argument("--no-intermediates", action="store_true",
                    help="Skip per-op intermediate dump (only inputs/outputs)")
    pa.add_argument("--seed", type=int, default=0,
                    help="Global torch seed for reproducible input_ids (default: 0)")
    args = pa.parse_args()

    gen_golden_with_dump(
        model_type=args.model_type,
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        dtype=args.dtype,
        num_layers=args.num_layers,
        use_embeddings=not args.no_embeddings,
        dump_intermediates=not args.no_intermediates,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
