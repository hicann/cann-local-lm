#!/usr/bin/env python3
"""Generate golden reference data with per-op intermediate dump.

CPU eager execution via torch.fx.Interpreter — every FX node's output is
saved as a .bin file, mirroring ONNX Runtime's tensor dump capability.

Usage:
  # gemma4 5-layer prefill (fp16, no quant)
  python gen_golden_with_dump.py --model-type gemma4 \
      --hf-path /mnt/workspace/models/unsloth/gemma-4-E2B-it \
      --seq-len 128 --num-layers 5 --dtype float16 \
      --output-dir out/golden_gemma4_5l

  # gemma4 5-layer quantized (fakequant W4A8, dump 量化后中间张量)
  python gen_golden_with_dump.py --model-type gemma4 \
      --hf-path /mnt/workspace/models/unsloth/gemma-4-E2B-it \
      --quant-path /path/to/dopt_output \
      --seq-len 128 --num-layers 5 --dtype float16 \
      --output-dir out/golden_gemma4_5l_quant

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
  golden_meta.json          — metadata (names, shapes, dtypes, targets, quant info)
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

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
    dt = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
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
    return {"file": path.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}


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
    quant_path: str | None = None,
    skip_embed_quant: bool = True,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from cann_local_lm.converter.loader import load_model

    artifact = load_model(hf_path, dtype=dtype, device="cpu")

    if num_layers is not None:
        artifact.model.model.layers = artifact.model.model.layers[:num_layers]
        layer_types = getattr(artifact.model.config, "layer_types", None)
        if layer_types:
            artifact.model.config.layer_types = layer_types[:num_layers]

    wrapper = artifact.wrapper_cls(
        artifact.model, kv_cache_max_len=2048, batch=1
    ).eval()

    # ── 量化接入: dopt fakequant (W4A8: 权重 int4 反量化 + 激活 int8 量化) ──
    quant_info: dict[str, Any] | None = None
    if quant_path:
        from cann_local_lm.converter.swap import (
            load_dopt_state_dict,
            _read_quant_param_map,
        )

        quant_path = Path(quant_path).resolve()
        weight_pth = quant_path / "trained_quant_weight.pth"

        print(f"[quant] loading dopt weights: {weight_pth}")
        # 手动调 swap 组件 (swap_from_dopt 内部 load_dopt_state_dict + _read_quant_param_map 用同一 path
        # 但 load_dopt_state_dict 要 .pth 文件, _read_quant_param_map 要 dopt_config.json 目录)
        import cann_local_lm.quant.dopt_lm.quantizer.qmin_max  # noqa: F401  # 注册量化器
        import cann_local_lm.quant.dopt_lm.quantizer.qeval  # noqa: F401
        from cann_local_lm.converter.swap import (
            _swap_linear_from_sd,
            swap_embeddings_from_dopt,
            DEFAULT_SKIP,
        )
        from cann_local_lm.quant.dopt_lm.do_opt import (
            set_quant_state,
            set_calibrate_state,
        )

        sd = load_dopt_state_dict(str(weight_pth))
        param_map = _read_quant_param_map(str(quant_path))
        # gen_golden_with_dump 在 CPU 跑 DumpInterpreter → cann_local_lm::quant_matmul CPU impl 要 unpacked [K,N] int8
        # 所以 swap 用 packed=False (不走 nibble-packed), 产出 unpacked int8 weight_q [K,N]
        swap_info = _swap_linear_from_sd(
            artifact.model, sd, skip=DEFAULT_SKIP, param_map=param_map
        )
        # swap 默认 packed=True (int8 nibble [K//2,N]); CPU DumpInterpreter 要 unpacked
        # 手动 re-materialize 每个 QLinear 的 weight 为 unpacked
        from cann_local_lm.converter.export.quant_swap.materialize import (
            materialize_int4_per_group_symmetric,
        )
        from cann_local_lm.converter.export.quant_swap.quant_linear import QuantLinear

        for name, m in artifact.model.named_modules():
            if isinstance(m, QuantLinear):
                # 从 QLinear 的现有 packed weight_q 解 nibble → unpacked
                # 更简单: 从 sd 重新 materialize unpacked
                wkey = f"{name}.weight"
                skey = f"{name}.quant_op.weight_quantizer.s"
                if wkey in sd and skey in sd:
                    w_fp = sd[wkey].float()
                    out_f, in_f = w_fp.shape
                    groups = sd[skey].numel() // out_f
                    gs = in_f // groups
                    scale2d = sd[skey].reshape(out_f, groups)
                    wq_unpacked, wsc = materialize_int4_per_group_symmetric(
                        w_fp, scale2d, gs, signed=True, nbits=4, packed=False
                    )
                    m.weight_q = torch.nn.Parameter(
                        wq_unpacked.transpose(-1, -2).contiguous(), requires_grad=False
                    )
                    m.weight_scale = torch.nn.Parameter(
                        wsc.transpose(-1, -2).contiguous().to(torch.float32),
                        requires_grad=False,
                    )
        _ = swap_embeddings_from_dopt(
            sd, artifact.model
        )  # 副作用: swap model embeddings to int8; 返回值未用
        swap_info["dopt_path"] = str(quant_path)

        set_quant_state(artifact.model, weight_state=True, input_state=True)
        set_calibrate_state(artifact.model, True)
        artifact.model.load_state_dict(
            torch.load(str(weight_pth), map_location="cpu", weights_only=True),
            strict=False,
        )
        set_calibrate_state(artifact.model, False)

        # 跳过 embedding 量化 (vocab×10752 的 fakequant 中间张量 ~11GB, CPU 也重)
        if skip_embed_quant:
            for name, m in artifact.model.named_modules():
                if hasattr(m, "quant_op") and m.quant_op is not None:
                    if "embed" in name or "per_layer_model_projection" in name:
                        m.quant_op.set_quant_state(weight=False, input=False)

        artifact.model.eval()
        quant_info = {
            "quant_path": str(quant_path),
            "scheme": swap_info.get("scheme"),
            "n_quantized": swap_info.get("n_quantized"),
            "weight_state": True,
            "input_state": True,
            "skip_embed_quant": skip_embed_quant,
        }
        print(
            f"[quant] {swap_info.get('n_quantized')} linears quantized, "
            f"skip_embed_quant={skip_embed_quant}"
        )

    # 固定 input_ids 种子 (get_sample_inputs 用 torch.randint), 保证跨 run 可复现
    # (ring vs absolute 对比需相同 input_ids; 否则 argmax 不可比)
    torch.manual_seed(0)
    sample_inputs = _get_sample_inputs(wrapper, seq_len, use_embeddings)

    if use_embeddings:
        wrapper.detach_externalized_lookups()

    print(
        f"[1/4] torch.export ({model_type}, seq_len={seq_len}, "
        f"layers={num_layers or 'all'})"
    )
    with torch.no_grad():
        ep = torch.export.export(wrapper, sample_inputs, strict=False)
    gm = ep.module()

    input_names = _get_input_names(wrapper)

    print(
        f"[2/4] {'DumpInterpreter' if dump_intermediates else 'forward (no-intermediates)'}: "
        f"{len(list(gm.graph.nodes))} FX nodes"
    )
    with torch.no_grad():
        if dump_intermediates:
            interp = DumpInterpreter(gm)
            output = interp.run(*sample_inputs)
        else:
            # --no-intermediates: 直接跑图, 不 capture 中间 (DumpInterpreter 会把所有 node 输出
            # 存 self.dump, seq 大时 OOM; 无需中间时跳过 capture, 只取最终 output)
            output = gm(*sample_inputs)

    meta: dict[str, Any] = {
        "model_type": model_type,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "dtype": str(dtype),
        "use_embeddings": use_embeddings,
        "quant": quant_info,
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
                info.update(
                    {
                        "node_name": node.name,
                        "op": node.op,
                        "target": str(node.target),
                        "sub_index": j,
                    }
                )
                node_meta.append(info)
        meta["intermediates"] = node_meta
        print(f"       {len(node_meta)} intermediate tensors saved")
    else:
        print("[4/4] Skipping intermediates (--no-intermediates)")

    meta_path = output_dir / "golden_meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone. Golden data -> {output_dir}")
    print(
        f"  Inputs: {len(meta['inputs'])}, Outputs: {len(meta['outputs'])}, "
        f"Intermediates: {len(meta['intermediates'])}"
    )
    print(f"  Meta: {meta_path}")
    return output_dir


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Generate golden reference data with per-op intermediate dump."
    )
    pa.add_argument(
        "--model-type", required=True, choices=["gemma4", "qwen3_5"], help="Model type"
    )
    pa.add_argument("--hf-path", required=True, help="HF model directory")
    pa.add_argument("--output-dir", required=True, help="Output directory")
    pa.add_argument(
        "--seq-len", type=int, default=128, help="Sequence length (default: 128)"
    )
    pa.add_argument(
        "--dtype",
        type=_dtype_arg,
        default=torch.float16,
        help="Data type (float16/float32, default: float16)",
    )
    pa.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate to N layers (default: all)",
    )
    pa.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Use input_ids in-graph (default: embeddings out-of-graph)",
    )
    pa.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Skip per-op intermediate dump (only inputs/outputs)",
    )
    pa.add_argument(
        "--quant-path",
        default=None,
        help="Path to dopt quantized output dir (trained_quant_weight.pth + dopt_config.json). "
        "When provided, applies fakequant W4A8 to the model before dump.",
    )
    pa.add_argument(
        "--no-skip-embed-quant",
        action="store_true",
        help="Do NOT skip embedding quantization (default: skip to avoid OOM from huge embedding tables)",
    )
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
        quant_path=args.quant_path,
        skip_embed_quant=not args.no_skip_embed_quant,
    )


if __name__ == "__main__":
    main()
