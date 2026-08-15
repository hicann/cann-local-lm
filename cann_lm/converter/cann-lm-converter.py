#!/usr/bin/env python3
"""CANN-LM Converter — 将 HuggingFace 模型转换为 epair 格式.

Model type is auto-detected from config.json; no --model-type needed.

Usage:
  python cann-lm-converter.py \
      --hf-path /path/to/gemma-4-e2b-it \
      --output-dir /path/to/output \
      --dtype float16 \
      --device cpu \
      --kv-cache-max-len 2048 \
      --seq-len 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.converter.loader import load_model
from cann_lm.converter.saver import save_epair, save_epair_prefill_decode

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024  # 50GB


def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    device: str = "cpu",
    max_file_size: int = _DEFAULT_MAX_FILE_SIZE,
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = True,   # 默认纯 embeddings (图外查表); False 走图内查表
    mode: str = "prefill-decode",
    num_layers: int | None = None,
    quant_path: str | None = None,
    quant: str | None = None,  # 量化策略, 当前仅支持 "w4a8"; 与 quant_path 互斥
) -> None:
    """程序化入口：HF 模型 → epair

    Args:
        mode: "single" — one shape-specialized graph (seq_len 由参数指定);
              "prefill-decode" — prefill (seq_len 由参数指定) + decode (seq_len=1)
              two graphs, shared weights, for actual inference.
        seq_len: shape-specialized seq_len for torch.export. single 模式 = 图 seq_len;
              prefill-decode 模式 = prefill 图 seq_len (decode 固定 1, 不参数化).
        quant_path: dopt GPTQ 产物路径 (safetensors/.pth/dir). 场景 1: converter 加载 dopt 产物,
            自动推断 group_size, swap QuantLinear (W4A8), 自动检测 embed 量化.
            精细调整 (层策略/group_size/算法参数) 由 dopt 分步完成 (见 dopt_design.md). 与 quant 互斥.
        quant: 量化策略 (当前仅支持 "w4a8"). 对 HF 原始权重走场景 2: converter 直接
            min-max W4A8 量化 (不经 dopt GPTQ), 固定 group_size=128. 精度不如 GPTQ,
            适合快速验证. 与 quant_path 互斥.
    """
    print(f"[1/3] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device=device)
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Params: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")
    print(f"      Wrapper: {artifact.wrapper_cls.__name__ if artifact.wrapper_cls else 'None'}")
    print(f"      Embedding mode: {'external (no embed_tokens in graph)' if use_embeddings else 'internal (embed_tokens in graph)'}")
    print(f"      Export mode: {mode}")

    if num_layers is not None:
        text_model = artifact.model.model
        total = len(text_model.layers)
        if num_layers > total:
            raise ValueError(f"--num-layers {num_layers} exceeds total {total} layers")
        if num_layers < 1:
            raise ValueError(f"--num-layers must be >= 1, got {num_layers}")
        text_model.layers = text_model.layers[:num_layers]
        print(f"      Truncated to first {num_layers}/{total} layers (config.num_hidden_layers unchanged)")

    embed_quant = None
    if quant_path:
        from cann_lm.converter.quant import (
            swap_to_quant_linear_from_dopt, swap_embeddings_from_dopt, load_dopt_state_dict,
        )
        print(f"      Quant: port dopt GPTQ from {quant_path} -> materialize int4 (auto group_size) -> swap QuantLinear (W4A8)")
        info = swap_to_quant_linear_from_dopt(artifact.model, quant_path)
        artifact.quant_info = info
        print(f"      Quantized {info['n_quantized']} linears (skipped {info['n_skipped_no_dopt_key']}, group_size={info['group_size']})")
        # embed 量化自动检测: dopt 产物含 embed scale 则自动做 (不靠 --quant-embeddings 开关)
        sd = load_dopt_state_dict(quant_path)
        embed_quant = swap_embeddings_from_dopt(sd, artifact.model)
        if embed_quant:
            print(f"      Embedding int8 (auto-detected from dopt): {list(embed_quant)}")
    elif quant == "w4a8":
        from cann_lm.converter.quant import swap_to_quant_linear
        print(f"      Quant: W4A8 min-max (HF raw weight, group=128, no GPTQ) -> swap QuantLinear")
        swap_to_quant_linear(artifact.model, group_size=128, act_quant=True)
        artifact.quant_info = {
            "scheme": "W4A8-s8s4", "group_size": 128, "act_quant": True,
            "scope": "all nn.Linear except ['embed_tokens', 'embed_tokens_per_layer']",
            "source": "min-max (no GPTQ, HF raw weight)",
        }
        print(f"      Quantized via min-max W4A8 (group=128)")

    print(f"[2/3] Saving epair to {output_dir}")
    if mode == "prefill-decode":
        print(f"      seq_len={seq_len}, decode_seq_len=1, shared weights")
        save_epair_prefill_decode(
            artifact,
            output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            seq_len=seq_len,
            use_embeddings=use_embeddings,
            embed_quant=embed_quant,
        )
    else:
        print(f"      single graph, seq_len={seq_len}")
        save_epair(
            artifact,
            output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            seq_len=seq_len,
            use_embeddings=use_embeddings,
            embed_quant=embed_quant,
        )

    print(f"[3/3] Verifying output")
    out = Path(output_dir)
    if mode == "prefill-decode":
        expected = ["prefill_ep_json_with_ascend_ir.json",
                    "decode_ep_json_with_ascend_ir.json",
                    "model_config.json", "weight_info.json"]
    else:
        expected = ["ep_json_with_ascend_ir.json",
                    "model_config.json", "weight_info.json"]
    for name in expected:
        fp = out / name
        status = "[OK]" if fp.exists() else "[MISSING]"
        print(f"      {status} {name}")
    print(f"      Weight files: {len(list(out.glob('weights_*.bin')))}")
    print(f"\nDone. Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CANN-LM Converter: HF model → epair format",
    )
    parser.add_argument("--hf-path", required=True,
                        help="HuggingFace model path (local directory)")
    parser.add_argument("--output-dir", required=True,
                        help="epair output directory")
    parser.add_argument("--dtype", default="float16",
                        choices=list(_DTYPE_MAP),
                        help="Weight dtype (default: float16)")
    parser.add_argument("--device", default="cpu",
                        help="Load device (default: cpu)")
    parser.add_argument("--max-file-size", type=int, default=_DEFAULT_MAX_FILE_SIZE,
                        help="Max bytes per bin file (default: 50GB)")
    parser.add_argument("--kv-cache-max-len", type=int, default=2048,
                        help="KV cache max length (default: 2048)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Shape-specialized seq_len for torch.export (default: 128). "
                             "single 模式 = 图 seq_len; prefill-decode 模式 = prefill 图 seq_len (decode 固定 1)")
    parser.add_argument("--use-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Use external embeddings (no embed_tokens in graph). 默认 True; --no-use-embeddings 走图内查表")
    parser.add_argument("--mode", default="prefill-decode", choices=["single", "prefill-decode"],
                        help="Export mode: prefill-decode (two graphs: prefill+decode, shared weights, default) "
                             "or single (one graph, seq_len=--seq-len, for quick verify)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Only export first N layers (slice/debug). None = all layers. "
                             "Truncates model.model.layers in place, config.num_hidden_layers unchanged "
                             "(PLE/delta_cache/layer_types stay full)")
    # 量化: 两参数互斥 (--quant-path 已量化场景 / --quant w4a8 HF 原始权重场景)
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quant-path", default=None,
                             help="dopt GPTQ 产物路径 (safetensors/.pth/dir). 场景 1: converter 加载 dopt 产物, "
                                  "自动推断 group_size, swap QuantLinear (W4A8), 自动检测 embed 量化. "
                                  "精细调整 (层策略/group_size/算法参数) 由 dopt 分步完成 (见 dopt_design.md). "
                                  "与 --quant 互斥.")
    quant_group.add_argument("--quant", default=None, choices=["w4a8"],
                             help="量化策略 (当前仅支持 w4a8). 场景 2: 对 HF 原始权重走 min-max W4A8 "
                                  "(不经 dopt GPTQ), 固定 group_size=128. 精度不如 GPTQ, 适合快速验证. "
                                  "与 --quant-path 互斥.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        dtype=_DTYPE_MAP[args.dtype],
        device=args.device,
        max_file_size=args.max_file_size,
        kv_cache_max_len=args.kv_cache_max_len,
        seq_len=args.seq_len,
        use_embeddings=args.use_embeddings,
        mode=args.mode,
        num_layers=args.num_layers,
        quant_path=args.quant_path,
        quant=args.quant,
    )


if __name__ == "__main__":
    main()
