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
      --sample-seq-len 128
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
    sample_seq_len: int = 128,
    use_embeddings: bool = False,
    mode: str = "single",
    prefill_seq_len: int = 128,
    num_layers: int | None = None,
) -> None:
    """程序化入口：HF 模型 → epair

    Args:
        mode: "single" — one shape-specialized graph (seq_len=sample_seq_len);
              "prefill-decode" — two graphs (prefill seq_len=prefill_seq_len
              + decode seq_len=1) with shared weights, for actual inference.
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

    print(f"[2/3] Saving epair to {output_dir}")
    if mode == "prefill-decode":
        print(f"      prefill_seq_len={prefill_seq_len}, decode_seq_len=1, shared weights")
        save_epair_prefill_decode(
            artifact,
            output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            prefill_seq_len=prefill_seq_len,
            use_embeddings=use_embeddings,
        )
    else:
        print(f"      single graph, seq_len={sample_seq_len}")
        save_epair(
            artifact,
            output_dir,
            max_file_size=max_file_size,
            kv_cache_max_len=kv_cache_max_len,
            sample_seq_len=sample_seq_len,
            use_embeddings=use_embeddings,
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
    parser.add_argument("--sample-seq-len", type=int, default=128,
                        help="Sample input seq len for torch.export (default: 128)")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use external embeddings (no embed_tokens in graph, for embedding quantization)")
    parser.add_argument("--mode", default="single", choices=["single", "prefill-decode"],
                        help="Export mode: single (one graph, seq_len=sample-seq-len) "
                             "or prefill-decode (two graphs: prefill+decode, shared weights, for actual inference)")
    parser.add_argument("--prefill-seq-len", type=int, default=128,
                        help="Prefill graph seq_len for prefill-decode mode (default: 128, pad prompt to this)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Only export first N layers (slice/debug). None = all layers. "
                             "Truncates model.model.layers in place, config.num_hidden_layers unchanged "
                             "(PLE/delta_cache/layer_types stay full)")
    # 量化参数: TBD
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
        sample_seq_len=args.sample_seq_len,
        use_embeddings=args.use_embeddings,
        mode=args.mode,
        prefill_seq_len=args.prefill_seq_len,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
