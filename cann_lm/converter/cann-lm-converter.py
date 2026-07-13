#!/usr/bin/env python3
"""CANN-LM Converter — 将 HuggingFace 模型转换为 epair 格式.

Usage:
  python cann-lm-converter.py \
      --hf-path /path/to/gemma-4-e2b-it \
      --model-type gemma4 \
      --output-dir /path/to/output \
      --dtype float16 \
      --max-file-size 536870912
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
from cann_lm.converter.saver import save_epair

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CANN-LM Converter: HF model → epair format",
    )
    parser.add_argument("--hf-path", required=True,
                        help="HuggingFace model path (local directory)")
    parser.add_argument("--model-type", required=True,
                        choices=["gemma4", "qwen3_5", "qwen3"],
                        help="Model type")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--dtype", default="float16",
                        choices=list(_DTYPE_MAP),
                        help="Weight dtype (default: float16)")
    parser.add_argument("--device", default="cpu",
                        help="Load device (default: cpu)")
    parser.add_argument("--max-file-size", type=int, default=512 * 1024 * 1024,
                        help="Max bytes per bin file (default: 512MB)")
    parser.add_argument("--sample-seq-len", type=int, default=128,
                        help="Sample input seq len for torch.export (default: 128)")
    parser.add_argument("--no-weights", action="store_true",
                        help="Only export ep_json, skip weights")
    parser.add_argument("--partition-layers", type=int, default=None,
                        help="Partition: layers per partition (not yet implemented)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/3] Loading model: {args.model_type} from {args.hf_path}")
    artifact = load_model(
        args.hf_path,
        args.model_type,
        dtype=_DTYPE_MAP[args.dtype],
        device=args.device,
    )
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Params: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")

    print(f"[2/3] Saving epair to {args.output_dir}")
    if args.partition_layers:
        print(f"      Partition mode: {args.partition_layers} layers/partition")
        raise NotImplementedError("Partition mode not yet implemented")
    else:
        save_epair(
            artifact,
            args.output_dir,
            max_file_size=args.max_file_size,
            sample_seq_len=args.sample_seq_len,
            save_weights=not args.no_weights,
        )

    print(f"[3/3] Verifying output")
    output_dir = Path(args.output_dir)
    expected = ["ep_json_with_ascend_ir.json", "model_config.json"]
    if not args.no_weights:
        expected.append("weight_info.json")
    for name in expected:
        fp = output_dir / name
        status = "[OK]" if fp.exists() else "[MISSING]"
        print(f"      {status} {name}")

    bin_files = list(output_dir.glob("weights_*.bin"))
    print(f"      Weight files: {len(bin_files)}")
    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
