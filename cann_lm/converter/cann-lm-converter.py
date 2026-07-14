#!/usr/bin/env python3
"""CANN-LM Converter — 将 HuggingFace 模型转换为 epair 格式.

Model type is auto-detected from config.json; no --model-type needed.

Usage:
  python cann-lm-converter.py \
      --hf-path /path/to/gemma-4-e2b-it \
      --output-dir /path/to/output \
      --dtype float16 \
      --device cpu \
      --max-file-size 53687091200 \
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
from cann_lm.converter.saver import save_epair

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
    sample_seq_len: int = 128,
) -> None:
    """程序化入口：HF 模型 → epair，返回 None。

    模型类型由 load_model 内部自动探测（从 config.json 的 model_type），
    调用方无需传入。量化参数后续在此签名上扩展。
    """
    print(f"[1/3] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device=device)
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Params: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")

    print(f"[2/3] Saving epair to {output_dir}")
    save_epair(
        artifact,
        output_dir,
        max_file_size=max_file_size,
        sample_seq_len=sample_seq_len,
    )

    print(f"[3/3] Verifying output")
    out = Path(output_dir)
    for name in ["ep_json_with_ascend_ir.json", "model_config.json", "weight_info.json"]:
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
    parser.add_argument("--sample-seq-len", type=int, default=128,
                        help="Sample input seq len for torch.export (default: 128)")
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
        sample_seq_len=args.sample_seq_len,
    )


if __name__ == "__main__":
    main()
