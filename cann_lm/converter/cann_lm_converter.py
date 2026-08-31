#!/usr/bin/env python3
"""CANN-LM Converter — convert a HuggingFace model to epair format.

Model type is auto-detected from config.json; no --model-type needed.
load_model is hardwired to CPU (cann_lm/models/fused_ops/ only has CPU impls), no --device.
GPTQ runs via --quant-method gptq (subprocess isolation); cpu also works (small models).

Usage:
  cann-lm-converter \          # after install (console_scripts); or python -m cann_lm.converter.cann_lm_converter
      --hf-path /path/to/gemma-4-e2b-it \
      --output-dir /path/to/output \
      --dtype float16 \
      --kv-cache-max-len 2048 \
      --seq-len 128
"""

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
    max_file_size: int = _DEFAULT_MAX_FILE_SIZE,
    kv_cache_max_len: int = 2048,
    seq_len: int = 128,
    use_embeddings: bool = True,   # default: pure embeddings (out-of-graph lookup); False = in-graph lookup
    mode: str = "prefill-decode",
    num_layers: int | None = None,
    quant_path: str | None = None,
    quant_method: str | None = None,  # quant algo, currently only "gptq" (auto-spawns dopt); mutually exclusive with quant_path
) -> None:
    """Programmatic entry: HF model -> epair.

    load_model is hardwired to CPU (fused_ops only has CPU impls). GPTQ runs in an
    NPU/GPU/CPU subprocess (--quant-method gptq); --quant-path consumes
    offline dopt output. See quant_design.md for the scene 1/2 split.
    """
    # Scene 2 (--quant-method gptq): spawn dopt subprocess first; don't pre-load cann_lm to avoid 2x peak
    # memory (cann_lm model + subprocess's transformers model). Scene 1 (--quant-path): consume offline output.
    dopt_out_path = None
    dopt_out_is_temp = False  # True = converter temp dir (scene 2), cleaned after swap; False = user path (scene 1)
    if quant_method == "gptq":
        from cann_lm.converter.quant_entry import run_auto_quant
        print(f"[1/3] Quant: --quant-method gptq, spawn dopt subprocess (GPTQ)")
        dopt_out_path = run_auto_quant(hf_path)
        dopt_out_is_temp = True
        print(f"      dopt GPTQ done: {dopt_out_path}")
        print(f"      Loading cann_lm model from {hf_path}")
    elif quant_path:
        dopt_out_path = quant_path
        print(f"[1/3] Loading model from {hf_path}")
        print(f"      Quant: port dopt GPTQ from {quant_path} (consumer-side swap)")
    else:
        print(f"[1/3] Loading model from {hf_path}")

    # load_model (scene 2: after subprocess exits; scene 1/no-quant: first). Hardwired to CPU.
    artifact = load_model(hf_path, dtype=dtype, device="cpu")
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
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

    # swap (scene 1: quant_path; scene 2: run_auto_quant's temp dir) — after load_model.
    embed_quant = None
    if dopt_out_path:
        from cann_lm.converter.quant_entry import swap_from_dopt
        print(f"      Quant: materialize int4 (auto group_size) -> swap QuantLinear (W4A8)")
        info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path)
        artifact.quant_info = info
        print(f"      Quantized {info['n_quantized']} linears (skipped {info['n_skipped_no_dopt_key']}, group_size={info['group_size']})")
        if embed_quant:
            print(f"      Embedding int8 (auto-detected from dopt): {list(embed_quant)}")
        if dopt_out_is_temp:
            import shutil
            shutil.rmtree(dopt_out_path, ignore_errors=True)
            print(f"      Cleaned temp dopt output: {dopt_out_path}")
            dopt_out_path = None

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
    parser.add_argument("--max-file-size", type=int, default=_DEFAULT_MAX_FILE_SIZE,
                        help="Max bytes per bin file (default: 50GB)")
    parser.add_argument("--kv-cache-max-len", type=int, default=2048,
                        help="KV cache max length (default: 2048)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Shape-specialized seq_len for torch.export (default: 128). "
                             "single mode = graph seq_len; prefill-decode mode = prefill graph seq_len (decode fixed at 1)")
    parser.add_argument("--use-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Use external embeddings (no embed_tokens in graph). Default True; --no-use-embeddings uses in-graph lookup")
    parser.add_argument("--mode", default="prefill-decode", choices=["single", "prefill-decode"],
                        help="Export mode: prefill-decode (two graphs: prefill+decode, shared weights, default) "
                             "or single (one graph, seq_len=--seq-len, for quick verify)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Only export first N layers (slice/debug). None = all layers. "
                             "Truncates model.model.layers in place, config.num_hidden_layers unchanged "
                             "(PLE/delta_cache/layer_types stay full)")
    # Quant: --quant-path (consume dopt output) and --quant-method (auto-spawn dopt) are mutually exclusive.
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quant-path", default=None,
                             help="dopt GPTQ output path (safetensors/.pth/dir). Scene 1: converter loads the dopt "
                                  "output, auto-infers group_size, swaps QuantLinear (W4A8), auto-detects embed quant. "
                                  "Fine-tuning (layer strategy/group_size/algo params) is done dopt-side, step by step "
                                  "(see dopt_design.md). Mutually exclusive with --quant-method.")
    quant_group.add_argument("--quant-method", default=None, choices=["gptq"],
                             help="Quant algo (currently only gptq). Scene 2: converter auto-spawns a dopt subprocess to "
                                  "run GPTQ (NPU/GPU/CPU env; CPU only for small models). Mutually exclusive with --quant-path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        dtype=_DTYPE_MAP[args.dtype],
        max_file_size=args.max_file_size,
        kv_cache_max_len=args.kv_cache_max_len,
        seq_len=args.seq_len,
        use_embeddings=args.use_embeddings,
        mode=args.mode,
        num_layers=args.num_layers,
        quant_path=args.quant_path,
        quant_method=args.quant_method,
    )


if __name__ == "__main__":
    main()
