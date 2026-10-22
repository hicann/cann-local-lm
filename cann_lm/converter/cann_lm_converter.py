#!/usr/bin/env python3
r"""CANN-LM Converter — convert a HuggingFace model to epair format.

Model type is auto-detected from config.json; no --model-type needed.
load_model is hardwired to CPU (cann_lm/models/fused_ops/ only has CPU impls), no --device.
GPTQ runs via --quant-method gptq (subprocess isolation); cpu also works (small models).

Usage:
  cann-lm-converter \          # after install (console_scripts); or python -m cann_lm.converter.cann_lm_converter
      --hf-path /path/to/gemma-4-e2b-it \
      --output-dir /path/to/output \
      --target-soc Ascend910B \
      --kv-cache-max-len 2048
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _derive_dtype(target_soc: str) -> torch.dtype:
    """Derive weight dtype from --target-soc.

    Ascend family (910/950) → bfloat16: ChunkGatedDeltaRule (qwen3.5 linear_attn)
    op-info on Ascend declares input dtype=bfloat16 only (no float16 kernel).
    Kirin (KirinX90) → float16: kirinx90 op-info supports float16.
    """
    s = (target_soc or "")
    if s.startswith("Kirin"):
        return torch.float16
    return torch.bfloat16

_DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024  # 50GB


def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    max_file_size: int = _DEFAULT_MAX_FILE_SIZE,
    kv_cache_max_len: int = 2048,
    seq_len: int = 64,
    use_embeddings: bool = True,   # default: pure embeddings (out-of-graph lookup); False = in-graph lookup
    mode: str = "prefill-decode",
    num_layers: int | None = None,
    quant_path: str | None = None,
    quant_method: str | None = None,  # quant algo, currently only "gptq" (auto-spawns dopt); mutually exclusive with quant_path
    target_soc: str = "KirinX90",
) -> None:
    """Programmatic entry: HF model -> epair.

    load_model is hardwired to CPU (fused_ops only has CPU impls). GPTQ runs in an
    NPU/GPU/CPU subprocess (--quant-method gptq); --quant-path consumes
    offline dopt output. See quant_design.md for the scene 1/2 split.
    """
    from cann_lm.converter.loader import load_model
    from cann_lm.converter.saver import save_epair, save_epair_prefill_decode
    # Scene 2 (--quant-method gptq): spawn dopt subprocess first; don't pre-load cann_lm to avoid 2x peak
    # memory (cann_lm model + subprocess's transformers model). Scene 1 (--quant-path): consume offline output.
    dopt_out_path = None
    dopt_out_is_temp = False  # True = converter temp dir (scene 2), cleaned after swap; False = user path (scene 1)
    if quant_method == "gptq":
        from cann_lm.converter.quant_entry import run_auto_quant
        print(f"[1/2] Quant: --quant-method gptq, spawn dopt subprocess (GPTQ)")
        dopt_out_path = run_auto_quant(hf_path)
        dopt_out_is_temp = True
        print(f"      dopt GPTQ done: {dopt_out_path}")
        print(f"      Loading cann_lm model from {hf_path}")
    elif quant_path:
        dopt_out_path = quant_path
        print(f"[1/2] Loading model from {hf_path}")
        print(f"      Quant: port dopt GPTQ from {quant_path} (consumer-side swap)")
    else:
        print(f"[1/2] Loading model from {hf_path}")

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
        info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path, target_soc=target_soc)
        artifact.quant_info = info
        print(f"      Quantized {info['n_quantized']} linears (skipped {info['n_skipped_no_dopt_key']}, group_size={info['group_size']})")
        if embed_quant:
            print(f"      Embedding int8 (auto-detected from dopt): {list(embed_quant)}")
        if dopt_out_is_temp:
            import shutil
            shutil.rmtree(dopt_out_path, ignore_errors=True)
            print(f"      Cleaned temp dopt output: {dopt_out_path}")
            dopt_out_path = None

    print(f"[2/2] Saving epair to {output_dir} (target_soc={target_soc})")
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
            target_soc=target_soc,
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
            target_soc=target_soc,
        )

    print(f"\nDone. Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CANN-LM Converter: HF model → epair format",
    )
    parser.add_argument("--hf-path", required=True,
                        help="HuggingFace model path (local directory)")
    parser.add_argument("--output-dir", required=True,
                        help="epair output directory")
    parser.add_argument("--target-soc", default="KirinX90",
                        help="Target SoC for epair export (default: KirinX90). Determines dtype: "
                             "Ascend→bfloat16, Kirin→float16.")
    parser.add_argument("--max-file-size", type=int, default=_DEFAULT_MAX_FILE_SIZE,
                        help="Max bytes per bin file (default: 50GB)")
    parser.add_argument("--kv-cache-max-len", type=int, default=2048,
                        help="KV cache max length (default: 2048)")
    parser.add_argument("--use-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Use external embeddings (no embed_tokens in graph). Default True; --no-use-embeddings uses in-graph lookup")
    parser.add_argument("--mode", default="prefill-decode", choices=["single", "prefill-decode"],
                        help="Export mode: prefill-decode (two graphs: prefill+decode, shared weights, default) "
                             "or single")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Only export first N layers (slice/debug). None = all layers. "
                             "Truncates model.model.layers in place, config.num_hidden_layers unchanged "
                             "(PLE/delta_cache/layer_types stay full)")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Prefill sequence length (default: 64). Decode graph is always seq_len=1.")

    # Quant: --quant-path (consume lightweight-tool output) and --quant-method (auto-spawn the tool) are mutually exclusive.
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quant-path", default=None,
                             help="GPTQ output path of the CANN lightweight quantization tool "
                                  "(safetensors/.pth/dir). Scene 1: converter loads the tool output, "
                                  "auto-infers group_size, swaps QuantLinear (W4A8), auto-detects "
                                  "embed quant. Fine-tuning (layer strategy/group_size/algo params) "
                                  "is done in the tool CLI, step by step. Mutually exclusive with "
                                  "--quant-method.")
    quant_group.add_argument("--quant-method", default=None, choices=["gptq"],
                             help="Quant algo (currently only gptq). Scene 2: converter auto-spawns a tool subprocess to "
                                  "run GPTQ (NPU/GPU/CPU env; CPU only for small models). Mutually exclusive with --quant-path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _derive_dtype(args.target_soc)
    print(f"[dtype] target_soc={args.target_soc} -> {str(dtype)} (derived from target-soc)")
    convert(
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        dtype=dtype,
        max_file_size=args.max_file_size,
        kv_cache_max_len=args.kv_cache_max_len,
        seq_len=args.seq_len,
        use_embeddings=args.use_embeddings,
        mode=args.mode,
        num_layers=args.num_layers,
        quant_path=args.quant_path,
        quant_method=args.quant_method,
        target_soc=args.target_soc,
    )


if __name__ == "__main__":
    main()
