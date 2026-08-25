#!/usr/bin/env python3
"""CANN-LM Converter — 将 HuggingFace 模型转换为 epair 格式.

Model type is auto-detected from config.json; no --model-type needed.
load_model 写死 CPU (融合算子 cann_lm/models/fused_ops/ 只在 CPU 实现), 不接受 --device.
NPU 跑 GPTQ 用 --quant-method gptq + --quant-device npu (subprocess 隔离).

Usage:
  cann-lm-converter \          # 装包后 (console_scripts); 或 python -m cann_lm.converter.cann_lm_converter
      --hf-path /path/to/gemma-4-e2b-it \
      --output-dir /path/to/output \
      --dtype float16 \
      --kv-cache-max-len 2048 \
      --seq-len 128
"""

from __future__ import annotations

import argparse
import os
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
    use_embeddings: bool = True,   # 默认纯 embeddings (图外查表); False 走图内查表
    mode: str = "prefill-decode",
    num_layers: int | None = None,
    quant_path: str | None = None,
    quant_method: str | None = None,  # 量化算法, 当前仅支持 "gptq" (自动 spawn dopt); 与 quant_path 互斥
    quant_bits: str = "w4a8",  # 格式/位宽 (w4a8); 仅 quant_method 时生效
    quant_device: str | None = None,  # 量化 subprocess 设备 ('npu'/'cuda'), 跳过环境检测; None 则继承父进程 DEVICE 或检测
) -> None:
    """程序化入口：HF 模型 → epair

    load_model 写死 CPU (cann_lm 自定义融合算子 cann_lm/models/fused_ops/ 只在 CPU 实现),
    不接受 --device 参数. NPU 跑 GPTQ 通过 --quant-method gptq + subprocess 隔离.

    Args:
        mode: "single" — one shape-specialized graph (seq_len 由参数指定);
              "prefill-decode" — prefill (seq_len 由参数指定) + decode (seq_len=1)
              two graphs, shared weights, for actual inference.
        seq_len: shape-specialized seq_len for torch.export. single 模式 = 图 seq_len;
              prefill-decode 模式 = prefill 图 seq_len (decode 固定 1, 不参数化).
        quant_path: dopt GPTQ 产物路径 (safetensors/.pth/dir). 场景 1: converter 加载 dopt 产物,
            自动推断 group_size, swap QuantLinear (W4A8), 自动检测 embed 量化.
            精细调整 (层策略/group_size/算法参数) 由 dopt 分步完成 (见 dopt_design.md). 与 quant_method 互斥.
        quant_method: 量化算法 (当前仅支持 "gptq"). 场景 2: converter 自动 spawn dopt subprocess
            跑 GPTQ (需 NPU 环境, 待实现, 见 quant_design.md "一键式量化方案选型"). 与 quant_path 互斥.
        quant_bits: 格式/位宽 (w4a8, 默认 w4a8). 仅 quant_method 时生效, 传给 dopt_config 的 strategy.
        quant_device: 量化 subprocess 设备 ('npu'/'cuda'), 跳过环境检测. None 则继承父进程 DEVICE
            环境变量 (set_env.sh npu 模式 export) 或 spawn 检测. 与 set_env.sh 等价 (后设覆盖先设).
    """
    # CLI --quant-device 显式指定: 写到 os.environ["DEVICE"] (主进程), subprocess 经 env=os.environ.copy()
    # 自动继承, _detect_device_type 优先读它跳过 spawn 检测. 与 set_env.sh npu 模式等价.
    if quant_device:
        os.environ["DEVICE"] = quant_device
    # 场景 2 (--quant-method gptq): 先 spawn dopt subprocess 跑 GPTQ,主进程不预 load cann_lm 模型,
    # 避免主进程 cann_lm 模型与 subprocess transformers 模型同时持有 (2x 内存峰值).
    # 场景 1 (--quant-path): dopt 已离线跑完,直接消费产物,无 subprocess.
    dopt_out_path = None
    dopt_out_is_temp = False  # True = converter 生成的临时盘 (场景 2),swap 后清理; False = 用户路径 (场景 1)
    if quant_method == "gptq":
        from cann_lm.converter.quant.auto_quant import run_auto_quant
        print(f"[1/3] Quant: --quant-method gptq (bits={quant_bits}), spawn dopt subprocess (NPU GPTQ)")
        dopt_out_path = run_auto_quant(hf_path, quant_bits)
        dopt_out_is_temp = True
        print(f"      dopt GPTQ done: {dopt_out_path}")
        print(f"      Loading cann_lm model from {hf_path}")
    elif quant_path:
        dopt_out_path = quant_path
        print(f"[1/3] Loading model from {hf_path}")
        print(f"      Quant: port dopt GPTQ from {quant_path} (consumer-side swap)")
    else:
        print(f"[1/3] Loading model from {hf_path}")

    # load_model: 场景 2 时在 run_auto_quant 之后 (subprocess 已退出,主进程才 load cann_lm);
    # 场景 1 / 无量化时在前. 三场景都需 cann_lm 模型实例 (swap 与 export 都依赖)
    # load_model 写死 CPU (融合算子 cann_lm/models/fused_ops/ 只在 CPU 实现)
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

    # swap: 场景 1 (--quant-path) + 场景 2 (--quant-method gptq) 共用统一路径,在 load_model 之后.
    # dopt_out_path 来源不同: 场景 1 = 用户给的 quant_path, 场景 2 = run_auto_quant 返回的临时盘.
    embed_quant = None
    if dopt_out_path:
        from cann_lm.converter.quant import swap_from_dopt
        print(f"      Quant: materialize int4 (auto group_size) -> swap QuantLinear (W4A8)")
        info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path)
        artifact.quant_info = info
        print(f"      Quantized {info['n_quantized']} linears (skipped {info['n_skipped_no_dopt_key']}, group_size={info['group_size']})")
        if embed_quant:
            print(f"      Embedding int8 (auto-detected from dopt): {list(embed_quant)}")
        # swap 全部成功,临时盘产物已消费成 QuantLinear + embed int8/scale (内存副本),磁盘可清理
        # 失败时 (swap 抛异常) 跳过此分支,临时盘自动保留供调试
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
    # 量化: --quant-path (消费 dopt 产物) 与 --quant-method (自动 spawn dopt) 互斥
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quant-path", default=None,
                             help="dopt GPTQ 产物路径 (safetensors/.pth/dir). 场景 1: converter 加载 dopt 产物, "
                                  "自动推断 group_size, swap QuantLinear (W4A8), 自动检测 embed 量化. "
                                  "精细调整 (层策略/group_size/算法参数) 由 dopt 分步完成 (见 dopt_design.md). "
                                  "与 --quant-method 互斥.")
    quant_group.add_argument("--quant-method", default=None, choices=["gptq"],
                             help="量化算法 (当前仅支持 gptq). 场景 2: converter 自动 spawn dopt subprocess 跑 GPTQ "
                                  "(需 NPU 环境, 待实现). 与 --quant-path 互斥.")
    parser.add_argument("--quant-bits", default="w4a8", choices=["w4a8"],
                        help="格式/位宽 (w4a8, 默认 w4a8). 仅 --quant-method 时生效, 传给 dopt_config 的 strategy.")
    parser.add_argument("--quant-device", default=None, choices=["npu", "cuda"],
                        help="量化 subprocess 设备 (npu/cuda), 跳过环境检测. 默认 None 继承父进程 DEVICE "
                             "(set_env.sh npu 模式 export) 或 spawn 检测. 与 source scripts/set_env.sh npu 等价.")
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
        quant_bits=args.quant_bits,
        quant_device=args.quant_device,
    )


if __name__ == "__main__":
    main()
