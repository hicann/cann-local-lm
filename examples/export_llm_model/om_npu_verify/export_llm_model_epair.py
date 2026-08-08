#!/usr/bin/env python3
"""Export LLM model (gemma4/qwen3/qwen3.5) to epair via cann_lm model definitions.

通用导出脚本: --model 选择模型, --hf-path 指定 HF 权重目录, 复用
cann_lm.converter.model_registry (ModelEntry: config/model/load_weights/wrapper)
+ ExportWrapper.get_sample_inputs 统一 I/O 契约。

Pipeline: config -> model -> weights -> wrapper -> torch.export -> epair
          -> strip _assert_tensor_metadata -> consolidated_weights_save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.converter.model_registry import get_entry, available_models
from cann_lm.converter.loader import build_config
from cann_lm.converter.export.ge_converter import register_ge_converters
from cann_lm.converter.export.epair_utils import consolidated_weights_save
from cann_lm.converter.saver import _extract_host_side_weights
import torch_delegate

ROOT = Path(__file__).resolve().parent
_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _build_simple_sample_inputs(wrapper, seq_len: int) -> tuple:
    """简单导出输入 (past=None, 模型内部创建 cache, 非 scatter-write 全 KV buffer 模式).

    适用于未完成 scatter-write 改造的模型 (torch.cat 拼接 past+new, K=seq_len);
    scatter-write 模型也兼容 (past=None 走模型内部 cache 路径). 与 ExportWrapper.forward
    签名对齐: (input_ids, attention_mask, position_ids, past_key_values,
    new_kv_cache_pos, embeddings, output_pos).
    """
    config = wrapper.config
    batch = wrapper.batch
    dtype = next(wrapper.model.parameters()).dtype
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), dtype=torch.int64)
    min_val = torch.finfo(dtype).min
    q = torch.arange(seq_len, dtype=dtype).unsqueeze(1)
    k = torch.arange(seq_len, dtype=dtype).unsqueeze(0)
    mask = torch.where(k <= q, torch.tensor(0.0, dtype=dtype), min_val)
    attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq_len, seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(batch, 1)
    new_kv_cache_pos = torch.arange(seq_len, dtype=torch.int64)
    return (input_ids, attention_mask, position_ids, None, new_kv_cache_pos, None, None)


def export_model(
    model_name: str,
    hf_path: str,
    seq_len: int,
    dtype: torch.dtype,
    output_dir: str,
    kv_cache_max_len: int,
    use_embeddings: bool = False,
) -> None:
    register_ge_converters()
    entry = get_entry(model_name)

    print(f"[1/5] Loading config from {hf_path} (model={model_name})")
    with open(Path(hf_path) / "config.json") as f:
        hf_config = json.load(f)
    config = build_config(hf_config, entry)
    print(f"      hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"head_dim={config.head_dim}, vocab={config.vocab_size}")

    print(f"[2/5] Instantiating model (dtype={dtype})")
    model = entry.model_cls(config)
    model = model.to(dtype=dtype).eval()
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[3/5] Loading weights from {hf_path}")
    model = entry.load_weights_fn(model, hf_path, device="cpu", dtype=dtype)

    print(f"[4/5] Building export wrapper (kv_cache_max_len={kv_cache_max_len}, "
          f"use_embeddings={use_embeddings})")
    wrapper = entry.wrapper_cls(model, kv_cache_max_len=kv_cache_max_len, batch=1).eval()
    if use_embeddings:
        # 纯 embeddings: 走 wrapper.get_sample_inputs (含 gemma4 PLE 的 per_layer_token_identity)
        sample_inputs = wrapper.get_sample_inputs(seq_len=seq_len, use_embeddings=True)
    else:
        # 非 embeddings: 同样走 wrapper.get_sample_inputs (适配各 wrapper 自身 forward 签名,
        # 如 gemma4 新 wrapper 的 mask_full/mask_sliding 10 参; 旧 _build_simple_sample_inputs
        # 7 参仅适配 base forward, 对 scatter+host-mask 的 wrapper 不兼容)
        sample_inputs = wrapper.get_sample_inputs(seq_len=seq_len, use_embeddings=False)
    # 纯 embeddings: detach 外置查表子模块 (embed_tokens / gemma4 embed_tokens_per_layer),
    # 使 torch.export 不 lift 它们 → weights_*.bin 不含冗余查表权重 (对齐 saver.py)
    if use_embeddings:
        wrapper.detach_externalized_lookups()

    print(f"[5/5] Exporting to epair (seq_len={seq_len})")
    exported = torch.export.export(wrapper, sample_inputs, strict=False)
    epair = torch_delegate.export_to_epair(exported)

    # 纯 embeddings: 抽取 host 查表权重 (embed_tokens / gemma4 embed_tokens_per_layer)
    host_side_weights = _extract_host_side_weights(wrapper, use_embeddings) if use_embeddings else []

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    consolidated_weights_save(epair, out, save_weights=True,
                              max_file_size=512 * 1024 * 1024,
                              host_side_weights=host_side_weights)

    pt2_path = out.parent / f"{out.name}.pt2"
    torch_delegate.epair_save(epair, str(pt2_path))
    print(f"Saved: {pt2_path}")

    # 保存主用户输入供 C++ 验证: use_embeddings 时存 embeddings (pos 5), 否则 input_ids (pos 0)
    inp = sample_inputs[5] if use_embeddings else sample_inputs[0]
    inp_path = out.parent / f"input_0_{out.name}.bin"
    inp.cpu().detach().numpy().tofile(inp_path)
    print(f"Saved input[0] shape={list(inp.shape)} -> {inp_path}")
    print(f"\nDone. Output: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LLM model (gemma4/qwen3/qwen3.5) to epair via cann_lm definitions.")
    parser.add_argument("--model", choices=available_models(), required=True,
                        help="model name (gemma4/qwen3/qwen3_5)")
    parser.add_argument("--hf-path", required=True, help="HF model directory (with config.json + weights)")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dtype", default="float16", choices=list(_DTYPE_MAP))
    parser.add_argument("--output-dir", default="epair_model")
    parser.add_argument("--kv-cache-max-len", type=int, default=2048)
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Pure-embeddings export: input_ids not in graph, host feeds embeddings "
                             "(+ gemma4 per_layer_token_identity for PLE). Lookup weights packed "
                             "to host_weights_*.bin additionally.")
    args = parser.parse_args()

    export_model(args.model, args.hf_path, args.seq_len, _DTYPE_MAP[args.dtype],
                 args.output_dir, args.kv_cache_max_len, use_embeddings=args.use_embeddings)


if __name__ == "__main__":
    main()
