#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPL 对比: 浮点模型 vs dopt GPTQ 量化产物 (hf-type-save 格式).

原理: GPTQ 产物由 --hf-type-save 落盘, 即普通 HF 权重格式 (W4 伪量化值已写回
weight), 可直接 AutoModelForCausalLM 加载, 无需 dopt 运行时。两侧使用相同
dtype / 数据 / 滑窗, PPL 差值即量化带来的损伤。

用法 (quant/ 目录下):
  # CPU (无需 NPU)
  DEVICE=cpu python ppl_compare.py \
      --float-model /path/Qwen3-0.6B \
      --quant-model /path/dopt_qwen3_0dot6_w4a8
  # NPU (需已装 torch_npu 的 python)
  DEVICE=npu python ppl_compare.py --float-model ... --quant-model ...

参数:
  --data        评测 jsonl (每行需含 text 字段),
                默认 calibrate_sets/calibrate_coig_cqia.jsonl
                (注意: 默认集即校准集, 量化侧偏乐观; 公正对比可换 held-out 数据)
  --max-tokens  参与 PPL 计算的 token 上限 (默认 16384)
  --seq-len     滑窗长度 (默认 2048)
  --stride      窗口步长 (默认 = seq-len, 窗口不重叠)
  --dtype       auto (默认: cpu->float32, 其他->float16) | float32 | float16 | bfloat16
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = os.getenv("DEVICE", "cpu").strip().lower()
if DEVICE == "npu":
    import torch_npu  # noqa: F401 - 注册 NPU 后端


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--float-model", required=True, help="浮点 HF 模型目录 (同时提供 tokenizer)")
    p.add_argument("--quant-model", required=True, help="dopt GPTQ 产物目录 (hf-type-save)")
    p.add_argument("--data", default=os.path.join(here, "calibrate_sets", "calibrate_coig_cqia.jsonl"))
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--stride", type=int, default=None, help="默认 = seq-len (窗口不重叠)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    return p.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    return torch.float32 if DEVICE == "cpu" else torch.float16


def build_tokens(tokenizer, data_file: str, max_tokens: int) -> torch.Tensor:
    """读 jsonl 的 text 字段拼接送 tokenizer, 截断到 max_tokens。"""
    texts, n_chars = [], 0
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            n_chars += len(texts[-1])
            if n_chars > max_tokens * 8:  # 粗估字符预算, 避免全文 tokenize
                break
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    return ids[:max_tokens]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seq_len: int, stride: int):
    """滑窗 PPL: 位置 i 的 logits 预测 token i+1, NLL 总和 / 预测数 后取 exp。"""
    total_nll, n_pred = 0.0, 0
    for begin in range(0, len(ids) - 1, stride):
        window = ids[begin : begin + seq_len].unsqueeze(0).to(DEVICE)
        logits = model(window).logits
        vocab = logits.shape[-1]
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab).float(),
            window[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        n_pred += window.numel() - 1
    return math.exp(total_nll / n_pred), n_pred


def load_model(path: str, dtype: torch.dtype):
    """量化产物 state_dict 含 dopt quantizer 参数 (scale/offset 等), HF 加载时
    会被忽略并刷大量 unexpected-keys 告警 —— 此处降级日志, 属预期行为。"""
    from transformers.utils import logging as hf_logging

    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True
        ).to(DEVICE).eval()
    finally:
        hf_logging.set_verbosity(verbosity)


def main():
    args = parse_args()
    stride = args.stride or args.seq_len
    dtype = resolve_dtype(args.dtype)
    print(f"[ppl] device={DEVICE} dtype={str(dtype).removeprefix('torch.')} "
          f"seq={args.seq_len} stride={stride}")

    tokenizer = AutoTokenizer.from_pretrained(args.float_model, trust_remote_code=True)
    ids = build_tokens(tokenizer, args.data, args.max_tokens)
    print(f"[ppl] data={args.data} tokens={len(ids)}")

    results = {}
    for tag, path in (("float", args.float_model), ("quant", args.quant_model)):
        model = load_model(path, dtype)
        results[tag] = perplexity(model, ids, args.seq_len, stride)
        del model
        print(f"[ppl] {tag:5s} {path}")
        print(f"      ppl={results[tag][0]:.4f} (n_pred={results[tag][1]})")

    fp, qp = results["float"][0], results["quant"][0]
    print(f"[ppl] 量化影响: {fp:.4f} -> {qp:.4f}  (+{(qp - fp) / fp * 100:.2f}%)")


if __name__ == "__main__":
    main()
