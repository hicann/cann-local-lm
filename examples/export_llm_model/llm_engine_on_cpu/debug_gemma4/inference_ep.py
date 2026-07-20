#!/usr/bin/env python3
"""使用 ep 模型进行文本推理

加载 gemma4 → ExportWrapper → torch.export → ep.module() 推理
输入文本 → BOS + tokenize → pad → ep prefill → wrapper decode → 输出文本

环境变量设置 (运行前必须 source):

  # 1. CANN 环境
  source /home/developer/Ascend/cann/set_env.sh

  # 2. Python 路径
  export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"

  # 3. 禁用 fallback codegen 和 backend autoload
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0

  # 4. 动态链接库路径 (CANN + torchair stubs + Python lib)
  #    CANN lib64 必须在 torchair stubs 之前 (否则 liberror_manager.so 被覆盖)
  #    Python base lib 必须在最前 (否则 libpython3.12.so 被覆盖)
  _PYTHON_LIB_DIR="/opt/buildtools/python-3.12.9/lib"
  _TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs"
  export LD_LIBRARY_PATH="${_PYTHON_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}:${LD_LIBRARY_PATH:-}"

  # 5. 输出不缓冲 (实时显示)
  export PYTHONUNBUFFERED=1

  # 6. Venv Python
  PYTHON="/mnt/workspace/torchair_lazy_venv/bin/python3"

完整一键命令:
  source /home/developer/Ascend/cann/set_env.sh && \
  export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}" && \
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0 && \
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0 && \
  _PYTHON_LIB_DIR="/opt/buildtools/python-3.12.9/lib" && \
  _TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs" && \
  export LD_LIBRARY_PATH="${_PYTHON_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}:${LD_LIBRARY_PATH:-}" && \
  export PYTHONUNBUFFERED=1 && \
  cd /mnt/workspace/project/CANN-LM_5883/examples/export_llm_model/llm_engine_on_cpu/debug_gemma4 && \
  /mnt/workspace/torchair_lazy_venv/bin/python3 inference_ep.py

环境变量说明:
  ASCEND_HOME_PATH          — CANN 安装路径 (set_env.sh 设置)
  PYTHONPATH                — 项目根目录 (import cann_lm 需要)
  TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0 — 禁用 fallback codegen (已删除该功能)
  TORCH_DEVICE_BACKEND_AUTOLOAD=0         — 阻止 torch autoload npu backend
  LD_LIBRARY_PATH           — 动态链接库搜索路径
    顺序: Python lib → CANN lib64 → torchair stubs
    CANN lib64 必须在 stubs 前: 否则 stubs 的 liberror_manager.so (空实现)
    会覆盖 CANN 的真实实现, 导致 GE 编译报错
    Python lib 必须在最前: 否则 CANN 的 libpython3.12.so (不同编译版本)
    会覆盖系统的正确版本, 导致 Python 崩溃
  PYTHONUNBUFFERED=1        — 禁用 Python 输出缓冲, 实时显示日志
  Venv: /mnt/workspace/torchair_lazy_venv  — torch + torch_delegate + torchair 安装的 venv

用法:
  python inference_ep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper
from cann_lm.models.fake_ops import register_ge_converters
from transformers import AutoTokenizer

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B"
MAX_NEW_TOKENS = 20
EXPORT_SEQ_LEN = 8
BOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0
# Sampling 参数 (来自 generation_config.json)
TEMPERATURE = 1.0
TOP_K = 64
TOP_P = 0.95
DO_SAMPLE = True

PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "人工智能的未来发展方向是",
]


def make_causal_mask_with_padding(q_len, k_len, real_q_len, real_k_len, dtype):
    min_val = torch.finfo(dtype).min
    mask = torch.triu(torch.full((q_len, k_len), min_val, dtype=dtype), diagonal=1)
    if real_q_len < q_len:
        mask[real_q_len:, :] = min_val
    if real_k_len < k_len:
        mask[:, real_k_len:] = min_val
    return mask.unsqueeze(0).unsqueeze(0)


def reconstruct_kv(flat_kv):
    return [(flat_kv[i], flat_kv[i + 1]) for i in range(0, len(flat_kv), 2)]


def load_and_export():
    """加载模型 → ExportWrapper → torch.export"""
    with open(f"{HF_PATH}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    import dataclasses
    valid = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    config = Gemma4TextConfig(**{k: v for k, v in text_cfg.items() if k in valid})
    if hasattr(config, "__post_init__"):
        config.__post_init__()

    model = Gemma4ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model = load_gemma4_weights(model, HF_PATH, device="cpu", dtype=torch.bfloat16)

    register_ge_converters()
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=2048, batch=1).eval()
    sample_inputs = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN)
    ep = torch.export.export(wrapper, sample_inputs, strict=False)

    return ep, wrapper, config


def sample_next_token(logits, temperature, top_k, top_p):
    """从 logits 中采样下一个 token"""
    if temperature == 0:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    logits = logits[:, -1, :] / temperature

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(ep, wrapper, input_ids, real_seq_len, max_new_tokens, dtype):
    """用 ep 模型推理：ep prefill + wrapper decode (with sampling)"""
    module = ep.module()

    # Pad 到 EXPORT_SEQ_LEN
    if real_seq_len < EXPORT_SEQ_LEN:
        pad_len = EXPORT_SEQ_LEN - real_seq_len
        input_ids_padded = torch.cat([
            input_ids,
            torch.full((1, pad_len), PAD_TOKEN_ID, dtype=torch.long),
        ], dim=1)
    else:
        input_ids_padded = input_ids

    # 构建 4D mask (causal + padding 屏蔽)
    mask = make_causal_mask_with_padding(
        EXPORT_SEQ_LEN, EXPORT_SEQ_LEN, real_seq_len, real_seq_len, dtype)
    output_pos = torch.tensor([real_seq_len - 1], dtype=torch.int32)

    # Prefill: ep.module()
    sample = wrapper.get_sample_inputs(seq_len=EXPORT_SEQ_LEN)
    traced_inputs = list(sample)
    traced_inputs[0] = input_ids_padded
    traced_inputs[1] = mask
    traced_inputs[6] = output_pos

    with torch.no_grad():
        out = module(*traced_inputs)
        logits = out[0]
        kv_flat = list(out[1:])
    kv = reconstruct_kv(kv_flat)

    # Decode: wrapper (with sampling)
    tokens = []
    for _ in range(max_new_tokens):
        if DO_SAMPLE:
            next_token = sample_next_token(logits, TEMPERATURE, TOP_K, TOP_P)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == 1:  # EOS
            break
        cached_len = EXPORT_SEQ_LEN + len(tokens)
        mask_dec = torch.zeros(1, 1, 1, cached_len, dtype=dtype)
        with torch.no_grad():
            logits, *kv_flat = wrapper(
                next_token, mask_dec, None, kv,
                torch.tensor([cached_len - 1], dtype=torch.long),
                None, None,
            )
        kv = reconstruct_kv(kv_flat)

    return tokens


def main():
    dtype = torch.bfloat16

    print("[1/3] 加载模型 + 导出 ep")
    ep, wrapper, config = load_and_export()
    print(f"  模型: Gemma4ForCausalLM (params={sum(p.numel() for p in wrapper.model.parameters()):,})")
    print(f"  层数: {config.num_hidden_layers} (KV共享: {getattr(config, 'num_kv_shared_layers', 0)})")
    print(f"  导出: {len(ep.graph.nodes)} 图节点")

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    print(f"\n[2/3] 推理 (seq_len={EXPORT_SEQ_LEN}, max_new_tokens={MAX_NEW_TOKENS})")

    results = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[BOS_TOKEN_ID] + ids], dtype=torch.long)
        real_seq_len = input_ids.shape[1]

        tokens = generate(ep, wrapper, input_ids, real_seq_len, MAX_NEW_TOKENS, dtype)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append((prompt, text))

        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Input tokens: {real_seq_len} (含 BOS)")
        print(f"  Output tokens: {len(tokens)}")
        print(f"  Output: \"{text}\"")

    print(f"\n[3/3] 推理完成")
    print(f"{'='*60}")
    for prompt, text in results:
        print(f"  \"{prompt}\" → \"{text}\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
