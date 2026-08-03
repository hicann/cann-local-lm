"""Export the first 5 transformer layers of Gemma4-E2B as ONNX with KV cache I/O.

直接使用 Gemma4ExportWrapper (cann_lm/models/gemma4/export_wrapper.py) 导出。
sample_inputs 和 input/output names 由 base ExportWrapper 的通用方法生成
（make_sample_inputs + make_input_output_names），无需在脚本里重复定义。

past_key_values 用固定 shape 的零 tensor + mask 屏蔽 → prefill 正确
embeddings=None, output_pos=None → Python 分支在 tracing 时解析, 不是 ONNX 输入

环境变量设置:
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH="/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}"
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  _PYTHON_LIB_DIR="/opt/buildtools/python-3.12.9/lib"
  _TORCHAIR_STUBS="/mnt/workspace/torchair/build/stubs"
  export LD_LIBRARY_PATH="${_PYTHON_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${_TORCHAIR_STUBS}:${LD_LIBRARY_PATH:-}"
  export PYTHONUNBUFFERED=1

用法:
  python export_gemma4_5layers_onnx_kv.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import dataclasses

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.gemma4.export_wrapper import Gemma4ExportWrapper


def _register_cann_lm_decompositions():
    """注册 cann_lm 自定义算子的 ONNX decomposition (分解为 aten ops → ONNX symbolic).

    仅对有 CPU eager kernel 的算子 (rms_norm/add_rms_norm/rotary_mul) 注册,
    fake-only 算子 (rope_in_place/quant_matmul/fused_infer_attention_score/beam_search)
    无 eager kernel, 需单独 ONNX symbolic (此处不涉及, gemma4 不用).
    """
    from torch._decomp import register_decomposition

    @register_decomposition(torch.ops.cann_lm.rms_norm.default)
    def _rms_norm_decomp(x, gamma, *, epsilon=1e-6):
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(ms + epsilon)
        y = (x.float() * rstd * gamma.float()).to(x.dtype)
        return y, rstd

    @register_decomposition(torch.ops.cann_lm.add_rms_norm.default)
    def _add_rms_norm_decomp(x1, x2, gamma, *, epsilon=1e-6):
        s = x1 + x2
        ms = s.float().pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(ms + epsilon)
        y = (s.float() * rstd * gamma.float()).to(s.dtype)
        return y, rstd, s

    @register_decomposition(torch.ops.cann_lm.rotary_mul.default)
    def _rotary_mul_decomp(x, cos, sin, *, rotary_mode="half"):
        if rotary_mode != "half":
            raise NotImplementedError(f"rotary_mode={rotary_mode!r} not supported")
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos + rotated * sin).to(x.dtype)

HF_PATH = "/mnt/workspace/data/models/gemma-4-E2B-it"
OUT_ONNX = Path(__file__).resolve().parent / "generate" / "gemma4_5layers_kv.onnx"
SEQ_LEN = 8
NUM_LAYERS = 5
KV_CACHE_MAX_LEN = 2048


def main() -> None:
    _register_cann_lm_decompositions()
    dtype = torch.bfloat16

    # 1. Build a 5-layer config
    with open(f"{HF_PATH}/config.json") as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg["text_config"]
    valid_fields = {f.name for f in dataclasses.fields(Gemma4TextConfig)}
    filtered = {k: v for k, v in text_cfg.items() if k in valid_fields}
    filtered["num_hidden_layers"] = NUM_LAYERS
    filtered["layer_types"] = text_cfg["layer_types"][:NUM_LAYERS]
    filtered["num_kv_shared_layers"] = 0
    filtered["hidden_size_per_layer_input"] = 0
    cfg = Gemma4TextConfig(**filtered)
    if hasattr(cfg, "__post_init__"):
        cfg.__post_init__()

    print(f"[1/4] Build Gemma4ForCausalLM (5 layers, {cfg.layer_types})")
    model = Gemma4ForCausalLM(cfg).to(dtype).eval()
    print(f"      params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[2/4] Load weights from {HF_PATH}")
    model = load_gemma4_weights(model, HF_PATH, device="cpu", dtype=dtype)

    # 3. 用 Gemma4ExportWrapper + base 通用方法构造 sample 和 names
    print(f"[3/4] Create Gemma4ExportWrapper (kv_cache_max_len={KV_CACHE_MAX_LEN})")
    wrapper = Gemma4ExportWrapper(model, kv_cache_max_len=KV_CACHE_MAX_LEN, batch=1).to(dtype).eval()

    # base ExportWrapper 通用方法：make_sample_inputs + make_input_output_names
    sample_inputs = wrapper.make_sample_inputs(seq_len=SEQ_LEN, dtype=dtype)
    input_names, output_names = wrapper.make_input_output_names()

    # Verify forward
    with torch.no_grad():
        out = wrapper(*sample_inputs)
    num_outputs = len(out)
    print(f"      forward OK: {num_outputs} outputs (1 logits + {num_outputs-1} KV)")
    print(f"      logits: {list(out[0].shape)}")
    print(f"      kv_out[0]: {list(out[1].shape)} (cache_len={out[1].shape[2]})")

    # 4. Export ONNX
    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] torch.onnx.export → {OUT_ONNX}")
    torch.onnx.export(
        wrapper,
        sample_inputs,
        str(OUT_ONNX),
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
    )
    sz = OUT_ONNX.stat().st_size
    print(f"      Done. {OUT_ONNX} ({sz / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
