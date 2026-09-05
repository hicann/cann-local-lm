---
name: cann-lm-epair-export
description: |
  CANN-LM 模型 epair 导出技能。当用户提到以下场景时使用：
  - 导出 CANN-LM 模型 (gemma4/qwen3/qwen3_5) 为 epair 格式
  - cann_lm 自定义算子 (fused_ops: add_rms_norm/rms_norm/rotary_mul/rope_in_place/quant_matmul/fused_infer_attention_score/beam_search)
  - GE 转换器注册 (converter/export/ge_converter.py)
  - torchair 打桩库在无 NPU 设备时编译 epair
version: 2.0.0
metadata:
  author: assistant
  category: model-export
  last_updated: 2026-07-25
allowedTools:
  - read
  - write
  - bash
  - task
  - question
---

# CANN-LM epair 导出技能

CANN-LM 模型导出为 epair 的完整流水线。**详细设计见 `docs/cann_lm_converter/converter_design.md`**（目录结构/调用链/兼容性处理/精度差异分析）。

## 当前流程 (converter CLI)

```
cann_lm_converter.py  →  epair 目录 (ep_json + weights + model_config + tokenizer)
```

```bash
# 单图导出 (形状特化, seq_len=seq-len)
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/gemma-4-E2B-it \
    --output-dir /path/to/output \
    --dtype bfloat16 --mode single --seq-len 8

# prefill/decode 分离 (实际推理用, 两图 + 共享权重)
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model --output-dir /path/to/output \
    --mode prefill-decode --seq-len 128 --dtype bfloat16
```

## 文件结构 (当前)

```
cann_lm/
├── models/fused_ops/                # 自定义算子 (torch.ops.cann_lm.*)
│   ├── ops.py                       # 聚合 import (注册所有算子)
│   ├── _library.py                  # torch.library.Library("cann_lm", "DEF")
│   ├── add_rms_norm.py              # schema + meta + CPU eager kernel
│   ├── rms_norm.py                  # schema + meta + CPU eager kernel
│   ├── rotary_mul.py                # schema + meta + CPU eager kernel
│   ├── rope_in_place.py             # schema + meta (fake-only)
│   ├── quant_matmul.py              # schema + meta (fake-only, export-only)
│   ├── fused_infer_attention_score.py  # schema + meta (fake-only)
│   └── beam_search.py               # schema + meta (fake-only)
└── converter/
    ├── cann_lm_converter.py         # CLI 入口 + convert() 程序化入口
    ├── loader.py                    # load_model + build_config + collect_tokenizer_files
    ├── saver.py                     # save_epair / save_epair_prefill_decode
    ├── export_wrapper.py            # ExportWrapper 基类 (forward + get_sample_inputs)
    └── export/
        ├── epair_utils.py          # pack_weights + consolidated_weights_save
        ├── ge_converter.py          # register_ge_converters (cann_lm.* + aten.* GE 映射)
        └── _quant_matmul_converter.py # cann_lm::quant_matmul GE converter
```

## 算子分类

| 算子 | CPU eager kernel | GE converter | 用途 |
|------|:---:|------|------|
| `add_rms_norm` | ✓ | `ge.AddRmsNorm` | fused add + RMSNorm |
| `rms_norm` | ✓ | `ge.RmsNorm` | RMSNorm |
| `rotary_mul` | ✓ | `ge.RotaryMul` | fused rotate-half RoPE |
| `rope_in_place` | ✗ (fake) | `custom_op("RopeInPlace")` | inplace RoPE |
| `quant_matmul` | ✗ (fake) | `ge.QuantBatchMatmulV4` | 量化 matmul (**未实现真实量化**, fake-only) |
| `fused_infer_attention_score` | ✗ (fake) | `ge.FusedInferAttentionScore` | 融合注意力 |
| `beam_search` | ✗ (fake) | `custom_op("BeamSearch")` | 束搜索 |

> 前 3 个有 CPU eager kernel（可 eager 跑 + 导出）；后 4 个仅 fake/meta（export-only, 真实计算靠 GE converter 映射到 NPU 算子）。详见 `docs/cann_lm_models/fused_ops/fused_ops.md`。

## 环境准备 (CPU-only, 无 NPU)

```bash
# source 基底 (① converter: CPU venv + torchair stubs; 详见 llm_models_run_on_cpu/check_env.sh)
source ../../check_env.sh   # 或单独: source <proj_root>/scripts/set_env_export.sh
```

- **venv**: torch 2.9.0+cpu, torchair 源码编译 NO_ASCEND_SDK=1, torch_delegate 0.1.0, transformers 5.14.1
- **stubs**: 必须用 torchair 编译生成的 `build/stubs` (ABI 匹配), 不要用真实 CANN lib64 (会 `aclrtSetDevice 107001` 无设备)
- **量化未实现**: `quant_matmul` 是 fake-only 算子 (schema + meta, export-only), `_quant_matmul_converter.py` 仅映射到 `ge.QuantBatchMatmulV4`, 无 calibration / 权重量化流程

## 常见问题

### libhccl.so not found
torchair `_torchair.so` 链接 `libhccl.so`。用打桩库 `${TORCHAIR_ROOT}/build/stubs/libhccl.so`。

### torch_npu not available
epair 编译**不需要** `torch_npu`,打桩库提供 GE/hccl stub。venv 无 torch_npu 包 → transformers 不 probe, 不需 `import hf_cpu`。

### move_to_device_pass fails
跳过 `move_to_device_pass(ep, "npu")` (需 torch_npu 注册 npu 设备), 直接 CPU 调 `torch_delegate.export_to_epair`, 打桩库处理 GE graph 构建。

### _assert_tensor_metadata 节点
C++ 侧 `PlanNodes` 顶部 `IsAssertTensorMetadataNode` 跳过 (无需 Python 侧 strip ep_json)。
