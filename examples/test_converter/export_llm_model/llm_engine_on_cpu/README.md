# llm_engine_on_cpu — 三模型 CPU 侧调试与验证脚本集

cann_lm 三个模型（gemma4 / qwen3 / qwen3.5）在 **converter stubs 环境（CPU-only，无 NPU）** 下的纯 Python 级调试与验证脚本：单元测试、端到端推理、vs HF 数值对齐、chunk 推理、epair 回环（`ep.module()`）、多轮对话、wrapper 自测。

> 这里验证的是 **PyTorch/epair 层**（导出前后、数值对齐、回环）。NPU 上的 **OM 编译 + 出字** 验证见 [`../om_npu_verify/`](../om_npu_verify)（`run_*_om.py` / `run_*_gen.py`）。

## 目录结构

```
llm_engine_on_cpu/
├── hf_cpu.py                 # 禁 transformers 的 torch_npu probe 的 shim（跑 HF 前先 import）
├── README.md                 # 本文件（总索引）
├── skills/                   # 通用技能（跨模型）
│   ├── cann-lm-epair-export/SKILL.md
│   ├── torch-delegate-epair/SKILL.md
│   └── epair-om-binary-compare/SKILL.md
├── debug_gemma4/             # gemma4-E2B 调试脚本（8 .py + README + skills/）
├── debug_qwen3/              # qwen3-0.6B 调试脚本（5 .py + README + skills/）
└── debug_qwen3_5/            # qwen3.5-0.8B 调试脚本（9 .py + README + skills/）
```

各 `debug_*/README.md` 有该模型每个脚本的详细用法/参数/验证结果——本文件只做跨模型总览 + 公共环境。

## 公共环境（三个目录通用，跑任何脚本前先 source）

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0           # 阻止 torch autoload npu backend
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0  # 不走 fallback codegen
VENV=/mnt/workspace/torch-delegate-harmony/.venv   # torch 2.9.0+cpu + torchair + torch_delegate + transformers
```

- **stubs 必须用** `/mnt/workspace/torchair/build/stubs`（ABI 匹配）；勿用真实 CANN lib64（会 `aclrtSetDevice 107001` 无设备）。
- 跑 HF transformers 对比时，先 `import hf_cpu`（禁 torch_npu probe，shim 在本目录 `hf_cpu.py`）。

## 跨模型用例对照表

| 用例类型 | gemma4 | qwen3 | qwen3.5 |
|---|---|---|---|
| 单元测试 | `verify_gemma4.py --test-only`（45 项） | `verify_qwen3.py --test-only`（35 项） | `test_modeling_qwen3_5.py`（68 项） |
| 端到端推理 | `verify_gemma4.py` | `verify_qwen3.py` | `run_qwen3_5.py` |
| vs HF 数值对齐 | `compare_hf_vs_ep.py` / `verify_gemma4_chunk.py --vs-hf` | `verify_qwen3_vs_hf.py`（full/wrapper/embeddings） | `verify_qwen3_5_vs_hf.py` |
| chunk 推理 | `verify_gemma4_chunk.py` | `verify_qwen3_chunk.py` | `verify_qwen3_5_chunk.py` |
| epair 回环（ep.module） | `verify_gemma4_epair.py` | `verify_qwen3_epair.py`（含 `--dynamic`） | `verify_qwen3_5_epair.py` / `verify_qwen3_5_all.py` |
| 多轮对话 | — | `verify_qwen3_multiturn.py` | `verify_qwen3_5_multiturn.py` |
| wrapper 自测（vs model） | `verify_gemma4_wrapper_bypass.py` | （含在 `verify_qwen3.py`） | `verify_wrapper_qwen3_5.py` + `verify_wrapper_qwen3_5_decode.py` |
| wrapper 导出验证（图无 masked_fill） | `verify_gemma4_wrapper_export.py` | — | — |
| tiny e2e epair→OM 产物 | `verify_gemma4_tiny_epair_om.py` | — | — |
| ONNX 导出（带 KV cache） | `export_gemma4_5layers_onnx_kv.py` | — | — |
| 全量验证（all-in-one） | — | — | `verify_qwen3_5_all.py` |

> 各脚本参数/用法/验证结果见对应 `debug_*/README.md`。上表含部分 `debug_*/README.md` 未单列的脚本（gemma4 的 `verify_gemma4_wrapper_bypass/export/tiny_epair_om`、qwen3.5 的 `verify_wrapper_qwen3_5_decode`）——这些 README 的脚本表略有滞后，以本表 + 实际文件为准。

## 模型与路径

| 模型 | 参数 | 层数 | 架构特点 | HF 路径 |
|---|---|---|---|---|
| gemma4-E2B-it | 5B | 35 | 20 层 KV 共享 + sliding/full attention + PLE | `/mnt/workspace/data/models/gemma-4-E2B-it` |
| qwen3-0.6B | 0.596B | 28 | 标准 GQA + thinking 模式 | `/mnt/workspace/data/models/qwen3-0.6B` |
| Qwen3.5-0.8B | 0.75B | 24 | 6 full + 18 linear（GatedDeltaNet + delta_cache） | `/mnt/workspace/data/models/Qwen3.5-0.8B` |

## 通用调试经验（三个模型共通）

1. **必须 bfloat16**（float16 输出退化重复）。
2. **验证 logits 用 greedy 而非 sampling**（sampling 随机性无法判 bug）。
3. **gemma4 必须加 BOS**（id=2，tokenizer 不自动加）。
4. **HF 对比需 `attn_implementation="eager"`**（对齐 cann_lm eager attention，非 sdpa）。
5. **scatter-write**（KV cache 用 `index_copy` 写入预分配 buffer，K=kv_max）已改造三个模型。
6. 详见各 `debug_*/skills/` 下的 SKILL.md。

## 文档关联

- 各 `debug_*/README.md`：单模型脚本详情。
- `../../../../../docs/cann_lm_converter/converter_design.md`：converter 全流程 + 精度差异分析 + 遗留任务。
- [`../om_npu_verify/README.md`](../om_npu_verify/README.md)：NPU 侧 OM 编译 + 出字验证。
