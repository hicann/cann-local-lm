# dopt 使用手册（vendored GPTQ 权重量化框架）

> 提取自 [dopt_design.md](./dopt_design.md) 的介绍与使用部分（架构解析 / 历史调试档案 / converter 导出细节不在本手册范围，见原设计文档）。
>
> 状态: 当前采用 | 最后核对: 2026-10-10

## 目录

- [dopt 是什么](#dopt-是什么)
- [环境准备](#环境准备)
- [使用三步](#使用三步)
- [产物](#产物)
- [可设置参数](#可设置参数)
- [已知约束](#已知约束)
- [相关文档](#相关文档)

---

## dopt 是什么

`dopt_lm/` 是华为 vendored 的 **GPTQ 风格权重 PTQ 框架**（命名 `dopt` = "do optimize"，Quantization Optimization）。它本身是自洽的、可独立运行的 Python 包（入口 `opt_main.py`，CLI 模式），在 **NPU / GPU / CPU** 上对 HuggingFace 模型做 **GPTQ 权重量化**，产出 per-layer `group_scale`（per-channel × per-group 对称，offset=0）+ GPTQ 修正后的权重，最终序列化成 `trained_quant_weight.pth`（或 `--hf_type_save` 的 HF 目录）。

**调用关系**：dopt_lm 在主仓代码中**不被 `import`**，而是作为独立 CLI 工具运行，其产物（.pth/safetensors）由 converter 的 `--quant-path` 在导出时加载。dopt_lm 内部通过相对导入 `from dopt_lm.xxx import yyy` 自引用（需将 `cann_lm/converter/quant` 放进 `PYTHONPATH`）。

> **说明**：算法本质是 GPTQ（Frantar et al. ICLR 2023，基于 Hessian 矩阵的 PTQ）。

## 环境准备

dopt GPTQ 通过 `DEVICE=npu|cuda|cpu` 选择执行设备。NPU 需要 `torch_npu`，CUDA 需要可用的 CUDA PyTorch；CPU 不依赖设备扩展（fp16 配置自动回退 FP32，适合功能验证和小模型，大模型极慢）。

```bash
# NPU 环境
source /home/developer/Ascend/cann/set_env.sh
# dopt_lm 内部相对 import `from dopt_lm.xxx` → 需 cann_lm/converter/quant 在 PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/cann_lm/converter/quant:${PYTHONPATH:-}

# CPU 示例（若配置中 fp16=true，运行时会自动回退到 FP32）
DEVICE=cpu python -m dopt_lm.opt_main --help
```

> 环境配置详见 [AGENTS.md](../../../../AGENTS.md) "NPU 环境（torch_npu，已验证）"。

## 使用三步

### Step 1: 生成 dopt_config 脚手架

首次运行时 `--dopt-config` 指向的文件不存在，dopt 会生成脚手架 JSON（所有层 `quant_strategy="float"`）并 `exit()`，需手动编辑策略后重跑。

```bash
cd cann_lm/converter/quant
python -m dopt_lm.opt_main \
    --model-path /path/to/hf-model \
    --dopt-config dopt_config.json
# → 生成 dopt_config.json (所有层 float) 后 exit
```

### Step 2: 编辑 dopt_config.json 设置层策略

编辑 `layer_strategy`，为每层选 `quant_strategy`（策略名见 [可设置参数](#可设置参数) ②）：

| 模块类型 | 推荐 `quant_strategy` | 说明 |
|---------|----------------------|------|
| `nn.Linear`（q/k/v/o/gate/up/down_proj 等） | `Quant_INT8-4_ECO` | W4A8 主路径（默认 group 128）；可用 `;group_size` 按层覆盖 |
| `nn.Linear`（lm_head） | `Quant_INT8-4_ECO` | W4A8 主路径 |
| `nn.Embedding` | `Quant_Embed_MinMax` | embed int8 per-channel，映射 `QEmbedding` |
| 不量化的层 | `float` | 从 GPTQ 任务移除，保留原始 module |

示例（`dopt_config.json` 片段）：

```json
{
  "layer_strategy": {
    "model.embed_tokens": {"quant_strategy": "Quant_Embed_MinMax"},
    "model.layers.0.self_attn.q_proj": {"quant_strategy": "Quant_INT8-4_ECO"},
    "model.layers.0.mlp.gate_proj": {"quant_strategy": "Quant_INT8-4_ECO"},
    "lm_head": {"quant_strategy": "Quant_INT8-4_ECO"}
  }
}
```

### Step 3: 运行 GPTQ

推荐用 `dopt_wrapper.py`（自动生成 `dopt_config.json` 到 `--output-dir`，免手动 Step 1/2）：

```bash
python cann_lm/converter/quant/dopt_wrapper.py \
    --model-path /path/to/hf-model \
    --optimize-config config.yaml \
    --output-dir /path/to/dopt_out \
    --hf-type-save
```

或手动 `python -m dopt_lm.opt_main`（需自备 `--dopt-config`）：

```bash
python -m dopt_lm.opt_main \
    --model-path /path/to/hf-model \
    --dopt-config dopt_config.json \
    --optimize-config config.yaml \
    --output-dir /path/to/dopt_out \
    --hf_type_save
```

- `--optimize-config config.yaml`：校准配置（详见 [可设置参数](#可设置参数) ③）
- `calibrate_sets/*.jsonl`：校准语料（由 `download_dataset.py` 下载生成，相对 CWD 解析，需在 `cann_lm/converter/quant/` 目录下运行）
- `--hf_type_save`：以 HF 格式保存（`save_pretrained`，产出 `model.safetensors`）；不带则存 `trained_quant_weight.pth`
- 设备：`DEVICE=npu|cuda|cpu` 环境变量

## 产物（--output-dir 指定目录）

| 文件 | 说明 |
|------|------|
| `model.safetensors` | GPTQ 权重 + per-layer `quant_op.weight_quantizer.s`（group_scale）+ `{bit, unsigned_quant, offset}` 元信息（大小随模型/dtype 变化） |
| `config.json` | HF 模型 config 副本 |
| `config.yaml` | dopt 写的训练配置回放 |

产物 state_dict keys 示例：

```
model.layers.N.self_attn.q_proj.weight                    (GPTQ 权重)
model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.s   (group_scale, 扁平 [out*in//group])
model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.{bit,unsigned_quant,offset}
model.embed_tokens.weight                                 (fakequant fp)
model.embed_tokens.quant_op.weight_quantizer.s            (per-channel scale)
```

> 产物由 converter `--quant-path` 消费（物化 int4 + swap `QuantLinear`），详见 [quant_design.md](./quant_design.md)。PPL 对比评估可用 [ppl_compare.py](../../cann_lm/converter/quant/ppl_compare.py)（用法见 [ppl_compare_usage.md](./ppl_compare_usage.md)）。

## 可设置参数

dopt 可设置的量化参数分布在 4 个层面，**用户实际可设的只有 ①②③，④ GPTQ 算法参数大多硬编码不暴露**。

### ① dopt CLI 参数（`opt_main.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | (必填) | HF 模型路径 |
| `--dopt-config` | (必填) | 层策略配置文件（`dopt_config.json`） |
| `--optimize-config` | `None` | 训练配置（`config.yaml`） |
| `--output-dir` | `None` | 输出目录 |
| `--hf_type_save` | `False` | 以 HF 格式保存（`save_pretrained`） |

### ② `dopt_config.json` 层策略

`quant_strategy` 从 `strategy_dict` 选（6 个预设），支持 `"name;group_size"` 语法按层覆盖 `group_size`：

| 策略名 | 用途 | weight | act | 备注 |
|--------|------|--------|-----|------|
| `float` | 不量化 | — | — | |
| `Quant_aigc_ptq` | W8A8 | 8(per-channel min_max) | 8(ema_min_max, unsigned) | |
| `Quant_aigc_qat` | W8A8 | 8(omni_min_max) | 8(omni, unsigned) | QAT 用 |
| `Quant_INT8-4_ECO` | **W4A8（主路径）** | 4(group 128, block 128, omni_group) | 8(dyn_s8, signed) | converter 默认 |
| `Quant_Embed_MinMax` | embed int8 | 8(per-channel min_max) | — | host gather |
| `Quant_Dummy` | 占位 | — | — | |

### ③ `config.yaml` 训练配置（GPTQ 校准）

| 字段 | 当前仓库值 | 说明 |
|------|-----------|------|
| `dataset.train_files` | `["calibrate_sets/calibrate_coig_cqia.jsonl"]` | 校准语料（`download_dataset.py` 下载，可替换/多选，支持 `path:ratio` 配比） |
| `dataset.ptq_samples` | `256` | 采样数 |
| `cutoff_len` | `1024` | 截断长度 |
| `extra_training_config.fp16` | `true` | fp16 训练+存盘（CPU dispatch 自动回落 float32） |

### ④ GPTQ 算法参数 — 硬编码，不暴露 CLI

`percdamp`(0.01) / `act_order`(False) / `static_groups`(False) / `sym`(True) / `blocksize`(128) / `seed`(1234) 等默认值硬编码在 `QuantArgs` / `fasterquant` / `configure` 中。要改需改源码（`ptq_loop.py` 的 `QuantArgs` 默认值或 `weight_quant_loop` 调用参数）；`blocksize` 不支持 `;` 语法覆盖，改它要动 `strategy_dict`。完整清单见 [dopt_design.md](./dopt_design.md) "可设置参数清单 ④"。

## 已知约束

1. **多模态 `ForConditionalGeneration` 检查点可直接跑**：`train.py` 的 `model_define` 检测 `model.model.language_model` 后自动抽取 `text_config` 重挂为文本-only ForCausalLM（lm_head/tie_word_embeddings 一并处理）；`ptq_loop` 带 gemma4/PLE 结构支持（`unique_layer_types` / `get_per_layer_inputs` / `per_layer_projection` / `embed_tokens_per_layer`）。已实测：`gemma-4-E4B-it` 多模态原始权重（`Gemma4ForConditionalGeneration`）在 CPU 上全量 GPTQ 跑通（2026-10-12，42 层 + PLE + lm_head）。前提是 transformers 版本支持该模型（gemma4 需新版如 5.x；Qwen3 4.51.3 即可）。
2. **CPU 模式为功能验证用途**：fp32 慢且吃主机内存，大模型建议 NPU/GPU。
3. **校准语料相对路径**基于 CWD 解析，需在 `cann_lm/converter/quant/` 目录下运行。
4. **产物不含 tokenizer**：converter/PPL 评估等下游消费时用原始 HF 模型目录的 tokenizer。

## 相关文档

| 文档 | 关联内容 |
|------|---------|
| [dopt_design.md](./dopt_design.md) | 完整设计文档：架构分层 / 文件树 / 调用链 / 历史调试与路径选型档案 |
| [quant_design.md](./quant_design.md) | converter 侧量化接入（`materialize` / `swap` / `quant_linear` / 端到端流程） |
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题 |
| [ppl_compare_usage.md](./ppl_compare_usage.md) | 浮点 vs 量化产物 PPL 对比脚本使用说明 |
