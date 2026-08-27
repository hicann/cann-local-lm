# 量化接入 Converter 设计文档

> 将 HuggingFace 模型经 GPTQ 权重量化后导出为含 s8s4 量化 matmul 算子的 epair 格式
>
> 量化方案：**W4A8 s8s4**（激活 s8 per-token 动态对称 + 权重 s4 per-group 对称）
>
> 算子实现：**路径 C' — 复用 CANN 原生 `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4`（当前采用）**。早期曾试方案 A（自研 `cann_lm::quant_matmul_s8s4`），后回退到 C'：converter 侧 `_quant_matmul_converter.py` 用 V4 **full IR**（所有输入含 None placeholder，让 V4→V3 TBE 融合正确把 `x2_scale` 映射到 V3 的 `scale` 输入）+ **dummy bias zeros[N] workaround**（绕开 CANN V4 `InferNDimWithBias` 不支持 bias=nullptr 的限制），使整数 int8/int4 s8s4 经 V4→V3 融合后落到预编译 kernel 可达。详见下文 [路径 C' 算子规格与实现状态](#路径-c--npu_quant_matmul--v4-算子规格与实现状态)。
>
> 状态: 当前采用 | 最后核对: 2026-08-25 | ADR: 待记（方案 A→路径 C' 历史见本文"历史档案"段）

## 目录

- [概述](#概述)
- [量化工具链分工设计](#量化工具链分工设计)
- [quant 包架构](#quant-包架构)
- [量化策略](#量化策略)
- [端到端流程](#端到端流程)
- [quant 顶层文件详解](#quant-顶层文件详解)
- [dopt_lm 子包详解](./dopt_design.md)（已抽取到独立文档）
- [路径 C' — npu_quant_matmul → V4 算子规格与实现状态](#路径-c--npu_quant_matmul--v4-算子规格与实现状态)
- [C++ 注册与产物](#c-注册与产物)
- [weight_scale 落盘格式：fp32→uint64 位模式转换](#weight_scale-落盘格式fp32uint64-位模式转换)
- [已知问题与遗留](#已知问题与遗留)
- [附录 A — CANN QuantBatchMatmulV3 / V4 op_graph 源码分析（历史档案）](#附录-a--cann-quantbatchmatmulv3--v4-op_graph-源码分析历史档案)
- [附录 B — 历史调试与路径选型档案](./dopt_design.md)（已抽取到 dopt_design.md "附录"）

---

## 概述

> 量化接入的 CLI 参数（`--quant-path` / `--quant-method` / `--quant-bits`）见 [cli_design.md](../cli/cli_design.md)；量化 epair 的产物布局、host_weights 机制、C++ 加载侧见 [export_design.md](../export/export_design.md)。embedding 量化的 W8 host gather 子主题见 [embedding_quant_design.md](./embedding_quant_design.md)。

`cann_lm/converter/quant/` 下内置两块：

1. **dopt_lm（vendored GPTQ 框架）**：在 **NPU/GPU/CPU** 上对 HF 模型做 GPTQ 权重量化，产出 per-layer `group_scale`（per-channel × per-group 对称，offset=0）+ GPTQ 修正后的 fp16 权重，落盘 `trained_quant_weight.pth` 或 `model.safetensors`。dopt 自身只做 fakequant（权重留在 fp16），不物化真 int4，不接 converter 导出。**dopt 的完整设计（文件树 / 入口 / GPTQ 调用链 / 可设置参数清单 / 历史调试档案）见 [dopt_design.md](./dopt_design.md)**。
2. **converter 侧量化接入（`materialize.py` / `quant_linear.py` / `swap.py` / `auto_quant.py` / `dopt_wrapper.py` / `__init__.py`）**：在 **CPU venv + torchair stubs** 上加载 dopt 产物，物化成真 int4 payload + fp32 scale，把 `nn.Linear` 替换成 `QuantLinear`（forward 调 `npu::npu_dynamic_quant` + `npu::npu_quant_matmul`），然后 `torch.export` 产出含 `npu_quant_matmul` + `DynamicQuant` 节点的 epair。

### 能力边界

- **输入**：HF 模型目录 + 校准数据（`calibrate_sets/` 下 jsonl，由 `download_dataset.py` 下载生成）+ dopt GPTQ 产物（`model.safetensors` 或 `.pth`，目录由 `--quant-path`/dopt `--output-dir` 指定）。
- **输出**：epair 模型目录，其 `*_ep_json_with_ascend_ir.json` 含 `npu::npu_quant_matmul` + `npu::npu_dynamic_quant` IR 节点，权重 bin 含 nibble-packed int4（int8 存储）+ fp32 scale，`model_config.json` 含 `quant_info` 字段。
- **量化范围**：全部 `nn.Linear`（attn q/k/v/o_proj + mlp gate/up/down_proj + gemma4 的 `per_layer_input_gate` / `per_layer_projection` / `per_layer_model_projection` + `lm_head`），**仅 `embed_tokens` / `embed_tokens_per_layer` 不走 Linear swap**——它们是 `nn.Embedding`，由 `swap_embeddings_from_dopt` 走 W8 host gather（converter 自动检测 dopt 产物含 embed scale 即做）。
- **不包含**：GPTQ 与导出的内存链路一体化（仍两步磁盘桥接，见 [端到端流程](#端到端流程)）。CPU 跑 GPTQ **已支持**（dopt_lm `SUPPORTED_DEVICE_TYPES` 含 `cpu`，走 `--quant-device cpu` 或无加速器时自动回落），但 7B+ Hessian 极慢，仅小模型/调试适用。

---

## 量化工具链分工设计

量化参数众多（层策略 / group_size / bit / 算法参数 percdamp/act_order 等），全部暴露到 converter CLI 会让用户难以驾驭且易传错。工具链按 **dopt 分步精细调整 + CLI 一键导出** 分工：

### 设计原则

1. **dopt 分步使用（精细调整）**：dopt 作为独立 CLI 工具（[dopt_design.md](./dopt_design.md) "使用方法"），用户可调整所有量化参数——层策略（`dopt_config.json` 的 `quant_strategy`，由 `_build_layer_strategy` 自动生成后可手改）、校准（`config.yaml` 的 `ptq_samples`/`cutoff_len`，`train_files` 指向 `calibrate_sets/`）、group_size（`"name;group_size"` 语法）。产物落 `--output-dir` 指定目录。
2. **CLI 一键导出（少量重要参数）**：converter CLI 只暴露 **两个互斥** 量化入口 + 一个格式参数 + 一个设备参数，精细参数不暴露（走 dopt 分步）：
   - `--quant-path <dopt产物>`：场景 1，dopt 已 GPTQ，converter 消费产物
   - `--quant-method gptq`：场景 2，converter 自动 spawn dopt subprocess 跑 GPTQ（已实现 `auto_quant.py`，见 [一键式量化方案选型](#一键式量化方案选型)）
   - `--quant-bits w4a8|w4a16`：格式/位宽（默认 `w4a8`，`--quant-method` 时生效，传给 dopt_config 的 strategy）
   - `--quant-device npu|cuda|cpu`：场景 2 subprocess 设备（默认 `None` 继承父进程 `DEVICE` 或 spawn 检测，无加速器回落 `cpu`；`cpu` 仅小模型适用）
3. **converter 自动推断**：场景 1 的 `group_size` 从 dopt 产物 `group_scale` shape 自动反推（`swap.py`），`embed` 量化自动检测（dopt 产物含 `embed_tokens.quant_op...s` 即做），无需用户重复传参。

### 两场景对比

| | 场景 1: `--quant-path` | 场景 2: `--quant-method gptq` |
|---|---|---|
| 输入 | dopt GPTQ 产物（`--output-dir` 指定目录） | HF 原始权重 |
| 谁跑 dopt | 用户手动跑 dopt（`dopt_wrapper.py`，dopt_config 自动生成） | converter 自动 spawn dopt subprocess |
| 量化算法 | GPTQ（dopt，精度高） | GPTQ（dopt，精度高） |
| 格式 | 从产物推断 | `--quant-bits` 配（w4a8/w4a16，默认 w4a8） |
| group_size | 自动推断（从 dopt 产物） | dopt_config 自动生成（默认 128） |
| embed 量化 | 自动检测 | dopt_config 自动生成（Quant_Embed_MinMax） |
| 环境 | converter CPU stubs（消费离线产物） | converter CPU stubs + subprocess（NPU/GPU/CPU，`--quant-device` 选） |
| 状态 | ✅ 已实现 | ✅ 已实现（方案 C） |
| 精细调整 | dopt 分步（层策略等，dopt_config 自动生成后可手改） | dopt 分步（或 `--quant-bits` 简配） |

### 互斥约束

`--quant-path` 和 `--quant-method` 互斥（argparse `mutually_exclusive_group`），二选一：
- 都不传 → 不量化（导出 fp 原始权重 epair）
- 传 `--quant-path <产物>` → 场景 1
- 传 `--quant-method gptq` → 场景 2（+ `--quant-bits` 配格式）
- 都传 → argparse 报错

`--quant-bits` 仅 `--quant-method` 时生效（场景 1 从产物推断格式）。

> CLI 量化参数精简的设计取舍：`--quant-group-size` / `--quant-act` / `--quant-embeddings` / `--no-quant-linears` 已移除（group_size/act/embed 改自动推断或 dopt_config 配）。converter 的 min-max 快速路径（`swap_to_quant_linear` + `materialize_int4`）已删除（dopt 只支持 GPTQ，min-max 精度差无实用价值）。精细调整全部由 dopt 分步完成。

## quant 包架构

```
cann_lm/converter/quant/
├── __init__.py              包门面，re-export 7 个公共符号
├── materialize.py           权重物化纯函数：fp → (int4 payload, fp32 scale) + transdata_scale/restore_scale
├── quant_linear.py          导出用 QuantLinear nn.Module，forward 发射 npu::npu_quant_matmul IR
├── swap.py                  算子 IR 替换 + dopt 产物加载 + 量化元信息读取 + 层策略解析
├── auto_quant.py            一键式量化（方案 C：spawn dopt subprocess，检测 NPU/GPU/CPU）
├── dopt_wrapper.py          dopt subprocess 主体（model_define + weight_quant_pipline + 自动生成 dopt_config）
├── download_dataset.py      校准数据集下载脚本（下载/格式化为 jsonl，默认输出到 calibrate_sets/）
├── config.yaml              dopt GPTQ 校准训练配置（train_files 指向 calibrate_sets/，dopt_wrapper via --optimize-config）
├── calibrate_sets/                   校准数据集子目录（jsonl，由 download_dataset.py 下载生成，可替换/多选）
└── dopt_lm/                 vendored GPTQ 框架，独立 CLI 运行
```

> **dopt_config.json 不在 quant/ 包目录**：层策略配置（`{层名: quant_strategy}`）由 `_build_layer_strategy` 遍历模型自动生成，写入 dopt 产物目录（`--output-dir`/临时盘），converter 侧 `swap._read_quant_param_map` 从该产物目录读取。**两场景均自动生成**，无需用户手写。`download_dataset.py` 默认输出目录是 `calibration_data/`，但本仓库实际产物落在 `calibrate_sets/`（config.yaml 的 `train_files` 指向）。

### 依赖关系

```
__init__.py (门面, re-export 7 符号)
    │
    ├── materialize.py      ← 叶子，仅 import torch
    ├── quant_linear.py     ← import cann_lm.models.fused_ops.ops (注册副作用: npu_quant_matmul + npu_dynamic_quant)
    └── swap.py             ← 相对 import materialize + quant_linear

auto_quant.py               ← 场景 2 入口，subprocess 调 dopt_wrapper.py (独立进程，不 import swap)
                                    ↑
                                    │ 外部消费者
                cann_lm_converter.py:
                    场景 2:        from cann_lm.converter.quant.auto_quant import run_auto_quant
                    场景 1/2 共用:  from cann_lm.converter.quant import swap_from_dopt
```

- **无循环依赖**：`materialize ← swap → quant_linear` 的有向无环图；`materialize` 与 `quant_linear` 互不依赖；`auto_quant.py` 与 `swap.py` 解耦（subprocess 不 import swap）。
- **`__all__` 暴露 7 个符号**（见 `__init__.py` 的 `__all__`）：其中 **2 个** 被 `cann_lm_converter.py` 实际 import —— `swap_from_dopt`（主路径，场景 1/2 共用，一次 load sd 同时做 Linear swap + Embedding 抽取）+ `run_auto_quant`（从 `auto_quant` 直接 import，场景 2 spawn subprocess）。其余 5 个对外暴露为测试/离线/将来直接调用预留：
  - `swap_to_quant_linear_from_dopt`（向后兼容薄 wrapper，场景 1 离线用户或单独 Linear swap 用，已非主路径）
  - `swap_embeddings_from_dopt`（embedding int8+scale 抽取，被 `swap_from_dopt` 内部组合调用）
  - `load_dopt_state_dict`（dopt 产物加载，被 `swap_from_dopt`/`swap_to_quant_linear_from_dopt` 内部调用）
  - `materialize_int4_per_group_symmetric` / `materialize_int8_per_channel_symmetric` / `QuantLinear`（经 `swap.py` 内部相对 import 使用）

### I/O 定位

| 文件 | 谁读 | 怎么读 |
|------|------|--------|
| `config.yaml` | `dopt_wrapper.py`（via `--optimize-config`） | dopt subprocess 内部，**不经 converter 侧 .py** |
| `calibrate_sets/*.jsonl` | dopt GPTQ 校准（`config.yaml` 的 `train_files` 指向，相对 CWD = `quant/`） | dopt subprocess 内部，**不经 converter 侧 .py**；由 `download_dataset.py` 下载生成 |
| `dopt_config.json`（产物目录内） | dopt `--dopt-config`（dopt_wrapper 自动生成写入产物目录） + converter 侧 `swap._read_quant_param_map` 解析层策略 | **converter 侧 swap.py 也读**（从 dopt 产物目录解析 `{层名: 量化参数}`）；两场景均自动生成，非手写 |
| `<产物目录>/model.safetensors` | `swap.load_dopt_state_dict(quant_path)`（`swap.py` 的 `load_dopt_state_dict`，用 `safetensors.safe_open`） | swap.py 读 |

**结论**：converter 侧 5 个 .py 里，`swap.py`（`load_dopt_state_dict` + `_read_quant_param_map`）和 `auto_quant.py`（spawn subprocess）做 I/O；`materialize.py` / `quant_linear.py` / `dopt_wrapper.py` 完全无 I/O、无状态（`dopt_wrapper.py` 在 subprocess 内跑，I/O 走 dopt_lm）。`config.yaml` / `calibrate_sets/*.jsonl` 全是 dopt GPTQ 阶段输入；`dopt_config.json` 由 `_build_layer_strategy` 自动生成到产物目录，dopt 内部消费 + converter 侧 `swap.py` 解析层策略（`_read_quant_param_map`，方案 C 的 group_size/bit 来源）。

---

## 量化策略

### 方案：W4A8 s8s4（对称）

| 侧 | 位宽 | 粒度 | 对称/非对称 | scale 来源 | offset |
|----|------|------|-------------|-----------|--------|
| 权重 (x2) | int4 (s4) | per-output-channel × per-group(group_size, 默认 128) | 对称 | GPTQ 离线求（`group_scale`） | 0（对称，不存） |
| 激活 (x1) | int8 (s8) | per-token 动态 | 对称 | 运行时从 x 算（`npu_dynamic_quant` → `DynamicQuant` IR） | 0 |
| 输出 | fp16/bf16 | — | — | — | — |

对应 dopt `strategy.py` 的 **`Quant_act_weight_eco`**（`dopt_lm/quantizer/strategy.py`）：

```python
"Quant_act_weight_eco": {
    "weight": {"blocksize": 128, "group_size": 128, "bit": 4, "per_channel": True,
               "weight_algo": "omni_group_min_max"},
    "input":  {"bit": 8, "per_channel": False, "input_algo": "dyn_s8", "unsigned_quant": False},
}
```

- `group_size` 可经 `"Quant_act_weight_eco;128"` 语法覆盖（`strategy.get_optimize_cfg`）。
- `dyn_s8` = 激活动态 per-token int8（signed）—— 对应 W4A8 的 A8 侧，运行时算 per-token scale 传给 `npu_quant_matmul` 的 `pertoken_scale`。
- `omni_group_min_max` = LSQ 风格可学习 `quant_alpha` + sigmoid 自适应 scale 的 per-group 权重量化器（`dopt_lm/quantizer/qmin_max.py:OmniGroupMinMaxGridWeight`）。

### 量化范围与校准

- **范围**：全部 `nn.Linear`（`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/per_layer_input_gate/per_layer_projection/per_layer_model_projection` + `lm_head`）。
- **lm_head 量化**（重要）：`swap.DEFAULT_SKIP = ("embed_tokens", "embed_tokens_per_layer")`（`swap.py` 的 `DEFAULT_SKIP`）**不含 `lm_head`** → `lm_head` 作为大 NPU-side Linear 按 W4 QuantLinear 消费。converter 部署形态为 host-embed + NPU-lm_head（`tie_word_embeddings` 必须解 tie：embed 在 host、lm_head 在 NPU，不同计算域无法共享权重）。
- **embed_tokens / embed_tokens_per_layer**：是 `nn.Embedding`（非 Linear，Linear swap 不触及），由 `swap_embeddings_from_dopt` 消费 W8 host gather——converter 自动检测 dopt 产物含 embed scale 即做（embed 外置，图内无 embed_tokens 节点，`--use-embeddings` 默认 True）。
- **校准数据**：`calibrate_sets/` 子目录下的 jsonl 文件（由 `download_dataset.py` 下载生成，支持 ultrachat_200k / fineweb_edu / toolbench / bfclv4 / c4 / coig_cqia / mc4 七种；`config.yaml` 的 `train_files` 指向具体文件，可替换/多选）。
- **`config.yaml` 实际内容**（`quant/config.yaml`）：

  ```yaml
  dataset:
    # 校准集: calibrate_sets/ 下 jsonl (由 download_dataset.py 下载生成, 可替换/多选)
    train_files:
      - "calibrate_sets/calibrate_c4.jsonl"
    ptq_samples: 1024
  cutoff_len: 512

  extra_training_config:
    fp16: true   # GPTQ 跑 fp16 + 存盘 fp16,避免 dopt 默认 fp32 导致产物翻倍 (CPU dispatch 自动回落 float32)
  ```

  > `ptq_samples=1024`（非历史文档误记的 128）。CPU 设备时 dopt_lm `dispatch_dtype` 自动回落 `float32`（`train.py` CPU 分支），fp16 标志不冲突。

- **GPTQ 跑在 HF 模型**：dopt 现成依赖 `transformers.AutoModelForCausalLM` + `accelerate` dispatch，不改 dopt；量化参数按层名 port 到 cann_lm 模型（两者命名一致：`model.layers.N.self_attn.q_proj` 等）。

### dopt 实测量化范围（gemma4-E4B，346 个 quant 模块）

自动生成的 `dopt_config.json` 配置：`embed_tokens` → `Quant_Embed_MinMax`（bit=8 int8 per-channel，host gather）、`lm_head` → `Quant_lm_head`（bit=4）、其余 Linear → `Quant_act_weight_eco`（默认 group 128）。dopt 产物里 `lm_head.weight` 与 `model.embed_tokens.weight` 已**实质 untie**（max diff ≈0.01 = GPTQ 修正量，lm_head 是 GPTQ 修正版给 W4、embed 是原始版给 W8）。

- **bit 分布**：bit=4 共 344 个（attn q/k/v/o_proj + mlp gate/up/down_proj + PLE per_layer_input_gate/per_layer_projection/per_layer_model_projection + lm_head），bit=8 共 2 个（embed_tokens + embed_tokens_per_layer）。
- **gemma4 KV 共享**：`k_proj`/`v_proj` 只 0..23（24 层）有独立权重，后 18 层（24..41）复用前层 KV 无独立 k/v_proj——模型结构决定，非漏量化。

---

## 端到端流程

### 两步流程（现状，已实现，磁盘桥接）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OFFLINE (NPU/GPU/CPU, 系统 Python + torch_npu(NPU 时)) — Step A: dopt GPTQ │
│                                                                         │
│  两种触发:                                                               │
│   场景 1 (--quant-path): 用户手动跑 dopt_wrapper (dopt_config 自动生成) │
│   场景 2 (--quant-method gptq): auto_quant.run_auto_quant 检测 NPU/GPU/CPU │
│           → spawn dopt_wrapper.py subprocess (subprocess env, cwd=quant/)│
│           → dopt_wrapper 内 _build_layer_strategy 自动生成 dopt_config  │
│                                                                         │
│  输入:                                                                   │
│   - HF 模型 (--hf-path / --model-path)                                  │
│   - cann_lm/converter/quant/config.yaml      ← --optimize-config         │
│       (dataset.train_files=calibrate_sets/*.jsonl, ptq_samples=1024,    │
│        cutoff_len=512, fp16=true)                                       │
│   - calibrate_sets/*.jsonl ← 校准语料 (download_dataset.py 下载, CWD)   │
│   - dopt_config.json ← --dopt-config (层策略, _build_layer_strategy 自动生成)│
│       (layer_strategy: Linear→Quant_act_weight_eco, group_size=128,        │
│        lm_head→Quant_lm_head, Embedding→Quant_Embed_MinMax)             │
│                                                                         │
│  执行: python dopt_wrapper.py --model-path ... --hf-type-save           │
│        (两场景都用 dopt_wrapper; 或手动 dopt_lm.opt_main + generate_dopt_config)│
│                                                                         │
│  产物 (--output-dir 指定目录 或 场景 2 临时盘):                          │
│   - model.safetensors (~3GB) ← load_dopt_state_dict 读这个               │
│     keys:                                                               │
│       model.layers.N.self_attn.q_proj.weight            (GPTQ fp16)     │
│       model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.s       │
│                                  (扁平 [out*in//group] group_scale)    │
│       ....weight_quantizer.{bit, unsigned_quant, offset}                │
│                                  (_read_dopt_quant_meta 读)            │
│       model.embed_tokens.weight (fakequant fp)                         │
│       model.embed_tokens.quant_op.weight_quantizer.s (per-channel)     │
│   - config.json (HF 模型 config 副本)                                   │
│   - config.yaml (dopt 写的训练回放)                                     │
│   - dopt_config.json (层策略, _build_layer_strategy 自动生成; converter  │
│     侧 _read_quant_param_map 读它解析 group_size/bit)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 磁盘桥接 (场景 1: 用户产物; 场景 2: 临时盘)
┌─────────────────────────────────────────────────────────────────────────┐
│  CONVERTER (CPU venv, torchair stubs) — Step B-E                         │
│                                                                         │
│  cann_lm_converter.py:                                                  │
│   B. load_model(hf_path) → nn.Module (cann_lm, fp16)                    │
│      (场景 2: 在 run_auto_quant 之后, subprocess 已退出才 load)          │
│   C+D. swap_from_dopt(model, dopt_out_path):  # 主路径, 一次 load       │
│      a. load_dopt_state_dict(dopt_out_path) → sd: dict                   │
│      b. _read_quant_param_map(dopt_out_path) → param_map (层策略)       │
│      c. _swap_linear_from_sd(model, sd, param_map=param_map):           │
│         for each nn.Linear (非 DEFAULT_SKIP):                           │
│           - sd[name.weight] (GPTQ fp) + sd[...weight_quantizer.s] (s)  │
│           - group_size: param_map[name] 优先, fallback _infer_group_size│
│           - _read_dopt_quant_meta → bit/uq/offset (校验 uq==0, offset 全0)│
│           - materialize_int4_per_group_symmetric(w, scale2d, gs,       │
│             signed=True, nbits=bit, packed=True)                        │
│             → (weight_q int8 [n,k//2] nibble-packed, weight_scale fp32) │
│           - 预转置: weight_q [n,k//2]→[k//2,n], scale [n,k//g]→[k//g,n] │
│           - QuantLinear(weight_q, weight_scale, gs, bias, act_quant=True)│
│           - setattr 替换 nn.Linear                                      │
│         return quant_info dict → artifact.quant_info                    │
│      d. swap_embeddings_from_dopt(sd, model) → embed_quant dict         │
│         → saver 经 wrapper.set_quant_embeddings() 注入图入口             │
│      return (quant_info, embed_quant)                                  │
│   E. save_epair_prefill_decode(...)                                      │
│      → torch.export(model) → ep_json (含 npu::npu_quant_matmul +        │
│         npu::npu_dynamic_quant IR) + weights_*.bin (int4 packed +       │
│         fp32 scale, weight_scale 落盘经 transdata_scale 转 int64)       │
│         + model_config.json (含 quant_info) + weight_info.json         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  最终产物 (epair 格式, output-dir/)                                      │
│   - prefill_ep_json_with_ascend_ir.json  (含 npu::npu_quant_matmul 节点  │
│     + npu::npu_dynamic_quant 节点, QuantLinear 在 nn_module_stack)      │
│   - decode_ep_json_with_ascend_ir.json                                  │
│   - weights_*.bin  (int4 nibble-packed int8 + fp32 scale buffer,        │
│     weight_scale 落盘为 int64 位模式)                                    │
│   - model_config.json (quant_info: scheme/group_size/bit/n_quantized/...)│
│   - weight_info.json                                                    │
│   (host-embeddings 模式额外: embed_tokens_q.bin int8 +                  │
│    embed_tokens_scale.bin fp32, host 查表)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

后续编译成 OM 是下游 `epair → OM` 编译器的事，不在 quant/ 范围。

### 一键式量化方案选型

> **设计原则**：dopt 直接用 `transformers` 库源码进行量化（`AutoModelForCausalLM.from_pretrained` + Catcher/hooks），**不改 dopt 适配 cann_lm 模型**。GPTQ 跑 HF transformers 模型，converter 消费其 state_dict 产物 swap 到 cann_lm 模型。这保证 dopt 与 transformers 上游一致，维护成本低。

现状两步靠磁盘桥接（`--quant-path`）。要实现 CLI 一键式量化（用户传一条命令，converter 自动跑 GPTQ + swap + export），有三个方案：

#### 三方案对比

| 维度 | A. 改 dopt 跑 cann_lm | B. 内存桥接（进程内） | C. subprocess 调 dopt ✅选定 |
|------|------|------|------|
| 是否根治两实例 | ✅ 根治（单 cann_lm 实例） | ❌ 仍两实例（HF vs cann_lm） | ❌ 仍两实例（subprocess 内 HF） |
| 内存峰值 | ✅ 1× | ❌ 3-4×（cann_lm + dopt 2 份 HF + sd） | ⚠️ 2×（subprocess 内 dopt 2 份 + 主进程 cann_lm，分进程） |
| 环境冲突 | ⚠️ 单进程但仍需 NPU | ❌ converter 进程需同时 CPU stubs(export) + NPU(GPTQ)，矛盾 | ✅ 规避（subprocess NPU/GPU/CPU，主进程 CPU stubs） |
| 改动量 | ❌ 最大（dopt 适配三模型 + vendored diff） | ✅ 小（三处改动点） | ✅ 小（converter 加 subprocess 调用） |
| dopt 代码改动 | ❌ 大（get_llama/llama_sequential/visible_type_dict） | ⚠️ 中（加 ptq_quant_produce_state_dict） | ✅ 无（复用 dopt CLI） |
| dopt 全局状态风险 | ❌ 高（进程内 args 全局单例/重入） | ❌ 高（进程内） | ✅ 隔离（subprocess 独立） |
| 真一步化 | ✅ 是 | ✅ 是（进程内） | ❌ 否（自动化两步，converter 触发 subprocess + swap） |
| 维护成本 | ❌ 高（跟进 dopt vendored diff） | ⚠️ 中（dopt 加函数维护） | ✅ 低（dopt 不改） |
| 7B+ 适用 | ✅ 好（1×） | ❌ 差（3-4×） | ⚠️ 中（2× 分进程） |

#### 选定方案 C：subprocess 调 dopt

**理由**：
1. **守住设计原则**：dopt 用 transformers 源码量化，零改动（方案 A 改 dopt 适配 cann_lm 违背此原则）
2. **规避环境冲突**：converter 主进程保持 CPU stubs（export 段），subprocess 用 NPU/GPU/CPU 环境（GPTQ 段，dopt_lm `SUPPORTED_DEVICE_TYPES` 含 `cpu`），两套环境隔离（方案 B 环境冲突无解）
3. **dopt 全局状态隔离**：subprocess 独立，规避 `ptq_loop.py` 的 `args` 全局单例/重入风险
4. **维护成本低**：dopt 不改，跟进 transformers 上游即可

**代价**：非真一步化（subprocess 跑 dopt 本质仍是两步，只是用户视角一键）；产物传输（临时盘 IO 或内存管道）

#### 方案 C 设计

converter `--quant-method gptq` 一键触发：主进程 spawn `dopt_wrapper.py` subprocess（NPU/GPU/CPU 环境，由 `--quant-device` 选，无加速器回落 `cpu`）跑 GPTQ → 产物到临时盘 → converter swap + export。实现位于 `cann_lm/converter/quant/auto_quant.py`（`generate_dopt_config` + `run_auto_quant`）。

```
converter 主进程 (CPU stubs)                      subprocess (NPU/GPU/CPU 环境)
─────────────────────────────                     ─────────────────────────
convert(--quant-method gptq)                      python dopt_wrapper.py
  ├─ generate_dopt_config(artifact, quant_bits)    --model-path <hf> --optimize-config config.yaml
  │   遍历 cann_lm 模型生成 dopt_config             --output-dir <临时盘> --hf-type-save --quant-bits w4a8
  ├─ run_auto_quant → spawn subprocess ───────→     ├─ _build_layer_strategy 自动生成 dopt_config
  │   (系统 Python + subprocess env + cwd=quant/)  ├─ weight_quant_loop (GPTQ, NPU/GPU/CPU)
  │                                                  └─ 产物 → 临时盘 model.safetensors
  ├─ wait subprocess 完成
  ├─ swap_to_quant_linear_from_dopt(临时盘)         # 复用 --quant-path 逻辑
  ├─ swap_embeddings_from_dopt(sd, model)            # embed 自动检测
  └─ save_epair_prefill_decode(...)                  # export
```

目标命令：
```bash
DEVICE=npu python cann_lm_converter.py \
  --hf-path /path/to/hf --output-dir /path/to/out --dtype float16 \
  --quant-method gptq
# converter 自动: 检测设备 → spawn dopt subprocess (GPTQ) → 临时盘 → swap → export
# (--quant-method gptq 与 --quant-path 互斥; --quant-bits 配格式; --quant-device 显式选 npu/cuda/cpu)
```

**两阶段实现**：
1. **阶段 1（临时盘，已实现）**：subprocess 产物到临时盘（`tempfile`），converter 读临时盘 swap。简单，复用 `--quant-path` 逻辑。代价：临时盘 IO。
2. **阶段 2（内存管道，待实现）**：subprocess 产物到内存管道（pickle/mmap/共享内存），converter 读管道 swap。省 IO 但复杂。7B+ 大模型受益。

**实现状态**（`auto_quant.py` + `dopt_wrapper.py`）：
1. **subprocess 环境** ✅：`PYTHONPATH` 加 `quant/` + `DEVICE=<npu|cuda|cpu>` + `LD_LIBRARY_PATH` 去 stubs；`DOPT_PYTHON` 环境变量可覆盖系统 Python 路径
2. **calibration 路径** ✅：subprocess `cwd=cann_lm/converter/quant/`（config.yaml 的 `train_files` 指向 `calibrate_sets/*.jsonl`，相对 CWD）
3. **dopt_config 生成** ✅：两场景均由 `_build_layer_strategy` 遍历模型层名自动生成（写入产物目录），**无需用户手写**；场景 2 在 subprocess 内（dopt_wrapper），场景 1 离线由 `auto_quant.generate_dopt_config` 遍历 cann_lm `artifact.model` 生成临时 cfg 供 raw `dopt_lm.opt_main` 用（`examples/run_pd4_cases.sh:gen_dopt_config` 依赖）
4. **subprocess 失败处理** ✅：退出码非 0 raise `RuntimeError`，保留临时产物路径供调试
5. **设备检测** ✅：`_detect_device_type` 优先读父进程 `DEVICE`（含 `cpu`），未设才 spawn 检测（`torch_npu`/`torch.cuda`）；关 `ASCEND_SLOG_PRINT_TO_STDOUT` 防 CANN 日志污染 stdout 解析；**无 NPU/GPU 时回落 `cpu`**（dopt_lm 支持 CPU float32，7B+ Hessian 极慢，仅小模型/调试适用，不再报错）
6. **dopt_wrapper 主体** ✅：`model_define`（transformers fp16）+ `weight_quant_pipline`（GPTQ）已落地，`--hf-type-save` 产 safetensors
7. **w4a8** ✅：`dopt_wrapper.parse_args` 的 `--quant-bits` default=`w4a8`，choices=`["w4a8"]`
8. **w4a16** ❌ 不支持：`choices=["w4a8"]` 限制，无 preset W4A16 Linear strategy
9. **临时盘清理** ⏳ 待实现：当前不清理（保留供调试），7B+ 大文件需 `shutil.rmtree`
10. **实测验证** ⏳ 待做：cann_lm 层名 vs HF 匹配、subprocess 环境跑通、calibration 匹配

> `--quant-path`（消费已存在的 dopt 磁盘产物）保留作兜底，与 `--quant-method gptq`（自动 spawn dopt）互斥。`--quant-bits` 仅 `--quant-method` 时生效；`--quant-device`（`npu`/`cuda`/`cpu`，默认检测/回落）选 subprocess 设备。
### `materialize.py` — 权重物化纯函数

纯函数模块，把 fp16/bf16/fp32 的 `nn.Linear` 权重 `[out, in]` 物化成量化 payload + per-group/per-channel scale。**无状态、无副作用、无外部 I/O**。是整个量化链路的"数值核心"。

| 函数/常量 | 签名 | 参数数 | 作用 |
|------|------|--------|------|
| `_SCALE_MASK` | 模块级常量 `0x00000000FFFFE000` | — | 位掩码：截断尾数低 13 位 + 高 32 位清零（落盘格式） |
| `transdata_scale` | `(scale_fp32: Tensor) -> Tensor` | 1 | fp32 → int64 位模式 + `& _SCALE_MASK`（落盘格式转换）。torch 无 uint64，用 int64（位模式相同） |
| `restore_scale` | `(scale_packed: Tensor) -> Tensor` | 1 | int64（位模式 + 掩码）→ fp32（尾数低 13 位为 0，精度降低）。方案 A 遗留，当前不调但保留 |
| `pack_int4` | `(q: Tensor) -> Tensor` | 1 | Nibble-pack signed int4（存为 int8）成 `[..., k//2]` int8；byte[i] = int4(k=2i+1)<<4 \| int4(k=2i)。私有辅助，不在 `__all__` |
| `materialize_int4_per_group_symmetric` | `(weight, scale, group_size, signed=True, nbits=4, packed=False) -> (Tensor, Tensor)` | **6** | **主路径**：用 dopt 预算好的 per-group scale（GPTQ-optimized）量化，scale 原样返回。被 `_swap_linear_from_sd` 调用 |
| `materialize_int8_per_channel_symmetric` | `(weight, scale, signed=True) -> (Tensor, Tensor)` | 3 | 8-bit per-row（per-vocab-channel）对称量化（匹配 dopt `Quant_Embed_MinMax`），产出 (int8 `[vocab,hidden]`, fp32 scale `[vocab]`)。被 `swap_embeddings_from_dopt` 调用 |

**内部数值管线**（对 `materialize_int4_per_group_symmetric`，4 步）：
1. 预处理：`w = weight.detach().float()` + shape 校验（`in_f % group_size == 0`，`scale.shape == (out_f, in_f // group_size)`）。
2. scale 广播：`scale_exp = scale.float().repeat_interleave(group_size, dim=-1)` 到 `[out, in]`。
3. 量化 + 截断：`q = torch.round(w / scale_exp)`；signed → `clamp(-qmax, qmax)`（4-bit → `[-7, 7]`）；unsigned → `clamp(0, 2*qmax+1)`。
4. packing + 返回：`q.to(int8)`；`packed=True` → `pack_int4(q)` 返回 `[out, in//2]`；否则返回 `[out, in]`。返回 `(payload, scale.to(float32))`。

> **代码风格观察**：`materialize_int4_per_group_symmetric` 有 **6 个参数**（超 AGENTS.md "参数不超过 5 个" 原则）。建议聚合 `signed`/`nbits`/`packed` 三个量化格式参数为 `Int4Spec`/`QuantFormat` dataclass（三者共同描述"int4 编码格式"，语义内聚），让签名变成 4 参数。当前未拆。

> **docstring 遗留**：文件头 docstring 仍写 "int4 signed values stored 1/byte = unpack" + "本模块为 unpack+signed 占位实现，先跑通 Python 流程，CANN 对齐后续"，但实际 `pack_int4` 已实现 nibble-pack、`materialize_int4_per_group_symmetric` 支持 `packed=True`，早已非占位实现。docstring 待更新。

### `quant_linear.py` — 导出用 QuantLinear

定义 `QuantLinear(nn.Module)`，作为导出路径上 `nn.Linear` 的替身。其 `forward` 显式调用 `torch.ops.npu.npu_dynamic_quant`（激活量化）+ `torch.ops.npu.npu_quant_matmul`（权重 int4 matmul），使得 `torch.export` 产出的图里直接是这两个 custom op 节点（而非 `aten.mm`/`MatMul`），后续经 `_quant_matmul_converter.py` 映射到 `ge.QuantBatchMatmulV4`。

```python
class QuantLinear(nn.Module):
    def __init__(self, weight_q: Tensor, weight_scale: Tensor,
                 group_size: int, bias: Tensor | None = None,
                 act_quant: bool = True):  # 5 参数（不含 self）
        # act_quant=False 直接 raise NotImplementedError (W4A16 不支持)
        # weight_q: int8 [k//2, n] nibble-packed (预转置) → register_buffer
        # weight_scale: fp32 [k//g, n] (预转置) → register_buffer
        # bias is None → self.bias = None (forward 传 None, GE converter 侧塞 dummy)

    def forward(self, x: Tensor) -> Tensor:
        x_q, x_scale = torch.ops.npu.npu_dynamic_quant(x)          # 激活 s8 per-token
        out = torch.ops.npu.npu_quant_matmul(
            x_q, self.weight_q, self.weight_scale,
            pertoken_scale=x_scale,                      # per-token act scale (kwarg)
            bias=self.bias,                              # None when source Linear has no bias
            group_sizes=[1, 1, self.group_size],         # V4 3-element [m,n,k]; K-group at index [2]
            output_dtype=_OUT_DTYPE_CODE.get(x.dtype, 5),  # fp16=5, bf16=7, fp32=6
            transpose_x1=False,
            transpose_x2=False,                           # x2 已预转置到 [k//2, n], meta 取 N=x2.size(-1)
        )
        return out
```

模块级常量 `_OUT_DTYPE_CODE = {torch.float16: 5, torch.bfloat16: 7, torch.float32: 6}` 把 torch dtype 映射到 GE output_dtype 编码。`from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401` 为注册 `npu_quant_matmul` + `npu_dynamic_quant` 两个 custom op 的副作用，不直接用其符号。

**约定**（路径 C'，s8s4 对称，offset=0）：
- weight_q / weight_scale **预转置** 到 `[k//2, n]` / `[k//g, n]`（swap.py 物化后 `transpose(-1,-2)`）；`transpose_x2=False`（meta 取 `N = x2.size(-1)`，GE converter 硬编码 `transpose_x2=False`）。
- 激活：`npu_dynamic_quant` → int8 x1 + per-token fp32 x1_scale（作为 `pertoken_scale` kwarg 传入）。
- bias 可选 `[n]`（output dtype）；源 Linear 无 bias 时 forward 传 None，但 GE converter `_quant_matmul_converter.py` 的 dummy bias workaround 侧塞 dummy zeros[N]（CANN V4 `InferNDimWithBias` 不支持 bias=nullptr 的限制）。
- `group_sizes=[1, 1, g]`：V4 的 3-element [m,n,k] 48-bit 编码，K-group 在 index [2]。

### `swap.py` — 算子 IR 替换 + dopt 产物加载

在 `torch.export` 之前，**就地遍历** `model.named_modules()`，把目标 `nn.Linear` 替换成 `QuantLinear`（物化 int4 权重）、把目标 `nn.Embedding` 的 int8+scale 抽出来供 host gather（不替换成图模块）。还提供 dopt GPTQ 产物的加载器 + 量化元信息读取器 + 层策略解析器。是 `cann_lm_converter.py` 主路径（`swap_from_dopt`）唯一直接调用的模块。

| 函数/常量 | 签名 | 参数数 | 作用 |
|------|------|--------|------|
| `DEFAULT_SKIP` | 模块级常量 `("embed_tokens", "embed_tokens_per_layer")` | — | Linear swap 跳过名单。**`lm_head` 不在 skip 里**——会被量化成 QuantLinear W4A8 |
| `swap_embeddings_from_dopt` | `(sd: dict, model: Module) -> dict[str, tuple[Tensor, Tensor]]` | 2 | 遍历 `nn.Embedding`，从 sd 读 `{name}.weight` (fakequant) + `{name}.quant_op.weight_quantizer.s`，`materialize_int8_per_channel_symmetric` → `{wrapper_fqn: (int8, scale)}`。**不** swap 成图模块（embedding 外置给 host gather）。跳过无 scale 的（如 gemma4 per_layer 未量化） |
| `_resolve_parent` | `(model: Module, dotted_name: str) -> tuple[Module, str]` | 2 | 私有辅助：把 `"a.b.c"` 拆成 `(parent=a.b module, leaf="c")`，供 setattr 替换 |
| `load_dopt_state_dict` | `(dopt_path: str \| Path) -> dict` | 1 | 接受 `.safetensors` / `.pth` / 目录(含 `model.safetensors`)。规范化 `model.language_model.*` 前缀（VLM → text-only，cann_lm 匹配）。返回 `{name: tensor}` dict |
| `_read_dopt_quant_meta` | `(sd, name, out_f, in_f, group_size) -> tuple[int, float, Tensor \| None]` | 5 | 私有辅助：读 `{name}.quant_op.weight_quantizer.{bit, unsigned_quant, offset}`。bit=int64 标量，unsigned_quant=fp32 标量，offset=fp32 扁平 `[out*groups]`（reshape 到 `[out, in//group]`）。老版 dopt 无 buffer 时 fallback (4, 0.0, None) |
| `_STRATEGY_DEFAULT_PARAMS` | 模块级常量 dict | — | dopt 策略默认参数硬编码（`Quant_act_weight_eco` / `Quant_lm_head` 的 group_size=128/bit=4/blocksize=128/per_channel/weight_algo=omni_group_min_max）。converter venv 不装 dopt_lm 不能 import，硬编码；新增带 weight 量化的策略时需同步更新 |
| `_read_quant_param_map` | `(dopt_path: str \| Path \| None) -> dict[str, dict]` | 1 | 从 dopt 产物目录的 `dopt_config.json` 解析 `{层名: weight 侧量化参数 dict}`。后缀解析（`"Quant_act_weight_eco;128"` → strategy + gs_override）+ `_STRATEGY_DEFAULT_PARAMS` 默认值。无 dopt_config.json / 层未配置 / 策略无默认参数 → 不在该 dict |
| `_infer_group_size` | `(s_flat: Tensor, out_f: int, in_f: int, name: str) -> tuple[int, int]` | 4 | 从 dopt scale s_flat（扁平 `[out*groups]`）推断 (group_size, groups)。支持 pergroup/perchannel；拒绝 pertensor（numel==1）/ 非 1D / numel 不是 out_f 整数倍 |
| `_swap_linear_from_sd` | `(model, sd, skip=DEFAULT_SKIP, act_quant=True, param_map=None) -> dict` | 5 | **核心 swap 逻辑**（接收已加载 sd，避免重复 IO）。遍历 `nn.Linear`（非 skip）：读 w_gptq/s_flat；group_size 来源优先级 `param_map[name]` > `_infer_group_size`；reshape s_flat → scale2d；`_read_dopt_quant_meta` → bit/uq/offset；校验 uq==0（symmetric only，否则 ValueError）；校验 offset 全 0（否则 NotImplementedError）；`materialize_int4_per_group_symmetric(packed=True)` → 预转置 `[k//2,n]`/`[k//g,n]` → `QuantLinear` → setattr 替换。返回 quant-info dict |
| `swap_to_quant_linear_from_dopt` | `(model, dopt_path, skip=DEFAULT_SKIP, act_quant=True) -> dict` | 4 | **向后兼容薄 wrapper**：`load_dopt_state_dict` + `_read_quant_param_map` + 调 `_swap_linear_from_sd`。仅供场景 1 离线用户或单独 Linear swap 用；主进程场景 1/2 用 `swap_from_dopt` |
| `swap_from_dopt` | `(model, dopt_path, skip=DEFAULT_SKIP, act_quant=True) -> tuple[dict, dict]` | 4 | **主路径**（场景 1/2 共用）：一次 `load_dopt_state_dict` + `_read_quant_param_map` → `_swap_linear_from_sd`（Linear swap）+ `swap_embeddings_from_dopt`（Embedding 抽取）。避免 `swap_to_quant_linear_from_dopt` + `swap_embeddings_from_dopt` 二次 load IO。返回 `(quant_info, embed_quant)` |

> **代码风格观察**：`_swap_linear_from_sd` 函数体超 50 行（超 AGENTS.md 原则）。该函数承担太多职责（遍历 Linear + 读 scale + 读 meta + 校验 uq + 校验 offset + materialize + 预转置 + 实例化 + 替换 + 计数 + 构造返回 dict）。建议按单一职责拆成 `_validate_dopt_quant_meta(uq, offset)` + `_build_quant_info(...)` 等。当前未拆。

### `__init__.py` — 包门面

re-export 7 个公共符号（见 `__init__.py` 的 `__all__`）：

```python
from .materialize import materialize_int4_per_group_symmetric, materialize_int8_per_channel_symmetric
from .quant_linear import QuantLinear
from .swap import (
    load_dopt_state_dict,
    swap_to_quant_linear_from_dopt,
    swap_from_dopt,
    swap_embeddings_from_dopt,
)

__all__ = [
    "materialize_int4_per_group_symmetric",
    "materialize_int8_per_channel_symmetric",
    "QuantLinear",
    "swap_to_quant_linear_from_dopt",
    "swap_from_dopt",
    "swap_embeddings_from_dopt",
    "load_dopt_state_dict",
]
```

顶部 docstring 概述了量化两条入口路径与 dopt_lm 离线运行的关系。7 个符号中只有 **2 个** 被 `cann_lm_converter.py` 实际 import —— `swap_from_dopt`（主路径，从 `__init__` import）+ `run_auto_quant`（从 `auto_quant` 直接 import，场景 2）。其余 5 个对外暴露为测试/离线/将来直接调用预留。

### `auto_quant.py` — 一键式量化（方案 C）

converter `--quant-method gptq` 一键触发：检测 NPU/GPU/CPU → spawn `dopt_wrapper.py` subprocess（subprocess env，`--quant-device` 选设备）跑 GPTQ → 临时盘产物 → converter swap。与主进程 CPU stubs 环境隔离。

| 函数 | 签名 | 参数数 | 作用 |
|------|------|--------|------|
| `_build_layer_strategy` | `(model, model_type: str) -> dict` | 2 | 遍历 `model.named_modules` 赋量化策略（Linear→`Quant_act_weight_eco`，lm_head→`Quant_lm_head`，Embedding→`Quant_Embed_MinMax`）。cann_lm 与 transformers 层名一致，通用 |
| `generate_dopt_config` | `(artifact, quant_bits: str) -> str` | 2 | 场景 1 离线辅助：遍历 cann_lm `artifact.model` 生成 dopt_config，返回临时文件路径（`examples/run_pd4_cases.sh:gen_dopt_config` 依赖） |
| `_run_dopt_subprocess` | `(hf_path: str, quant_bits: str, device: str) -> str` | 3 | 方案 C 主体：spawn `dopt_wrapper.py` subprocess（subprocess env，`PYTHONPATH` 加 `quant/`、`DEVICE=<npu\|cuda\|cpu>`、`LD_LIBRARY_PATH` 去 stubs；`DOPT_PYTHON` 可覆盖系统 Python 路径）。退出码非 0 raise `RuntimeError`，保留临时产物供调试 |
| `_detect_device_type` | `() -> str` | 0 | 检测系统计算设备：优先读父进程 `DEVICE`（含 `cpu`），未设才 spawn 检测（`torch_npu`/`torch.cuda`）。关 `ASCEND_SLOG_PRINT_TO_STDOUT` 防 CANN 日志污染 stdout 解析；**无 NPU/GPU 时回落 `cpu`**（不再返回 None/报错） |
| `run_auto_quant` | `(hf_path: str, quant_bits: str) -> str` | 2 | **主入口**：检测 NPU/GPU/CPU → 方案 C (subprocess)；CPU 时 warn 慢（dopt_lm 支持 CPU float32，7B+ 极慢）。返回 dopt_out_dir 路径 |

### `dopt_wrapper.py` — dopt subprocess 主体

被 `auto_quant._run_dopt_subprocess` 通过 `python <path>/dopt_wrapper.py ...` 调用。subprocess 用系统 Python（有 torch_npu(NPU 时) + transformers），不走 cann_lm 的 venv。

| 函数 | 签名 | 参数数 | 作用 |
|------|------|--------|------|
| `_read_model_type` | `(model_path: str) -> str` | 1 | 从 `config.json` 读 `model_type` |
| `_build_layer_strategy` | `(model, model_type: str) -> dict` | 2 | 同 `auto_quant._build_layer_strategy`（subprocess 内独立实现，不 import auto_quant） |
| `parse_args` | `() -> Namespace` | 0 | `--model-path` / `--optimize-config` / `--output-dir` / `--hf-type-save` / `--quant-bits`（default=`w4a8`, choices=`["w4a8"]`，w4a16 不支持） |
| `main` | `() -> None` | 0 | 读 model_type → `model_define`（transformers fp16）→ 生成 dopt_config.json → `del model` + `gc.collect` → `weight_quant_pipline`（GPTQ）→ 产物到 `--output-dir` |

### `config.yaml` — dopt GPTQ 训练配置

是 `dopt_wrapper.py` 的 `--optimize-config` 参数文件，描述 GPTQ 校准的训练配置：校准语料是 `calibrate_sets/` 下的 jsonl（由 `download_dataset.py` 下载生成，`train_files` 指向具体文件，可替换/多选），采样 1024 条，截断长度 512，fp16 训练+存盘（CPU dispatch 自动回落 float32）。内容见 [量化范围与校准](#量化范围与校准)。

> `ptq_samples=1024`。`cutoff_len` 在当前版本只定义一次。`train_files` 可配多文件（YAML 列表）。

---

## dopt_lm 子包详解（已抽取）

> dopt_lm 的完整设计（文件树 / 入口 / GPTQ 调用链 / common_tools / modules / quantizer / opt_funcs/weight_quant_ptq / 代码风格 / 模型加载约束 / 可设置参数清单 / 历史调试档案）已抽取到 [dopt_design.md](./dopt_design.md)。
## 路径 C' — npu_quant_matmul → V4 算子规格与实现状态

> 当前采用：复用 CANN 原生 `npu::npu_quant_matmul`，FX→GE converter `_quant_matmul_converter.py` 映射到 `ge.QuantBatchMatmulV4`，经 V4→V3 TBE 融合落到预编译 kernel。
>
> **历史**：早期曾选方案 A（自研 `cann_lm::quant_matmul_s8s4`，用 fp32 scale 做 `q*scale` 反量化绕开 CANN V4 的 uint64/FP19 与 V4→V3 融合降级）。后回退到 C'：converter 侧 `_quant_matmul_converter.py` 用 V4 **full IR**（所有输入含 None placeholder）让 V4→V3 融合正确把 `x2_scale` 映射到 V3 的 `scale` 输入 + **dummy bias zeros[N] workaround** 绕开 `InferNDimWithBias` 不支持 bias=nullptr，使整数 int8/int4 s8s4 经 V4→V3 融合后落到预编译 kernel 可达。方案 A 的自研 kernel 规格与流水参考作为历史档案保留在本节末尾。

### 实现状态（已落地）

路径 C' 的算子定义 + converter 链路改造：

| # | 改造项 | 实现位置 | 状态 |
|---|--------|---------|------|
| 1 | `materialize_int4_per_group_symmetric(..., packed=True)` 产出 nibble-packed int8 `[n, k//2]` | `cann_lm/converter/quant/materialize.py` 的 `pack_int4` / `materialize_int4_per_group_symmetric` | ✅ |
| 2 | `QuantLinear` 权重/scale **预转置** 到 `[k//2, n]`/`[k//g, n]`，`transpose_x2=False` | `cann_lm/converter/quant/quant_linear.py` 的 `__init__`/`forward` + `swap.py` 的预转置 | ✅ |
| 3 | `group_sizes=[1,1,g]` 三元组（V4 48-bit 编码，K-group at index [2]） | `cann_lm/converter/quant/quant_linear.py` 的 `group_sizes` | ✅ |
| 4 | bias=None 时 forward 传 None，GE converter 侧塞 dummy zeros[N] | `quant_linear.py` 的 `forward` + `_quant_matmul_converter.py` 的 dummy bias workaround | ✅ |
| 5 | `npu::npu_quant_matmul`（CANN 原生，取代自研 `quant_matmul_s8s4`） | `cann_lm/converter/quant/quant_linear.py` 的 `forward` | ✅ |
| 6 | `_quant_matmul_converter.py` 映射到 `ge.QuantBatchMatmulV4`（V4 full IR + dummy bias） | `cann_lm/converter/export/_quant_matmul_converter.py` | ✅ |
| 7 | `reg_register.h` 注册 `QuantBatchMatmulV4` + `DynamicQuant` + `Bitcast`（`QuantMatmulS8S4` 方案 A 遗留仍注册，见下） | `examples/test_converter/export_llm_model/reg_register.h`、`examples/test_converter/export_and_load_epair/reg_register.h`、`examples/test_converter/export_llm_model/om_npu_verify/reg_register.h` | ✅ |
| 8 | custom op schema + fake-meta + CPU impl 自提供（converter venv 无 torch_npu） | `cann_lm/models/fused_ops/quant_matmul.py`（schema + fake-meta + CPU impl） | ✅ |

custom op 定义在 `cann_lm/models/fused_ops/quant_matmul.py`（schema + fake-meta + CPU PoC impl），GE 转换器 `_quant_matmul_converter.py` 把 `npu::npu_quant_matmul` 映射到 `ge.QuantBatchMatmulV4`。converter venv 不装 torch_npu，由 `_library.py` 自提供 schema+fake-meta（`_HAS_TORCH_NPU=False` 路径），torch.export 能 trace。

### 计算语义（s8s4 对称，offset=0）

记激活 `x1` `[b, m, k]`、权重 `x2` `[n, k]`（nn.Linear 自然朝向，物化后预转置到 `[k, n]`/`[k//2, n]`）、group_size `g`（`k%g==0`）：

```
y[b,m,n] = Σ_k ( x1_q[b,m,k] · x1_scale[b,m] ) · ( x2_q[n,k] · x2_scale[n, k//g] ) + bias[n]
```

- `x1_q` int8（per-token 动态对称，每行一个 scale）、`x2_q` int4（per-channel×per-group 对称，每 (n,group) 一个 scale）。
- `x1_scale[b,m]` 是 per-token（V4 的 `pertoken_scale` 输入），**可提到 k 求和之外**。
- `x2_scale[n, k//g]` 是 per-group（V4 的 `x2_scale`/V3 的 `scale` 输入），**在 k 求和之内**（每组 g 个元素共用一个，反量化时按组广播，无法提到求和之外）。

### 算子签名（`npu::npu_quant_matmul`）

| 方向 | 名称 | dtype | shape | 语义/来源 |
|---|---|---|---|---|
| in | `x1` | int8 | `[b, m, k]` | 来自 `npu_dynamic_quant` 的已量化激活 |
| in | `x2` | int4（packed int8） | `[k//2, n]`（**预转置**） | nn.Linear 权重物化后 int4 nibble-packed |
| in | `scale` | fp32 | `[k//g, n]`（**预转置**） | per-K-group × per-output-channel，来自 GPTQ（V4 的 `x2_scale`） |
| in（opt） | `pertoken_scale` | fp32 | `[b, m]` | per-token scale，来自 `npu_dynamic_quant`（V4 的 `x1_scale`） |
| in（opt） | `offset` | fp32 | `[k//g, n]` | per-group offset（V4 的 `x2_offset`）；当前校验必须全 0（symmetric only） |
| in（opt） | `bias` | 同输出 | `[n]` | 融合 add；None 时 GE converter 塞 dummy zeros[N] |
| attr | `group_sizes` | int[3] | — | V4 48-bit 编码 `[m, n, k]`，K-group at index [2]（`[1, 1, g]`） |
| attr | `output_dtype` | int | — | 5=fp16 / 7=bf16 / 6=fp32 |
| attr | `transpose_x1` | bool | — | 默认 False（激活恒不转） |
| attr | `transpose_x2` | bool | — | 默认 False（x2 已预转置到 `[k//2, n]`，meta 取 `N=x2.size(-1)`） |
| out | `y` | fp16/bf16 | `[b, m, n]` | — |

> **schema 全貌**（`quant_matmul.py` 的 schema 定义）：`npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset, Tensor? pertoken_scale, Tensor? bias, int? output_dtype, int? x1_dtype, int? x2_dtype, int? pertoken_scale_dtype, int? scale_dtype, int[]? group_sizes, Tensor? y_scale, bool transpose_x1=False, bool transpose_x2=False) -> (Tensor y)`。`x1_dtype`/`x2_dtype`/`*_scale_dtype` 用于 Bitcast（mxfp4/a8w4 浮点 dtype 分支，已删）。

### GE converter 关键技术（`_quant_matmul_converter.py`）

1. **V4 full IR**：`ge.QuantBatchMatmulV4(x1, x2, bias, x1_scale=pertoken_scale, x2_scale=scale, y_scale, x1_offset=None, x2_offset=offset, y_offset=None, x2_table=None, dtype, transpose_x1=False, transpose_x2=False, group_size)`。所有输入含 None placeholder，让 V4→V3 TBE 融合正确把 `x2_scale` 映射到 V3 的 `scale` 输入（见附录 A.2.2 索引映射）。
2. **dummy bias zeros[N] workaround**：bias=None 时塞 `ge.Const([0.0]*N, dtype=1)`。CANN V4 `InferNDimWithBias` 不支持 bias=nullptr（会从 x1 的 M 维推断 bias shape 并与 N 比较，报 "dimensions a(M) and b(N) must be same"）。dummy 值 0 不影响 matmul 结果。
3. **group_sizes 48-bit 编码**：`group_size = (group_m << 32) + (group_n << 16) + group_k`，校验三元素 ∈ [0, 65535]。`[1, 1, g]` → K 轴分组 g。
4. **删 a8w4/mxfp4 分支**：为去 torch_npu 依赖，删了 `float4_e2m1fn_x2`/`float8_e8m0fnu` 等 torch 无等价物的 dtype 常量分支（`x1_dtype`/`x2_dtype`/`*_scale_dtype` 的 Bitcast 路径保留但 converter 不主动用）。恢复 a8w4 需重新引入 torch_npu 依赖或建立 dtype 常量映射（见 `quant_matmul.py` 文件头注释）。
5. **transpose_x1/x2 硬编码 False**：meta 取 `N = x2.size(-1)`，故 x2 必须 swap.py 预转置到 `[k//2, n]` 使 `size(-1) = n = out_features`。

### CANN V4 限制与 workaround 清单

| 限制 | workaround | 位置 |
|------|-----------|------|
| `InferNDimWithBias` 不支持 bias=nullptr | 塞 dummy zeros[N]（值 0 不影响结果） | `_quant_matmul_converter.py` 的 dummy bias workaround |
| V4→V3 融合需 full IR 才正确映射 x2_scale | 用 `ge.QuantBatchMatmulV4` full IR（所有输入含 None placeholder） | `_quant_matmul_converter.py` 的 V4 full IR 构造 |
| meta 取 `N = x2.size(-1)`，忽略 transpose_x2 | swap.py 物化后预转置 weight_q/weight_scale 到 `[k//2, n]`/`[k//g, n]` | `swap.py` 的预转置 |
| 无 torch_npu 则无 schema/fake-meta | `_library.py` 自提供（`_HAS_TORCH_NPU=False` 路径）+ CPU PoC impl | `quant_matmul.py` 的 fake-meta + CPU impl |

### 历史档案：方案 A 自研 kernel 规格（已弃用，保留供参考）

> 以下为方案 A `cann_lm::quant_matmul_s8s4` 自研 kernel 的 I/O 约束与流水参考，**已弃用**（代码已切回 `npu::npu_quant_matmul`），仅作历史决策档案保留。若未来 CANN V4 整数 int8/int4 路径再遇瓶颈，可参考此规格重启自研。

**方案 A 与路径 C' 的关键差异**：

| 维度 | 方案 A（自研，已弃用） | 路径 C'（当前采用） |
|---|---|---|
| 算子 | `cann_lm::quant_matmul_s8s4` | `npu::npu_quant_matmul` |
| GE op | 自研 GE 算子 | `ge.QuantBatchMatmulV4`（CANN 原生） |
| weight 朝向 | 自然 `[n, k//2]`，不预转置 | 预转置 `[k//2, n]` |
| `transpose_x2` | True（kernel 内转置） | False（swap 已预转置） |
| `group_size` attr | 单 int | `group_sizes=[1,1,g]` 三元组 |
| bias | 真正可选（无 dummy） | None 时 GE converter 塞 dummy zeros[N] |
| scale 反量化 | fp32 `q*scale`（绕开 V4 uint64） | 走 CANN V4→V3 融合（依赖预编译 kernel） |

**方案 A kernel 内部流水参考**（自研时供实现方参考）：

权重侧 per-group scale 在 k 求和之内，故 kernel 须在 cube 累加前把 int4 权重按组反量化：

1. L1/L0B load 时 int4→int8 unpack，按组取 `x2_scale` 广播；
2. cube 做 **int8×int8 → int32** 累加（命中 W8A8 已验证的 int8 cube 路径，规避 FP19/uint64）；
3. fixpipe 阶段乘 `x1_scale`（per-row，可后乘）+ `bias` → cast fp16/bf16。

此流水让 s8s4 数值落在"int8 cube + fp32 scale 反量化"语义上，是绕开 CANN V4 失败点的关键。

---

## C++ 注册与产物

### reg_register.h（三文件均已完成注册）

| 文件路径 | 注册算子 | 状态 |
|---------|---------|------|
| `examples/test_converter/export_llm_model/reg_register.h` | 全量（含 `QuantMatmulS8S4` 方案 A 遗留 + `QuantBatchMatmulV3` + `QuantBatchMatmulV4` + `DynamicQuant` + `Bitcast` + 80+ 其他） | ✅ |
| `examples/test_converter/export_and_load_epair/reg_register.h` | 合并 + IR 校准版（73 个有 IR 定义的算子 + 8 个无 IR 自研算子含 `QuantMatmulS8S4`） | ✅ |
| `examples/test_converter/export_llm_model/om_npu_verify/reg_register.h` | 同上合并版 | ✅ |

> **路径勘误**：早期文档写 `examples/reg_register.h` / `examples/export_llm_model/reg_register.h`（均不存在），实际在 `examples/test_converter/` 下。`export_llm_model/reg_register.h` 是全量主文件（含所有 `REG_OP`）；`export_and_load_epair/` 与 `export_llm_model/om_npu_verify/` 是合并 + IR 校准版（以 `om_npu_verify` 版为基底合并）。

> **方案 A 遗留**：`QuantMatmulS8S4` 在三文件中**仍注册**（方案 A 自研算子，代码已切回 `npu::npu_quant_matmul` 不再使用）。注释标注为"8 个无 IR 的自研算子"之一。当前代码链路实际只用 `QuantBatchMatmulV4` + `DynamicQuant` + `Bitcast`（V4 converter 的 dtype Bitcast 路径）。清理 `QuantMatmulS8S4` 注册是低风险纯删除，但会破坏旧 epair（含 `cann_lm::quant_matmul_s8s4` 节点）的加载，故保留。

### GE converter 与权重落盘

- **GE converter**：`cann_lm/converter/export/_quant_matmul_converter.py`（无 `_s8s4` 后缀）把 `npu::npu_quant_matmul` 映射到 `ge.QuantBatchMatmulV4`（V4 full IR + dummy bias，详见 [路径 C' GE converter 关键技术](#ge-converter-关键技术_quant_matmul_converterpy)）。
- **权重落盘（无需改 pack）**：`epair_utils.pack_weights` 是**通用**的——序列化 state_dict 里任意 tensor（int8 weight_q + fp32 scale 自动进 bin，bf16→int16 view 已处理，weight_scale fp32→int64 经 `transdata_scale` 转换见 [weight_scale 落盘格式](#weight_scale-落盘格式fp32uint64-位模式转换)）。`weight_info.json` 按 name 记 offset/size。**量化权重无需改打包逻辑**（weight_scale 的 fp32→int64 转换在 `_pack_weights_to_dir` 识别 name 含 `weight_scale` 时调 `transdata_scale`）。
- **C++ ge_graph_builder**：`PreparePlaceholders` 通用喂权重常量到任意算子输入（`ReadTensorData`→Const 节点）→ `npu_quant_matmul` 的 `x2`/`scale`/`bias` 自动喂入。`ge_weight_loader` 支持 int8/uint8/fp32/int64 等 dtype（`ElementSizeForPt2Dtype` / `Pt2DtypeFromGeString`）。

### 最终产物（epair 格式）

- `prefill_ep_json_with_ascend_ir.json` / `decode_ep_json_with_ascend_ir.json`：含 `npu::npu_quant_matmul` 节点 + `npu::npu_dynamic_quant` 节点，`QuantLinear` 在 `nn_module_stack` 元信息里。weight_scale 的 `tensor_values` dtype 后处理改标 `DT_INT64`（`inline_weight_scale_dtype`）。
- `weights_*.bin`：int4 nibble-packed int8 + fp32 scale 的二进制权重（weight_scale 落盘为 int64 位模式 bytes）。
- `model_config.json`：含 `quant_info` 字段（`scheme`/`group_size`/`bit`/`blocksize`/`per_channel`/`weight_algo`/`act_quant`/`scope`/`n_quantized`/`n_skipped_no_dopt_key`/`dopt_path`/`weight_int4_encoding`/`op`/`dopt_unsigned_quant`/`offset_supported`，由 `swap_from_dopt`/`_swap_linear_from_sd` 返回）。
- `weight_info.json`：权重 offset/size 索引。
- host-embeddings 模式额外：`embed_tokens_q.bin` int8 + `embed_tokens_scale.bin` fp32，host 查表。

后续编译成 OM 是下游 `epair → OM` 编译器的事，不在 quant/ 范围。

---

## weight_scale 落盘格式：fp32→uint64 位模式转换

> 将 `QuantLinear.weight_scale`（GPTQ 产出的 per-group fp32 scale）按位掩码 `& 0x00000000FFFFE000` 转 uint64 位模式落盘。截断尾数低 13 位（精度 23→10 位）+ 截断高 32 位（值只用低 32 位）。集成位置：`materialize.transdata_scale` / `restore_scale` + `epair_utils.inline_weight_scale_dtype` + saver `_pack_weights_to_dir`。
>
> 最后更新：2026-08-25
>
> **注意**：本节的"方案 A/B"是 **weight_scale 落盘格式的两种集成方案**（方案 A 改 QuantLinear buffer / 方案 B 仅落盘转换），与[算子选型](#路径-c--npu_quant_matmul--v4-算子规格与实现状态)的"方案 A 自研 quant_matmul_s8s4"是不同概念维度。方案 B（仅落盘）当前采用。

### 背景：位掩码 `0x00000000FFFFE000` 的含义

```
掩码 0x00000000FFFFE000 (64 位)
     └──────┬──────┘└──────┬──────┘
       高 32 位=0      低 32 位=0xFFFFE000
                    └──┬──┘└──┬──┘
                     全 1    bit12-0=0
                   (bit31-13 保留)
```

| 位段 | 掩码位 | 结果 | float32 含义 |
|------|--------|------|-------------|
| bit 63-32 | 全 0 | 清零 | （值只用低 32 位，等效 uint32 范围） |
| bit 31 | 1 | 保留 | 符号位 |
| bit 30-23 | 全 1 | 保留 | 指数（8 位） |
| bit 22-13 | 全 1 | 保留 | 尾数高 10 位 |
| bit 12-0 | 全 0 | 清零 | **尾数低 13 位**（精度 23→10 位） |

→ 保留符号 + 指数 + 尾数高 10 位，截断尾数低 13 位。反量化还原后仍是合法 fp32，但精度降低（~7 位→~3 位十进制）。

### 两种集成方案

#### 方案 A：改 QuantLinear（buffer 存 int64）

`QuantLinear.weight_scale` buffer 直接存 int64，forward 时 `restore_scale`（int64→int32→fp32 位还原）再传 `npu_quant_matmul`。

```python
# materialize.py
def transdata_scale(scale_fp32):  # fp32 位模式 → int64 + 掩码
    bits = scale_fp32.contiguous().view(torch.int32).to(torch.int64)
    return bits & 0x00000000FFFFE000

def restore_scale(scale_packed):  # int64 → fp32 (尾数低 13 位为 0)
    return scale_packed.to(torch.int32).view(torch.float32)
```

```python
# quant_linear.py (方案 A)
self.register_buffer("weight_scale", weight_scale.to(torch.int64))  # int64
def forward(self, x):
    weight_scale_fp32 = restore_scale(self.weight_scale)  # int64 → fp32
    out = torch.ops.npu.npu_quant_matmul(x_q, self.weight_q, weight_scale_fp32, ...)
```

- **swap.py**：物化后调 `transdata_scale` 把 fp32→int64，存入 QuantLinear buffer。
- **torch.export trace**：`restore_scale` 生成 Cast(int64→int32) + view(int32→fp32) 图节点。
- **ep_json dtype**：`DT_INT64`（torch buffer 是 int64，torch.export 自动推断）。
- **saver**：直接打包 int64 tensor 的 bytes（无需特殊处理）。
- **C++**：按 ep_json 的 `DT_INT64` 读（`DecodeBytesAs<int64_t>`），等效 uint64。

#### 方案 B：仅落盘转换（buffer 保持 fp32，saver 打包时转 int64）✅ 当前采用

`QuantLinear.weight_scale` buffer 保持 fp32（计算路径不变，forward 直接传 `npu_quant_matmul`，无 Cast 图节点）。仅 saver 打包时识别 weight_scale 调 `transdata_scale` 转 int64；ep_json 后处理把 dtype 从 `DT_FLOAT` 改标 `DT_INT64`。

```python
# quant_linear.py (方案 B) — 不改
self.register_buffer("weight_scale", weight_scale.to(torch.float32))  # fp32
def forward(self, x):
    out = torch.ops.npu.npu_quant_matmul(x_q, self.weight_q, self.weight_scale, ...)
```

```python
# epair_utils.py: _pack_weights_to_dir — 识别 weight_scale 转 int64
for name in names:
    raw = state_dict[name].detach().cpu().contiguous()
    if raw.dtype == torch.bfloat16:
        raw = raw.view(torch.int16)
    elif "weight_scale" in name and raw.dtype == torch.float32:
        raw = transdata_scale(raw)  # fp32 → int64 (位模式 + 掩码, 落盘格式)
    data = raw.numpy().tobytes()
```

```python
# epair_utils.py: inline_weight_scale_dtype — ep_json 后处理改 dtype
def inline_weight_scale_dtype(ep_json_path):
    tv = doc["graph_module"]["graph"]["tensor_values"]
    for name, meta in tv.items():
        if "weight_scale" in name and meta.get("dtype") == 7:  # float32
            meta["dtype"] = 5  # int64 (kDtypeLong)
```

```python
# saver.py: save_epair_prefill_decode — 调后处理
inline_fused_attn_seq_constants(prefill_ep_json, seq_len)
inline_weight_scale_dtype(prefill_ep_json)  # ← 新增
```

### 方案对比

| 维度 | 方案 A（改 quant_linear）❌未采用 | 方案 B（仅落盘）✅ 当前采用 |
|------|---------------------------|-------------------|
| **QuantLinear buffer** | int64 | **fp32**（计算路径不变） |
| **forward** | `restore_scale`→fp32→op | **直接用 fp32**→op |
| **Cast 图节点（weight_scale 相关）** | 277 个（restore_scale） | **0 个** |
| **torch.export 图复杂度** | 多 Cast+view 节点 | 简洁（无额外节点） |
| **saver 打包** | 直存 int64 bytes | 识别 weight_scale 转 int64 |
| **ep_json dtype** | `DT_INT64`（torch buffer 推断） | `DT_INT64`（后处理改标） |
| **C++ 读取** | 按 `DT_INT64`（`DecodeBytesAs<int64_t>`） | 同左（等效 uint64） |
| **weights_0.bin 大小** | 1484.6 MB | 1484.6 MB（相同，int64 bytes） |
| **改动文件** | quant_linear.py + swap.py + materialize.py | materialize.py + epair_utils.py + saver.py（quant_linear/swap 不改） |

→ **方案 B 更优**：QuantLinear 不改（forward 路径不变，图简洁），仅落盘层做转换 + ep_json 后处理同步 dtype。计算与存储分离。

### 三层分离（方案 B）

```
计算层 (QuantLinear)       存储层 (saver)              标注层 (ep_json)
─────────────────         ─────────────              ──────────────
weight_scale: fp32    ──→ transdata_scale         ──→ dtype=DT_INT64
(forward 直接用)         (fp32→int64+掩码, 落盘)     (inline_weight_scale_dtype
                              ↓                       后处理改标)
                         weights_0.bin
                         (int64 bytes = uint64)
                              ↓
                         C++ ReadTensorData
                         (按 ep_json DT_INT64 读,
                          DecodeBytesAs<int64_t>,
                          等效 uint64 位模式)
```

### C++ 读取保证链

```
ep_json tensor_values: weight_scale dtype = "DT_INT64" (kDtypeLong=5)
    ↓ C++ Pt2DtypeFromGeString("DT_INT64") → pt2::kDtypeLong
    ↓ ElementSizeForPt2Dtype(kDtypeLong) → sizeof(int64_t) = 8 bytes/元素
    ↓ DecodeBytesAs<int64_t>(bytes, size) → 按 int64 解码
int64 bytes = uint64 bytes (8 字节小端序, 位模式相同)
    ↓ C++ 可 reinterpret_cast<uint64_t*> 解释
等效按 uint64 读 ✓
```

**torch 无 uint64 dtype**，用 int64 存储（位模式相同，位操作 `&` 结果一致）。`BuildTensorHolder` 还校验 `Product(dims) * element_size == byte_size`（`WEIGHT_SIZE_MISMATCH` 兮底），dtype 标错会直接报错。

### 验证结果（case6 gemma4 quant-method gptq, 2026-08-24）

| 检查项 | 结果 |
|--------|------|
| weight_q dtype | int8 × 277 层 ✓ |
| weight_scale dtype（ep_json） | int64 × 277 层 ✓（后处理改标成功） |
| size 匹配（ep_json dtype×shape = weight_info.json size） | ✓ 554 个 buffer 全匹配 |
| Cast 算子涉及 weight_scale | 0 个 ✓（方案 B 无 restore_scale） |
| weights_0.bin | 1484.6 MB（scale fp32→int64 翻倍：+71.2 MB） |
| 导出 | ✓ Done |

### 改动文件清单（方案 B）

| 文件 | 改动 |
|------|------|
| `materialize.py` | 加 `transdata_scale`（fp32→int64+掩码）+ `restore_scale`（int64→fp32，方案 A 遗留，方案 B 不调但保留） |
| `epair_utils.py` | `_pack_weights_to_dir` 识别 weight_scale 调 `transdata_scale` 转 int64 + 加 `inline_weight_scale_dtype` 后处理函数 |
| `saver.py` | `save_epair_prefill_decode` 调 `inline_weight_scale_dtype`（prefill + decode） |
| `quant_linear.py` | **不改**（weight_scale 保持 fp32） |
| `swap.py` | **不改**（物化后不调 transdata_scale） |

---

## 已知问题与遗留

1. **unsigned int4 不支持**：`_swap_linear_from_sd`（`swap.py` 的 unsigned_quant 校验）读 dopt 产物的 `unsigned_quant`，校验必须 = 0（symmetric signed `[-8, 7]`）。非 0（unsigned int4 `[0, 15]`）触及 CANN int4 signed/unsigned 编码约定，当前直接 `ValueError` 拒绝。dopt 默认 `Quant_act_weight_eco` 的 `unsigned_quant=False`，故主路径不触发。
2. **非对称 offset 不支持**：`_swap_linear_from_sd`（`swap.py` 的 offset 校验）读 dopt 产物的 `offset`，校验必须全 0（symmetric）。非零 offset（asymmetric）`NotImplementedError`——`npu::npu_quant_matmul` 当前校验 symmetric offset=0（op schema 有 `offset` 输入但 converter 不主动用）。dopt `weight_q_params_adapter_into_dopt` 把 offset 置零（`offset=torch.zeros_like(...)`），故主路径不触发。dopt `QuantizerFuncIntOffset` 反量化 `w_fp=(q-offset)*scale`；若未来支持非对称，需对齐 V4 的 `x2_offset` 语义 + 按 group `repeat_interleave` 扩维 + GE converter 把 offset 喂入 `x2_offset` 输入。
3. **CPU 跑 GPTQ 已支持但慢**：dopt_lm `SUPPORTED_DEVICE_TYPES` 含 `cpu`（`train.py`/`modelutils.py`，CPU 走 `dispatch_dtype=float32`）。`--quant-method gptq` 经 `auto_quant.run_auto_quant` 检测，无 NPU/GPU 时**自动回落 `cpu`**（不再报错），或显式 `--quant-device cpu`。**代价**：7B+ Hessian 矩阵数 GB，CPU 极慢（仅小模型/调试适用）；生产建议 NPU/GPU 或 `--quant-path` 消费离线产物。
4. **一键式量化（方案 C）已实现**：converter `--quant-method gptq`（`auto_quant.py` + `dopt_wrapper.py`：检测设备 NPU/GPU/CPU → spawn subprocess → 临时盘 → `swap_from_dopt`）。w4a8 ✅；w4a16 ❌ 不支持（`dopt_wrapper.parse_args` 的 `--quant-bits` choices=`["w4a8"]`）；临时盘清理 + 实测验证待做，详见 [一键式量化方案选型](#一键式量化方案选型)。
5. **dopt 函数超规**：`swap.py _swap_linear_from_sd` 函数体超 50 行（承担遍历+读 scale+读 meta+校验+materialize+预转置+实例化+替换+计数+构造 dict），`materialize.py materialize_int4_per_group_symmetric` 6 参数（超 5 参数原则）。两者建议按单一职责拆分 + `Int4Spec` dataclass 收敛，当前未拆。
6. **dopt gemma4 多模态约束**：dopt GPTQ 主循环按文本 `ForCausalLM` 结构访问 `model.model.layers`；`model_define` 会从多模态模型抽取并重挂语言模型。`QGemma4ScaledEmbedding` 已通过 `visible_type_dict` 的原始类名分支接入。
7. ~~**dopt_lm 拼写错误**~~ **已修复**：源码已用正确拼写 `ptq_algo_hessian` / `WeightQuantHessian`（`dopt_lm/opt_funcs/weight_quant_ptq/ptq_algo_hessian/`）。仅 `dopt_lm/weight_quant_ptq/ptq_algo_heissen/__pycache__/` 遗留旧编译产物（heissen 拼写），可删。
8. **materialize.py docstring 过时**：文件头 docstring 仍写 "int4 signed values stored 1/byte = unpack" + "本模块为 unpack+signed 占位实现，先跑通 Python 流程，CANN 对齐后续"，但实际 `pack_int4` 已实现 nibble-pack、`materialize_int4_per_group_symmetric` 支持 `packed=True`，早已非占位实现。docstring 待更新。
9. **reg_register.h 仍注册 QuantMatmulS8S4（方案 A 遗留）**：三文件 `reg_register.h` 仍注册 `QuantMatmulS8S4`（方案 A 自研算子，代码已切回 `npu::npu_quant_matmul` 不再使用），注释标注为"8 个无 IR 的自研算子"之一。清理是低风险纯删除，但会破坏旧 epair（含 `cann_lm::quant_matmul_s8s4` 节点）的加载，故保留。
10. **_quant_matmul_converter.py 删 a8w4/mxfp4 分支**：为去 torch_npu 依赖，删了 `float4_e2m1fn_x2`/`float8_e8m0fnu` 等 torch 无等价物的 dtype 常量分支（`x1_dtype`/`x2_dtype`/`*_scale_dtype` 的 Bitcast 路径保留但 converter 不主动用）。恢复 a8w4 需重新引入 torch_npu 依赖或建立 dtype 常量映射（见 `quant_matmul.py` 文件头注释）。

---

## 附录 A — CANN QuantBatchMatmulV3 / V4 op_graph 源码分析（历史档案）

> 源码位置：`cann/ops-nn/matmul/quant_batch_matmul_v3/op_graph/` 与 `cann/ops-nn/matmul/quant_batch_matmul_v4/op_graph/`
>
> 本附录记录对 CANN 官方 V3/V4 op_graph 层代码的逐文件分析，作为 converter 早期选型（路径 C'）与对齐的依据。**注意：方案 A 已绕开 V4 整数 int4 路径，本附录作为历史决策档案保留**，对当前实现不再直接相关。

### A.1 QuantBatchMatmulV3 op_graph 结构

```
quant_batch_matmul_v3/op_graph/
├── CMakeLists.txt                                      构建脚本
├── quant_batch_matmul_v3_proto.h                       算子原型注册
├── quant_batch_matmul_v3_op_graph.cpp                  OpSelectFormat 实现
├── quant_batch_matmul_v3_fallback.cpp                  Host 侧 fallback 执行
└── fusion_pass/
    ├── quant_batch_matmul_v3_transpose_fusion_pass.{h,cpp}  Transpose 融合 pass
    └── z_z_matmul_t_qbmmv3_fusion_pass.{h,cpp}             MatMul→QBMMV3 融合 pass
```

#### A.1.1 算子原型（quant_batch_matmul_v3_proto.h）

通过 `REG_OP(QuantBatchMatmulV3)` 注册，定义：

- **输入**：`x1`、`x2`（必选）；`scale`、`offset`、`bias`、`pertoken_scale`（可选）
  - x1/x2 类型：int8, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1
  - scale 类型：uint64, float32, int64, bfloat16, float8_e8m0
- **属性**：
  - `dtype`（必选 Int）：输出类型（0=float32, 1=float16, 2=int8, 3=int32, 27=bfloat16）
  - `transpose_x1`/`transpose_x2`（Bool，默认 false）
  - `group_size`（Int，默认 0）：48 位编码（k/n/m 各 16 位）
- **输出**：`y`（float16/int8/bfloat16/int32/float32）

proto 注释详细说明了各平台（Atlas Inference、A2/A3、Ascend 950）支持的 dtype 组合与 quant 模式表。

#### A.1.2 OpSelectFormat（quant_batch_matmul_v3_op_graph.cpp）

根据 `x2` 是否为常量输入，动态生成 `OpSelectFormat` 结果（JSON 描述的 dtype/format 组合表）：

- `listWeightNZ`：x2 为常量（常量折叠场景），weight 只保留 NZ 格式
- `listQbmmV3`：非常量场景，与原始算子信息库一致

`QbmmV3OpCheckHelper` 构造函数根据 `context->IsConstInput("x2")` 选择列表；`OpSelectFormat` 用 lambda 抽取每个张量的 dtype/format/unknownshapeFormat 列表，组装成 `TensorSpec`，用 `nlohmann::json` 序列化为字符串返回。

通过 `IMPL_OP(QuantBatchMatmulV3).OpSelectFormat(QbmmV3OpSelectFormat)` 注册到 optiling。

#### A.1.3 Host 侧 Fallback（quant_batch_matmul_v3_fallback.cpp）

当算子在设备侧无法直接执行时，通过 `IMPL_OP(QuantBatchMatmulV3).OpExecuteFunc(...)` 注册 Host 回退函数 `QuantBatchMatmulV3ExecuteFunc`，根据输入 dtype/shape 路由到 `aclnnQuantMatmulV5` 或 `aclnnQuantMatmulV4`：

**V5 路径**：当满足以下任一条件时调用 `aclnnQuantMatmulV5`：
- x1 dtype ∈ {FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1}（v5 输入类型）
- 或为 "A8W8GB" 场景：x1/x2=int8、y=bf16、pertoken_scale/scale=float32、bias 为 null 或 float，且 x1 与 pertoken_scale、x2 与 scale 的存储维度数一致

V5 调用还会读取 `group_size` 属性并传入，且补齐 yScale/x1Offset/yOffset 为 nullptr。

**V4 路径**：其余情况调用 `aclnnQuantMatmulV4`。

属性索引硬编码：`transpose_x1=1`、`transpose_x2=2`、`group_size=3`。

#### A.1.4 Transpose 融合 Pass（quant_batch_matmul_v3_transpose_fusion_pass）

**目标**：删除 `QuantBatchMatmulV3` 之前的 `Transpose`/`TransposeD`/`Reshape`（以及 Bitcast 模式下的同类节点），并翻转 `transpose_x1`/`transpose_x2` 属性。

两个 Pass 类：
- `QuantBatchMatmulV3TransposeFusionPass`：常规 pass
- `QuantBatchMatmulV3TransposeLimitFusionPass`：继承前者，在"禁止关闭融合规则"场景下兜底，允许在内轴 > 65535 而外轴 ≤ 65535 时强制融合

主要流程：
1. 版本检查（仅 GE compiler ≥ 9.1.0 才生效）
2. 前置检查：校验是 QBMMV3 节点、平台能力（supportL12btBf16、supportMmadS8S4）、dtype 合法性、shape 维度 ≥ 2、是否动态 shape、是否 Bitcast 模式
3. 查找待删节点：根据内轴/外轴与 65535 的关系、平台是否支持 l12bt-bf16、x2 是否 NZ，决定 isDelTransx1/isDelTransx2；取出 x1/x2 上的 Transpose 节点；进一步尝试取 scale 上的等价 Transpose/Reshape（对 mx 量化的 FLOAT8_E8M0 scale 做了特殊等价判定）
4. 执行融合：翻转 transpose 属性、绕过 Transpose 重连上游节点、上报融合、删节点

#### A.1.5 MatMul→QBMMV3 融合 Pass（z_z_matmul_t_qbmmv3_fusion_pass）

**目标**：把 `MatMul / MatMulV2 / MatMulV3 / BatchMatMul / BatchMatMulV2 / BatchMatMulV3` 转换成 `QuantBatchMatmulV3`，用一个常量 `scale=1.0`（`0x3F800000`）当作"伪量化"输入，从而复用 QBMMV3 的高性能实现。

模式与构造：`Patterns()` 为 2/3/4 输入构造 6 类共 18 个匹配 pattern。

准入条件：GE compiler 版本 ≥ 9.0.0；排除 hifloat8；Ascend950 平台还排除 `int8+int8→int32` 组合。

替换流程：读取原节点的转置属性映射到 QBMMV3 的 `transpose_x1/transpose_x2`；创建常量 scale（uint64 1.0，shape={1}）；调用 `QuantBatchMatmulV3(...)` 生成新节点；更新输入/输出 desc；复制其他属性。

### A.2 QuantBatchMatmulV4 op_graph 结构

```
quant_batch_matmul_v4/op_graph/
├── CMakeLists.txt                                      构建脚本
├── quant_batch_matmul_v4_proto.h                       算子原型注册
└── fusion_pass/
    ├── quant_batch_matmul_v4_to_v3_fusion_pass.{h,cpp}     V4 → V3 转换 pass
    └── quant_batch_matmul_v4_transpose_fusion_pass.{h,cpp}  Transpose 融合 pass
```

相比 V3 的 op_graph，V4 这里**只包含算子原型和两个融合 pass**，没有 `OpSelectFormat` 和 `fallback` 实现——因为 V4 在 fallback 时会被转成 V3。

#### A.2.1 算子原型（quant_batch_matmul_v4_proto.h）

通过 `REG_OP(QuantBatchMatmulV4)` 注册，注释定位为"antiquant + matmul 的融合算子，支持 A8W4 / MxA8W4 / A8W8 / A4W4"。

**输入（10 个，2 必选 + 8 可选）**：

| Index | 名称 | 必选 | 类型 |
|---|---|---|---|
| 0 | `x1` | 是 | float8_e5m2 / float8_e4m3fn / int8 / int4 |
| 1 | `x2` | 是 | float4_e2m1 / int4 / int8 |
| 2 | `bias` | 否 | bfloat16 / float16 / float32 |
| 3 | `x1_scale` | 否 | float16 / bfloat16 / float8_e8m0 / float32 |
| 4 | `x2_scale` | 否 | float16 / bfloat16 / float8_e8m0 / uint64 / float32 |
| 5 | `y_scale` | 否 | uint64 |
| 6 | `x1_offset` | 否 | float16 / bfloat16（not supported yet） |
| 7 | `x2_offset` | 否 | float16 / bfloat16 |
| 8 | `y_offset` | 否 | float16 / bfloat16 / float32（not supported yet） |
| 9 | `x2_table` | 否 | int8（not supported yet） |

**属性**：
- `dtype`（必选 Int）：输出类型，仅支持 1(float16) 和 27(bfloat16)
- `compute_type`（Int，默认 -1）：-1(auto) 或 4(f8f4)
- `transpose_x1`（Bool，默认 false）：注释说明"目前应当总是 false"
- `transpose_x2`（Bool，默认 false）：注释说明"目前应当总是 true"
- `group_size`（Int，默认 **-1**）：注意 V3 默认 0，V4 默认 -1

**与 V3 的关键差异**：V4 把 scale 拆成了 `x1_scale`/`x2_scale`/`y_scale`/`x1_offset`/`x2_offset`/`y_offset`/`x2_table` 共 8 个可选输入，对应"反量化 + matmul"融合语义。

#### A.2.2 V4→V3 转换 Pass（quant_batch_matmul_v4_to_v3_fusion_pass）

**目标**：当 V4 在当前平台/场景下无需独立实现时，把它重写为 V3。

**索引映射**（V4_V3_INDEX_MAP）：

| V4 输入 | V4 idx | → | V3 输入 | V3 idx |
|---|---|---|---|---|
| x1 | 0 | → | x1 | 0 |
| x2 | 1 | → | x2 | 1 |
| bias | 2 | → | bias | 4 |
| x1_scale | 3 | → | pertoken_scale | 5 |
| x2_scale | 4 | → | scale | 2 |
| x2_offset | 7 | → | offset | 3 |

V4 的 `y_scale / x1_offset / y_offset / x2_table` 在转 V3 时被丢弃。

**Patterns**：构造 9 个 pattern（optionalInputNumber 从 0 到 8），匹配 V4 节点不同可选输入数目的实际形态。

**MeetRequirements**——决定是否需要把 V4 转 V3。两个"无需转换"的条件，满足任一则 `return false`（不做融合）：

1. **V4 原生支持（MxA8W4）**：
   ```
   x1 == FLOAT8_E4M3FN && (x2 == FLOAT4_E2M1 || x2 == FLOAT) && (y == BF16 || y == FLOAT16)
   ```
   即 MxA8W4/MxA8W8FP 场景，V4 本身能跑，不需要降级。

2. **A8W8 G-B 量化**：当 x1/x2 scale 都存在，且
   ```
   x1=int8, x2=int8, x1_scale=float32, x2_scale=float32, y=bf16
   && x1 与 x1_scale 维度数相同 && x2 与 x2_scale 维度数相同
   ```
   即 perblock/pergroup 量化，V4 本身能跑。

否则返回 true，进入 `Replacement` 把 V4 重写为 V3。

**Replacement**：用 `EsGraphBuilder` 建新子图；遍历 `V4_V3_INDEX_MAP`，对每个 V4 实际存在的输入在 V3 侧按序创建输入端口；读取 V4 属性（`group_size` 为 -1 时改为 0）；调用 `es::QuantBatchMatmulV3(...)` 生成新节点；按映射表回写各输入 desc 和输出 desc。

#### A.2.3 V4 Transpose 融合 Pass（quant_batch_matmul_v4_transpose_fusion_pass）

**目标**：删除 `QuantBatchMatmulV4` 的 `x2` 与 `x2_scale` 输入前的 `Transpose` / `Reshape`（含动态 shape 的 `Pack-Gather-Shape` 链），并翻转 `transpose_x2` 属性。

与 V3 transpose pass 的差异：
- V3 pass 同时处理 x1/x2/scale/pertoken_scale 四个候选；V4 pass **只处理 x2 和 x2_scale**
- V4 pass 仅支持 `x1=FLOAT8_E4M3FN`、`x2 ∈ {FLOAT4_E2M1, FLOAT}`、`y ∈ {BF16, FLOAT16}`（即 MxA8W4 场景）
- V4 pass 额外处理了**动态 shape 下的 `Reshape(Shape→Gather→Pack)` 链**

### A.3 V4 与 V3 的整体关系

| 维度 | V3 op_graph | V4 op_graph |
|---|---|---|
| 算子原型 | scale/offset/pertoken_scale 3 量化输入 | 拆成 x1_scale/x2_scale/y_scale/x1_offset/x2_offset/y_offset/x2_table 8 个可选 |
| OpSelectFormat | 有（按 weight 是否常量分两表） | **无**（直接用算子信息库） |
| Fallback | 有，路由到 aclnnQuantMatmulV4/V5 | **无**（V4 通过 V4→V3 pass 走 V3 的 fallback） |
| Transpose 融合 | x1/x2/x2_scale/pertoken_scale 四候选，Bitcast 模式，平台 l12bt/mmad 能力判定 | 仅 x2/x2_scale 两候选，仅 MxA8W4 dtype，额外处理 Pack-Gather-Shape 动态链 |
| MatMul→V3 融合 | 有（z_z_matmul_t_qbmmv3，scale=1.0） | 无对应 |
| V4→V3 转换 | 不适用 | 有（核心 pass，按 dtype/场景决定是否降级） |

**整体设计**：V4 是更新的"反量化+matmul 融合"算子，原生支持 MxA8W4/A8W8GB 等新场景；当场景超出当前平台 V4 实现能力时，由 `QuantBatchMatmulV4ToV3FusionPass` 降级到 V3 走老路径。

**两个 V4 pass 的互斥性**：`QuantBatchMatmulV4TransposeFusionPass` 的合法 dtype 恰好等于 to_v3 pass 里 `isV4Supported` 的条件。满足 MxA8W4 → transpose 融合 pass 生效，to_v3 pass 不生效；不满足 MxA8W4 且非 A8W8GB → to_v3 pass 生效，transpose 融合 pass 不生效。

### A.4 MxA8W4 与 A8W8GB 场景详解

这两个名字是量化场景的缩写，规则是 `<scale-mode>A<xbits>W<ybits>`：

- **A8 / W4 / W8**：Activation 是 8-bit、Weight 是 4-bit 或 8-bit。这里的"8/4"指的是数据位宽，不限定 int 还是 float——float8_e4m3fn 算 8-bit，float4_e2m1 算 4-bit。
- **Mx 前缀**：Microscaling 量化模式，使用低比特浮点（float8/float4）做激活和权重、用 float8_e8m0 做 scale，scale 按 k 轴 32 元素一组（对应 V3 proto 里 `mx quant` 表，group_size `[1,1,32]`）。
- **GB 后缀**：Group-Block，即 pergroup-perblock 或 perblock-perblock 量化（V3 proto 里 group_size `[1,128,128]` / `[128,128,128]`）。区别于 perchannel/pertensor（scale 是 1D），GB 模式 scale 的形状与数据同维数。

| 场景 | x1 | x2 | scale | 输出 |
|---|---|---|---|---|
| MxA8W4 | float8_e4m3fn（8b float） | float4_e2m1（4b float） | x1_scale/x2_scale 一般是 float8_e8m0 | bf16/fp16 |
| A8W8GB | int8 | int8 | float32（与 x1/x2 同维数） | bf16 |

**A8W8GB 的判定关键**：scale 的维数必须和数据相同。
- perchannel/pertensor 时 scale 是 1D（如 `(n,)`），与 x2 的 2D 不匹配 → 不算 GB
- perblock/pergroup 时 scale 是 2D（如 `(ceil(k/128), ceil(n/128))`），与 x2 同是 2D → 命中 GB

**完整路径**：

| 场景 | 图算子 | 最终 opapi |
|---|---|---|
| MxA8W4 / A8W8GB | **保留 V4**（不融合） | V4 自己的 tiling/kernel |
| 其他场景 | V4 → V3（to_v3 pass） | V3 fallback → `aclnnQuantMatmulV4`（多数情况） |

### A.5 V3 的 perchannel 量化支持

V3 支持 perchannel 量化。proto 里明确列了两种含 perchannel 的量化模式（针对 Ascend 950）：

命名规则是 `<x1-side>-<x2-side>`，对应 proto 里 x1 用 `pertoken_scale`、x2 用 `scale`：
- `pertensor-perchannel`：x1=pertensor（或 null），x2=perchannel
- `pertoken-perchannel`：x1=pertoken，x2=perchannel

**perchannel 时 scale 的形状**（`quant_batch_matmul_v3_proto.h:59-62`，外部依赖 CANN 9.2.0）：
```
When the quant mode of x2 is perchannel or pertensor, the shape is 1D (t,),
with t equal to 1 or n, where n is the same as that of x2.
```
- `t = n`：真正的 perchannel（每通道一个 scale）
- `t = 1`：实际是 pertensor

**与 V4 的关系**：V4 的 A8W8GB 判定（`quant_batch_matmul_v4_to_v3_fusion_pass.cpp:127-128`，外部依赖 CANN 9.2.0）要求 scale 维数与数据相同，这**排除了 perchannel**——因为 perchannel 时 scale 是 1D，而 x2 是 2D/3D，维数不等。所以 A8W8 的 perchannel/pertensor 场景下，V4 会被转成 V3 来执行。

| A8W8 量化模式 | scale 形状 | V4 处理方式 |
|---|---|---|
| pertensor | 1D, t=1 | V4→V3 转换（走 V3 路径） |
| perchannel | 1D, t=n | V4→V3 转换（走 V3 路径） |
| perblock/pergroup | 2D/3D（同维数） | 保留 V4（A8W8GB） |

### A.6 pertoken / pergroup / perchannel / perblock 概念

用一句话概括：**这几个"per-X"就是回答"多少个元素共用一个 scale（缩放因子）"的问题。颗粒度从粗到细排列。**

假设激活矩阵 x1 形状是 `(m, k)`（m 行 k 列），权重矩阵 x2 形状是 `(n, k)`。

```
pertensor  —— 整个矩阵共用 1 个 scale
┌────────────────────┐
│  全部用 S0          │   → 1 个 scale
└────────────────────┘

pertoken   —— 每一行（每个 token）一个 scale
┌────────────────────┐
│  S0 S0 S0 S0 S0 S0 S0 S0 │  ← 第 0 行用 S0
│  S1 S1 S1 S1 S1 S1 S1 S1 │  ← 第 1 行用 S1
│  S2 S2 S2 S2 S2 S2 S2 S2 │
│  S3 S3 S3 S3 S3 S3 S3 S3 │
└────────────────────┘   → m 个 scale

perchannel —— 每一列（每个 output channel）一个 scale
┌────────────────────┐
│ S0 S1 S2 S3 S4 S5 S6 S7│  ← 第 0 列用 S0, 第 1 列用 S1...
│ S0 S1 S2 S3 S4 S5 S6 S7│
│ S0 S1 S2 S3 S4 S5 S6 S7│
│ S0 S1 S2 S3 S4 S5 S6 S7│
└────────────────────┘   → k 个 scale（或 n 个，取决于谁被当"channel"）

pergroup   —— 沿某一轴每连续 group_size（如 128）个元素一个 scale
┌────────────────────┐
│ S0 S0 S0 S0 S1 S1 S1 S1│  ← 每 4 个一组（示意，实际是 128）
│ S2 S2 S2 S2 S3 S3 S3 S3│
│ S4 S4 S4 S4 S5 S5 S5 S5│
│ S6 S6 S6 S6 S7 S7 S7 S7│
└────────────────────┘   → m × ceil(k/group) 个 scale

perblock   —— 切成 2D 小块，每块一个 scale
┌───────┬───────┐
│  S0   │  S1   │   ← 左上 4×4 块用 S0, 右上 4×4 块用 S1
├───────┼───────┤
│  S2   │  S3   │
└───────┴───────┘   → ceil(m/gs) × ceil(k/gs) 个 scale
```

| 模式 | 谁共用一个 scale | scale 数量 | 颗粒度 |
|---|---|---|---|
| **pertensor** | 整个张量 | 1 | 最粗 |
| **perchannel** | 同一通道（一整列/一整行） | n 或 k | 中 |
| **pertoken** | 同一行（同一个 token） | m | 中 |
| **perblock** | 一个 2D 小块（如 128×128） | ceil(m/128)×ceil(k/128) | 较细 |
| **pergroup** | 沿某轴连续一维的一段（如每 128 个） | m×ceil(k/128) | 最细 |

**在 V3 里谁用哪个**：
- **x1（激活）** 用 `pertoken_scale`：可选 pertoken / pertensor / perblock / pergroup
- **x2（权重）** 用 `scale`：可选 perchannel / pertensor / perblock

V3 权重侧没有"pergroup"模式——最细就是 **perblock**（按 K 和 N 两个轴都分组）。如果只想沿 K 轴分组、N 轴不分组，V3 没有现成模式支持。

### A.7 pergroup 的判定机制

pergroup 是通过**两个东西组合判定**的：scale 张量的**形状** + 算子的 **group_size 属性**。没有任何显式的"quant_mode"枚举字段。

#### 判定依据 1：pertoken_scale 的形状

对比 x1（激活）侧四种模式的 pertoken_scale 形状（`quant_batch_matmul_v3_proto.h:77-83`，外部依赖 CANN 9.2.0）：

| 模式 | pertoken_scale 形状 | 关键特征 |
|---|---|---|
| pertensor | 1D `(1,)` | 只有一个值 |
| pertoken | 1D `(m,)` | 每行一个值 |
| **pergroup** | 3D `(batch, m, ceil(k/128))` | **m 是完整的，k 被分组** |
| perblock | 3D `(batch, ceil(m/128), ceil(k/128))` | m 和 k 都被分组 |

pergroup 和 perblock 的**唯一形状差异**：

```
pergroup :  (batch, m          , ceil(k/128))   ← m 没除 128
perblock :  (batch, ceil(m/128), ceil(k/128))   ← m 也除了 128
```

#### 判定依据 2：group_size 属性

`group_size` 是一个 48 位编码的 int 属性（`quant_batch_matmul_v3_proto.h:93-105`，外部依赖 CANN 9.2.0）：

```
bits  0-15  → group_size_k   （沿 K 轴分组大小）
bits 16-31  → group_size_n   （沿 N 轴分组大小）
bits 32-47  → group_size_m   （沿 M 轴分组大小）
```

pergroup vs perblock 由 **group_size_m** 决定：

| group_size_m | 含义 | x1 模式 |
|---|---|---|
| **1** | M 轴不分组（每组 1 个 m → 等价于不分组） | **pergroup** |
| **128** | M 轴每 128 个一组 | perblock |

支持表（`:231-243`）也印证：

```
pergroup-perblock : [group_size_m, group_size_n, group_size_k] = [1,   128, 128]
perblock-perblock : [group_size_m, group_size_n, group_size_k] = [128, 128, 128]
```

两者 x2 侧都是 perblock（n=128, k=128），**唯一区别就是 group_size_m 是 1 还是 128**。

#### 判定依据 3：dtype 限制

pergroup-perblock 模式还要求特定 dtype（`:212-218`）：

```
| x1                       | x2                       | pertoken_scale | scale  |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32        | float32 |
| hifloat8                  | hifloat8                 | float32        | float32 |
```

#### 总结：pergroup 是怎么被"看出来"的

当图编译器看到一个 V3 节点时，它通过下面这套组合判断 x1 是不是 pergroup：

```
pertoken_scale 是 3D && 第 2 维 == m（完整，没除 128）&& 第 3 维 == ceil(k/128)
    && group_size 属性解出 group_size_m == 1
    && dtype ∈ {float8_e4m3fn, float8_e5m2, hifloat8} + float32 scale
        → 判定为 pergroup
```

没有任何一个字段直接写着"pergroup"，全靠**形状结构 + group_size 编码 + dtype** 三者联合推断。

### A.8 W4A8 pergroup + pertoken 场景的可行性分析

如果权重使用 W4A8 的 pergroup 量化、激活使用 pertoken 量化，需要澄清以下关键点：

#### 问题 1："pergroup for weight" 在 V3 里不是独立模式

V3 对 x2（weight）侧只支持三种量化模式：
- pertensor：scale 1D `(1,)`
- perchannel：scale 1D `(n,)`
- perblock：scale `(batch, ceil(k/128), ceil(n/128))`

而"pergroup"在 V3 里是 **x1（activation）侧**的模式。"weight pergroup（沿 K 轴每 128 个一组）"最接近的是 x2 的 **perblock**（但 perblock 同时按 K 和 N 分组）。V3 不支持只按 K 分组、不按 N 分组的 x2 量化。

#### 问题 2：int8/int4（W4A8）的 dtype 组合不被支持

V3 和 V4 的支持表中，x1 和 x2 的 dtype 必须**成对相同**：

| 平台 | int8 行 | int4 行 |
|---|---|---|
| A2/A3 | int8/int8 | int4/int4 |
| 950 | int8/int8 | 无 int4 |

**没有 int8/int4 这一行**。V4 虽然注册了 `x1=DT_INT8` 和 `x2=DT_INT4`，但 proto 注释里 A8W4 场景的 x2 是 `float4_e2m1`（float4），不是 int4；int4 只在 A4W4（int4/int4）里出现。

#### 可行的替代方案

| 方案 | x1 dtype | x2 dtype | 量化模式 | 平台 | 说明 |
|---|---|---|---|---|---|
| A. MxA8W4（V4 原生） | float8_e4m3fn | float4_e2m1 | Mx（float8_e8m0 scale） | 950 | x1_scale/x2_scale 是 float8_e8m0，group_size [1,1,32] |
| B. A4W4 pertoken-perchannel（V3） | int4 | int4 | pertoken + perchannel | A2/A3 | x1 必须也是 int4，scale 1D |
| C. int8/int4 pergroup | — | — | — | — | **当前 V3/V4 均不支持** |

> **注意**：本项目早期曾选路径 C'（V4 整数 int4 per-group），实测 CANN 9.2.0 预编译 kernel 目录含 `QuantBatchMatmulV4_ND_ND_int8_int4_{bf16,fp16}`（tilingKey=24），即 V4 runtime kernel 覆盖了整数 int8/int4 组合。proto 的 REG_OP dtype 列表与 runtime kernel 的覆盖范围不完全一致，proto 只列了 mxfp 分支但 runtime kernel 覆盖整数 dtype。但经实测 eager 路径 `npu_quant_matmul` int8/int4 结果错（详见附录 B），最终转方案 A 自研算子。

---

## 附录 B — 历史调试与路径选型档案（已抽取）

> 附录 B 内容（路径 C' 选型 / V4 NPU 验证 / OM 编译调试 / 工具链 / 环境约束）已抽取到 [dopt_design.md](./dopt_design.md) "附录 — 历史调试与路径选型档案"。

## 文档关联

| 文档 | 关联内容 |
|------|---------|
| [cli_design.md](../cli/cli_design.md) | CLI 入口 `--quant-path` / `--quant-method` / `--quant-bits` 参数 / `convert()` 程序化入口 |
| [export_design.md](../export/export_design.md) | epair 产物布局 / host_weights 机制 / C++ `GeWeightLoader` / GE 算子注册两层 |
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题（`swap_embeddings_from_dopt` + `materialize_int8_per_channel_symmetric`） |
| [dopt_design.md](./dopt_design.md) | dopt_lm vendored GPTQ 框架（文件树 / 入口 / GPTQ 调用链 / 可设置参数清单 / 历史调试档案） |
| [converter_design.md](../converter_design.md) | 跨模块调用链总览 / 模型差异总览 |
| [reference_qwen3_5_export.md](../reference_qwen3_5_export.md) | qwen3.5 导出专题（参考） |
