# 量化接入 Converter 设计文档

> 将 HuggingFace 模型经 GPTQ 权重量化后导出为含 s8s4 量化 matmul 算子的 epair 格式
>
> 量化方案：**W4A8 s8s4**（激活 s8 per-token 动态对称 + 权重 s4 per-group 对称）
>
> 算子实现：**方案 A — 自研 `cann_lm::quant_matmul_s8s4`（已落地）**。早期路径 C'（复用 CANN `QuantBatchMatmulV4` 整数 int8/int4）经实测不可达 s8s4 正确结果（V4 要求 uint64 deqScale，且 int8/int4 在 V4→V3 融合 pass 中降级、无预编译 kernel），故转为自研算子，用 **fp32 scale** 做 `q*scale` 反量化绕开 CANN V4 失败点。详见下文 [方案 A 算子规格与实现状态](#方案-a--自研-quant_matmul_s8s4-算子规格与实现状态)。
>
> 最后更新：2026-08-15

## 目录

- [概述](#概述)
- [quant 包架构](#quant-包架构)
- [量化策略](#量化策略)
- [端到端流程](#端到端流程)
- [quant 顶层文件详解](#quant-顶层文件详解)
- [dopt_lm 子包详解](./dopt_design.md)（已抽取到独立文档）
- [方案 A — 自研 quant_matmul_s8s4 算子规格与实现状态](#方案-a--自研-quant_matmul_s8s4-算子规格与实现状态)
- [C++ 注册与产物](#c-注册与产物)
- [已知问题与遗留](#已知问题与遗留)
- [附录 A — CANN QuantBatchMatmulV3 / V4 op_graph 源码分析（历史档案）](#附录-a--cann-quantbatchmatmulv3--v4-op_graph-源码分析历史档案)
- [附录 B — 历史调试与路径选型档案](./dopt_design.md)（已抽取到 dopt_design.md "附录"）

---

## 概述

> 量化接入的 CLI 参数（`--quant-path` / `--quant-method` / `--quant-bits`）见 [cli_design.md](../cli/cli_design.md)；量化 epair 的产物布局、host_weights 机制、C++ 加载侧见 [export_design.md](../export/export_design.md)。embedding 量化的 W8 host gather 子主题见 [embedding_quant_design.md](./embedding_quant_design.md)。

`cann_lm/converter/quant/` 下内置两块：

1. **dopt_lm（vendored GPTQ 框架）**：在 **NPU/GPU** 上对 HF 模型做 GPTQ 权重量化，产出 per-layer `group_scale`（per-channel × per-group 对称，offset=0）+ GPTQ 修正后的 fp16 权重，落盘 `trained_quant_weight.pth` 或 `model.safetensors`。dopt 自身只做 fakequant（权重留在 fp16），不物化真 int4，不接 converter 导出。**dopt 的完整设计（文件树 / 入口 / GPTQ 调用链 / 可设置参数清单 / 历史调试档案）见 [dopt_design.md](./dopt_design.md)**。
2. **converter 侧量化接入（`materialize.py` / `quant_linear.py` / `swap.py` / `__init__.py`）**：在 **CPU venv + torchair stubs** 上加载 dopt 产物，物化成真 int4 payload + fp32 scale，把 `nn.Linear` 替换成 `QuantLinear`（forward 调 `cann_lm::quant_matmul_s8s4`），然后 `torch.export` 产出含 `quant_matmul_s8s4` + `DynamicQuant` 节点的 epair。

### 能力边界

- **输入**：HF 模型目录 + 校准数据（`quant/calibration.json`）+ dopt GPTQ 产物（`quant_output/model.safetensors` 或 `.pth`）。
- **输出**：epair 模型目录，其 `*_ep_json_with_ascend_ir.json` 含 `quant_matmul_s8s4` + `DynamicQuant` IR 节点，权重 bin 含 nibble-packed int4（int8 存储）+ fp32 scale，`model_config.json` 含 `quant_info` 字段。
- **量化范围**：全部 `nn.Linear`（attn q/k/v/o_proj + mlp gate/up/down_proj + gemma4 的 `per_layer_input_gate` / `per_layer_projection` / `per_layer_model_projection` + `lm_head`），**仅 `embed_tokens` / `embed_tokens_per_layer` 不走 Linear swap**——它们是 `nn.Embedding`，由 `swap_embeddings_from_dopt` 走 W8 host gather（converter 自动检测 dopt 产物含 embed scale 即做）。
- **不包含**：CPU-only 环境跑 GPTQ（GPTQ 需 NPU/GPU + 校准，CPU 跑不了，见 [已知问题](#已知问题与遗留)）；GPTQ 与导出的内存链路一体化（仍两步磁盘桥接，见 [端到端流程](#端到端流程)）。

---

## 量化工具链分工设计

量化参数众多（层策略 / group_size / bit / 算法参数 percdamp/act_order 等），全部暴露到 converter CLI 会让用户难以驾驭且易传错。工具链按 **dopt 分步精细调整 + CLI 一键导出** 分工：

### 设计原则

1. **dopt 分步使用（精细调整）**：dopt 作为独立 CLI 工具（[dopt_design.md](./dopt_design.md) "使用方法"），用户可调整所有量化参数——层策略（`dopt_config.json` 的 `quant_strategy`）、校准（`config.yaml` 的 `ptq_samples`/`cutoff_len`）、group_size（`"name;group_size"` 语法）。产出 `quant_output/` GPTQ 产物。
2. **CLI 一键导出（少量重要参数）**：converter CLI 只暴露 **两个互斥** 量化入口 + 一个格式参数，精细参数不暴露（走 dopt 分步）：
   - `--quant-path <dopt产物>`：场景 1，dopt 已 GPTQ，converter 消费产物
   - `--quant-method gptq`：场景 2，converter 自动 spawn dopt subprocess 跑 GPTQ（已实现 `auto_quant.py`，见 [一键式量化方案选型](#一键式量化方案选型)）
   - `--quant-bits w4a8|w4a16`：格式/位宽（默认 `w4a8`，`--quant-method` 时生效，传给 dopt_config 的 strategy）
3. **converter 自动推断**：场景 1 的 `group_size` 从 dopt 产物 `group_scale` shape 自动反推（`swap.py`），`embed` 量化自动检测（dopt 产物含 `embed_tokens.quant_op...s` 即做），无需用户重复传参。

### 两场景对比

| | 场景 1: `--quant-path` | 场景 2: `--quant-method gptq` |
|---|---|---|
| 输入 | dopt GPTQ 产物（`quant_output/`） | HF 原始权重 |
| 谁跑 dopt | 用户手动跑 dopt | converter 自动 spawn dopt subprocess |
| 量化算法 | GPTQ（dopt，精度高） | GPTQ（dopt，精度高） |
| 格式 | 从产物推断 | `--quant-bits` 配（w4a8/w4a16，默认 w4a8） |
| group_size | 自动推断（从 dopt 产物） | dopt_config 配（默认 128） |
| embed 量化 | 自动检测 | dopt_config 配（Quant_Embed_MinMax） |
| 环境 | converter CPU stubs | converter CPU stubs + subprocess NPU |
| 状态 | ✅ 已实现 | ⏳ 待实现（方案 C） |
| 精细调整 | dopt 分步（`dopt_config.json` 层策略等） | dopt 分步（或 `--quant-bits` 简配） |

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
├── __init__.py              (47 行)  包门面，re-export 6 个公共符号
├── materialize.py           (162 行) 权重物化纯函数：fp → (int4 payload, fp32 scale)
├── quant_linear.py          (81 行)  导出用 QuantLinear nn.Module，forward 发射 quant_matmul_s8s4 IR
├── swap.py                  (226 行) 算子 IR 替换 + dopt 产物加载 + 量化元信息读取
├── config.yaml              (10 行)  dopt GPTQ 校准训练配置（dopt_lm.opt_main 的 --optimize-config）
├── calibration.json         (557KB)  GPTQ 校准语料（dopt 内部消费，不经 4 个 .py 文件）
├── dopt_config.json         (34KB)   dopt 层策略配置（dopt 内部 --dopt-config）
├── dopt_lm/                 (28 文件) vendored GPTQ 框架（独立 CLI 运行）
└── quant_output/            (dopt 产物) model.safetensors + config.json + config.yaml
```

### 依赖关系

```
__init__.py (门面)
    │  re-export
    ├── materialize.py      ← 叶子，仅 import torch
    ├── quant_linear.py     ← import cann_lm.models.fused_ops.ops (注册副作用)
    └── swap.py             ← 相对 import materialize + quant_linear
                                ↑
                                │ 唯一外部消费者
                cann_lm_converter.py:90, 105
                    from cann_lm.converter.quant import (
                        swap_to_quant_linear_from_dopt,     # 主 swap 路径
                        swap_embeddings_from_dopt,          # embedding host gather
                        load_dopt_state_dict,               # dopt 产物加载
                    )
```

- **无循环依赖**：`materialize ← swap → quant_linear` 的有向无环图；`materialize` 与 `quant_linear` 互不依赖。
- **`__all__` 暴露 6 个符号**（`__init__.py:38-47`）：其中只有 3 个（`swap_to_quant_linear_from_dopt` / `swap_embeddings_from_dopt` / `load_dopt_state_dict`）被 `cann_lm_converter.py` 实际 import；其余 3 个（`materialize_int4_with_scale` / `materialize_int8_per_channel` / `QuantLinear`）经 `swap.py` 内部相对 import 使用，对外暴露为测试/将来直接调用预留。

### I/O 定位

| 文件 | 谁读 | 怎么读 |
|------|------|--------|
| `config.yaml` | `dopt_lm.opt_main`（via `--optimize-config`，见 `scripts/quant_export.sh`） | dopt 内部，**不经 4 个 .py 文件** |
| `calibration.json` | dopt GPTQ 校准（相对 CWD = `quant/`） | dopt 内部，**不经 4 个 .py 文件** |
| `dopt_config.json` | `dopt_lm.opt_main`（via `--dopt-config`） | dopt 内部，**不经 4 个 .py 文件** |
| `quant_output/model.safetensors` | `swap.load_dopt_state_dict(quant_path)`（`swap.py:97-122`，用 `safetensors.safe_open`） | **swap.py 唯一读的 I/O** |

**结论**：4 个 Python 文件里，**只有 `swap.py` 的 `load_dopt_state_dict` 做 I/O**（读 safetensors/.pth），其他 3 个文件完全无 I/O、无状态、纯计算。`config.yaml` / `calibration.json` / `dopt_config.json` 全是 dopt GPTQ 阶段的输入，与 converter 这 4 个文件只间接相关（经 dopt 产物 `model.safetensors` 影响 converter）。

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
- `dyn_s8` = 激活动态 per-token int8（signed）—— 对应 W4A8 的 A8 侧，运行时算 per-token scale 传给 `quant_matmul_s8s4` 的 `x1_scale`。
- `omni_group_min_max` = LSQ 风格可学习 `quant_alpha` + sigmoid 自适应 scale 的 per-group 权重量化器（`dopt_lm/quantizer/qmin_max.py:OmniGroupMinMaxGridWeight`）。

### 量化范围与校准

- **范围**：全部 `nn.Linear`（`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/per_layer_input_gate/per_layer_projection/per_layer_model_projection` + `lm_head`）。
- **lm_head 量化**（重要）：`swap.DEFAULT_SKIP = ("embed_tokens", "embed_tokens_per_layer")`（`swap.py:33`）**不含 `lm_head`** → `lm_head` 作为大 NPU-side Linear 按 W4 QuantLinear 消费。converter 部署形态为 host-embed + NPU-lm_head（`tie_word_embeddings` 必须解 tie：embed 在 host、lm_head 在 NPU，不同计算域无法共享权重）。
- **embed_tokens / embed_tokens_per_layer**：是 `nn.Embedding`（非 Linear，Linear swap 不触及），由 `swap_embeddings_from_dopt` 消费 W8 host gather——converter 自动检测 dopt 产物含 embed scale 即做（embed 外置，图内无 embed_tokens 节点，`--use-embeddings` 默认 True）。
- **校准数据**：`cann_lm/converter/quant/calibration.json`（557KB）。
- **`config.yaml` 实际内容**（`quant/config.yaml`，10 行）：

  ```yaml
  dataset:
    train_files:
      - "calibration.json"
    ptq_samples: 1024
  cutoff_len: 512

  extra_training_config:
    fp16: true   # GPTQ 跑 fp16 + 存盘 fp16，避免 dopt 默认 fp32 导致产物翻倍

  cutoff_len: 512
  ```

  > **注意**：`cutoff_len: 512` 在文件里**重复定义两次**（行 5 和行 10，YAML 后者覆盖前者）。`ptq_samples=1024`（非历史文档误记的 128）。

- **GPTQ 跑在 HF 模型**：dopt 现成依赖 `transformers.AutoModelForCausalLM` + `accelerate` dispatch，不改 dopt；量化参数按层名 port 到 cann_lm 模型（两者命名一致：`model.layers.N.self_attn.q_proj` 等）。

### dopt 实测量化范围（gemma4-E4B，346 个 quant 模块）

`dopt_config.json`（796 行）配置：`embed_tokens` → `Quant_Embed_MinMax`（bit=8 int8 per-channel，host gather）、`lm_head` → `Quant_lm_head`（bit=4）、其余 Linear → `Quant_act_weight_eco`（默认 group 128）。dopt 产物里 `lm_head.weight` 与 `model.embed_tokens.weight` 已**实质 untie**（max diff ≈0.01 = GPTQ 修正量，lm_head 是 GPTQ 修正版给 W4、embed 是原始版给 W8）。

- **bit 分布**：bit=4 共 344 个（attn q/k/v/o_proj + mlp gate/up/down_proj + PLE per_layer_input_gate/per_layer_projection/per_layer_model_projection + lm_head），bit=8 共 2 个（embed_tokens + embed_tokens_per_layer）。
- **gemma4 KV 共享**：`k_proj`/`v_proj` 只 0..23（24 层）有独立权重，后 18 层（24..41）复用前层 KV 无独立 k/v_proj——模型结构决定，非漏量化。

---

## 端到端流程

### 两步流程（现状，已实现，磁盘桥接）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OFFLINE (NPU, 系统 Python 3.11.4 + 真 torch_npu) — Step A: dopt GPTQ   │
│                                                                         │
│  输入:                                                                   │
│   - HF 模型 (--hf-path)                                                  │
│   - cann_lm/converter/quant/config.yaml      ← --optimize-config         │
│       (dataset.train_files=calibration.json, ptq_samples=1024,            │
│        cutoff_len=512, fp16=true)                                       │
│   - cann_lm/converter/quant/calibration.json ← 校准语料 (相对 CWD)        │
│   - cann_lm/converter/quant/dopt_config.json ← --dopt-config (层策略)    │
│       (layer_strategy: Linear→Quant_act_weight_eco, group_size=128,        │
│        Embedding→Quant_Embed_MinMax)                                    │
│                                                                         │
│  执行: python -m dopt_lm.opt_main --model-path ... --hf_type_save        │
│                                                                         │
│  产物 (quant_output/):                                                   │
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
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ 磁盘桥接
┌─────────────────────────────────────────────────────────────────────────┐
│  CONVERTER (CPU venv, torchair stubs) — Step B-E                         │
│                                                                         │
│  cann_lm_converter.py:                                                  │
│   B. load_model(hf_path) → nn.Module (HF 原始, fp16)                     │
│   C+D. swap_to_quant_linear_from_dopt(model, quant_path=quant_output,    │
│                                      group_size=128, act_quant=True):  │
│      a. load_dopt_state_dict(quant_output) → sd: dict                    │
│      b. for each nn.Linear (非 skip):                                   │
│         - sd[name.weight] (GPTQ fp) + sd[...weight_quantizer.s] (scale) │
│         - _read_dopt_quant_meta → bit/uq/offset (校验 uq==0, offset 全 0)│
│         - materialize_int4_with_scale(w, scale2d, gs, packed=True)       │
│           → (weight_q int8 [n,k//2] nibble-packed, weight_scale fp32)   │
│         - QuantLinear(weight_q, weight_scale, gs, bias, act_quant=True) │
│         - setattr 替换 nn.Linear                                        │
│      c. return quant_info dict → artifact.quant_info                    │
│   (embed 自动检测, embed_quant 非空时):                                  │
│      swap_embeddings_from_dopt(sd, model) → embed_quant dict            │
│      → saver 经 wrapper.set_quant_embeddings() 注入图入口                │
│   E. save_epair_prefill_decode(...)                                      │
│      → torch.export(model) → ep_json (含 quant_matmul_s8s4 +            │
│         DynamicQuant IR) + weights_*.bin (int4 packed + fp32 scale)    │
│         + model_config.json (含 quant_info) + weight_info.json         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  最终产物 (epair 格式, output-dir/)                                      │
│   - prefill_ep_json_with_ascend_ir.json  (含 cann_lm::quant_matmul_s8s4 │
│     节点 + npu::npu_dynamic_quant 节点, QuantLinear 在 nn_module_stack) │
│   - decode_ep_json_with_ascend_ir.json                                  │
│   - weights_*.bin  (int4 nibble-packed int8 + fp32 scale buffer)        │
│   - model_config.json (quant_info: scheme/group_size/n_quantized/...)   │
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
| 环境冲突 | ⚠️ 单进程但仍需 NPU | ❌ converter 进程需同时 CPU stubs(export) + NPU(GPTQ)，矛盾 | ✅ 规避（subprocess NPU，主进程 CPU stubs） |
| 改动量 | ❌ 最大（dopt 适配三模型 + vendored diff） | ✅ 小（三处改动点） | ✅ 小（converter 加 subprocess 调用） |
| dopt 代码改动 | ❌ 大（get_llama/llama_sequential/visible_type_dict） | ⚠️ 中（加 ptq_quant_produce_state_dict） | ✅ 无（复用 dopt CLI） |
| dopt 全局状态风险 | ❌ 高（进程内 args 全局单例/重入） | ❌ 高（进程内） | ✅ 隔离（subprocess 独立） |
| 真一步化 | ✅ 是 | ✅ 是（进程内） | ❌ 否（自动化两步，converter 触发 subprocess + swap） |
| 维护成本 | ❌ 高（跟进 dopt vendored diff） | ⚠️ 中（dopt 加函数维护） | ✅ 低（dopt 不改） |
| 7B+ 适用 | ✅ 好（1×） | ❌ 差（3-4×） | ⚠️ 中（2× 分进程） |

#### 选定方案 C：subprocess 调 dopt

**理由**：
1. **守住设计原则**：dopt 用 transformers 源码量化，零改动（方案 A 改 dopt 适配 cann_lm 违背此原则）
2. **规避环境冲突**：converter 主进程保持 CPU stubs（export 段），subprocess 用 NPU 环境（GPTQ 段），两套环境隔离（方案 B 环境冲突无解）
3. **dopt 全局状态隔离**：subprocess 独立，规避 `ptq_loop.py` 的 `args` 全局单例/重入风险
4. **维护成本低**：dopt 不改，跟进 transformers 上游即可

**代价**：非真一步化（subprocess 跑 dopt 本质仍是两步，只是用户视角一键）；产物传输（临时盘 IO 或内存管道）

#### 方案 C 设计

converter `--quant-method gptq` 一键触发：主进程 spawn `python -m dopt_lm.opt_main` subprocess（NPU 环境）跑 GPTQ → 产物到临时盘 → converter swap + export。实现位于 `cann_lm/converter/quant/auto_quant.py`（`generate_dopt_config` + `run_auto_quant`）。

```
converter 主进程 (CPU stubs)                      subprocess (NPU 环境)
─────────────────────────────                     ─────────────────────────
convert(--quant-method gptq)                      python -m dopt_lm.opt_main
  ├─ generate_dopt_config(artifact, quant_bits)    --model-path <hf> --dopt-config <临时cfg>
  │   遍历 cann_lm 模型生成 dopt_config             --optimize-config config.yaml
  ├─ run_auto_quant → spawn subprocess ───────→     --output-dir <临时盘> --hf_type_save
  │   (系统 Python + NPU env + cwd=quant/)          ├─ weight_quant_loop (GPTQ, NPU)
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
# converter 自动: generate_dopt_config → spawn dopt subprocess (NPU GPTQ) → 临时盘 → swap → export
# (--quant-method gptq 与 --quant-path 互斥; --quant-bits 配格式)
```

**两阶段实现**：
1. **阶段 1（临时盘，已实现）**：subprocess 产物到临时盘（`tempfile`），converter 读临时盘 swap。简单，复用 `--quant-path` 逻辑。代价：临时盘 IO。
2. **阶段 2（内存管道，待实现）**：subprocess 产物到内存管道（pickle/mmap/共享内存），converter 读管道 swap。省 IO 但复杂。7B+ 大模型受益。

**实现状态**（`auto_quant.py`）：
1. **subprocess 环境** ✅：`PYTHONPATH` 加 `quant/` + `DEVICE=npu` + `LD_LIBRARY_PATH` 去 stubs；`DOPT_PYTHON` 环境变量可覆盖系统 Python 路径
2. **calibration 路径** ✅：subprocess `cwd=cann_lm/converter/quant/`（config.yaml 的 `train_files` 相对 CWD）
3. **dopt_config 生成** ✅：`generate_dopt_config` 遍历 cann_lm 模型自动生成（按 `quant_bits` 映射 strategy），不依赖 `quant/dopt_config.json`
4. **subprocess 失败处理** ✅：退出码非 0 raise `RuntimeError`，保留临时产物路径供调试
5. **w4a16 strategy** ⏳ 待实现：无 preset W4A16 Linear strategy，当前 `NotImplementedError`
6. **临时盘清理** ⏳ 待实现：当前不清理（保留供调试），7B+ 大文件需 `shutil.rmtree`
7. **实测验证** ⏳ 待做：cann_lm 层名 vs HF 匹配、subprocess 环境跑通、calibration 匹配

> `--quant-path`（消费已存在的 dopt 磁盘产物，CPU-only 机或用户已手动跑 dopt）保留作兜底，与 `--quant-method gptq`（自动 spawn dopt）互斥。`--quant-bits` 仅 `--quant-method` 时生效。
### `materialize.py`（162 行）— 权重物化纯函数

纯函数模块，把 fp16/bf16/fp32 的 `nn.Linear` 权重 `[out, in]` 物化成量化 payload + per-group/per-channel scale。**无状态、无副作用、无外部 I/O**。是整个量化链路的"数值核心"。

| 函数 | 行 | 签名 | 参数数 | 作用 |
|------|----|------|--------|------|
| `pack_int4` | 30-44 | `(q: Tensor) -> Tensor` | 1 | Nibble-pack signed int4（存为 int8）成 `[n, k//2]` int8；byte[i] = int4(k=2i+1)<<4 \| int4(k=2i)。私有辅助，**不在 `__all__`** |
| ~~`materialize_int4`~~ | ~~47-87~~ | ~~`(weight, group_size=128, signed=True, nbits=4, packed=False)`~~ | ~~5~~ | **已删除**（min-max 路径移除；converter 只走 dopt GPTQ via `materialize_int4_with_scale`） |
| `materialize_int4_with_scale` | 90-129 | `(weight, scale, group_size, signed=True, nbits=4, packed=False) -> (Tensor, Tensor)` | **6** | 用 dopt 预算好的 per-group scale（GPTQ-optimized）量化，scale 原样返回。**主路径**，被 `swap_to_quant_linear_from_dopt` 调用 |
| `materialize_int8_per_channel` | 132-162 | `(weight, scale, signed=True) -> (Tensor, Tensor)` | 3 | 8-bit per-row 对称量化（匹配 dopt `Quant_Embed_MinMax`），产出 (int8 `[vocab,hidden]`, fp32 scale `[vocab]`)。被 `swap_embeddings_from_dopt` 调用 |

**内部数值管线**（4 步，对 `materialize_int4` / `materialize_int4_with_scale` 同构）：
1. 预处理：`w = weight.detach().float()` + shape 校验（`in_f % group_size == 0`，`scale.shape == (out_f, in_f // group_size)`）。
2. scale 计算/广播：`materialize_int4` 用 `wg.abs().amax(-1)/qmax`；`materialize_int4_with_scale` 直接用入参 scale；都 `repeat_interleave(group_size)` 到 `[out, in]`。
3. 量化 + 截断：`q = torch.round(w / scale_exp)`；signed → `clamp(-qmax, qmax)`（4-bit → `[-7, 7]`）；unsigned → `clamp(0, 2*qmax+1)`。
4. packing + 返回：`q.to(int8)`；`packed=True` → `pack_int4(q)` 返回 `[out, in//2]`；否则返回 `[out, in]`。返回 `(payload, scale.to(float32))`。

> **代码风格观察**：`materialize_int4_with_scale` 有 **6 个参数**（超 AGENTS.md "参数不超过 5 个" 原则）。建议聚合 `signed`/`nbits`/`packed` 三个量化格式参数为 `Int4Spec`/`QuantFormat` dataclass（三者共同描述"int4 编码格式"，语义内聚），让签名变成 4 参数。当前未拆。

### `quant_linear.py`（81 行）— 导出用 QuantLinear

定义 `QuantLinear(nn.Module)`，作为导出路径上 `nn.Linear` 的替身。其 `forward` 显式调用 `torch.ops.npu.npu_dynamic_quant`（激活量化）+ `torch.ops.cann_lm.quant_matmul_s8s4`（权重 int4 matmul），使得 `torch.export` 产出的图里直接是这两个 custom op 节点（而非 `aten.mm`/`MatMul`），后续经 `_quant_matmul_s8s4_converter.py` 映射到方案 A 自研 GE 算子。

```python
class QuantLinear(nn.Module):
    def __init__(self, weight_q: Tensor, weight_scale: Tensor,
                 group_size: int, bias: Tensor | None = None,
                 act_quant: bool = True):  # 5 参数（不含 self）
        # act_quant=False 直接 raise NotImplementedError (W4A16 不支持)
        # weight_q: int8 [n, k//2] nibble-packed → register_buffer
        # weight_scale: fp32 [n, k//g] → register_buffer
        # bias is None → self.bias = None (无 dummy-zeros workaround)

    def forward(self, x: Tensor) -> Tensor:
        x_q, x_scale = torch.ops.npu.npu_dynamic_quant(x)          # 激活 s8 per-token
        out = torch.ops.cann_lm.quant_matmul_s8s4(
            x_q, x_scale, self.weight_q, self.weight_scale,
            bias=self.bias,                                          # None when source Linear has no bias
            group_size=self.group_size,                              # 单 int attr (非 V4 的 [m,n,k] 48-bit pack)
            output_dtype=_OUT_DTYPE_CODE.get(x.dtype, 5),           # fp16=5, bf16=7, fp32=6
            transpose_x1=False,
            transpose_x2=True,                                       # x2 自然朝向, kernel 内转置
        )
        return out
```

模块级常量 `_OUT_DTYPE_CODE = {torch.float16: 5, torch.bfloat16: 7, torch.float32: 6}` 把 torch dtype 映射到 GE output_dtype 编码。`from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401`（行 36）为注册 `quant_matmul_s8s4` + `npu_dynamic_quant` 两个 custom op 的副作用，不直接用其符号。

**约定**（方案 A，s8s4 对称，offset=0）：
- weight_q / weight_scale 在 **自然** nn.Linear 朝向 `[n, k]` / `[n, k//g]`（**无预转置**）；`transpose_x2=True`（kernel 内部转置喂 cube）。
- 激活：`npu_dynamic_quant` → int8 x1 + per-token fp32 x1_scale。
- bias 可选 `[n]`（output dtype）；源 Linear 无 bias 时为 None（**无 dummy-zeros workaround**——方案 A 的 op bias 真正可选）。

### `swap.py`（226 行）— 算子 IR 替换 + dopt 产物加载

在 `torch.export` 之前，**就地遍历** `model.named_modules()`，把目标 `nn.Linear` 替换成 `QuantLinear`（物化 int4 权重）、把目标 `nn.Embedding` 的 int8+scale 抽出来供 host gather（不替换成图模块）。还提供 dopt GPTQ 产物的加载器 + 量化元信息读取器。是 `cann_lm_converter.py` 唯一直接调用的模块。

| 函数 | 行 | 签名 | 参数数 | 作用 |
|------|----|------|--------|------|
| `swap_embeddings_from_dopt` | 36-58 | `(sd: dict, model: Module) -> dict[str, tuple[Tensor, Tensor]]` | 2 | 遍历 `nn.Embedding`，从 sd 读 `{name}.weight` (fakequant) + `{name}.quant_op.weight_quantizer.s`，`materialize_int8_per_channel` → `{wrapper_fqn: (int8, scale)}`。**不** swap 成图模块（embedding 外置给 host gather）。跳过无 scale 的（如 gemma4 per_layer 未量化） |
| `_resolve_parent` | 61-67 | `(model: Module, dotted_name: str) -> tuple[Module, str]` | 2 | 私有辅助：把 `"a.b.c"` 拆成 `(parent=a.b module, leaf="c")`，供 setattr 替换 |
| ~~`swap_to_quant_linear`~~ | ~~70-94~~ | ~~`(model, group_size=128, skip=DEFAULT_SKIP, act_quant=True)`~~ | ~~4~~ | **已删除**（min-max 快速路径移除；dopt 只支持 GPTQ，min-max 精度差无实用价值） |
| `load_dopt_state_dict` | 97-122 | `(dopt_path: str \| Path) -> dict` | 1 | 接受 `.safetensors` / `.pth` / 目录(含 `model.safetensors`)。返回 `{name: tensor}` dict |
| `_read_dopt_quant_meta` | 125-148 | `(sd, name, out_f, in_f, group_size) -> tuple[int, float, Tensor \| None]` | 5 | 私有辅助：读 `{name}.quant_op.weight_quantizer.{bit, unsigned_quant, offset}`。bit=int64 标量，unsigned_quant=fp32 标量，offset=fp32 扁平 `[out*groups]`（reshape 到 `[out, in//group]`）。老版 dopt 无 buffer 时 fallback (4, 0.0, None) |
| `swap_to_quant_linear_from_dopt` | 151-226 | `(model, dopt_path, skip=DEFAULT_SKIP, act_quant=True) -> dict` | 5 | **主路径**（流程 C+D）：① `load_dopt_state_dict` → sd；② 遍历 `nn.Linear`（非 skip）：读 wkey/skey，跳过缺 key 的；从 scale shape 自动推断 group_size；reshape s_flat → scale2d；`_read_dopt_quant_meta` → bit/uq/offset；校验 uq==0（symmetric only，否则 ValueError）；校验 offset 全 0（否则 NotImplementedError：方案 A 不支持非对称）；`materialize_int4_with_scale(packed=True)` → `QuantLinear` → setattr 替换；③ 返回 quant-info dict（scheme/group_size/scope/counts/op/encoding/dopt_bit/uq/offset_supported）→ 被 `cann_lm_converter.py` 写进 `model_config.json` 的 `quant_info` 字段 |

模块级常量 `DEFAULT_SKIP = ("embed_tokens", "embed_tokens_per_layer")`（行 33）。**注意 `lm_head` 不在 skip 里**——会被量化成 QuantLinear W4A8（行 30 注释明确）。

> **代码风格观察**：`swap_to_quant_linear_from_dopt` 函数体超 50 行（超 AGENTS.md 原则）。该函数承担太多职责（加载 sd + 遍历 Linear + 读 scale + 读 meta + 校验 uq + 校验 offset + materialize + 实例化 + 替换 + 计数 + 构造返回 dict），按单一职责拆成：
> - `swap_to_quant_linear_from_sd(model, sd, ...)`（收内存 dict，做核心 swap）
> - `swap_to_quant_linear_from_dopt(model, dopt_path, ...)`（前者 load 后转发）
> - `_validate_dopt_quant_meta(uq, offset)`（校验 uq==0 / offset 全 0）
> - `_build_quant_info(...)`（构造返回 dict）
>
> 当前未拆。

### `__init__.py`（47 行）— 包门面

re-export 6 个公共符号（`__init__.py:38-47` 的 `__all__`）：

```python
from .materialize import materialize_int4_with_scale, materialize_int8_per_channel
from .quant_linear import QuantLinear
from .swap import (
    load_dopt_state_dict,
    swap_to_quant_linear_from_dopt,
    swap_embeddings_from_dopt,
)

__all__ = [
    "materialize_int4_with_scale",
    "materialize_int8_per_channel",
    "QuantLinear",
    "swap_to_quant_linear_from_dopt",
    "swap_embeddings_from_dopt",
    "load_dopt_state_dict",
]
```

顶部 docstring（行 15-27）概述了量化两条入口路径与 dopt_lm 离线运行的关系。6 个符号中只有 **3 个**（`swap_to_quant_linear_from_dopt` / `swap_embeddings_from_dopt` / `load_dopt_state_dict`）被 `cann_lm_converter.py` 实际 import；其余 3 个经 `swap.py` 内部相对 import 使用，对外暴露为测试/将来直接调用预留。

### `config.yaml`（10 行）— dopt GPTQ 训练配置

**不是** 4 个 Python 文件的输入！它是 `dopt_lm.opt_main` 的 `--optimize-config` 参数文件，描述 GPTQ 校准的训练配置：校准语料是同目录的 `calibration.json`，采样 1024 条，截断长度 512，fp16 训练+存盘。内容见 [量化范围与校准](#量化范围与校准)。

---

## dopt_lm 子包详解（已抽取）

> dopt_lm 的完整设计（文件树 / 入口 / GPTQ 调用链 / common_tools / modules / quantizer / opt_funcs/weight_quant_ptq / 代码风格 / 模型加载约束 / 可设置参数清单 / 历史调试档案）已抽取到 [dopt_design.md](./dopt_design.md)。
## 方案 A — 自研 quant_matmul_s8s4 算子规格与实现状态

> 路径 C'（复用 CANN V4 整数 int8/int4）经实测不可达 s8s4 正确结果（详见 [附录 B](#附录-b--历史调试与路径选型档案)），故转方案 A：自研 AscendC kernel `quant_matmul_s8s4`，用 **fp32 scale** 做 `q*scale` 反量化，绕开 CANN V4 的 uint64/FP19 与 V4→V3 融合降级。本节给定算子输入输出与计算语义，供实现方对齐；**kernel 内部流水为实现方自由**。

### 实现状态（已全部落地）

方案 A 的算子定义 + converter 链路改造**7 项全部完成**：

| # | 改造项 | 实现位置 | 状态 |
|---|--------|---------|------|
| 1 | `materialize_int4_with_scale(..., packed=True)` 产出 nibble-packed int8 `[n, k//2]` | `cann_lm/converter/quant/materialize.py:30-44, 90-129`（`pack_int4` + `packed` 参数） | ✅ 已实现 |
| 2 | `QuantLinear` 删预转置，权重/scale 自然朝向 `[n, k]`/`[n, k//g]`，`transpose_x2=True` | `cann_lm/converter/quant/quant_linear.py:24-30, 73-81` | ✅ 已实现 |
| 3 | `group_size=g` 单 int attr（非 V4 的 `[m,n,k]` 48-bit pack） | `cann_lm/converter/quant/quant_linear.py:76` | ✅ 已实现 |
| 4 | 源 Linear 无 bias 时 `bias=None`（无 dummy-zeros workaround） | `cann_lm/converter/quant/quant_linear.py:30, 66-69, 75` | ✅ 已实现 |
| 5 | `cann_lm::quant_matmul_s8s4` + `transpose_x2=True`（取代 `npu::npu_quant_matmul`） | `cann_lm/converter/quant/quant_linear.py:73-81` | ✅ 已实现 |
| 6 | `_quant_matmul_s8s4_converter.py` 映射到方案 A 自定义 GE op | `cann_lm/converter/export/_quant_matmul_s8s4_converter.py` | ✅ 已实现 |
| 7 | 三个 `reg_register.h` 注册 `QuantMatmulS8S4` + `QuantBatchMatmulV4` + `DynamicQuant` | `examples/reg_register.h`、`examples/export_and_load_epair/reg_register.h`、`examples/export_llm_model/om_npu_verify/reg_register.h` | ✅ 已实现 |

custom op 定义在 `cann_lm/models/fused_ops/quant_matmul_s8s4.py`（fake/meta only），GE 转换器 `_quant_matmul_s8s4_converter.py` 把 `cann_lm::quant_matmul_s8s4` 映射到方案 A 自研 GE 算子。

### 计算语义（s8s4 对称，offset=0）

记激活 `x1` `[b, m, k]`、权重 `x2` `[n, k]`（nn.Linear 自然朝向）、group_size `g`（`k%g==0`）：

```
y[b,m,n] = Σ_k ( x1_q[b,m,k] · x1_scale[b,m] ) · ( x2_q[n,k] · x2_scale[n, k//g] ) + bias[n]
```

- `x1_q` int8（per-token 动态对称，每行一个 scale）、`x2_q` int4（per-channel×per-group 对称，每 (n,group) 一个 scale）。
- `x1_scale[b,m]` 是 per-token，**可提到 k 求和之外**（kernel 里后乘即可）。
- `x2_scale[n, k//g]` 是 per-group，**在 k 求和之内**（每组 g 个元素共用一个，反量化时按组广播，无法提到求和之外）。

### 算子签名（已落地实现）

| 方向 | 名称 | dtype | shape | 语义/来源 |
|---|---|---|---|---|
| in | `x1` | int8 | `[b, m, k]` | 来自 DynamicQuant 的已量化激活 |
| in | `x1_scale` | fp32 | `[b, m]` | per-token scale，来自 DynamicQuant |
| in | `x2` | int4（packed） | `[n, k]` 逻辑 ≈ `[n, k//2]` int8 物理 | nn.Linear 权重物化后的 int4 |
| in | `x2_scale` | fp32 | `[n, k//g]` | per-channel×per-group，来自 GPTQ |
| in（opt） | `bias` | 同输出 | `[n]` | 融合 add，省一个节点 |
| attr | `group_size` | int | — | K 轴分组 g |
| attr | `output_dtype` | int | — | 1=fp16 / 27=bf16 |
| attr | `transpose_x1` | bool | — | 默认 False（激活恒不转） |
| attr | `transpose_x2` | bool | — | 默认 True（x2 给 `[n,k]`，kernel 内部转置喂 cube，对齐 CANN 约定） |
| out | `y` | fp16/bf16 | `[b, m, n]` | — |

### 硬需求 vs 设计选择

- **硬需求（s8s4 spec，不可改）**：`x1=int8 per-token 对称`、`x2=int4 per-channel×per-group 对称`、输出 fp16/bf16、offset=0。
- **设计选择（推荐，实现方可改）**：
  1. **激活量化拆出 op 外**：DynamicQuant 独立（已可用、已注册 `ge.DynamicQuant`），本 op 只做"已量化 int8×int4 → fp16"。权衡：若要更少图节点可把 per-token max/div 融进 op，但 cube+vector 混合融合更复杂。
  2. **scale 用 fp32**：精度参考 W8A8 已验证路径（`examples/verify_quant_matmul_v4_w8a8.py` maxdiff≈0 用 fp32 pertoken）；x2_scale 沿用 dopt/GPTQ 产出的 fp32。fp16 占带宽更小但有精度风险，实现方可据 kernel 实测定。
  3. **x2 nibble-packed `[n, k//2] int8`**：比 unpack 省 2x 带宽。kernel 内部 L1/L0B 的具体布局是实现方自由（CANN 用 NZ/int4pack，不必照搬）。
  4. **transpose_x2=True**：让 converter 不必预转置权重（更干净），kernel 在 L1 load 时处理转置。
  5. **不带 offset 输入**：对称=0；非对称未来再加 dormant 字段。

### kernel 内部流水参考（非 I/O 约束）

权重侧 per-group scale 在 k 求和之内，故 kernel 须在 cube 累加前把 int4 权重按组反量化。推荐流水（供实现方参考）：

1. L1/L0B load 时 int4→int8 unpack，按组取 `x2_scale` 广播；
2. cube 做 **int8×int8 → int32** 累加（命中 W8A8 已验证的 int8 cube 路径，规避 FP19/uint64）；
3. fixpipe 阶段乘 `x1_scale`（per-row，可后乘）+ `bias` → cast fp16/bf16。

此流水让 s8s4 数值落在"int8 cube + fp32 scale 反量化"语义上，是绕开 CANN V4 失败点的关键。

---

## C++ 注册与产物

### reg_register.h（三文件均已完成注册）

| 文件路径 | 注册算子 | 状态 |
|---------|---------|------|
| `examples/reg_register.h` | `QuantMatmulS8S4` + `QuantBatchMatmulV4` + `DynamicQuant` | ✅ |
| `examples/export_and_load_epair/reg_register.h` | 同上 | ✅ |
| `examples/export_llm_model/om_npu_verify/reg_register.h` | 同上 | ✅ |

> **路径勘误**：早期文档反复写 `examples/export_llm_model/reg_register.h`（不存在），实际应为 `examples/export_llm_model/om_npu_verify/reg_register.h`。文档同时遗漏顶层 `examples/reg_register.h`（第三个文件，也已注册新算子）。

### GE converter 与权重落盘

- **GE converter**：`cann_lm/converter/export/_quant_matmul_s8s4_converter.py` 把 `cann_lm::quant_matmul_s8s4` 映射到方案 A 自研 GE 算子。
- **权重落盘（无需改 pack）**：`epair_utils.pack_weights` 是**通用**的——序列化 state_dict 里任意 tensor（int8 weight_q + fp32 scale 自动进 bin，bf16→int16 view 已处理）。`weight_info.json` 按 name 记 offset/size。**量化权重无需改打包逻辑**。
- **C++ ge_graph_builder**：`PreparePlaceholders` 通用喂权重常量到任意算子输入（`ReadTensorData`→Const 节点）→ `quant_matmul_s8s4` 的 `x2`/`x2_scale` 自动喂入。`ge_weight_loader` 支持 int8/uint8/fp32 等 dtype（`ElementSizeForPt2Dtype` / `Pt2DtypeFromGeString`）。

### 最终产物（epair 格式）

- `prefill_ep_json_with_ascend_ir.json` / `decode_ep_json_with_ascend_ir.json`：含 `cann_lm::quant_matmul_s8s4` 节点 + `npu::npu_dynamic_quant` 节点，`QuantLinear` 在 `nn_module_stack` 元信息里。
- `weights_*.bin`：int4 nibble-packed int8 + fp32 scale 的二进制权重。
- `model_config.json`：含 `quant_info` 字段（`scheme`/`group_size`/`act_quant`/`scope`/`n_quantized`/`n_skipped_no_dopt_key`/`dopt_path`/`weight_int4_encoding`/`op`/`dopt_bit`/`dopt_unsigned_quant`/`offset_supported`，由 `swap_to_quant_linear_from_dopt` 返回）。
- `weight_info.json`：权重 offset/size 索引。
- host-embeddings 模式额外：`embed_tokens_q.bin` int8 + `embed_tokens_scale.bin` fp32，host 查表。

后续编译成 OM 是下游 `epair → OM` 编译器的事，不在 quant/ 范围。

---

## 已知问题与遗留

1. **unsigned int4 不支持**：`swap_to_quant_linear_from_dopt`（`swap.py:191-195`）读 dopt 产物的 `unsigned_quant`，校验必须 = 0（symmetric signed `[-8, 7]`）。非 0（unsigned int4 `[0, 15]`）触及 CANN int4 signed/unsigned 编码约定，当前直接 `ValueError` 拒绝。dopt 默认 `Quant_act_weight_eco` 的 `unsigned_quant=False`，故主路径不触发。
2. **非对称 offset 不支持**：`swap_to_quant_linear_from_dopt`（`swap.py:196-201`）读 dopt 产物的 `offset`，校验必须全 0（symmetric）。非零 offset（asymmetric）`NotImplementedError`——方案 A `quant_matmul_s8s4` 只支持 symmetric offset=0。dopt `weight_q_params_adapter_into_dopt` 把 offset 置零（`offset=torch.zeros_like(...)`），故主路径不触发。dopt `QuantizerFuncIntOffset` 反量化 `w_fp=(q-offset)*scale`；若未来支持非对称，需对齐 op 的 offset 语义 + 按 group `repeat_interleave` 扩维。
3. **CPU-only 适配**：converter 侧 GPTQ 跑不了（需 NPU/GPU + 校准）。**`--quant-path` 已作兜底**：离线机跑 dopt GPTQ 产出（`quant_output/model.safetensors` 或 `trained_quant_weight.pth`），converter 加载它只做物化 + swap + 导出（不跑 GPTQ）。dopt `opt_main` 在本机 NPU 上跑 GPTQ 曾静默卡住（converter 一键式接入待解，见 [一键式量化方案选型](#一键式量化方案选型)）。
4. **一键式量化（方案 C）已实现**：converter `--quant-method gptq`（`auto_quant.py`：`generate_dopt_config` + `run_auto_quant` spawn dopt subprocess + swap）。w4a16/实测待做，详见 [一键式量化方案选型](#一键式量化方案选型)。
5. **dopt 函数超规**：`swap.py swap_to_quant_linear_from_dopt` 函数超 50 行，`materialize.py materialize_int4_with_scale` 6 参数（超 5 参数原则）。两者建议按单一职责拆分 + `Int4Spec` dataclass 收敛，当前未拆。
6. **dopt gemma4 多模态约束**：dopt GPTQ 主循环按文本 `ForCausalLM` 结构访问 `model.model.layers`；`model_define` 会从多模态模型抽取并重挂语言模型。`QGemma4ScaledEmbedding` 已通过 `visible_type_dict` 的原始类名分支接入。
7. **dopt_lm 拼写错误**：目录名 `ptq_algo_heissen` 与类名 `WeightQuantHeissen` 中的 `heissen` 是 `Hessian` 的系统性拼写错误（遍布目录、类、属性 `heissen_matrix`、局部变量、甚至版权头）。因 dopt_lm 不被主仓 import（独立 CLI 运行，产物按文件路径加载），重命名 `heissen→hessian` 不影响 converter 消费链路——只需同步 dopt_lm 内部 import 与 `train.py` 的 import 即可，是低风险纯内部重构。当前未改。

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

**perchannel 时 scale 的形状**（`quant_batch_matmul_v3_proto.h:59-62`）：
```
When the quant mode of x2 is perchannel or pertensor, the shape is 1D (t,),
with t equal to 1 or n, where n is the same as that of x2.
```
- `t = n`：真正的 perchannel（每通道一个 scale）
- `t = 1`：实际是 pertensor

**与 V4 的关系**：V4 的 A8W8GB 判定（`quant_batch_matmul_v4_to_v3_fusion_pass.cpp:127-128`）要求 scale 维数与数据相同，这**排除了 perchannel**——因为 perchannel 时 scale 是 1D，而 x2 是 2D/3D，维数不等。所以 A8W8 的 perchannel/pertensor 场景下，V4 会被转成 V3 来执行。

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

对比 x1（激活）侧四种模式的 pertoken_scale 形状（`quant_batch_matmul_v3_proto.h:77-83`）：

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

`group_size` 是一个 48 位编码的 int 属性（`quant_batch_matmul_v3_proto.h:93-105`）：

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
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题（`swap_embeddings_from_dopt` + `materialize_int8_per_channel`） |
| [dopt_design.md](./dopt_design.md) | dopt_lm vendored GPTQ 框架（文件树 / 入口 / GPTQ 调用链 / 可设置参数清单 / 历史调试档案） |
| [converter_design.md](../converter_design.md) | 跨模块调用链总览 / 模型差异总览 |
| [reference_qwen3_5_export.md](../reference_qwen3_5_export.md) | qwen3.5 导出专题（参考） |
