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
- [dopt_lm 子包详解](#dopt_lm-子包详解)
- [方案 A — 自研 quant_matmul_s8s4 算子规格与实现状态](#方案-a--自研-quant_matmul_s8s4-算子规格与实现状态)
- [C++ 注册与产物](#c-注册与产物)
- [已知问题与遗留](#已知问题与遗留)
- [附录 A — CANN QuantBatchMatmulV3 / V4 op_graph 源码分析（历史档案）](#附录-a--cann-quantbatchmatmulv3--v4-op_graph-源码分析历史档案)
- [附录 B — 历史调试与路径选型档案](#附录-b--历史调试与路径选型档案)

---

## 概述

> 量化接入的 CLI 参数（`--quant-path` / `--quant-embeddings`）见 [cli_design.md](../cli/cli_design.md)；量化 epair 的产物布局、host_weights 机制、C++ 加载侧见 [export_design.md](../export/export_design.md)。embedding 量化的 W8 host gather 子主题见 [embedding_quant_design.md](./embedding_quant_design.md)。

`cann_lm/converter/quant/` 下内置两块：

1. **dopt_lm（vendored GPTQ 框架）**：在 **NPU/GPU** 上对 HF 模型做 GPTQ 权重量化，产出 per-layer `group_scale`（per-channel × per-group 对称，offset=0）+ GPTQ 修正后的 fp16 权重，落盘 `trained_quant_weight.pth` 或 `model.safetensors`。dopt 自身只做 fakequant（权重留在 fp16），不物化真 int4，不接 converter 导出。
2. **converter 侧量化接入（`materialize.py` / `quant_linear.py` / `swap.py` / `__init__.py`）**：在 **CPU venv + torchair stubs** 上加载 dopt 产物，物化成真 int4 payload + fp32 scale，把 `nn.Linear` 替换成 `QuantLinear`（forward 调 `cann_lm::quant_matmul_s8s4`），然后 `torch.export` 产出含 `quant_matmul_s8s4` + `DynamicQuant` 节点的 epair。

### 能力边界

- **输入**：HF 模型目录 + 校准数据（`quant/calibration.json`）+ dopt GPTQ 产物（`quant_output/model.safetensors` 或 `.pth`）。
- **输出**：epair 模型目录，其 `*_ep_json_with_ascend_ir.json` 含 `quant_matmul_s8s4` + `DynamicQuant` IR 节点，权重 bin 含 nibble-packed int4（int8 存储）+ fp32 scale，`model_config.json` 含 `quant_info` 字段。
- **量化范围**：全部 `nn.Linear`（attn q/k/v/o_proj + mlp gate/up/down_proj + gemma4 的 `per_layer_input_gate` / `per_layer_projection` / `per_layer_model_projection` + `lm_head`），**仅 `embed_tokens` / `embed_tokens_per_layer` 不走 Linear swap**——它们是 `nn.Embedding`，由 `swap_embeddings_from_dopt` 走 W8 host gather（需 converter `--quant-embeddings`）。
- **不包含**：CPU-only 环境跑 GPTQ（GPTQ 需 NPU/GPU + 校准，CPU 跑不了，见 [已知问题](#已知问题与遗留)）；GPTQ 与导出的内存链路一体化（仍两步磁盘桥接，见 [端到端流程](#端到端流程)）。

---

## quant 包架构

```
cann_lm/converter/quant/
├── __init__.py              (47 行)  包门面，re-export 8 个公共符号
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
                cann-lm-converter.py:90, 105
                    from cann_lm.converter.quant import (
                        swap_to_quant_linear_from_dopt,     # 主 swap 路径
                        swap_embeddings_from_dopt,          # embedding host gather
                        load_dopt_state_dict,               # dopt 产物加载
                    )
```

- **无循环依赖**：`materialize ← swap → quant_linear` 的有向无环图；`materialize` 与 `quant_linear` 互不依赖。
- **`__all__` 暴露 8 个符号**（`__init__.py:38-47`）：其中只有 3 个（`swap_to_quant_linear_from_dopt` / `swap_embeddings_from_dopt` / `load_dopt_state_dict`）被 `cann-lm-converter.py` 实际 import；其余 5 个（`materialize_int4` / `materialize_int4_with_scale` / `materialize_int8_per_channel` / `QuantLinear` / `swap_to_quant_linear`）经 `swap.py` 内部相对 import 使用，对外暴露为测试/将来直接调用预留。

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
| 权重 (x2) | int4 (s4) | per-output-channel × per-group(group_size, 默认 256) | 对称 | GPTQ 离线求（`group_scale`） | 0（对称，不存） |
| 激活 (x1) | int8 (s8) | per-token 动态 | 对称 | 运行时从 x 算（`npu_dynamic_quant` → `DynamicQuant` IR） | 0 |
| 输出 | fp16/bf16 | — | — | — | — |

对应 dopt `strategy.py` 的 **`Quant_act_weight_eco`**（`dopt_lm/quantizer/strategy.py`）：

```python
"Quant_act_weight_eco": {
    "weight": {"blocksize": 128, "group_size": 256, "bit": 4, "per_channel": True,
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
- **embed_tokens / embed_tokens_per_layer**：是 `nn.Embedding`（非 Linear，Linear swap 不触及），由 `swap_embeddings_from_dopt` 消费 W8 host gather——需 converter `--quant-embeddings` + `--use-embeddings`（embed 外置，图内无 embed_tokens 节点）。
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

`dopt_config.json`（796 行）配置：`embed_tokens` → `Quant_Embed_MinMax`（bit=8 int8 per-channel，host gather）、`lm_head` → `Quant_lm_head`（bit=4）、其余 Linear → `Quant_act_weight_eco;256`。dopt 产物里 `lm_head.weight` 与 `model.embed_tokens.weight` 已**实质 untie**（max diff ≈0.01 = GPTQ 修正量，lm_head 是 GPTQ 修正版给 W4、embed 是原始版给 W8）。

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
│       (layer_strategy: Linear→Quant_act_weight_eco;256,                    │
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
│  cann-lm-converter.py:                                                  │
│   B. load_model(hf_path) → nn.Module (HF 原始, fp16)                     │
│   C+D. swap_to_quant_linear_from_dopt(model, quant_path=quant_output,    │
│                                      group_size=256, act_quant=True):  │
│      a. load_dopt_state_dict(quant_output) → sd: dict                    │
│      b. for each nn.Linear (非 skip):                                   │
│         - sd[name.weight] (GPTQ fp) + sd[...weight_quantizer.s] (scale) │
│         - _read_dopt_quant_meta → bit/uq/offset (校验 uq==0, offset 全 0)│
│         - materialize_int4_with_scale(w, scale2d, gs, packed=True)       │
│           → (weight_q int8 [n,k//2] nibble-packed, weight_scale fp32)   │
│         - QuantLinear(weight_q, weight_scale, gs, bias, act_quant=True) │
│         - setattr 替换 nn.Linear                                        │
│      c. return quant_info dict → artifact.quant_info                    │
│   (可选 --quant-embeddings):                                             │
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

### 一步化流程（GPTQ + 导出内存链路，待实现）

> 现状两步靠磁盘桥接。根因：dopt GPTQ 依赖 transformers `AutoModelForCausalLM`（`weight_quant_loop` 用 `model.model.layers`/Catcher/hooks，写死 HF 模型结构），无法直接跑在 cann_lm 自定义模型上；故 GPTQ（HF 模型）与导出（cann_lm 模型）是两个模型实例，磁盘 state_dict 作桥。

目标命令（一条）：

```bash
DEVICE=npu python cann-lm-converter.py \
  --hf-path /path/to/hf --output-dir /path/to/out --dtype float16 \
  --quant-gptq --quant-group-size 256
# 进程内：GPTQ(NPU) → 内存 dopt_sd → swap → export，无磁盘往返
```

`--quant-path`（消费磁盘产物，CPU-only 机用）保留作兜底，两者并存。

三处改动点（均未落地，仍待实现）：

| # | 文件:行 | 改动 |
|---|---|---|
| 1 | `dopt_lm/opt_funcs/eco_stage/train.py:530-581`（`ptq_quant_pipline`） | 拆"产出 state_dict"与"落盘"：新增 `ptq_quant_produce_state_dict(base_model, config) -> dict`（GPTQ + swap-to-QLinear + 注入 group_scale，**返回** `dopt_model.state_dict()`，不落盘）；`ptq_quant_pipline` 改为调它再落盘 |
| 2 | `cann_lm/converter/quant/swap.py:151-226`（`swap_to_quant_linear_from_dopt`） | 拆成 `swap_to_quant_linear_from_sd(model, sd: dict, ...)`（收内存 dict）+ `swap_to_quant_linear_from_dopt(model, dopt_path, ...)`（前者 load 后转发）。`swap_embeddings_from_dopt` 已直接收 sd，无需改 |
| 3 | `cann-lm-converter.py:89-99,189`（`convert()`） | 加 `quant_gptq` 分支（与 `quant_path` 并列）：调 `ptq_quant_produce_state_dict(hf_path, dopt_cfg)` → `swap_to_quant_linear_from_sd(...)`；CLI 加 `--quant-gptq`（store_true）+ 可选 `--dopt-config`/`--dopt-optimize-config`（默认 `cann_lm/converter/quant/dopt_config.json`+`config.yaml`） |

必须处理的两个细节：
1. **calibration 路径**：`config.yaml` 的 `train_files: ["calibration.json"]` 相对 CWD。converter 从项目根跑时需把 calibration 绝对化，或 `chdir` 到 `cann_lm/converter/quant/` 再调 GPTQ（否则 `get_loaders` 找不到文件）。
2. **环境**：converter 进程须 `DEVICE=npu` + `torch_npu` 可 import（GPTQ 段需要）。dopt 内部已自行做 layer CPU↔NPU 搬移，export 段仍在 CPU（`load_model` 默认 `device="cpu"`）。

内存权衡：进程内同时持有 cann_lm 模型（export）+ dopt 内部 2 份 HF 模型拷贝（GPTQ + dopt_model）+ 提取出的 sd dict ≈ **3–4× 模型大小**峰值。0.6B 无虞；7B+ 需 `del dopt_model; gc.collect()` 顺序释放，或退回 `--quant-path` 两步。

---

## quant 顶层文件详解

### `materialize.py`（162 行）— 权重物化纯函数

纯函数模块，把 fp16/bf16/fp32 的 `nn.Linear` 权重 `[out, in]` 物化成量化 payload + per-group/per-channel scale。**无状态、无副作用、无外部 I/O**。是整个量化链路的"数值核心"。

| 函数 | 行 | 签名 | 参数数 | 作用 |
|------|----|------|--------|------|
| `pack_int4` | 30-44 | `(q: Tensor) -> Tensor` | 1 | Nibble-pack signed int4（存为 int8）成 `[n, k//2]` int8；byte[i] = int4(k=2i+1)<<4 \| int4(k=2i)。私有辅助，**不在 `__all__`** |
| `materialize_int4` | 47-87 | `(weight, group_size=128, signed=True, nbits=4, packed=False) -> (Tensor, Tensor)` | 5 | min-max 对称量化（无 GPTQ），产出 (int8 `[out,in]` 或 packed `[out,in//2]`, fp32 scale `[out,in//group]`)。被 `swap_to_quant_linear` 调用（占位/测试路径） |
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

在 `torch.export` 之前，**就地遍历** `model.named_modules()`，把目标 `nn.Linear` 替换成 `QuantLinear`（物化 int4 权重）、把目标 `nn.Embedding` 的 int8+scale 抽出来供 host gather（不替换成图模块）。还提供 dopt GPTQ 产物的加载器 + 量化元信息读取器。是 `cann-lm-converter.py` 唯一直接调用的模块。

| 函数 | 行 | 签名 | 参数数 | 作用 |
|------|----|------|--------|------|
| `swap_embeddings_from_dopt` | 36-58 | `(sd: dict, model: Module) -> dict[str, tuple[Tensor, Tensor]]` | 2 | 遍历 `nn.Embedding`，从 sd 读 `{name}.weight` (fakequant) + `{name}.quant_op.weight_quantizer.s`，`materialize_int8_per_channel` → `{wrapper_fqn: (int8, scale)}`。**不** swap 成图模块（embedding 外置给 host gather）。跳过无 scale 的（如 gemma4 per_layer 未量化） |
| `_resolve_parent` | 61-67 | `(model: Module, dotted_name: str) -> tuple[Module, str]` | 2 | 私有辅助：把 `"a.b.c"` 拆成 `(parent=a.b module, leaf="c")`，供 setattr 替换 |
| `swap_to_quant_linear` | 70-94 | `(model, group_size=128, skip=DEFAULT_SKIP, act_quant=True) -> Module` | 4 | 无 GPTQ 路径（占位/测试）：遍历 `nn.Linear`（非 skip），`materialize_int4` (min-max) → `QuantLinear` → setattr 替换。就地修改 model |
| `load_dopt_state_dict` | 97-122 | `(dopt_path: str \| Path) -> dict` | 1 | 接受 `.safetensors` / `.pth` / 目录(含 `model.safetensors`)。返回 `{name: tensor}` dict |
| `_read_dopt_quant_meta` | 125-148 | `(sd, name, out_f, in_f, group_size) -> tuple[int, float, Tensor \| None]` | 5 | 私有辅助：读 `{name}.quant_op.weight_quantizer.{bit, unsigned_quant, offset}`。bit=int64 标量，unsigned_quant=fp32 标量，offset=fp32 扁平 `[out*groups]`（reshape 到 `[out, in//group]`）。老版 dopt 无 buffer 时 fallback (4, 0.0, None) |
| `swap_to_quant_linear_from_dopt` | 151-226 | `(model, dopt_path, group_size=256, skip=DEFAULT_SKIP, act_quant=True) -> dict` | 5 | **主路径**（流程 C+D）：① `load_dopt_state_dict` → sd；② 遍历 `nn.Linear`（非 skip）：读 wkey/skey，跳过缺 key 的；reshape s_flat → scale2d；`_read_dopt_quant_meta` → bit/uq/offset；校验 uq==0（symmetric only，否则 ValueError）；校验 offset 全 0（否则 NotImplementedError：方案 A 不支持非对称）；`materialize_int4_with_scale(packed=True)` → `QuantLinear` → setattr 替换；③ 返回 quant-info dict（scheme/group_size/scope/counts/op/encoding/dopt_bit/uq/offset_supported）→ 被 `cann-lm-converter.py` 写进 `model_config.json` 的 `quant_info` 字段 |

模块级常量 `DEFAULT_SKIP = ("embed_tokens", "embed_tokens_per_layer")`（行 33）。**注意 `lm_head` 不在 skip 里**——会被量化成 QuantLinear W4A8（行 30 注释明确）。

> **代码风格观察**：`swap_to_quant_linear_from_dopt` 函数体 **76 行**（151-226，超 AGENTS.md "单函数 50 行" 原则）。即便扣除 docstring（158-167，约 10 行），纯代码约 58 行。该函数承担太多职责（加载 sd + 遍历 Linear + 读 scale + 读 meta + 校验 uq + 校验 offset + materialize + 实例化 + 替换 + 计数 + 构造返回 dict），应按 [一步化流程](#一步化流程gptq--导出内存链路待实现) 改动点 #2 的拆分计划拆成：
> - `swap_to_quant_linear_from_sd(model, sd, ...)`（收内存 dict，做核心 swap）
> - `swap_to_quant_linear_from_dopt(model, dopt_path, ...)`（前者 load 后转发）
> - `_validate_dopt_quant_meta(uq, offset)`（校验 uq==0 / offset 全 0）
> - `_build_quant_info(...)`（构造返回 dict）
>
> 当前未拆。

### `__init__.py`（47 行）— 包门面

re-export 8 个公共符号（`__init__.py:38-47` 的 `__all__`）：

```python
from .materialize import materialize_int4, materialize_int4_with_scale, materialize_int8_per_channel
from .quant_linear import QuantLinear
from .swap import (
    load_dopt_state_dict,
    swap_to_quant_linear,
    swap_to_quant_linear_from_dopt,
    swap_embeddings_from_dopt,
)

__all__ = [
    "materialize_int4",
    "materialize_int4_with_scale",
    "materialize_int8_per_channel",
    "QuantLinear",
    "swap_to_quant_linear",
    "swap_to_quant_linear_from_dopt",
    "swap_embeddings_from_dopt",
    "load_dopt_state_dict",
]
```

顶部 docstring（行 15-27）概述了量化两条入口路径与 dopt_lm 离线运行的关系。8 个符号中只有 **3 个**（`swap_to_quant_linear_from_dopt` / `swap_embeddings_from_dopt` / `load_dopt_state_dict`）被 `cann-lm-converter.py` 实际 import；其余 5 个经 `swap.py` 内部相对 import 使用，对外暴露为测试/将来直接调用预留。

### `config.yaml`（10 行）— dopt GPTQ 训练配置

**不是** 4 个 Python 文件的输入！它是 `dopt_lm.opt_main` 的 `--optimize-config` 参数文件，描述 GPTQ 校准的训练配置：校准语料是同目录的 `calibration.json`，采样 1024 条，截断长度 512，fp16 训练+存盘。内容见 [量化范围与校准](#量化范围与校准)。

---

## dopt_lm 子包详解

`dopt_lm/` 是华为 vendored 的 **GPTQ 风格权重 PTQ 框架**（命名 `dopt` = "do optimize"，Quantization Optimization）。它本身是自洽的、可独立运行的 Python 包（入口 `opt_main.py`，CLI 模式），在 **NPU/GPU** 上对 HuggingFace 模型做 **GPTQ 权重量化**，产出 per-layer `group_scale`（per-channel × per-group 对称，offset=0）+ GPTQ 修正后的 fp16 权重，最终序列化成 `trained_quant_weight.pth`（或 `--hf_type_save` 的 HF 目录）。

**调用关系**：dopt_lm 在主仓代码中**不被 `import`**，而是作为独立 CLI 工具运行，其产物（.pth/safetensors）由 converter 的 `quant/swap.py` 在导出时加载。dopt_lm 内部通过相对导入 `from dopt_lm.xxx import yyy` 自引用（需将 `cann_lm/converter/quant` 放进 `PYTHONPATH`）。

> **拼写说明**：目录名 `ptq_algo_heissen` 中的 `heissen` 是 **`Hessian` 的拼写错误**（系统性同字错误，遍布目录名、类名 `WeightQuantHeissen`、属性 `heissen_matrix`、局部变量 `heissen_mat`/`heissen_cpu`，甚至版权头）。算法本质是 GPTQ（基于 Hessian 矩阵的 PTQ）。本节保留原拼写以对齐代码，但读者须知其为 `Hessian`。

### 文件树

```
dopt_lm/                                           (24 文件, ~2759 行)
├── __init__.py                          (3 行)    版权头
├── builder.py                          (31 行)   Quantizers/OPT_OPS Register + init_builder
├── do_opt.py                           (202 行)  核心调度：optimize_model / paser_quant_config / set_*_state
├── opt_main.py                         (56 行)   CLI 入口
├── utils.py                            (23 行)   replace_module_by_names
├── common_tools/
│   ├── __init__.py                     (0)
│   ├── log.py                          (135 行)  Logger / FileTxtLogger / visualize_dist
│   └── register.py                     (112 行)  Register 装饰器 + file_import
├── modules/
│   ├── __init__.py                     (0)
│   ├── qembed.py                      (132 行)  QEmbedding / QGemma4ScaledEmbedding
│   ├── qlinear.py                     (64 行)   QLinear (fakequant)
│   └── utils.py                        (16 行)   get_quantizer_cfg
├── opt_funcs/
│   ├── __init__.py                     (3 行)
│   └── eco_stage/
│       ├── __init__.py                 (0)
│       └── train.py                    (581 行)  weight_quant_pipline / ptq_quant_pipline / train / KD trainer
├── quantizer/
│   ├── __init__.py                     (0)
│   ├── qeval.py                        (90 行)   autograd 量化函数 + DynEvalGridS8
│   ├── qmanager.py                     (84 行)   QClass 容器
│   ├── qmin_max.py                     (420 行)  MinMaxGrid / Group / Omni / OmniGroup
│   ├── qutils.py                       (165 行)  clip range / min-max / NF 表 / fakequant
│   └── strategy.py                     (98 行)   strategy_dict + get_optimize_cfg
└── weight_quant_ptq/
    ├── __init__.py                     (0)
    └── ptq_algo_heissen/
        ├── __init__.py                 (0)
        ├── datautils.py                (109 行)  get_pangupi / get_pt_data / get_loaders
        ├── modelutils.py               (25 行)   find_layers / DEV
        ├── ptq_algo.py                 (185 行)  WeightQuantHeissen (GPTQ 核心)
        ├── ptq_loop.py                 (375 行)  llama_sequential / weight_quant_loop
        └── quant.py                    (134 行) Quantizer (GPTQ 专用 min-max)
```

### 入口与主流程（顶层 5 文件）

| 文件 | 作用 |
|------|------|
| `opt_main.py` | CLI 入口。`parse_args()` 接 `--model-path / --dopt-config / --optimize-config / --output-dir / --hf_type_save`。两步：① 若 `--dopt-config` 不存在 → `generate_quant_config` 生成脚手架 JSON（所有层 `quant_strategy="float"`）并 `exit()`，用户需手动编辑策略后重跑；② 否则组装 `optimize_info` 调 `llm_quant_pipline` → `train.weight_quant_pipline` |
| `opt_funcs/eco_stage/train.py` | **最大、最核心的 orchestrator**（581 行）。`weight_quant_pipline`：决策入口——`config["kd"]["enable"]` 走 `train()`（KD 训练），否则走 `ptq_quant_pipline()`（纯 GPTQ PTQ，无微调）。两者最终都产出 `trained_quant_weight.pth`。还含 `generate_config`/`prepare_model`/`model_define`/`seed_all`/`DoptTrainerCustomization` |
| `do_opt.py` | **核心调度层**（被多处 import）。模块加载时执行 `init_builder()`，定义 `visible_type_dict = {nn.Embedding: QEmbedding, "embedding_min_max": QEmbedding, nn.Linear: QLinear, "Linear": QLinear, "Gemma4TextScaledWordEmbedding": QGemma4ScaledEmbedding}`。导出 `optimize_model(model, config_file)`（按 `layer_strategy` 把目标 module wrap 成 QLinear/QEmbedding）、`paser_quant_config(model, config_file)`（GPTQ 专用，只加载 `layer_strategy` 并对命中层做 `parse_layer_config`，返回 dict 供 `ptq_loop.py` 取用）、`set_quant_state`/`set_calibrate_state`（切换每个 `quant_op` 的 fakequant 开关）、`generate_config_file`（遍历 `named_modules()` 生成脚手架 JSON） |
| `builder.py` | 注册中心初始化。定义两个全局 `Register`：`Quantizers = Register("Quantizers")`（量化器算法注册表）、`OPT_OPS = Register("OptimizedOps")`（优化算子注册表，当前未使用）。`init_builder(quantizer_path=None)` 递归 import `modules/` 与 `quantizer/` 目录触发注册 |
| `utils.py` | Stage1 minimal：仅 `replace_module_by_names(model, modules_to_replace) -> model`（递归 `add_module` 替换子模块） |

### GPTQ PTQ 主路径调用链

```
ptq_quant_pipline(base_model, config, output_dir, hf_type_save, quant_method='gptq')
  │
  ├─ [1] from ptq_algo_heissen.ptq_loop import weight_quant_loop
  │      weight_q_quantizer, weight_q_weight = weight_quant_loop(
  │          base_model, data_set=train_file, nsamples, seed=1234,
  │          seq_len, quant_config=config_path, torch_dtype)
  │         │
  │         ├─ get_llama(model, seq_len, torch_dtype)            # 加载到 NPU/GPU
  │         ├─ args.quant_config = paser_quant_config(model, quant_config)  # 解析 layer_strategy
  │         ├─ dataloader, _ = get_loaders(data_set, nsamples, seed, model, seqlen)
  │         │      ├─ .pt → get_pt_data
  │         │      └─ else → get_pangupi (json/jsonl/wikitext2)
  │         └─ llama_sequential(model, dataloader, DEV) → quantizers
  │                ├─ Catcher 捕 layer0 输入 → inps[nsamples, seqlen, hidden]
  │                ├─ (gemma4) unique_layer_types → causal_mask_map / position_embeddings_map
  │                ├─ (PLE) get_per_layer_inputs / project_per_layer_inputs
  │                ├─ 逐 layer i:
  │                │    ├─ find_layers(layer) → full
  │                │    ├─ sequential = sort_sequent(...) or [list(full.keys())]
  │                │    ├─ 逐 subset names:
  │                │    │    ├─ WeightQuantHeissen(subset[name]) + .quantizer = Quantizer()
  │                │    │    ├─ quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)
  │                │    │    ├─ register_forward_hook(add_batch) → 跑 nsamples 累积 Hessian
  │                │    │    ├─ groupsize/blocksize ← args.quant_config['model.layers.{i}.{name}']['weight']
  │                │    │    ├─ weight_algo[name].fasterquant(blocksize, percdamp, groupsize, actorder, static_groups)
  │                │    │    │      └─ quantize(w, scale, zero, maxq) + Cholesky 误差更新
  │                │    │    └─ quantizers['model.layers.{i}.{name}'] = quantizer; weight_algo.free()
  │                │    └─ capture_kv 跑 nsamples → outs (供下一层/ lm_head 用)
  │                ├─ (可选) model.per_layer_model_projection GPTQ
  │                └─ (可选) lm_head GPTQ
  │      返回 (quantizers, model.state_dict())
  │
  ├─ [2] dopt_model = prepare_model(base_model, quant_pth, config_path, config,
  │                                 requires_grad=True, trainable_keys=config.get("trainable_keys","lora"))
  │         └─ optimize_model(model, quant_config)  # nn.Linear → QLinear, nn.Embedding → QEmbedding
  │                ├─ get_optimized_op → parse_layer_config → get_target_op_cls → QLinear/QEmbedding(...)
  │                └─ replace_module_by_names
  │
  ├─ [3] set_quant_state(dopt_model, weight_state=True, input_state=False)   # 只权重量化
  │      set_calibrate_state(dopt_model, True)
  │
  ├─ [4] tie_lmhead 处理；dopt_model.load_state_dict(weight_q_weight, strict=False)
  │      weight_q_params_adapter_into_dopt(dopt_model, weight_q_quantizer, tie_lmhead)
  │         └─ 注入 group_scale → QLinear.quant_op.weight_quantizer.s；offset=zeros_like
  │      set_calibrate_state(dopt_model, False)
  │
  └─ [5] torch.save(dopt_model.state_dict(), {output_dir}/trained_quant_weight.pth)
         (或 dopt_model.save_pretrained(output_dir) if hf_type_save)
```

### common_tools（基础设施层）

| 文件 | 职责 |
|------|------|
| `log.py` | `Logger`（静态方法 `info/debug/warning/error`，按 `LOCAL_RANK==0` 过滤分布式日志）+ `add_file_handler`/`remove_file_handler`；`FileTxtLogger`（按 `layer_name` 记录每层量化分布的 txt 文件）；`set_record_mode(model, output_path, name_list=None)`（为每层注册 recoder）；`visualize_dist(ckpt)`（打印 state_dict 里 `*.quantizer.s` / `*.bit` / `*.offset` 分布） |
| `register.py` | `Register` 类：装饰器 `@register_func(name)` 注册 callable；`file_import(file_path, recursive, skip_libs_set)` 把目录/文件按 sys.path 推导成 `pkg.module` 名并 `importlib.import_module`（注册触发器）；`get(key)`/`__getitem__`/`__contains__`/`get_keys()` |

### modules（量化 module 适配层，fakequant）

dopt 的**推理期 fakequant 实现**，不是导出用算子。把 `nn.Linear`/`nn.Embedding` 包成 `QLinear`/`QEmbedding`/`QGemma4ScaledEmbedding`，内含 `quant_op`（`QClass`）做 fakequant 前向（量化后反量化回 fp，再 `F.linear`/`F.embedding`）。

| 文件 | 职责 |
|------|------|
| `qlinear.py` | `QLinear.__init__(m, optimize_cfg, layer_name=None, **argv)`：解析 weight/input/output cfg，建 `QClass`，`to(weight_device)`。`forward(x)`：`weight, x = self.quant_op(weight, x)` → `F.linear(x, weight, None)` → 可选 `output_quantizer(out)` → `+ bias`。支持 `dyn` 动态输入量化（解出 `s_denormalized` 反缩放）。`set_quant_state(weight, input)` 透传 `quant_op` |
| `qembed.py` | `QEmbedding(nn.Module).__init__(m: nn.Embedding, optimize_cfg, layer_name=None, **argv)`：强制 `per_channel=True`，建 `QClass(m, weight_cfg, input_cfg={})`；`tie_lmhead` 控制是否共享 `lm_head` 的 weight（不 clone）；`npu_simulator=None`（stub）。`forward(x)`：`weight, x = self.quant_op(weight, x)` → `F.embedding(...)`。`QGemma4ScaledEmbedding`：gemma4 专用，多 `embed_scale` buffer，forward 末尾 `out = out * self.embed_scale`。**注意：`QGemma4ScaledEmbedding` 未接入 `do_opt.py:visible_type_dict`，属预留未接线代码** |
| `utils.py` | `get_quantizer_cfg(cfg) -> (weight_cfg, input_cfg, output_cfg)`（dict 三段拆分） |

### quantizer（量化器算法层）

5 个文件分工——`strategy.py`（策略字典）/`qmanager.py`（QClass 容器）/`qeval.py`（autograd 量化函数）/`qmin_max.py`（min-max/group/omni 量化器实现，`Register` 注册）/`qutils.py`（clip range、min-max 统计、NF2/3/4 查找表等工具）。是典型的"注册表 + 字符串策略名"模式（非 GoF 策略类）。

| 文件 | 角色 | 关键导出 |
|------|------|---------|
| `strategy.py` | **策略字典** `strategy_dict` + `get_optimize_cfg()`。8 个预设策略（`Quant_aigc_ptq/qat`、`Quant_act_weight_eco`、`Quant_Embed_MinMax`、`Quant_Gemma4_Embed`、`Quant_lm_head`、`Quant_Dummy`、`float`），每项含 `weight`/`input`/`output` 子 cfg，子 cfg 里 `weight_algo`/`input_algo` 字符串指向 `Quantizers` Register 名。`get_optimize_cfg` 支持 `"name;group_size"` 字符串拼接语法覆盖 group_size；未注册策略 raise |
| `qmanager.py` | **容器** `QClass(nn.Module)`：把 weight/input/output 三种量化器 cfg 通过 `Quantizers.get(algo)(m, **cfg)` 工厂实例化聚合。`__init__(m, weight_quantizer_cfg=None, input_quantizer_cfg=None, output_quantizer_cfg=None)`；`set_quant_state(weight, input)` / `set_calibrate_state(state)` / `forward(weight, x) -> (weight, x)` / `is_empty() -> bool` |
| `qeval.py` | **autograd 量化原语** `QuantizerFuncIntOffset`（整数 offset 路径：`round(x/s+offset)` clamp 再 `(q-offset)*s`，STE backward）/ `QuantizerFuncFpOffset`（浮点 offset 路径：`round((x+offset)/s)`）/ `DynIntSymQuantizer`（动态 per-token 对称 int8，amax/127，返回 `(normalized, scale)`）。`@Quantizers.register_func("dyn_s8") class DynEvalGridS8`（动态 s8 包装） |
| `qmin_max.py` | **min-max 系列量化器实现**（4 个 `Register` 注册）：`MinMaxGrid`（`min_max`/`minmax`，基础 per-channel min-max，calibrate 阶段算 scale/offset，eval 用缓存）、`GroupMinMaxGridWeight`（`group_min_max`/`group_minmax`，per-group 维度 reshape 后调父类）、`OmniMinMaxGrid`（`omni_min_max`/`omni_minmax`，LSQ 可学习 `quant_alpha` + sigmoid 自适应 scale）、`OmniGroupMinMaxGridWeight`（`omni_group_min_max`，omni 的 group 版） |
| `qutils.py` | **纯工具**：常量 `MIN_SCALE_VALUE=1e-5`、`NF2/NF3/NF4`（normal_float 查找表，×127 量化）、`get_clip_range(bit, unsigned_quant, keep_neg_max=True)`（含 2.5bit 特例）、`RoundWithGrad`（round 的 STE）、`get_min_max`/`get_abs_max_per_channel`/`get_min_max_per_channel`、`do_quantize`/`do_antiquantize`/`fakequant_forward`、`get_grad_max`/`get_grad_min`、`fake_nf` |

**关键分工**：`strategy.py` 提供配置模板 → `qmanager.QClass` 按 cfg 里 `algo` 名查 `Quantizers` Register → `qmin_max.py`/`qeval.py` 提供注册的具体量化器 → `qutils.py` 提供底层 clip/min-max/round 工具。

### opt_funcs/eco_stage（训练/PTQ 流程编排层）

`train.py`（581 行，最大文件）是真正的 orchestrator——决策走 KD 训练还是 GPTQ PTQ，装配 trainer、数据集、teacher 模型、save 权重。关键导出：

| 函数/类 | 行 | 作用 |
|---------|----|------|
| `check_is_trainable(name, trainable_keys) -> bool` | — | 判断参数是否可训练 |
| `get_processed_dataset(dataset_files, tokenizer, dataset_config, train_samples=None) -> Dataset` | — | 支持 wikitext2/json/jsonl；`text_concat` 模式拼接全文再切 `cutoff_len` |
| `custom_model_generation(model, no_split_module_classes=None) -> model` | — | accelerate 多卡 device_map 平衡 |
| `class DoptTrainerCustomization(nn.Module)` | ~110-211 | KD trainer；`train()`→`training_loop_block()`（AdamW + cosine/linear scheduler + autocast + accelerate prepare）；`compute_loss(model, inputs, return_outputs=False)`（MSE hidden-states / KL-div logits） |
| `ce_loss(student_logits, teacher_logits, topk=-1, reduction="batchmean")` | — | topk 截断的 KL 散度 |
| `seed_all(seed_value=1234, deterministic=False)` | — | 全栈随机种子 |
| `model_define(model_path, torch_dtype=torch.float16) -> AutoModelForCausalLM` | ~290 | `from_pretrained(trust_remote_code=True)` |
| `generate_config(model_name_or_path, config_path)` | — | `model_define` + `generate_config_file` |
| `prepare_model(model, resume_ckpt=None, quant_config=None, config=None, requires_grad=False, trainable_keys=None, refresh=False, device=None) -> model` | 304-353 | 加载+`optimize_model`+可选 resume_ckpt+trainable_keys 冻结策略 |
| `print_trainable_parameters(model)` | — | 统计可训练比例 |
| `train(base_model, config, output_dir=None, hf_type_save=False, **argv)` | 368-477 | **KD 训练主入口**（`gptq_finetune`/`resume_finetune` 模式加载预训练 quant 权重，DoptTrainerCustomization 训练，存 `trained_quant_weight.pth`） |
| `weight_q_params_adapter_into_dopt(dopt_model, weight_q_quantizers, tie_lmhead)` | — | 把 GPTQ 产出的 `group_scale` 注入每个 `QLinear.quant_op.weight_quantizer.s`，`offset=zero_like(...)`（置零→对称），并对 weight 跑一遍 fakequant 写回 |
| `weight_quant_pipline(model_path, config_yaml, dopt_config=None, output_dir=None, hf_type_save=False, **argv)` | — | **决策入口**：`kd.enable` → `train()`；否则 → `ptq_quant_pipline()` |
| `ptq_quant_pipline(base_model, config, output_dir=None, hf_type_save=False, quant_method='gptq', **argv)` | 530-581 | 跑 `weight_quant_loop` → `prepare_model`→`optimize_model` → 注入 scale → 关 calibrate → save |

**KD 训练说明**（`train()`）：实现的是 **知识蒸馏（KD）微调**，用于 GPTQ 之后的精度恢复（不是从零训练，也不是 GPTQ 本身）：
- **学生模型**：先 `optimize_model` 把 `nn.Linear` 换成 `QLinear`（fakequant），加载 GPTQ 产出的 `trained_quant_weight.pth`（`gptq_finetune` 模式）或其二次微调版（`resume_finetune` 模式），按 `trainable_keys`（如 `quant_alpha`）冻结大部分参数只训可学习量化参数（LSQ `quant_alpha`）/norm。
- **教师模型**：`prepare_model`（`requires_grad=False`）加载同架构 FP 模型，`eval()`。
- **损失**：`compute_loss` 支持 `mse`（对学生/教师 `hidden_states[feature_layers]` 做 MSELoss）+ `kl_div`（对 logits 做 topk 截断 KL 散度 `ce_loss`）。
- **训练栈**：`AdamW` + cosine/linear warmup scheduler + autocast + accelerate `prepare`/device_map；存 `trained_quant_weight.pth`。

### weight_quant_ptq/ptq_algo_heissen（GPTQ PTQ 算法实现）

这是 **GPTQ**（Generative Pre-trained Transformer Quantization，Frantar et al. ICLR 2023）的 vendored 实现，命名特征完全对应参考实现：`fasterquant` = GPTQ 的逐列 Cholesky 求逆更新算法；`percdamp` = Hessian 对角阻尼系数（默认 0.01）；`actorder` = 按激活幅度降序重排列；`static_groups` = 预先按 groupsize 静态分组找 scale。

**Hessian 的本质**：`WeightQuantHeissen.add_batch` 累积的 `heissen_matrix` 实际是**输入激活的 Gram 矩阵** `X^T X`（GPTQ 论文里记作 Hessian 的近似，因为重建误差 `||W - Wq||_X^2` 的二阶项 Hessian 就是 `2 * X^T X`）。`fasterquant` 用 `H + damp*I` 后 Cholesky 求逆 `Hinv`，逐列按 `q = argmin_q ||w - q||_H` 的闭式解更新（误差沿 `hinv` 行反向传播到未量化列）。

| 文件 | 职责 |
|------|------|
| `quant.py` | **独立 min-max 量化器**（不注册到 `Quantizers`，与 `quantizer/` 体系平行）：`Quantizer` 类、`find_params`（per-channel min-max + 可选 mse shrink）、`quantize(x, scale, zero, maxq)`。GPTQ 算 int4 时 `maxq=2^bits-1=15`、`sym=True` → `scale=(xmax-xmin)/15`、`zero=(maxq+1)/2=8` |
| `ptq_algo.py` | **GPTQ 算法核心**：`WeightQuantHeissen` 类。`add_batch` 累积 Hessian（按 Linear/Conv2D/Conv1D 适配输入 reshape）；`fasterquant` 跑 Cholesky 逆 + 逐列量化更新；产出 `quantizer.group_scale`/`group_zero`（per-channel × per-group）。NPU 路径把 Hessian 搬 CPU 算 cholesky |
| `ptq_loop.py` | **逐层编排**：`get_llama`（加载并把 `model.model.layers` 留 CPU、其他 to DEV）、`sort_sequent`（按 layer 名分 4 子序列，true_sequential 模式）、`llama_sequential`（GPTQ 主循环，含 Catcher、多层类型预算、PLE/KV 共享、逐层 hook+`fasterquant`、可选 PLE 投影 / lm_head 量化）、`weight_quant_loop`（外层入口）。用模块级全局 `args`（`QuantArgs`）传参 |
| `datautils.py` | **校准数据加载**：`set_seed`、`get_pangupi`（多 json 按 `path:ratio` 拼比，tokenize 拼全文，切 `target_chunks × seqlen` 块）、`get_pt_data`（读预 tokenize 的 `.pt`）、`get_loaders`（调度） |
| `modelutils.py` | **工具**：`find_layers(module, layers=None, name='') -> {name: module}`（递归找 `nn.Conv2d`/`nn.Linear`）+ `DEV` |

### 代码风格观察（dopt_lm 内部）

dopt_lm 是 vendored 代码，不完全遵循 AGENTS.md 的"单函数 50 行 / 参数不超过 5 个"原则。主要超标点：

| 文件:行 | 函数 | 长度/参数数 | 备注 |
|---------|------|------------|------|
| `ptq_algo_heissen/ptq_loop.py:49` | `llama_sequential(model, dataloader, dev)` | **~284 行** | 含 Catcher、多层类型预算、PLE/KV 共享、逐层循环、PLE 投影、lm_head 五大块全塞一个函数，且依赖模块级全局 `args` |
| `opt_funcs/eco_stage/train.py:368` | `train(base_model, config, output_dir, hf_type_save, **argv)` | **~109 行** | 装配 trainer 全流程硬塞一处 |
| `ptq_algo_heissen/ptq_algo.py:60` | `fasterquant(blocksize, percdamp, groupsize, actorder, static_groups)` | **~118 行 / 6 参数** | GPTQ 主算法，dead 列/actorder/cholesky/逐列更新/group_scale 拼装五段混在一起 |
| `quantizer/qmin_max.py:72` | `MinMaxGrid.__init__` | **9 参数** | `(self, father, bit, per_channel, unsigned_quant, quant_id, channel_id, reserve_bit, fp16, **argv)` |
| `ptq_algo_heissen/quant.py:25` | `Quantizer.configure` | **9 参数** | `(self, bits, perchannel, sym, mse, norm, grid, maxshrink, trits)` |
| `ptq_algo_heissen/ptq_loop.py:354` | `weight_quant_loop` | **7 参数** | `(model_name_or_path, data_set, nsamples, seed, seq_len, quant_config, torch_dtype)` |
| `opt_funcs/eco_stage/train.py:304` | `prepare_model` | **8 参数** | `(model, resume_ckpt, quant_config, config, requires_grad, trainable_keys, refresh, device)` |

其他风格问题：
- **模块级全局可变状态**：`ptq_loop.py:351` `args = QuantArgs()` 是模块全局单例，`llama_sequential` 内部直接读写 `args.nsamples`/`args.quant_config`/`args.sym` 等，且 `weight_quant_loop` 也改它——非纯函数，并发/重入会出错。
- **绝对 import 与相对 import 混用**：`do_opt.py` 用相对 `.common_tools.log`，而 `opt_main.py`、`ptq_loop.py`、`qeval.py`、`qmanager.py`、`qmin_max.py`、`modules/*.py` 用绝对 `from dopt_lm.xxx`——要求 `cann_lm/converter/quant` 在 `PYTHONPATH`。
- **异常控制流**：`llama_sequential` 用 `raise ValueError` 在 `Catcher.forward` 里中断来抓输入（标准 GPTQ trick，但脆弱）。
- **拼写出错贯穿全局**：`paser_quant_config`（应为 `parse`）、`trainedable`（应为 `trainable`）、`pipline`（应为 `pipeline`，多处）。

### 模型加载约束（gemma4 多模态阻塞）

GPTQ 循环写死**文本-only ForCausalLM** 结构假设：
- `train.py:290` `model_define` = `AutoModelForCausalLM.from_pretrained(model_path, ...)`——对 `ForConditionalGeneration` 架构（多模态）多半直接找不到类报错；
- `ptq_loop.py:21` `get_llama` + `llama_sequential:54` 取 `model.model.layers` / `model.model.embed_tokens` / `model.model.norm` / `model.lm_head`——多模态模型的文本层在 `model.language_model.layers`（如 gemma4-E4B，42 层），非 `model.model.layers`。

**gemma4 适配现状**：`modules/qembed.py:70` 有 `QGemma4ScaledEmbedding` 类（处理 gemma4 的 `scalar_embed_scale`/`embed_scale`，forward 末尾 `out = out * self.embed_scale`），**但未接进 `do_opt.py:22-27` 的 `visible_type_dict`**（那里只有 `Embedding→QEmbedding`、`"embedding_min_max"→QEmbedding`），属**预留未接线**代码。说明 dopt 作者已预期 gemma4 **文本模型**（带 embed_scale 的 `Gemma4ForCausalLM`）会接入，但未完成 type_dict 分支。

**结论**：dopt 现状只支持文本-only `ForCausalLM`。要跑 gemma4 量化，正路是喂文本-only `Gemma4ForCausalLM` 导出（顶层 ForCausalLM、`model.model.layers`、`embed_tokens.embed_scale`），而非多模态 `Gemma4ForConditionalGeneration` 目录；或改 `model_define`/`get_llama` 抽 `model.language_model` 重挂顶层（dopt 改动，需确认）。

---

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
3. **CPU-only 适配**：converter 侧 GPTQ 跑不了（需 NPU/GPU + 校准）。**`--quant-path` 已作兜底**：离线机跑 dopt GPTQ 产出（`quant_output/model.safetensors` 或 `trained_quant_weight.pth`），converter 加载它只做物化 + swap + 导出（不跑 GPTQ）。dopt `opt_main` 在本机 NPU 上跑 GPTQ 曾静默卡住（step A 接入 converter 待解，见 [端到端流程 · 一步化流程](#一步化流程gptq--导出内存链路待实现)）。
4. **一步化流程未实现**：GPTQ（HF 模型）与导出（cann_lm 模型）仍是两步磁盘桥接。三处改动点（`train.py:530-581` / `swap.py:151-226` / `cann-lm-converter.py:89-99,189`）均未落地，详见 [端到端流程 · 一步化流程](#一步化流程gptq--导出内存链路待实现)。
5. **dopt 函数超规**：`swap.py:151-226 swap_to_quant_linear_from_dopt` 函数 76 行（超 50 行原则），`materialize.py:90-96 materialize_int4_with_scale` 6 参数（超 5 参数原则）。两者建议按 [一步化流程](#一步化流程gptq--导出内存链路待实现) 改动点 #2 与 `Int4Spec` dataclass 拆分收敛，当前未拆。
6. **dopt gemma4 多模态阻塞**：dopt GPTQ 循环写死文本-only `ForCausalLM` 结构（`train.py:290 model_define` 用 `AutoModelForCausalLM`、`ptq_loop.py:21 get_llama` 取 `model.model.layers`）。多模态 `ForConditionalGeneration` 文本层在 `model.language_model.layers`。`QGemma4ScaledEmbedding` 预留未接线（`do_opt.py:visible_type_dict` 无对应分支）。正路是喂文本-only `Gemma4ForCausalLM`，或改 dopt `model_define`/`get_llama` 抽 `model.language_model` 重挂顶层（dopt 改动，需确认）。
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

## 附录 B — 历史调试与路径选型档案

> 本附录记录方案 A 落地前的调试过程与路径选型决策。**这些内容是历史决策档案**，对当前实现不再直接相关，但保留了关键经验教训（uint64 deqScale 不可达、LxFusion 平台限制、stub GE 限制等）供未来类似选型参考。

### B.1 路径 C' 选型与弃用

早期路径 C' 试图复用 CANN V4 整数 int8/int4（`QuantBatchMatmulV4_ND_ND_int8_int4_{bf16,fp16}` 预编译 kernel，tilingKey=24）。

**实测推翻"V4 = MXFP only"结论**：本机 CANN 9.2.0 的预编译 kernel 目录 `opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_nn/quant_batch_matmul_v4/` 下有 **7 个预编译 .o**，其中含整数 int4 组合：

| 预编译 kernel | x1(act) | x2(weight) | out | 方案 | tilingKey |
|---|---|---|---|---|---|
| `ND_ND_int4_int4_{bf16,fp16}` | int4 | int4 | bf16/fp16 | A4W4 | 1040 |
| **`ND_ND_int8_int4_{bf16,fp16}`** | **int8** | **int4** | bf16/fp16 | **A8W4 = W4A8** | **24** |
| `ND_ND_int8_int8_bf16` | int8 | int8 | bf16 | A8W8 = W8A8（已验证跑通）| 529 |
| `ND_NZ_int8_int4_{bf16,fp16}` | int8 | int4(NZ) | bf16/fp16 | W4A8 (x2 NZ 格式) | 276 |

→ V4 同时支持整数 int4（`int8_int4` = W4A8）和 mxfp4，REG_OP 只列了 mxfp 分支但 runtime kernel 覆盖整数 dtype。V3（per-channel 无 group_size）不再必要。

**但路径 C' 经实测不可达 s8s4 正确结果**（详见 B.2），故转方案 A。

| 路径 | 选用 op | 弃用原因 |
|---|---|---|
| ~~C'. V4 整数 int4 per-group~~ | `QuantBatchMatmulV4` | 弃用：eager `npu_quant_matmul` int8/int4 结果错（x2_scale 要求 uint64 deqScale 非 fp32 `q*scale` 语义）；OM 编译经 V4→V3 融合降级（int8/int4 既非 MxA8W4 亦非 A8W8GB）+ V3 无 int8_int4 预编译 kernel → LxFusion 重编 → 910_9362 不支持 |
| ~~A. V3 整数 per-channel~~ | — | 弃用：V4 已有 int8_int4 kernel，V3 per-channel 粒度粗没必要 |
| ~~B. V4 mxfp4~~ | — | 弃用：V4 已支持整数 int4，无需转 mxfp4 |
| ~~D. 维持现状（旧结论）~~ | — | 旧结论"V4=mxfp4 不符、真 NPU OM 编译失败"已被预编译 kernel 实测推翻 |

### B.2 V4 NPU 验证状态（本机 Ascend910_9362 / ascend910_93 族）

`npu_quant_matmul`（= `aclnnQuantMatmulV5` = QuantBatchMatmulV4 eager 接口）在本机 NPU 实跑验证：

| 模式 | 状态 | 详情 |
|---|---|---|
| **W8A8（int8+int8）K-C** | ✅ **跑通，maxdiff≈0** | x1=INT8, x2=INT8, pertoken_scale=fp32(m,), scale=fp32(n,) per-channel → fp16。与 ref `(x1@(q*scale).T)*pertoken` 逐位一致（maxdiff=5.96e-08）。验证脚本 `examples/verify_quant_matmul_v4_w8a8.py`。证明 V4 算子 + scale/pertoken 方法论在本机可用。 |
| **W4A8 int4（A8W4 K-G）** | ⚠️ op 能执行，结果错 | int4pack(`npu_convert_weight_to_int4pack`→`(k,n//8) int32`) + UINT64 scale(`npu_trans_quant_param`) + group[0,0,256] + k=256 对齐 + pertoken(m,1) + yOffset。op 不报错但输出与 ref 偏差大，且 `(ref-y0)/pertoken` **非 per-channel 常量**（x1 相关）→ op 内部 int4 反量化 ≠ fp32 `q*scale`，是 packing 交错排布 + UINT64 fixed-point scale + yOffset 三者语义未对齐。 |

#### A8W4 int4 的关键工具（已验证可用）
- `torch_npu.npu_convert_weight_to_int4pack(int32_kn)` → `(k, n//8) int32`（每 int32 存 8 个 int4，交错排布）。DT_INT4=29。
- `torch.ops.npu.npu_trans_quant_param(fp32_scale_1d)` → int64（UINT64 deqScale，fp32→定点）。
- `torch.ops.npu.npu_quant_matmul`（V5 aclnn）A8W4 K-G 要求：x2Scale=UINT64/INT64、yOffset=FLOAT32(n,) 非空、k 256 对齐、group_sizes=[0,0,256]。

### B.3 OM 编译调试记录（W4A8 qwen3-0.6B 2 层，路径 C' 时期）

在 NPU（Ascend910_9362 / CANN 9.2.0）上对 quant epair 做 `aclgrphBuildModel`（OM 编译），逐步调试：

#### OM 编译步骤与状态

| 步骤 | 状态 | 修复 |
|------|------|------|
| Step 1: EpairModelLoader | ✅ PASS | — |
| Step 2: GetGEGraph | ✅ PASS（经多轮修复） | 见下"修复清单" |
| Step 3: aclgrphBuildModel (TBE) | ❌ FAIL | `LxFusion: Unsupported platform socVersion[Ascend910_9362]` |

#### 修复清单（Step 2 GetGEGraph 从 FAIL → PASS）

1. **重建 binary**：旧 binary 在 `reg_register.h` 加 V4 前构建 → 无 QuantBatchMatmulV4 注册 → C++ parser 创建未知 op → heap corruption（`corrupted size vs. prev_size`）。重建后 Step 2 PASS。
2. **`is_cann_compat` monkey-patch**（`ge_converter.py`）：stub GE 的 `_torchair.check_cann_compat` 拒绝 DynamicQuant（`OperatorFactory find optype DynamicQuant failed`）。Patch `is_cann_compat = lambda *a, **k: ""` 绕过 → torchair 原生 `custom/npu_dynamic_quant.py` converter（用 `ge.DynamicQuant` 全 IR）加载并注册 → ep_json 含完整 IR。
3. **删掉手写 DynamicQuant converter**：之前用 `custom_op("DynamicQuant")`（ir=None → 无 IR → TBE 编译器无法选 kernel）。改为依赖 torchair 原生 converter（`ge.DynamicQuant` 全 IR）。
4. **`_quant_matmul_converter` 用 `ge.QuantBatchMatmulV4`（全 IR）**：`custom_op("QuantBatchMatmulV4")` 只 3 输入 → V4→V3 融合输入映射丢失 → `Assert (scale) != nullptr`。`ge.QuantBatchMatmulV4` 全 IR（10 输入含 None 占位）→ 融合正确映射。
5. **`_strip_none_input_descs`（`epair_utils.py`）**：QBMM 的 OPTIONAL_INPUT bias/y_scale/offset 等 None inputDesc → C++ parser 创建 GE 输入（dtype=None, shape=[]）→ `InferShape: dimensions a(8) and b(2048) must be same`（bias 维度不匹配）。后处理：只对 QBMM strip `dtype=None AND input ref=""` 的 inputDesc + 对应 input 引用（不影响其他 op 如 Cast/Mul）。
6. **dummy bias**（`quant_linear.py`，路径 C' 时期）：QBMM `InferNDimWithBias` 不处理 bias=nullptr → 即使 strip 去掉 bias inputDesc，REG_OP 定义 bias 为 OPTIONAL_INPUT → InferShape 仍处理 → 报错。QuantLinear.forward 传 `dummy_bias = torch.zeros(n_out)` → bias 非空 → InferShape 通过。**方案 A 已废除 dummy bias workaround**（bias 真正可选）。
7. **DynamicQuant fake meta scale shape `(m,)`**（非 `(m,1)`）：V4 的 x1_scale 期望 1D `(m,)`（per-token），2D `(m,1)` 导致 InferShape 不匹配。

#### Step 3 TBE 编译失败根因

```
[WARNING] LxFusionFinalize: "Unsupported platform, socVersion[Ascend910_9362]"
[ERROR] ProcessAllFailedCompileTasks: failed to recompile single op[QuantBatchMatmulV3_0_QuantBatchMatmulV4_*]
```

- V4→V3 融合后，TBE 尝试用 LxFusion 实时编译 V3 kernel → **LxFusion 不支持 Ascend910_9362** → 4/12 个 QBMM 编译失败（8 个用预编译 binary 通过）。
- 这不是代码 bug——**CANN 9.2.0 的 LxFusion 编译器不支持本机芯片 Ascend910_9362**。
- **根因**（路径 C' 视角）：当时 converter 导出的 V4 节点 dtype/format 组合**没命中任何预编译 kernel**（7 个预编译 .o 见 B.1），触发 LxFusion 重编（910_9362 不支持）→ 失败。**若让 dtype 组合命中 `ND_ND_int8_int4` 预编译 kernel（x1=int8, x2=int4, out=fp16/bf16，tilingKey=24），runtime 直接用预编译 binary，跳过 LxFusion**，OM 编译应能过。
- 备选：更新 CANN 到支持 Ascend910_9362 的版本（让 LxFusion 也能用）。

> **方案 A 视角**：方案 A 已绕开 V4 整数 int4 路径，OM 编译不再依赖 V4 预编译 kernel 命中。方案 A 的 OM 编译状态取决于自研 kernel 的注册与 tiling 实现，与本附录记录的 C' 路径调试无关。

#### 关键 ep_json 修复效果对照（路径 C' 时期）

| 修复 | ep_json QBMM inputDesc | InferShape 结果 |
|------|----------------------|----------------|
| 无修复（custom_op） | 3 个（x1,x2,x2_scale），无 IR | TBE: Assert scale!=nullptr |
| ge.QuantBatchMatmulV4（全 IR，不 strip） | 10 个（含 None bias dtype=None, shape=[]） | InferShape: dimensions a(8) b(2048) |
| ge.QuantBatchMatmulV4 + strip None | 4 个（x1,x2,x1_scale,x2_scale） | 同上（bias REG_OP 仍处理） |
| ge.QuantBatchMatmulV4 + strip + dummy bias | 5 个（x1,x2,bias(DT_FLOAT16),x1_scale,x2_scale） | ✅ PASS → TBE: LxFusion unsupported |

### B.4 已验证的工具链（路径 C' 时期）

- ✅ `npu_convert_weight_to_int4pack` → `(k, n//8) int32`（顺序打包，非交错——0x11111111 = 8×int4=1）
- ✅ `npu_trans_quant_param` → int64（但 bit-reinterpret `fp32.view(int32).to(int64)` 更接近正确值）
- ✅ `npu_quant_matmul` eager W8A8 K-C（int8+int8+fp32 scale+fp32 pertoken → fp16）maxdiff=0
- ✅ `npu_quant_matmul` eager A8W4 K-G（int8+int32+UINT64 scale+yOffset+group[0,0,256]）能执行，但 K-G kernel 把 x1 当 int4 → 真 INT8 x1 被错误拆包
- ✅ torchair `custom/npu_dynamic_quant.py` converter（pertoken → `ge.DynamicQuant` 全 IR）
- ✅ `_strip_none_input_descs` 只对 QBMM strip None+空引用的 inputDesc
- ✅ `is_cann_compat` monkey-patch 让 stub GE 不拒绝 DynamicQuant/Bitcast 等
- ❌ TBE LxFusion 不支持 Ascend910_9362（CANN 9.2.0 版本限制）

### B.5 环境与执行约束

- **GPTQ（A 步）需 NPU/GPU**：dopt 依赖 `transformers.AutoModelForCausalLM` + `accelerate` dispatch + `autocast`/`GradScaler`，`DEVICE=cuda/npu`。当前 dev 机虽有 NPU，可跑；纯 CPU 机跑不了 GPTQ（见 [已知问题](#已知问题与遗留) #3）。
- converter 导出（B–E 步）本身 CPU 可跑（torch.export + pack + cpu eager 回归），不依赖 NPU。
- **OM 编译需 NPU + CANN**：`aclgrphBuildModel` 需真 NPU 环境（CANN toolkit + device）。本机 Ascend910_9362 + CANN 9.2.0 的 **LxFusion 不支持 910_9362**，但**预编译 kernel 不走 LxFusion**——只要 converter 导出的算子 dtype 组合命中预编译 kernel（路径 C' 视角）或方案 A 自研 kernel 已注册（方案 A 视角），OM 编译应能过。
- **stub GE 限制**：converter 导出在 stub 环境（venv + torchair stubs）上，`check_cann_compat` 拒绝 DynamicQuant/Bitcast 等 → 需 `is_cann_compat` monkey-patch 绕过（见 B.3 修复 #2）。

## 文档关联

| 文档 | 关联内容 |
|------|---------|
| [cli_design.md](../cli/cli_design.md) | CLI 入口 `--quant-path` / `--quant-embeddings` 参数 / `convert()` 程序化入口 |
| [export_design.md](../export/export_design.md) | epair 产物布局 / host_weights 机制 / C++ `GeWeightLoader` / GE 算子注册两层 |
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题（`swap_embeddings_from_dopt` + `materialize_int8_per_channel`） |
| [converter_design.md](../converter_design.md) | 跨模块调用链总览 / 模型差异总览 |
| [reference_qwen3_5_export.md](../reference_qwen3_5_export.md) | qwen3.5 导出专题（参考） |
