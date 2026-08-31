# dopt_lm 设计文档（vendored GPTQ 框架）

> 抽取自 [quant_design.md](./quant_design.md)。dopt 是华为 vendored 的 GPTQ 风格权重 PTQ 框架，作为独立 CLI 工具在 NPU/GPU 上对 HF 模型做 GPTQ 权重量化，产出 per-layer `group_scale` + GPTQ fp16 权重。
>
> converter 侧量化接入（`materialize` / `swap` / `quant_linear`）见 [quant_design.md](./quant_design.md)。
>
> 状态: 当前采用（部分核对：heissen 拼写已修正） | 最后核对: 2026-08-25

## 目录

- [概述](#概述)
- [使用方法](#使用方法)
- [可设置参数清单](#可设置参数清单)
- [文件树](#文件树)
- [入口与主流程（顶层 5 文件）](#入口与主流程顶层-5-文件)
- [GPTQ PTQ 主路径调用链](#gptq-ptq-主路径调用链)
- [common_tools（基础设施层）](#common_tools基础设施层)
- [modules（量化 module 适配层，fakequant）](#modules量化-module-适配层fakequant)
- [quantizer（量化器算法层）](#quantizer量化器算法层)
- [opt_funcs/eco_stage（训练/PTQ 流程编排层）](#opt_funcseco_stage训练ptq-流程编排层)
- [opt_funcs/weight_quant_ptq/ptq_algo_hessian（GPTQ PTQ 算法实现）](#opt_funcsweight_quant_ptqptq_algo_hessiangptq-ptq-算法实现)
- [代码风格观察（dopt_lm 内部）](#代码风格观察dopt_lm-内部)
- [模型加载约束（gemma4 多模态阻塞）](#模型加载约束gemma4-多模态阻塞)
- [附录 — 历史调试与路径选型档案](#附录-历史调试与路径选型档案)

---

## 概述

`dopt_lm/` 是华为 vendored 的 **GPTQ 风格权重 PTQ 框架**（命名 `dopt` = "do optimize"，Quantization Optimization）。它本身是自洽的、可独立运行的 Python 包（入口 `opt_main.py`，CLI 模式），在 **NPU/GPU** 上对 HuggingFace 模型做 **GPTQ 权重量化**，产出 per-layer `group_scale`（per-channel × per-group 对称，offset=0）+ GPTQ 修正后的 fp16 权重，最终序列化成 `trained_quant_weight.pth`（或 `--hf_type_save` 的 HF 目录）。

**调用关系**：dopt_lm 在主仓代码中**不被 `import`**，而是作为独立 CLI 工具运行，其产物（.pth/safetensors）由 converter 的 `quant/swap.py` 在导出时加载。dopt_lm 内部通过相对导入 `from dopt_lm.xxx import yyy` 自引用（需将 `cann_lm/converter/quant` 放进 `PYTHONPATH`）。

> **拼写说明**：目录名已使用正确拼写 `ptq_algo_hessian`，但类名 `WeightQuantHeissen` 等历史标识仍保留了 `Hessian` 的拼写错误。算法本质是 GPTQ（基于 Hessian 矩阵的 PTQ）。

## 使用方法

dopt 作为独立 CLI 工具运行，产出 GPTQ 量化权重，再由 converter 消费。GPTQ 支持 NPU、CUDA GPU 和 CPU；CPU 模式使用 FP32，适合功能验证和小模型，但大型模型会非常慢且需要较大的主机内存。

### 环境准备

dopt GPTQ 通过 `DEVICE=npu|cuda|cpu` 选择执行设备。NPU 需要 `torch_npu`，CUDA 需要可用的 CUDA PyTorch；CPU 不依赖设备扩展。

```bash
# NPU 环境（系统 Python 3.11.4 + torch_npu）
source /home/developer/Ascend/cann/set_env.sh
# dopt_lm 内部相对 import `from dopt_lm.xxx` → 需 cann_lm/converter/quant 在 PYTHONPATH
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883/cann_lm/converter/quant:${PYTHONPATH:-}

# CPU 示例（若配置中 fp16=true，运行时会自动回退到 FP32）
DEVICE=cpu python -m dopt_lm.opt_main --help
```

> 环境配置详见 [AGENTS.md](../../../../AGENTS.md) "NPU 环境（torch_npu，已验证）"。

### Step 1: 生成 dopt_config 脚手架

首次运行时 `--dopt-config` 指向的文件不存在，dopt 会 `generate_quant_config` 生成脚手架 JSON（所有层 `quant_strategy="float"`）并 `exit()`，用户需手动编辑策略后重跑。

```bash
cd cann_lm/converter/quant
python -m dopt_lm.opt_main \
    --model-path /path/to/hf-model \
    --dopt-config dopt_config.json
# → 生成 dopt_config.json (所有层 float) 后 exit
```

### Step 2: 编辑 dopt_config.json 设置层策略

编辑 `dopt_config.json` 的 `layer_strategy`，为每层选 `quant_strategy`（策略名见 [可设置参数清单](#可设置参数清单) ②）：

| 模块类型 | 推荐 `quant_strategy` | 说明 |
|---------|----------------------|------|
| `nn.Linear`（q/k/v/o/gate/up/down_proj 等） | `Quant_INT8-4_ECO` | W4A8 主路径（默认 group 128）；可用 `;group_size` 按层覆盖 |
| `nn.Linear`（lm_head） | `Quant_INT8-4_ECO` | W4A8 主路径 |
| `nn.Embedding` / `Gemma4TextScaledWordEmbedding` | `Quant_Embed_MinMax` | 按原始类名自动映射 `QEmbedding` / `QGemma4ScaledEmbedding`；embed int8 per-channel |
| 不量化的层 | `float` | 配置解析时从 GPTQ 任务中移除，并在 optimize 阶段保留原始 module |

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

推荐用 `dopt_wrapper.py`（自动生成 `dopt_config.json` 到 `--output-dir`）：

```bash
python cann_lm/converter/quant/dopt_wrapper.py \
    --model-path /path/to/hf-model \
    --optimize-config config.yaml \
    --output-dir /path/to/dopt_out \
    --hf-type-save \
    --quant-bits w4a8
```

或手动 `python -m dopt_lm.opt_main`（需先由 `auto_quant.generate_dopt_config` 生成 `--dopt-config`）：

```bash
python -m dopt_lm.opt_main \
    --model-path /path/to/hf-model \
    --dopt-config dopt_config.json \
    --optimize-config config.yaml \
    --output-dir /path/to/dopt_out \
    --hf_type_save
```

- `--optimize-config config.yaml`：校准配置（`dataset.train_files=calibrate_sets/*.jsonl`, `ptq_samples=1024`, `cutoff_len=512`, `fp16=true`，详见 [可设置参数清单](#可设置参数清单) ③）
- `calibrate_sets/*.jsonl`：校准语料（由 `download_dataset.py` 下载生成，相对 CWD = `cann_lm/converter/quant/`，需在此目录运行）
- `--hf_type_save`：以 HF 格式保存（`save_pretrained`，产出 `model.safetensors`）；不带则存 `trained_quant_weight.pth`
- 设备：`DEVICE=npu|cuda|cpu` 环境变量（dopt_lm `SUPPORTED_DEVICE_TYPES` 含 `cpu`，CPU 走 float32，7B+ 极慢）

### 产物（--output-dir 指定目录）

| 文件 | 说明 |
|------|------|
| `model.safetensors`（~3GB） | GPTQ fp16 权重 + per-layer `quant_op.weight_quantizer.s`（group_scale）+ `{bit, unsigned_quant, offset}` 元信息 |
| `config.json` | HF 模型 config 副本 |
| `config.yaml` | dopt 写的训练回放 |

产物 state_dict keys 示例：
```
model.layers.N.self_attn.q_proj.weight                    (GPTQ fp16)
model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.s   (group_scale, 扁平 [out*in//group])
model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.{bit,unsigned_quant,offset}
model.embed_tokens.weight                                 (fakequant fp)
model.embed_tokens.quant_op.weight_quantizer.s            (per-channel scale)
```

### converter 消费 dopt 产物

dopt 产物由 converter 的 `--quant-path` 加载（`swap.load_dopt_state_dict` 读 `model.safetensors`），物化 int4 + swap `QuantLinear`，详见 [quant_design.md](./quant_design.md) "端到端流程"。

```bash
# converter 环境（CPU venv + torchair stubs，详见 AGENTS.md "运行环境设置（converter）"）
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/hf-model \
    --output-dir /path/to/epair \
    --quant-path /path/to/dopt_out
    # embed 量化自动检测 (dopt 跑了 Quant_Embed_MinMax 则 converter 自动做, 无需额外参数)
```

> dopt GPTQ（Step A，NPU/GPU/CPU）与 converter 导出（Step B-E，CPU）是两步磁盘桥接，一键式流程（subprocess GPTQ→swap→export）详见 [quant_design.md](./quant_design.md) "一键式量化方案选型"。

### converter 进程内调 dopt 的环境验证（方案 A 探索，已放弃）

> 方案 C（subprocess 调 dopt）已实现（`auto_quant.py`），稳定。方案 A（进程内调 dopt）探索后**放弃**（torchair 带 CANN SDK export 链不完整 + fused_ops/真 torch_npu 兼容问题多），保留方案 C。

方案 A 要 converter 主进程同进程跑 torchair（IR 转换）+ torch_npu（dopt GPTQ 执行），需解决 torch 版本对齐 + torchair/torch_npu 模块冲突。

#### 环境对齐

converter venv（torch 2.9.0+cpu + torchair stubs `NO_ASCEND_SDK=1`）和 NPU 系统 Python（torch 2.7.1 + torch_npu 2.7.1）**torch 版本不一致**（2.9.0 vs 2.7.1），不能同进程。

方案 A 要统一 torch 版本：临时 venv 装 torch 2.9.0 + torch_npu 2.9.0 + torchair 带 CANN SDK 编译（`NO_ASCEND_SDK=0`）+ torch_delegate。

实测（临时 venv `/tmp/torch29_venv`）：
- ✅ torch_npu 2.9.0 + 本机 CANN 9.1.0 兼容（`torch.npu.is_available()`=True, count=2）
- ✅ torchair 带 CANN SDK 编译成功（有 `ge_graph_ascend` 模块，stubs 编译缺这个）
- ✅ torchair + torch_npu 同进程共存（加 import hack，见下）

#### torchair + torch_npu 同进程冲突 + hack

torch_npu 2.9.0 import 时注册 `sys.modules['torchair']`（劫持为 `torch_npu.dynamo.torchair`，不完整）。之后 `import torchair` 用劫持版（缺 `npu_format_cast_via_cpu` 等），`import torch_delegate` 失败。

**hack 解**：import torch_npu 后清 `sys.modules['torchair*']`，再 import 真 torchair（site-packages 版，完整）：

```python
import torch_npu
import sys
for k in list(sys.modules):
    if k == "torchair" or k.startswith("torchair."):
        del sys.modules[k]
import torchair      # 真 torchair（带 CANN SDK, 有 ge_graph_ascend）
import torch_delegate
```

实测加 hack 后 `import torchair + torch_delegate + torchair.ge.ge_graph_ascend` 成功。

#### 环境搭建坑

1. torchair `--force-reinstall` 会升 torch 到 2.13.0（whl 依赖 torch，pip 升级）—— 装 torchair 用 `--no-deps` 或后重装 torch 2.9.0
2. 缺依赖：pyyaml / numpy / wheel / setuptools<70（含 `pkg_resources`，新版 setuptools 去掉了）
3. torchair 带 CANN SDK 编译：`build.sh -c` 强制 `NO_ASCEND_SDK=1`（stubs），要手动 cmake + make（`ASCEND_SDK_PATH=${ASCEND_HOME_PATH}` + `bash configure` + `cmake ..` + `make torchair`）

#### 方案 A vs 方案 C

| | 方案 A（进程内调 dopt） | 方案 C（subprocess，当前） |
|---|---|---|
| 环境 | 统一 torch 2.9.0 + torch_npu 2.9.0 + torchair 带 CANN SDK + torch_delegate | converter venv（stubs）+ 系统 Python subprocess（torch_npu） |
| import hack | 需（清 sys.modules torchair） | 不需（分进程） |
| torchair 编译 | 带 CANN SDK（`NO_ASCEND_SDK=0`） | stubs（`NO_ASCEND_SDK=1`） |
| 调 dopt | `from dopt_lm import ...` 函数 | `subprocess` 跑 `python -m dopt_lm.opt_main` CLI |
| 产物传输 | 内存（state_dict 直接传） | 临时盘 IO |
| 状态 | **已放弃**（GPTQ 验证通过，但 export 卡 torchair hcom_broadcast + fused_ops 兼容） | 已实现 + 实测成功（qwen3-0.6B 完整 epair 产物） |

> **方案 A 已放弃**（torchair 带 CANN SDK 编译不完整 `hcom_broadcast` + fused_ops/真 torch_npu 兼容问题多）。当前用方案 C（`auto_quant.py` subprocess，已实测成功）。

## 可设置参数清单

dopt 可设置的量化参数分布在 4 个层面。**用户实际可设的只有 ①②③，④ GPTQ 算法参数大多硬编码不暴露**。

### ① dopt CLI 参数（`opt_main.py`，5 个）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | (必填) | HF 模型路径 |
| `--dopt-config` | (必填) | 层策略配置文件（`dopt_config.json`） |
| `--optimize-config` | `None` | 训练配置（`config.yaml`） |
| `--output-dir` | `None` | 输出目录 |
| `--hf_type_save` | `False` | 以 HF 格式保存（`save_pretrained`） |

### ② `dopt_config.json` 层策略（每层 `quant_strategy` 名）

`quant_strategy` 从 `strategy_dict` 选（6 个预设），支持 `"name;group_size"` 语法覆盖 `group_size`（`get_optimize_cfg`）：

| 策略名 | 用途 | weight | act | 备注 |
|--------|------|--------|-----|------|
| `float` | 不量化 | — | — | |
| `Quant_aigc_ptq` | W8A8 | 8(per-channel min_max) | 8(ema_min_max, unsigned) | |
| `Quant_aigc_qat` | W8A8 | 8(omni_min_max) | 8(omni, unsigned) | QAT 用 |
| `Quant_INT8-4_ECO` | **W4A8（主路径）** | 4(group 128, block 128, omni_group) | 8(dyn_s8, signed) | converter 默认 |
| `Quant_Embed_MinMax` | embed int8 | 8(per-channel min_max) | — | host gather |
| `Quant_Dummy` | 占位 | — | — | |

### ③ `config.yaml` 训练配置（GPTQ 校准）

| 字段　　　　　　　　　　　　 | 默认　　　　　　　　　 | 说明　　　　　　　　　　　　　　　　　　　　　|
| ------------------------------| ------------------------| -----------------------------------------------|
| `dataset.train_files`　　　　| `["calibrate_sets/calibrate_c4.jsonl"]` | 校准语料（download_dataset.py 下载，可替换/多选）|
| `dataset.ptq_samples`　　　　| `1024`　　　　　　　　 | 采样数　　　　　　　　　　　　　　　　　　　　|
| `cutoff_len`　　　　　　　　 | `512`　　　　　　　　　| 截断长度　　　　　　　　　　　　　　　　　　　　|
| `extra_training_config.fp16` | `true`　　　　　　　　 | fp16 训练+存盘（CPU dispatch 自动回落 float32）|

### ④ GPTQ 算法参数（`QuantArgs` / `fasterquant` / `configure`）— 大多硬编码，不暴露 CLI

| 参数　　　　　　　　　　　　　| 默认　　　　　　　　　　　　　　　　　　　　| 位置　　　　　　　　　　　　　　　| 说明　　　　　　　　 |
| -------------------------------| ---------------------------------------------| -----------------------------------| ----------------------|
| `nsamples`　　　　　　　　　　| `256`(QuantArgs) / `1024`(config.yaml 覆盖) | `QuantArgs` / `weight_quant_loop` | 校准采样数　　　　　 |
| `seed`　　　　　　　　　　　　| `1234`　　　　　　　　　　　　　　　　　　　| `QuantArgs`　　　　　　　　　　　 | 随机种子　　　　　　 |
| `seq_len`　　　　　　　　　　 | `2048`　　　　　　　　　　　　　　　　　　　| `weight_quant_loop`　　　　　　　 | 序列长度　　　　　　 |
| `percdamp`　　　　　　　　　　| `0.01`　　　　　　　　　　　　　　　　　　　| `QuantArgs` / `fasterquant`　　　 | Hessian 对角阻尼　　 |
| `act_order`　　　　　　　　　 | `False`　　　　　　　　　　　　　　　　　　 | `QuantArgs` / `fasterquant`　　　 | 按激活幅度降序重排列 |
| `static_groups`　　　　　　　 | `False`　　　　　　　　　　　　　　　　　　 | `QuantArgs` / `fasterquant`　　　 | 预静态分组　　　　　 |
| `true_sequential`　　　　　　 | `False`　　　　　　　　　　　　　　　　　　 | `QuantArgs`　　　　　　　　　　　 | true_sequential 模式 |
| `sym`　　　　　　　　　　　　 | `True`　　　　　　　　　　　　　　　　　　　| `QuantArgs` / `configure`　　　　 | 对称量化　　　　　　 |
| `nearest`　　　　　　　　　　 | `False`　　　　　　　　　　　　　　　　　　 | `QuantArgs`　　　　　　　　　　　 | nearest 量化　　　　 |
| `blocksize`　　　　　　　　　 | `128`　　　　　　　　　　　　　　　　　　　 | `fasterquant` / strategy_dict　　 | GPTQ 块大小　　　　　|
| `groupsize`　　　　　　　　　 | `-1`(默认)/strategy 覆盖　　　　　　　　　　| `fasterquant`　　　　　　　　　　 | group size　　　　　 |
| `bits`　　　　　　　　　　　　| strategy 决定　　　　　　　　　　　　　　　 | `configure`　　　　　　　　　　　 | 量化位宽　　　　　　 |
| `perchannel`　　　　　　　　　| strategy 决定　　　　　　　　　　　　　　　 | `configure`　　　　　　　　　　　 | per-channel　　　　　|
| `mse`　　　　　　　　　　　　 | `False`　　　　　　　　　　　　　　　　　　 | `configure`　　　　　　　　　　　 | mse shrink　　　　　 |
| `norm` / `grid` / `maxshrink` | `2.4` / `100` / `0.8`　　　　　　　　　　　 | `configure`　　　　　　　　　　　 | mse shrink 参数　　　|
| `trits`　　　　　　　　　　　 | `False`　　　　　　　　　　　　　　　　　　 | `configure`　　　　　　　　　　　 | trits　　　　　　　　|

> **结论**：④ 的参数（`percdamp` / `act_order` / `static_groups` / `true_sequential` / `sym` / `norm` / `grid` / `maxshrink` / `trits` 等）**硬编码在 `QuantArgs` / `configure` 默认值，CLI 和配置文件都不暴露**。要改得改源码（`ptq_loop.py` 的 `QuantArgs` 默认值，或 `weight_quant_loop` 调用参数）。`blocksize` 虽在 `strategy_dict` 里，但 `get_optimize_cfg` 只支持 `;group_size` 覆盖，**不支持 `;blocksize` 覆盖**——改 blocksize 要改 `strategy_dict`。

## 文件树

```
dopt_lm/
├── __init__.py                          版权头
├── builder.py                           Quantizers/OPT_OPS Register + init_builder
├── do_opt.py                             核心调度：optimize_model / paser_quant_config / set_*_state
├── opt_main.py                          CLI 入口
├── utils.py                              replace_module_by_names
├── common_tools/
│   ├── __init__.py
│   ├── log.py                            Logger / FileTxtLogger / visualize_dist
│   └── register.py                       Register 装饰器 + file_import
├── modules/
│   ├── __init__.py
│   ├── qembed.py                         QEmbedding / QGemma4ScaledEmbedding
│   ├── qlinear.py                        QLinear (fakequant)
│   └── utils.py                          get_quantizer_cfg
├── opt_funcs/
│   ├── __init__.py
│   ├── eco_stage/
│   │   ├── __init__.py
│   │   └── train.py                      weight_quant_pipline / ptq_quant_pipline / train / KD trainer
│   └── weight_quant_ptq/
│       ├── __init__.py
│       └── ptq_algo_hessian/
│           ├── __init__.py             (0)
│           ├── datautils.py            (109 行)  get_json_data / get_pt_data / get_loaders
│           ├── modelutils.py           (25 行)   find_layers / DEV
│           ├── ptq_algo.py             (185 行)  WeightQuantHeissen (GPTQ 核心)
│           ├── ptq_loop.py             (375 行)  llama_sequential / weight_quant_loop
│           └── quant.py                (134 行) Quantizer (GPTQ 专用 min-max)
├── quantizer/
│   ├── __init__.py
│   ├── qeval.py                          autograd 量化函数 + DynEvalGridS8
│   ├── qmanager.py                       QClass 容器
│   ├── qmin_max.py                       MinMaxGrid / Group / Omni / OmniGroup
│   ├── qutils.py                         clip range / min-max / NF 表 / fakequant
│   └── strategy.py                       strategy_dict + get_optimize_cfg
```

## 入口与主流程（顶层 5 文件）

| 文件　　　　　　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| --------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `opt_main.py`　　　　　　　　　| CLI 入口。`parse_args()` 接 `--model-path / --dopt-config / --optimize-config / --output-dir / --hf_type_save`。两步：① 若 `--dopt-config` 不存在 → `generate_quant_config` 生成脚手架 JSON（所有层 `quant_strategy="float"`）并 `exit()`，用户需手动编辑策略后重跑；② 否则组装 `optimize_info` 调 `llm_quant_pipline` → `train.weight_quant_pipline`　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `opt_funcs/eco_stage/train.py` | **最大、最核心的 orchestrator**。`weight_quant_pipline`：决策入口——`config["kd"]["enable"]` 走 `train()`（KD 训练），否则走 `ptq_quant_pipline()`（纯 GPTQ PTQ，无微调）。两者最终都产出 `trained_quant_weight.pth`。还含 `generate_config`/`prepare_model`/`model_define`/`seed_all`/`DoptTrainerCustomization` |
| `do_opt.py`　　　　　　　　　　| **核心调度层**（被多处 import）。模块加载时执行 `init_builder()`，定义唯一映射表 `visible_type_dict = {nn.Embedding: QEmbedding, nn.Linear: QLinear, "Gemma4TextScaledWordEmbedding": QGemma4ScaledEmbedding}`。导出 `optimize_model(model, config_file)`（按 `layer_strategy` 把目标 module wrap 成 QLinear/QEmbedding）、`paser_quant_config(model, config_file)`（GPTQ 专用，只加载 `layer_strategy` 并对命中层做 `parse_layer_config`，返回 dict 供 `ptq_loop.py` 取用）、`set_quant_state`/`set_calibrate_state`（切换每个 `quant_op` 的 fakequant 开关）、`generate_config_file`（遍历 `named_modules()` 生成脚手架 JSON） |
| `builder.py`　　　　　　　　　 | 注册中心初始化。定义两个全局 `Register`：`Quantizers = Register("Quantizers")`（量化器算法注册表）、`OPT_OPS = Register("OptimizedOps")`（优化算子注册表，当前未使用）。`init_builder(quantizer_path=None)` 递归 import `modules/` 与 `quantizer/` 目录触发注册　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| `utils.py`　　　　　　　　　　 | Stage1 minimal：仅 `replace_module_by_names(model, modules_to_replace) -> model`（递归 `add_module` 替换子模块）　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

## GPTQ PTQ 主路径调用链

```
ptq_quant_pipline(base_model, config, output_dir, hf_type_save, quant_method='gptq')
  │
  ├─ [1] from ptq_algo_hessian.ptq_loop import weight_quant_loop
  │      weight_q_quantizer, weight_q_weight = weight_quant_loop(
  │          base_model, data_set=train_file, nsamples, seed=1234,
  │          seq_len, quant_config=config_path, torch_dtype)
  │         │
  │         ├─ get_llama(model, seq_len, torch_dtype)            # 加载到 NPU/GPU
  │         ├─ args.quant_config = paser_quant_config(model, quant_config)  # 解析 layer_strategy
  │         ├─ dataloader, _ = get_loaders(data_set, nsamples, seed, model, seqlen)
  │         │      ├─ .pt → get_pt_data
  │         │      └─ else → get_json_data (json/jsonl/wikitext2)
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

## common_tools（基础设施层）

| 文件 | 职责 |
|------|------|
| `log.py` | `Logger`（静态方法 `info/debug/warning/error`，按 `LOCAL_RANK==0` 过滤分布式日志）+ `add_file_handler`/`remove_file_handler`；`FileTxtLogger`（按 `layer_name` 记录每层量化分布的 txt 文件）；`set_record_mode(model, output_path, name_list=None)`（为每层注册 recoder）；`visualize_dist(ckpt)`（打印 state_dict 里 `*.quantizer.s` / `*.bit` / `*.offset` 分布） |
| `register.py` | `Register` 类：装饰器 `@register_func(name)` 注册 callable；`file_import(file_path, recursive, skip_libs_set)` 把目录/文件按 sys.path 推导成 `pkg.module` 名并 `importlib.import_module`（注册触发器）；`get(key)`/`__getitem__`/`__contains__`/`get_keys()` |

## modules（量化 module 适配层，fakequant）

dopt 的**推理期 fakequant 实现**，不是导出用算子。把 `nn.Linear`/`nn.Embedding` 包成 `QLinear`/`QEmbedding`/`QGemma4ScaledEmbedding`，内含 `quant_op`（`QClass`）做 fakequant 前向（量化后反量化回 fp，再 `F.linear`/`F.embedding`）。

| 文件 | 职责 |
|------|------|
| `qlinear.py` | `QLinear.__init__(m, optimize_cfg, layer_name=None, **argv)`：解析 weight/input/output cfg，建 `QClass`，`to(weight_device)`。`forward(x)`：`weight, x = self.quant_op(weight, x)` → `F.linear(x, weight, None)` → 可选 `output_quantizer(out)` → `+ bias`。支持 `dyn` 动态输入量化（解出 `s_denormalized` 反缩放）。`set_quant_state(weight, input)` 透传 `quant_op` |
| `qembed.py` | `QEmbedding(nn.Module).__init__(m: nn.Embedding, optimize_cfg, layer_name=None, **argv)`：强制 `per_channel=True`，建 `QClass(m, weight_cfg, input_cfg={})`；`tie_lmhead` 控制是否共享 `lm_head` 的 weight（不 clone）；`npu_simulator=None`（stub）。`forward(x)`：`weight, x = self.quant_op(weight, x)` → `F.embedding(...)`。`QGemma4ScaledEmbedding`：gemma4 专用，多 `embed_scale` buffer，forward 末尾 `out = out * self.embed_scale`；已通过原始类名接入 `visible_type_dict`。 |
| `utils.py` | `get_quantizer_cfg(cfg) -> (weight_cfg, input_cfg, output_cfg)`（dict 三段拆分） |

## quantizer（量化器算法层）

5 个文件分工——`strategy.py`（策略字典）/`qmanager.py`（QClass 容器）/`qeval.py`（autograd 量化函数）/`qmin_max.py`（min-max/group/omni 量化器实现，`Register` 注册）/`qutils.py`（clip range、min-max 统计、NF2/3/4 查找表等工具）。是典型的"注册表 + 字符串策略名"模式（非 GoF 策略类）。

| 文件 | 角色 | 关键导出 |
|------|------|---------|
| `strategy.py` | **策略字典** `strategy_dict` + `get_optimize_cfg()`。6 个预设策略（`Quant_aigc_ptq/qat`、`Quant_INT8-4_ECO`、`Quant_Embed_MinMax`、`Quant_Dummy`、`float`），每项含 `weight`/`input`/`output` 子 cfg，子 cfg 里 `weight_algo`/`input_algo` 字符串指向 `Quantizers` Register 名。`get_optimize_cfg` 支持 `"name;group_size"` 字符串拼接语法覆盖 group_size；未注册策略 raise |
| `qmanager.py` | **容器** `QClass(nn.Module)`：把 weight/input/output 三种量化器 cfg 通过 `Quantizers.get(algo)(m, **cfg)` 工厂实例化聚合。`__init__(m, weight_quantizer_cfg=None, input_quantizer_cfg=None, output_quantizer_cfg=None)`；`set_quant_state(weight, input)` / `set_calibrate_state(state)` / `forward(weight, x) -> (weight, x)` / `is_empty() -> bool` |
| `qeval.py` | **autograd 量化原语** `QuantizerFuncIntOffset`（整数 offset 路径：`round(x/s+offset)` clamp 再 `(q-offset)*s`，STE backward）/ `QuantizerFuncFpOffset`（浮点 offset 路径：`round((x+offset)/s)`）/ `DynIntSymQuantizer`（动态 per-token 对称 int8，amax/127，返回 `(normalized, scale)`）。`@Quantizers.register_func("dyn_s8") class DynEvalGridS8`（动态 s8 包装） |
| `qmin_max.py` | **min-max 系列量化器实现**（4 个 `Register` 注册）：`MinMaxGrid`（`min_max`/`minmax`，基础 per-channel min-max，calibrate 阶段算 scale/offset，eval 用缓存）、`GroupMinMaxGridWeight`（`group_min_max`/`group_minmax`，per-group 维度 reshape 后调父类）、`OmniMinMaxGrid`（`omni_min_max`/`omni_minmax`，LSQ 可学习 `quant_alpha` + sigmoid 自适应 scale）、`OmniGroupMinMaxGridWeight`（`omni_group_min_max`，omni 的 group 版） |
| `qutils.py` | **纯工具**：常量 `MIN_SCALE_VALUE=1e-5`、`NF2/NF3/NF4`（normal_float 查找表，×127 量化）、`get_clip_range(bit, unsigned_quant, keep_neg_max=True)`（含 2.5bit 特例）、`RoundWithGrad`（round 的 STE）、`get_min_max`/`get_abs_max_per_channel`/`get_min_max_per_channel`、`do_quantize`/`do_antiquantize`/`fakequant_forward`、`get_grad_max`/`get_grad_min`、`fake_nf` |

**关键分工**：`strategy.py` 提供配置模板 → `qmanager.QClass` 按 cfg 里 `algo` 名查 `Quantizers` Register → `qmin_max.py`/`qeval.py` 提供注册的具体量化器 → `qutils.py` 提供底层 clip/min-max/round 工具。

## opt_funcs/eco_stage（训练/PTQ 流程编排层）

`train.py`（581 行，最大文件）是真正的 orchestrator——决策走 KD 训练还是 GPTQ PTQ，装配 trainer、数据集、teacher 模型、save 权重。关键导出：

| 函数/类 | 作用 |
|---------|------|
| `check_is_trainable(name, trainable_keys) -> bool` | 判断参数是否可训练 |
| `get_processed_dataset(dataset_files, tokenizer, dataset_config, train_samples=None) -> Dataset` | 支持 wikitext2/json/jsonl；`text_concat` 模式拼接全文再切 `cutoff_len` |
| `custom_model_generation(model, no_split_module_classes=None) -> model` | accelerate 多卡 device_map 平衡 |
| `class DoptTrainerCustomization(nn.Module)` | KD trainer；`train()`→`training_loop_block()`（AdamW + cosine/linear scheduler + autocast + accelerate prepare）；`compute_loss(model, inputs, return_outputs=False)`（MSE hidden-states / KL-div logits） |
| `ce_loss(student_logits, teacher_logits, topk=-1, reduction="batchmean")` | topk 截断的 KL 散度 |
| `seed_all(seed_value=1234, deterministic=False) | 全栈随机种子 |
| `model_define(model_path, torch_dtype=torch.float16) -> AutoModelForCausalLM` | `from_pretrained(trust_remote_code=True)` |
| `generate_config(model_name_or_path, config_path)` | `model_define` + `generate_config_file` |
| `prepare_model(model, resume_ckpt=None, quant_config=None, config=None, requires_grad=False, trainable_keys=None, refresh=False, device=None) -> model` | 加载+`optimize_model`+可选 resume_ckpt+trainable_keys 冻结策略 |
| `print_trainable_parameters(model)` | 统计可训练比例 |
| `train(base_model, config, output_dir=None, hf_type_save=False, **argv)` | **KD 训练主入口**（`gptq_finetune`/`resume_finetune` 模式加载预训练 quant 权重，DoptTrainerCustomization 训练，存 `trained_quant_weight.pth`） |
| `weight_q_params_adapter_into_dopt(dopt_model, weight_q_quantizers, tie_lmhead)` | 把 GPTQ 产出的 `group_scale` 注入每个 `QLinear.quant_op.weight_quantizer.s`，`offset=zero_like(...)`（置零→对称），并对 weight 跑一遍 fakequant 写回 |
| `weight_quant_pipline(model_path, config_yaml, dopt_config=None, output_dir=None, hf_type_save=False, **argv)` | **决策入口**：`kd.enable` → `train()`；否则 → `ptq_quant_pipline()` |
| `ptq_quant_pipline(base_model, config, output_dir=None, hf_type_save=False, quant_method='gptq', **argv)` | 跑 `weight_quant_loop` → `prepare_model`→`optimize_model` → 注入 scale → 关 calibrate → save |

**KD 训练说明**（`train()`）：实现的是 **知识蒸馏（KD）微调**，用于 GPTQ 之后的精度恢复（不是从零训练，也不是 GPTQ 本身）：
- **学生模型**：先 `optimize_model` 把 `nn.Linear` 换成 `QLinear`（fakequant），加载 GPTQ 产出的 `trained_quant_weight.pth`（`gptq_finetune` 模式）或其二次微调版（`resume_finetune` 模式），按 `trainable_keys`（如 `quant_alpha`）冻结大部分参数只训可学习量化参数（LSQ `quant_alpha`）/norm。
- **教师模型**：`prepare_model`（`requires_grad=False`）加载同架构 FP 模型，`eval()`。
- **损失**：`compute_loss` 支持 `mse`（对学生/教师 `hidden_states[feature_layers]` 做 MSELoss）+ `kl_div`（对 logits 做 topk 截断 KL 散度 `ce_loss`）。
- **训练栈**：`AdamW` + cosine/linear warmup scheduler + autocast + accelerate `prepare`/device_map；存 `trained_quant_weight.pth`。

## opt_funcs/weight_quant_ptq/ptq_algo_hessian（GPTQ PTQ 算法实现）

这是 **GPTQ**（Generative Pre-trained Transformer Quantization，Frantar et al. ICLR 2023）的 vendored 实现，命名特征完全对应参考实现：`fasterquant` = GPTQ 的逐列 Cholesky 求逆更新算法；`percdamp` = Hessian 对角阻尼系数（默认 0.01）；`actorder` = 按激活幅度降序重排列；`static_groups` = 预先按 groupsize 静态分组找 scale。

**Hessian 的本质**：`WeightQuantHessian.add_batch` 累积的 `hessian_matrix` 实际是**输入激活的 Gram 矩阵** `X^T X`（GPTQ 论文里记作 Hessian 的近似，因为重建误差 `||W - Wq||_X^2` 的二阶项 Hessian 就是 `2 * X^T X`）。`fasterquant` 用 `H + damp*I` 后 Cholesky 求逆 `Hinv`，逐列按 `q = argmin_q ||w - q||_H` 的闭式解更新（误差沿 `hinv` 行反向传播到未量化列）。

| 文件 | 职责 |
|------|------|
| `quant.py` | **独立 min-max 量化器**（不注册到 `Quantizers`，与 `quantizer/` 体系平行）：`Quantizer` 类、`find_params`（per-channel min-max + 可选 mse shrink）、`quantize(x, scale, zero, maxq)`。GPTQ 算 int4 时 `maxq=2^bits-1=15`、`sym=True` → `scale=(xmax-xmin)/15`、`zero=(maxq+1)/2=8` |
| `ptq_algo.py` | **GPTQ 算法核心**：`WeightQuantHeissen` 类。`add_batch` 累积 Hessian（按 Linear/Conv2D/Conv1D 适配输入 reshape）；`fasterquant` 跑 Cholesky 逆 + 逐列量化更新；产出 `quantizer.group_scale`/`group_zero`（per-channel × per-group）。NPU 路径把 Hessian 搬 CPU 算 cholesky |
| `ptq_loop.py` | **逐层编排**：`get_llama`（加载并把 `model.model.layers` 留 CPU、其他 to DEV）、`sort_sequent`（按 layer 名分 4 子序列，true_sequential 模式）、`llama_sequential`（GPTQ 主循环，含 Catcher、多层类型预算、PLE/KV 共享、逐层 hook+`fasterquant`、可选 PLE 投影 / lm_head 量化）、`weight_quant_loop`（外层入口）。用模块级全局 `args`（`QuantArgs`）传参 |
| `datautils.py` | **校准数据加载**：`set_seed`、`get_json_data`（多 json 按 `path:ratio` 拼比，tokenize 拼全文，切 `target_chunks × seqlen` 块）、`get_pt_data`（读预 tokenize 的 `.pt`）、`get_loaders`（调度） |
| `modelutils.py` | **工具**：`find_layers(module, layers=None, name='') -> {name: module}`（递归找 `nn.Conv2d`/`nn.Linear`）+ `DEV` |

## 代码风格观察（dopt_lm 内部）

dopt_lm 是 vendored 代码，不完全遵循 AGENTS.md 的"单函数 50 行 / 参数不超过 5 个"原则。主要超标点：

| 文件 | 函数 | 参数数 | 备注 |
|---------|------|------------|------|
| `ptq_algo_hessian/ptq_loop.py` 的 `llama_sequential` | `llama_sequential(model, dataloader, dev)` | 3 | 含 Catcher、多层类型预算、PLE/KV 共享、逐层循环、PLE 投影、lm_head 五大块全塞一个函数（远超 50 行原则），且依赖模块级全局 `args` |
| `opt_funcs/eco_stage/train.py` 的 `train` | `train(base_model, config, output_dir, hf_type_save, **argv)` | 5 | 装配 trainer 全流程硬塞一处（远超 50 行原则） |
| `ptq_algo_hessian/ptq_algo.py` 的 `fasterquant` | `fasterquant(blocksize, percdamp, groupsize, actorder, static_groups)` | 6 | GPTQ 主算法，dead 列/actorder/cholesky/逐列更新/group_scale 拼装五段混在一起（远超 50 行原则） |
| `quantizer/qmin_max.py` 的 `MinMaxGrid.__init__` | `MinMaxGrid.__init__` | 9 | `(self, father, bit, per_channel, unsigned_quant, quant_id, channel_id, reserve_bit, fp16, **argv)` |
| `ptq_algo_hessian/quant.py` 的 `Quantizer.configure` | `Quantizer.configure` | 9 | `(self, bits, perchannel, sym, mse, norm, grid, maxshrink, trits)` |
| `ptq_algo_hessian/ptq_loop.py` 的 `weight_quant_loop` | `weight_quant_loop` | 7 | `(model_name_or_path, data_set, nsamples, seed, seq_len, quant_config, torch_dtype)` |
| `opt_funcs/eco_stage/train.py` 的 `prepare_model` | `prepare_model` | 8 | `(model, resume_ckpt, quant_config, config, requires_grad, trainable_keys, refresh, device)` |

其他风格问题：
- **模块级全局可变状态**：`ptq_loop.py` 的 `args = QuantArgs()` 是模块全局单例，`llama_sequential` 内部直接读写 `args.nsamples`/`args.quant_config`/`args.sym` 等，且 `weight_quant_loop` 也改它——非纯函数，并发/重入会出错。
- **绝对 import 与相对 import 混用**：`do_opt.py` 用相对 `.common_tools.log`，而 `opt_main.py`、`ptq_loop.py`、`qeval.py`、`qmanager.py`、`qmin_max.py`、`modules/*.py` 用绝对 `from dopt_lm.xxx`——要求 `cann_lm/converter/quant` 在 `PYTHONPATH`。
- **异常控制流**：`llama_sequential` 用 `raise ValueError` 在 `Catcher.forward` 里中断来抓输入（标准 GPTQ trick，但脆弱）。
- **拼写出错贯穿全局**：`paser_quant_config`（应为 `parse`）、`trainedable`（应为 `trainable`）、`pipline`（应为 `pipeline`，多处）。

## 模型加载约束（gemma4 多模态阻塞）

GPTQ 循环写死**文本-only ForCausalLM** 结构假设：
- `train.py` 的 `model_define` = `AutoModelForCausalLM.from_pretrained(model_path, ...)`——对 `ForConditionalGeneration` 架构（多模态）多半直接找不到类报错；
- `ptq_loop.py` 的 `get_llama` + `llama_sequential` 取 `model.model.layers` / `model.model.embed_tokens` / `model.model.norm` / `model.lm_head`——多模态模型的文本层在 `model.language_model.layers`（如 gemma4-E4B，42 层），非 `model.model.layers`。

**gemma4 适配现状**：`modules/qembed.py` 的 `QGemma4ScaledEmbedding` 处理 gemma4 的 `scalar_embed_scale`/`embed_scale`，forward 末尾执行 `out = out * self.embed_scale`；`do_opt.py` 已通过原始类名 `Gemma4TextScaledWordEmbedding` 自动映射到该 wrapper。

**结论**：dopt 现状只支持文本-only `ForCausalLM`。要跑 gemma4 量化，正路是喂文本-only `Gemma4ForCausalLM` 导出（顶层 ForCausalLM、`model.model.layers`、`embed_tokens.embed_scale`），而非多模态 `Gemma4ForConditionalGeneration` 目录；或改 `model_define`/`get_llama` 抽 `model.language_model` 重挂顶层（dopt 改动，需确认）。

## 附录 — 历史调试与路径选型档案

> 本附录记录方案 A 落地前的调试过程与路径选型决策。**这些内容是历史决策档案**，对当前实现不再直接相关，但保留了关键经验教训（uint64 deqScale 不可达、LxFusion 平台限制、stub GE 限制等）供未来类似选型参考。

### 1. 路径 C' 选型与弃用

早期路径 C' 试图复用 CANN V4 整数 int8/int4（`QuantBatchMatmulV4_ND_ND_int8_int4_{bf16,fp16}` 预编译 kernel，tilingKey=24）。

**实测推翻"V4 = MXFP only"结论**：本机 CANN 9.2.0 的预编译 kernel 目录 `opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_nn/quant_batch_matmul_v4/` 下有 **7 个预编译 .o**，其中含整数 int4 组合：

| 预编译 kernel | x1(act) | x2(weight) | out | 方案 | tilingKey |
|---|---|---|---|---|---|
| `ND_ND_int4_int4_{bf16,fp16}` | int4 | int4 | bf16/fp16 | A4W4 | 1040 |
| **`ND_ND_int8_int4_{bf16,fp16}`** | **int8** | **int4** | bf16/fp16 | **A8W4 = W4A8** | **24** |
| `ND_ND_int8_int8_bf16` | int8 | int8 | bf16 | A8W8 = W8A8（已验证跑通）| 529 |
| `ND_NZ_int8_int4_{bf16,fp16}` | int8 | int4(NZ) | bf16/fp16 | W4A8 (x2 NZ 格式) | 276 |

→ V4 同时支持整数 int4（`int8_int4` = W4A8）和 mxfp4，REG_OP 只列了 mxfp 分支但 runtime kernel 覆盖整数 dtype。V3（per-channel 无 group_size）不再必要。

**但路径 C' 经实测不可达 s8s4 正确结果**（详见下节），故转方案 A。

| 路径 | 选用 op | 弃用原因 |
|---|---|---|
| ~~C'. V4 整数 int4 per-group~~ | `QuantBatchMatmulV4` | 弃用：eager `npu_quant_matmul` int8/int4 结果错（x2_scale 要求 uint64 deqScale 非 fp32 `q*scale` 语义）；OM 编译经 V4→V3 融合降级（int8/int4 既非 MxA8W4 亦非 A8W8GB）+ V3 无 int8_int4 预编译 kernel → LxFusion 重编 → 910_9362 不支持 |
| ~~A. V3 整数 per-channel~~ | — | 弃用：V4 已有 int8_int4 kernel，V3 per-channel 粒度粗没必要 |
| ~~B. V4 mxfp4~~ | — | 弃用：V4 已支持整数 int4，无需转 mxfp4 |
| ~~D. 维持现状（旧结论）~~ | — | 旧结论"V4=mxfp4 不符、真 NPU OM 编译失败"已被预编译 kernel 实测推翻 |

### 2. V4 NPU 验证状态（本机 Ascend910_9362 / ascend910_93 族）

`npu_quant_matmul`（= `aclnnQuantMatmulV5` = QuantBatchMatmulV4 eager 接口）在本机 NPU 实跑验证：

| 模式 | 状态 | 详情 |
|---|---|---|
| **W8A8（int8+int8）K-C** | ✅ **跑通，maxdiff≈0** | x1=INT8, x2=INT8, pertoken_scale=fp32(m,), scale=fp32(n,) per-channel → fp16。与 ref `(x1@(q*scale).T)*pertoken` 逐位一致（maxdiff=5.96e-08）。验证脚本 `examples/verify_quant_matmul_v4_w8a8.py`。证明 V4 算子 + scale/pertoken 方法论在本机可用。 |
| **W4A8 int4（A8W4 K-G）** | ⚠️ op 能执行，结果错 | int4pack(`npu_convert_weight_to_int4pack`→`(k,n//8) int32`) + UINT64 scale(`npu_trans_quant_param`) + group[0,0,256] + k=256 对齐 + pertoken(m,1) + yOffset。op 不报错但输出与 ref 偏差大，且 `(ref-y0)/pertoken` **非 per-channel 常量**（x1 相关）→ op 内部 int4 反量化 ≠ fp32 `q*scale`，是 packing 交错排布 + UINT64 fixed-point scale + yOffset 三者语义未对齐。 |

#### A8W4 int4 的关键工具（已验证可用）
- `torch_npu.npu_convert_weight_to_int4pack(int32_kn)` → `(k, n//8) int32`（每 int32 存 8 个 int4，交错排布）。DT_INT4=29。
- `torch.ops.npu.npu_trans_quant_param(fp32_scale_1d)` → int64（UINT64 deqScale，fp32→定点）。
- `torch.ops.npu.npu_quant_matmul`（V5 aclnn）A8W4 K-G 要求：x2Scale=UINT64/INT64、yOffset=FLOAT32(n,) 非空、k 256 对齐、group_sizes=[0,0,256]。

### 3. OM 编译调试记录（W4A8 qwen3-0.6B 2 层，路径 C' 时期）

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
- **根因**（路径 C' 视角）：当时 converter 导出的 V4 节点 dtype/format 组合**没命中任何预编译 kernel**（7 个预编译 .o 见上），触发 LxFusion 重编（910_9362 不支持）→ 失败。**若让 dtype 组合命中 `ND_ND_int8_int4` 预编译 kernel（x1=int8, x2=int4, out=fp16/bf16，tilingKey=24），runtime 直接用预编译 binary，跳过 LxFusion**，OM 编译应能过。
- 备选：更新 CANN 到支持 Ascend910_9362 的版本（让 LxFusion 也能用）。

> **方案 A 视角**：方案 A 已绕开 V4 整数 int4 路径，OM 编译不再依赖 V4 预编译 kernel 命中。方案 A 的 OM 编译状态取决于自研 kernel 的注册与 tiling 实现，与本附录记录的 C' 路径调试无关。

#### 关键 ep_json 修复效果对照（路径 C' 时期）

| 修复 | ep_json QBMM inputDesc | InferShape 结果 |
|------|----------------------|----------------|
| 无修复（custom_op） | 3 个（x1,x2,x2_scale），无 IR | TBE: Assert scale!=nullptr |
| ge.QuantBatchMatmulV4（全 IR，不 strip） | 10 个（含 None bias dtype=None, shape=[]） | InferShape: dimensions a(8) b(2048) |
| ge.QuantBatchMatmulV4 + strip None | 4 个（x1,x2,x1_scale,x2_scale） | 同上（bias REG_OP 仍处理） |
| ge.QuantBatchMatmulV4 + strip + dummy bias | 5 个（x1,x2,bias(DT_FLOAT16),x1_scale,x2_scale） | ✅ PASS → TBE: LxFusion unsupported |

### 4. 已验证的工具链（路径 C' 时期）

- ✅ `npu_convert_weight_to_int4pack` → `(k, n//8) int32`（顺序打包，非交错——0x11111111 = 8×int4=1）
- ✅ `npu_trans_quant_param` → int64（但 bit-reinterpret `fp32.view(int32).to(int64)` 更接近正确值）
- ✅ `npu_quant_matmul` eager W8A8 K-C（int8+int8+fp32 scale+fp32 pertoken → fp16）maxdiff=0
- ✅ `npu_quant_matmul` eager A8W4 K-G（int8+int32+UINT64 scale+yOffset+group[0,0,256]）能执行，但 K-G kernel 把 x1 当 int4 → 真 INT8 x1 被错误拆包
- ✅ torchair `custom/npu_dynamic_quant.py` converter（pertoken → `ge.DynamicQuant` 全 IR）
- ✅ `_strip_none_input_descs` 只对 QBMM strip None+空引用的 inputDesc
- ✅ `is_cann_compat` monkey-patch 让 stub GE 不拒绝 DynamicQuant/Bitcast 等
- ❌ TBE LxFusion 不支持 Ascend910_9362（CANN 9.2.0 版本限制）

### 5. 环境与执行约束

- **GPTQ（A 步）支持 NPU/GPU/CPU**：通过 `DEVICE=npu|cuda|cpu` 选择。CPU 模式自动使用 FP32，不调用设备同步或显存缓存 API；大型模型在 CPU 上耗时和内存开销会明显增加。
- converter 导出（B–E 步）本身 CPU 可跑（torch.export + pack + cpu eager 回归），不依赖 NPU。
- **OM 编译需 NPU + CANN**：`aclgrphBuildModel` 需真 NPU 环境（CANN toolkit + device）。本机 Ascend910_9362 + CANN 9.2.0 的 **LxFusion 不支持 910_9362**，但**预编译 kernel 不走 LxFusion**——只要 converter 导出的算子 dtype 组合命中预编译 kernel（路径 C' 视角）或方案 A 自研 kernel 已注册（方案 A 视角），OM 编译应能过。
- **stub GE 限制**：converter 导出在 stub 环境（venv + torchair stubs）上，`check_cann_compat` 拒绝 DynamicQuant/Bitcast 等 → 需 `is_cann_compat` monkey-patch 绕过（见上"修复清单" #2）。

## 文档关联

| 文档 | 关联内容 |
|------|---------|
| [quant_design.md](./quant_design.md) | converter 侧量化接入（`materialize` / `swap` / `quant_linear` / 端到端两步流程 / 方案 A 算子 / 已知问题） |
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题 |
| [cli_design.md](../cli/cli_design.md) | CLI `--quant-path` / `--quant-method` / `--quant-bits` 参数 |
