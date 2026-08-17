# 量化模块索引

> `cann_lm/converter/quant/` — W4A8 s8s4 量化接入：dopt GPTQ 权重量化 + converter 物化 int4 + `quant_matmul_s8s4` 算子导出。

## 模块文档

| 文档 | 内容 |
|------|------|
| [quant_design.md](./quant_design.md) | W4A8 s8s4 方案 / quant 包架构 / 量化策略 / 端到端两步流程 / `materialize` / `quant_linear` / `swap` / 方案 A 自研 `quant_matmul_s8s4` 算子规格 / C++ 注册 / CANN V3/V4 源码分析附录 |
| [dopt_design.md](./dopt_design.md) | dopt_lm vendored GPTQ 框架（文件树 / 入口 / GPTQ 调用链 / 可设置参数清单 / 历史调试档案） |
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题（`materialize_int8_per_channel` + `swap_embeddings_from_dopt`） |

## 代码文件

### converter 侧（`cann_lm/converter/quant/`）

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包门面，re-export 6 个公共符号 |
| `materialize.py` | 权重物化纯函数：`pack_int4` / `materialize_int4` / `materialize_int4_with_scale` / `materialize_int8_per_channel` |
| `quant_linear.py` | 导出用 `QuantLinear(nn.Module)`，forward 发射 `quant_matmul_s8s4` + `npu_dynamic_quant` IR |
| `swap.py` | `swap_to_quant_linear_from_dopt` / `swap_embeddings_from_dopt` / `load_dopt_state_dict` / `_read_dopt_quant_meta` |
| `config.yaml` | dopt GPTQ 校准训练配置（`--optimize-config`） |
| `calibration.json` | GPTQ 校准语料（557KB） |
| `dopt_config.json` | dopt 层策略配置（`--dopt-config`） |

### dopt_lm 子包（vendored GPTQ 框架）

| 文件 | 职责 |
|------|------|
| `dopt_lm/opt_main.py` | CLI 入口 |
| `dopt_lm/do_opt.py` | 核心调度：`optimize_model` / `paser_quant_config` |
| `dopt_lm/opt_funcs/eco_stage/train.py` | orchestrator：`weight_quant_pipline` / `ptq_quant_pipline` / `train`（KD） |
| `dopt_lm/weight_quant_ptq/ptq_algo_heissen/` | GPTQ 算法实现（`WeightQuantHeissen` / `fasterquant` / `llama_sequential`） |

### 算子定义与 GE converter

| 文件 | 职责 |
|------|------|
| `cann_lm/models/fused_ops/quant_matmul_s8s4.py` | `cann_lm::quant_matmul_s8s4` custom op（schema + fake/meta） |
| `cann_lm/models/fused_ops/npu_dynamic_quant.py` | `npu::npu_dynamic_quant` custom op（激活动态量化 s8） |
| `cann_lm/converter/export/_quant_matmul_s8s4_converter.py` | `quant_matmul_s8s4` → 方案 A 自研 GE 算子 converter |

## 关键函数 / 类

| 符号 | 作用 |
|------|------|
| `swap_to_quant_linear_from_dopt(model, dopt_path, ...)` | 主路径：加载 dopt 产物 → 物化 int4 → 替换 `nn.Linear` 为 `QuantLinear` |
| `swap_embeddings_from_dopt(sd, model)` | embed_tokens 的 W8 host gather（不替换成图模块） |
| `load_dopt_state_dict(dopt_path) -> dict` | 加载 dopt 产物（.safetensors/.pth） |
| `materialize_int4_with_scale(weight, scale, group_size, ...)` | 用 dopt 预算 scale 量化（主路径） |
| `materialize_int8_per_channel(weight, scale)` | 8-bit per-row 对称量化（embed 用） |
| `QuantLinear(nn.Module)` | 导出用替身，forward 调 `quant_matmul_s8s4` + `npu_dynamic_quant` |

## 量化方案

| 侧 | 位宽 | 粒度 | 对称 | scale 来源 |
|----|------|------|------|-------------|
| 权重 (x2) | int4 (s4) | per-channel × per-group(默认 256) | 对称(offset=0) | GPTQ 离线求（`group_scale`） |
| 激活 (x1) | int8 (s8) | per-token 动态 | 对称 | 运行时 `npu_dynamic_quant` → `DynamicQuant` IR |
| 输出 | fp16/bf16 | — | — | — |

> 方案 A 自研 `quant_matmul_s8s4` 算子（用 fp32 scale 做 `q*scale` 反量化，绕开 CANN V4 失败点）。路径 C'（复用 CANN V4 整数 int4）经实测不可达，已弃用。

## CLI 接入

| 参数 | 作用 |
|------|------|
| `--quant-path <dopt产物目录>` | 加载 dopt GPTQ 产物，物化 int4 + swap QuantLinear |
| `--quant-method` | `gptq` | 量化算法（当前仅 gptq；自动 spawn dopt，场景 2，待实现；与 --quant-path 互斥） |
| `--quant-bits` | `w4a8` | 格式/位宽（w4a8，默认 w4a8；--quant-method 时生效） |

> CLI 用法见 [INDEX.md](../cli/INDEX.md) / [cli_design.md](../cli/cli_design.md)。

## 关联模块

- **CLI**：[INDEX.md](../cli/INDEX.md) — `--quant-path` / `--quant-method` / `--quant-bits` 参数
- **导出**：[INDEX.md](../export/INDEX.md) — `QuantLinear` 替换后走 `save_epair` / `host_weights` 机制 / C++ `GeWeightLoader`
- **总览**：[converter_design.md](../converter_design.md) — 待实现与遗留（量化部分）
