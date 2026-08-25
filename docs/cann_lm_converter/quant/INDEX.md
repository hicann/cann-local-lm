# 量化模块索引

> `cann_lm/converter/quant/` — W4A8 s8s4 量化接入：dopt GPTQ 权重量化 + converter 物化 int4 + `npu::npu_quant_matmul` 算子导出。

## 模块文档

| 文档 | 内容 |
|------|------|
| [quant_design.md](./quant_design.md) | W4A8 s8s4 方案 / quant 包架构 / 量化策略 / 端到端两步流程 / `materialize` / `quant_linear` / `swap` / `auto_quant` / `dopt_wrapper` / 路径 C' `npu_quant_matmul`→V4 算子规格 / C++ 注册 / CANN V3/V4 源码分析附录 |
| [dopt_design.md](./dopt_design.md) | dopt_lm vendored GPTQ 框架（文件树 / 入口 / GPTQ 调用链 / 可设置参数清单 / 历史调试档案） |
| [embedding_quant_design.md](./embedding_quant_design.md) | embed_tokens 的 W8 host gather 量化子主题（`materialize_int8_per_channel_symmetric` + `swap_embeddings_from_dopt`） |

## 代码文件

### converter 侧（`cann_lm/converter/quant/`）

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包门面，re-export 7 个公共符号 |
| `materialize.py` | 权重物化纯函数：`_SCALE_MASK` / `transdata_scale` / `restore_scale` / `pack_int4` / `materialize_int4_per_group_symmetric` / `materialize_int8_per_channel_symmetric` |
| `quant_linear.py` | 导出用 `QuantLinear(nn.Module)`，forward 发射 `npu::npu_quant_matmul` + `npu::npu_dynamic_quant` IR |
| `swap.py` | `swap_from_dopt`（主路径）/ `swap_to_quant_linear_from_dopt`（薄 wrapper）/ `_swap_linear_from_sd` / `swap_embeddings_from_dopt` / `load_dopt_state_dict` / `_read_dopt_quant_meta` / `_read_quant_param_map` / `_infer_group_size` / `_STRATEGY_DEFAULT_PARAMS` |
| `auto_quant.py` | 一键式量化（方案 C）：`run_auto_quant` / `generate_dopt_config` / `_run_dopt_subprocess` / `_detect_device_type` |
| `dopt_wrapper.py` | dopt subprocess 主体：`main` / `_build_layer_strategy` / `parse_args` |
| `config.yaml` | dopt GPTQ 校准训练配置（`--optimize-config`） |
| `calibration.json` | GPTQ 校准语料（557KB） |
| `dopt_config.json` | dopt 层策略配置（场景 1 `--dopt-config`；场景 2 由 wrapper 自动生成；converter 侧 `_read_quant_param_map` 也读） |

### dopt_lm 子包（vendored GPTQ 框架）

| 文件 | 职责 |
|------|------|
| `dopt_lm/opt_main.py` | CLI 入口 |
| `dopt_lm/do_opt.py` | 核心调度：`optimize_model` / `paser_quant_config` |
| `dopt_lm/opt_funcs/eco_stage/train.py` | orchestrator：`weight_quant_pipline` / `ptq_quant_pipline` / `train`（KD） |
| `dopt_lm/opt_funcs/weight_quant_ptq/ptq_algo_hessian/` | GPTQ 算法实现（`WeightQuantHessian` / `fasterquant` / `llama_sequential`） |

### 算子定义与 GE converter

| 文件 | 职责 |
|------|------|
| `cann_lm/models/fused_ops/quant_matmul.py` | `npu::npu_quant_matmul` custom op（schema + fake/meta + CPU PoC impl） |
| `cann_lm/models/fused_ops/dynamic_quant.py` | `npu::npu_dynamic_quant` custom op（激活动态量化 s8） |
| `cann_lm/converter/export/_quant_matmul_converter.py` | `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4` converter（V4 full IR + dummy bias） |

## 关键函数 / 类

| 符号 | 作用 |
|------|------|
| `swap_from_dopt(model, dopt_path, ...)` | 主路径（场景 1/2 共用）：一次 load sd → Linear swap + Embedding 抽取，返回 `(quant_info, embed_quant)` |
| `swap_to_quant_linear_from_dopt(model, dopt_path, ...)` | 向后兼容薄 wrapper（场景 1 离线/单独 Linear swap） |
| `swap_embeddings_from_dopt(sd, model)` | embed_tokens 的 W8 host gather（不替换成图模块） |
| `load_dopt_state_dict(dopt_path) -> dict` | 加载 dopt 产物（.safetensors/.pth/目录） |
| `run_auto_quant(hf_path, quant_bits) -> str` | 一键式 GPTQ（场景 2）：检测 NPU/GPU → spawn dopt subprocess → 返回 dopt_out_dir |
| `materialize_int4_per_group_symmetric(weight, scale, group_size, ...)` | 用 dopt 预算 scale 量化（主路径） |
| `materialize_int8_per_channel_symmetric(weight, scale)` | 8-bit per-row 对称量化（embed 用） |
| `QuantLinear(nn.Module)` | 导出用替身，forward 调 `npu::npu_quant_matmul` + `npu::npu_dynamic_quant` |

## 量化方案

| 侧 | 位宽 | 粒度 | 对称 | scale 来源 |
|----|------|------|------|-------------|
| 权重 (x2) | int4 (s4) | per-channel × per-group(默认 128) | 对称(offset=0) | GPTQ 离线求（`group_scale`） |
| 激活 (x1) | int8 (s8) | per-token 动态 | 对称 | 运行时 `npu_dynamic_quant` → `DynamicQuant` IR |
| 输出 | fp16/bf16 | — | — | — |

> 路径 C'：复用 CANN 原生 `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4`（V4 full IR + dummy bias workaround）。早期方案 A（自研 `quant_matmul_s8s4`）已弃用，规格作历史档案保留。

## CLI 接入

| 参数 | 作用 |
|------|------|
| `--quant-path <dopt产物目录>` | 场景 1：加载 dopt GPTQ 产物，物化 int4 + swap QuantLinear（CPU-only 兜底） |
| `--quant-method` | `gptq` | 场景 2：自动 spawn dopt subprocess 跑 GPTQ（已实现，需 NPU/GPU；与 --quant-path 互斥） |
| `--quant-bits` | `w4a8` | 格式/位宽（w4a8，默认 w4a8；--quant-method 时生效；w4a16 不支持） |

> CLI 用法见 [INDEX.md](../cli/INDEX.md) / [cli_design.md](../cli/cli_design.md)。

## 关联模块

- **CLI**：[INDEX.md](../cli/INDEX.md) — `--quant-path` / `--quant-method` / `--quant-bits` 参数
- **导出**：[INDEX.md](../export/INDEX.md) — `QuantLinear` 替换后走 `save_epair` / `host_weights` 机制 / C++ `GeWeightLoader`
- **总览**：[converter_design.md](../converter_design.md) — 待实现与遗留（量化部分）
