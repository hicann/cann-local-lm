# 量化模块索引

简体中文 | [English](../../../en/cann_lm_converter/quant/INDEX.md)

> `cann_local_lm/quant/` — W4A8 s8s4 量化接入：CANN 轻量化工具 GPTQ 权重量化 + converter 物化 int4 + `npu::npu_quant_matmul` 算子导出。

## 模块文档

| 文档 | 内容 |
|------|------|
| [quant_design.md](./quant_design.md) | W4A8 s8s4 方案 / quant 包架构 / 量化策略 / 端到端两步流程 / `materialize` / `quant_linear` / `swap` / `auto_quant` / `dopt_wrapper` / 路径 C' `npu_quant_matmul`→V4 算子规格 / C++ 注册 / CANN V3/V4 源码分析附录 |
| [quant_manual.md](./quant_manual.md) | 轻量化工具使用手册：介绍 / 环境准备 / 使用三步 / 产物 / 可设置参数 / 已知约束（无调试与导出细节） |
| [ppl_compare_usage.md](./ppl_compare_usage.md) | `ppl_compare.py` 使用说明：浮点 vs GPTQ 量化产物 PPL 对比（原理 / 参数 / 实测参考） |

## 代码文件

### 轻量化工具编排 + 校准（`cann_local_lm/quant/`）

> converter 侧 `swap.py` 升至 `cann_local_lm/converter/swap.py`（converter 根，编排层 peer of loader/saver）；`materialize.py` / `quant_linear.py` 落 `cann_local_lm/converter/export/quant_swap/`（物化内部），详见 [../export/INDEX.md](../export/INDEX.md) "Python 代码文件"。`quant/__init__.py` 不再 re-export materialize/swap 符号。

| 文件 | 职责 |
|------|------|
| `auto_quant.py` | 一键式量化（方案 C）：`run_auto_quant` / `_run_dopt_subprocess` / `_detect_device_type` |
| `dopt_wrapper.py` | 轻量化工具 subprocess 主体：`main` / `build_layer_strategy` / `parse_args` |
| `config.yaml` | 轻量化工具 GPTQ 校准训练配置（`--optimize-config`） |
| `calibrate_sets/` | GPTQ 校准语料（jsonl，由 `download_dataset.py` 下载生成，可替换/多选） |
| `download_dataset.py` | 校准数据集下载脚本（ultrachat_200k/fineweb_edu/toolbench/bfclv4/c4/coig_cqia/mc4） |
| `ppl_compare.py` | 浮点 vs GPTQ 量化产物 PPL 对比脚本（用法见 [ppl_compare_usage.md](./ppl_compare_usage.md)） |
| `dopt_config.json` | 轻量化工具层策略配置（由 `build_layer_strategy` 自动生成到产物目录；converter 侧 `_read_quant_param_map` 从产物目录读） |

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
| `cann_local_lm/models/fused_ops/quant_matmul.py` | `npu::npu_quant_matmul` custom op（schema + fake/meta + CPU PoC impl） |
| `cann_local_lm/models/fused_ops/dynamic_quant.py` | `npu::npu_dynamic_quant` custom op（激活动态量化 s8） |
| `cann_local_lm/converter/export/ge_converters/_quant_matmul_converter.py` | `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4` converter（V4 full IR + dummy bias） |

## 关键函数 / 类

| 符号 | 作用 |
|------|------|
| `swap_from_dopt(model, dopt_path)` | 主路径（场景 1/2 共用）：一次 load sd → Linear swap + Embedding 抽取，返回 `(quant_info, embed_quant)` |
| `swap_embeddings_from_dopt(sd, model)` | embed_tokens 的 W8 host gather（不替换成图模块） |
| `load_dopt_state_dict(dopt_path) -> dict` | 加载轻量化工具产物（.safetensors/.pth/目录） |
| `run_auto_quant(hf_path) -> str` | 一键式 GPTQ（场景 2）：检测 NPU/GPU → spawn 轻量化工具 subprocess → 返回 dopt_out_dir |
| `materialize_int4_per_group_symmetric(weight, scale, group_size, ...)` | 用轻量化工具预算 scale 量化（主路径） |
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
| `--quant-path <轻量化工具产物目录>` | 场景 1：加载轻量化工具 GPTQ 产物，物化 int4 + swap QuantLinear（兜底） |
| `--quant-method gptq` | 场景 2：自动 spawn 轻量化工具 subprocess 跑 GPTQ（已实现，NPU/GPU/CPU；与 --quant-path 互斥） |

> CLI 用法见 [INDEX.md](../cli/INDEX.md) / [cli_design.md](../cli/cli_design.md)。

## 关联模块

- **CLI**：[INDEX.md](../cli/INDEX.md) — `--quant-path` / `--quant-method` 参数
- **导出**：[INDEX.md](../export/INDEX.md) — `QuantLinear` 替换后走 `save_epair` / `host_weights` 机制 / C++ `GeWeightLoader`
- **总览**：[README.md](../README.md) — 待实现与遗留（量化部分）
