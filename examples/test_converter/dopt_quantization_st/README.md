# dopt_quantization_st — gemma4/Qwen3.5 真实量化系统测试

本目录把 `demos/gemma4` 和 `demos/qwen3dot5_4B` 收敛为两个固定的真实模型 ST。模型权重仍由运行环境提供；校准数据固定使用仓库内的 `cann_lm/converter/quant/calibrate_sets/calibrate_c4.jsonl`，因此使用者只需要配置两个模型路径。

仓库不归档真实模型权重、量化产物或机器相关模型路径。默认输出目录 `output/` 已被 `.gitignore` 屏蔽。

## ST 用例

| ID | 模型 | demo 参数 | 默认设备 | 通过标准 |
|---|---|---|---|---|
| DOPT-ST-GEMMA4 | gemma4 E4B | `ptq_samples=8`、`cutoff_len=32` | NPU 1 | 真实 GPTQ 返回 0；新策略配置有效；state dict 非空 |
| DOPT-ST-QWEN3_5 | Qwen3.5 4B | `ptq_samples=1024`、`cutoff_len=2048` | NPU 0 | 真实 GPTQ 返回 0；`linear_attn` 链路通过；state dict 非空 |
| DOPT-ST-VERIFY | 两个输出目录 | 独立解析配置和产物 | 不需要设备 | `config.yaml`、`dopt_config.json`、量化权重满足契约 |
| DOPT-ST-CHECK | 两个用例定义 | 不加载权重 | 不需要设备 | 内置数据、demo 参数、命令构造及 fail-fast 检查通过 |

自动生成的普通 Linear 策略为 `Quant_INT8-4_ECO`；`lm_head` 和 embedding 分别使用 `Quant_lm_head`、`Quant_Embed_MinMax`。

## 运行两个真实 demo 用例

只需设置模型路径和带 NPU 依赖的 Python：

```bash
cd <CANN-LM>

export DOPT_PYTHON=<python_with_torch_npu>
export GEMMA4_MODEL_PATH=<runtime_gemma4_model_dir>
export QWEN3_5_MODEL_PATH=<runtime_qwen3_5_model_dir>

bash examples/test_converter/dopt_quantization_st/run.sh run
```

默认 `--profile demo`，严格使用上表中的两套 demo 参数并顺序执行。模型路径也可以直接从命令行传入：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh run \
  --gemma4-model <runtime_gemma4_model_dir> \
  --qwen3-5-model <runtime_qwen3_5_model_dir>
```

仅运行一个模型：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh run \
  --models gemma4 \
  --gemma4-model <runtime_gemma4_model_dir>
```

## 真实权重快速冒烟

提交前如果只验证“模型加载 → 自动策略 → NPU GPTQ → 权重保存 → 产物校验”链路，可使用真实权重的 smoke profile；两个模型都固定为 `ptq_samples=1`、`cutoff_len=32`：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh run --profile smoke
```

`smoke` 证明系统链路可用，不代替 demo 参数的正式量化效果测试。

## 真实执行记录（2026-08-27）

本版本已在 Ascend NPU 上使用运行时提供的两个真实模型目录执行：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh run --profile smoke
```

| 用例 | GPTQ 记录耗时 | 策略分布 | state dict 大小 | 结果 |
|---|---:|---|---:|---|
| DOPT-ST-GEMMA4 | 1065.42 s | `Quant_INT8-4_ECO` 343、embedding 2、lm_head 1 | 16,713,397,425 bytes | PASS |
| DOPT-ST-QWEN3_5 | 778.86 s | `Quant_INT8-4_ECO` 248、embedding 1、lm_head 1 | 10,082,095,389 bytes | PASS |

整套命令退出码为 0，随后独立执行 `verify` 再次 PASS。Qwen3.5 的 `linear_attn` 和 `self_attn` 均进入了真实逐层 GPTQ。执行记录不包含模型目录；本地权重产物由 `.gitignore` 排除。

## 独立验证和无权重检查

验证默认输出目录中的两个结果：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh verify
```

无权重、无设备检查：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh check
```

输出目录必须不存在或为空，避免旧权重造成假 PASS。需要保留上一轮结果时，使用新的输出根目录：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh run \
  --output-root examples/test_converter/dopt_quantization_st/output_second
```

## 环境与归档边界

- `DOPT_PYTHON` 应指向安装了 `torch`、设备插件、`transformers` 和 `PyYAML` 的 Python。
- 默认 `HF_DATASETS_OFFLINE=1`、`WANDB_DISABLED=true`。
- 设备号与原 demo 一致：gemma4 默认 1，Qwen3.5 默认 0；可用 `--gemma4-device-id`、`--qwen3-5-device-id` 覆盖。
- `run` 默认顺序执行两个模型，避免单进程同时占用大量主机内存。
- 运行配置、日志、自动生成的 `dopt_config.json` 和权重只存在于 `--output-root`，不得提交到仓库。


重新运行真实 smoke 时需指定新输出目录：

export DOPT_PYTHON=/opt/buildtools/Python-3.11.4/bin/python3
export GEMMA4_MODEL_PATH=/mnt/workspace/oreo/cann_lm/models/gemma-4-E4B-it
export QWEN3_5_MODEL_PATH=/mnt/workspace/oreo/cann_lm/models/Qwen3.5-4B

bash examples/test_converter/dopt_quantization_st/run.sh run \
  --profile smoke \
  --output-root examples/test_converter/dopt_quantization_st/output_next

不加 --profile smoke 就是完整 demo 参数：gemma4 使用 8×32，Qwen3.5 使用 1024×2048。完整 demo profile 尚未执行，本次真实执行的是缩小校准规模但不跳过任何量化环节的 smoke。
