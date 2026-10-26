# dopt_quantization_st — gemma4/Qwen3.5 真实量化系统测试

本目录把两个固定真实模型收敛为 ST。模型权重由运行环境提供；校准数据使用已安装的 `cann_lm` pip 包内的 `calibrate_sets/calibrate_coig_cqia.jsonl`，使用者只需配置模型路径。

仓库不归档真实模型权重、量化产物或机器相关模型路径。默认输出目录 `output/` 已被 `.gitignore` 屏蔽。

> **前置**：`.run` 包已安装（`/opt/cann-device-lm/`），`cann_lm` whl 已装进系统 python（`--no-deps --user`）。

## ST 用例

| ID | 模型 | demo 参数 | 默认设备 | 通过标准 |
|---|---|---|---|---|
| DOPT-ST-GEMMA4 | gemma4 E4B | `ptq_samples=8`、`cutoff_len=32` | NPU 1 | 真实 GPTQ 返回 0；新策略配置有效；state dict 非空 |
| DOPT-ST-QWEN3_5 | Qwen3.5 4B | `ptq_samples=8`、`cutoff_len=2048` | NPU 0 | 真实 GPTQ 返回 0；`linear_attn` 链路通过；state dict 非空 |
| DOPT-ST-VERIFY | 两个输出目录 | 独立解析配置和产物 | 不需要设备 | `config.yaml`、`dopt_config.json`、量化权重满足契约 |
| DOPT-ST-CHECK | 两个用例定义 | 不加载权重 | 不需要设备 | 内置数据、demo 参数、命令构造及 fail-fast 检查通过 |

自动生成的 Linear（包括 `lm_head`）策略为 `Quant_INT8-4_ECO`；embedding 使用 `Quant_Embed_MinMax`。

## 运行两个真实 demo 用例

只需设置模型路径（`DOPT_PYTHON` 由 `set_env.sh --npu` 自动探测）：

```bash
export GEMMA4_MODEL_PATH=<runtime_gemma4_model_dir>
export QWEN3_5_MODEL_PATH=<runtime_qwen3_5_model_dir>

bash examples/test_converter/dopt_quantization_st/run.sh run
```

默认 `--profile demo`，严格使用上表中的两套 demo 参数并顺序执行；同时默认截取前 2 层加速（`--num-layers 0` 恢复全量模型）。模型路径也可以直接从命令行传入：

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

提交前如果只验证"模型加载 → 自动策略 → NPU GPTQ → 权重保存 → 产物校验"链路，可使用 smoke profile；两个模型都固定为 `ptq_samples=1`、`cutoff_len=32`：

```bash
bash examples/test_converter/dopt_quantization_st/run.sh run --profile smoke
```

`smoke` 证明系统链路可用，不代替 demo 参数的正式量化效果测试。

## 截层 + 小样本快速运行（默认开启）

`run` 默认截取前 **2** 个 decoder 层（`--num-layers`，默认 2；传 `0` 跑全量模型），demo 档默认样本数两模型均为 8（`--ptq-samples` 可继续覆盖）：

```bash
# 默认：截前 2 层 + 8 条样本
bash examples/test_converter/dopt_quantization_st/run.sh run \
  --models qwen3_5 --qwen3-5-model <runtime_qwen3_5_model_dir> --device npu

# 全量模型 + 自定义样本数
bash examples/test_converter/dopt_quantization_st/run.sh run \
  --models qwen3_5 --qwen3-5-model <runtime_qwen3_5_model_dir> \
  --num-layers 0 --ptq-samples 64 --device npu
```

- 截层在 GPTQ 之前执行：CPU fp16 加载一次原模型，把前 N 层另存为文本-only HF 目录（多模态 gemma4 会按 `model_define` 同款逻辑重挂为 text-only），写入该用例输出目录的 `truncated_model/`，GPTQ 与产物校验都在截层目录上进行。
- 截层只改 `config.num_hidden_layers` 并裁剪 `model.layers`，其余逐层配置列表（如 `layer_types`）保持原样。
- 截层产物只服务于快速验证，不代表完整模型的量化效果；需要完整模型回归时用 `--num-layers 0`。

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

- `run.sh` 自动 `source /opt/cann-device-lm/set_env.sh --npu`（探测 `DOPT_PYTHON` + 设 `DEVICE=npu`）；探测失败时手动 `export DOPT_PYTHON=/path/to/python3`。
- `DOPT_PYTHON` 应指向安装了 `torch`、`torch_npu`、`transformers` 和 `PyYAML` 的 Python。
- 默认 `HF_DATASETS_OFFLINE=1`、`WANDB_DISABLED=true`。
- 设备号与原 demo 一致：gemma4 默认 1，Qwen3.5 默认 0；可用 `--gemma4-device-id`、`--qwen3-5-device-id` 覆盖。
- `run` 默认顺序执行两个模型，避免单进程同时占用大量主机内存。
- 运行配置、日志、自动生成的 `dopt_config.json` 和权重只存在于 `--output-root`，不得提交到仓库。
