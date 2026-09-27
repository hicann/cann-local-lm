# dopt_op_benchmark — 量化算子一致性标杆

> **验证方法论与现状**：整条量化过程怎么验证、本目录在体系里的位置、还缺什么，见 [`VERIFICATION.md`](VERIFICATION.md)。

把 dopt 参考算子（`QLinear` fakequant）的输入/输出固化为 **golden 标杆数据**，用于验证候选量化 matmul 算子实现（`cann_lm::quant_matmul` → GE `QuantBatchMatmulV4` → OM 真 int4 内核）的数值一致性。

## 核心思想：算子本身 vs 量化精度

参考算子与候选算子使用**同一套量化权重**（`trained_quant_weight.pth`），差异只能来自算子实现，**不来自 GPTQ 量化精度**——全程不引入 fp16 基线，从而把"算子本身"从"量化工程精度"隔离。

| 对比轴 | 参考算子 | 候选算子 |
|---|---|---|
| 实现 | `dopt.QLinear.fakequant`（quant→dequant→fp matmul，weight-only） | `cann_lm::quant_matmul` / OM int4 内核 |
| 权重 | `trained_quant_weight.pth`（**同一套**） | `trained_quant_weight.pth`（**同一套**） |
| 输入 | `op_io/<op>.inputs[i]`（**同一批固定激活**） | `op_io/<op>.inputs[i]`（**同一批固定激活**） |
| 输出 | `op_io/<op>.outputs[i]` = golden | 候选算子计算结果 |
| 判据 | — | `maxΔ ≤ atol + rtol·|golden|`（`atol=rtol=1e-2`） |

## 数据契约（`golden/manifest.json`）

```
golden/
├── manifest.json        # 算子实现/权重路径/输入/atol-rtol/对比说明
├── e2e_logits.pt        # [n_prompts, vocab] 末端 last-token logits（高层 golden）
└── op_io/<op_name>.pt   # {inputs:[act...], outputs:[act...]} 每算子 × 每 prompt
```

- 每个算子的 golden = `(input_act, output_act)`：`input_act` 是端到端前向中喂入该 `QLinear` 的真实激活，`output_act` 是参考算子在该激活下的输出。
- 候选算子只需：从 `trained_quant_weight.pth` 取**同一算子的权重 + s/offset**，喂 `inputs[i]`，算输出，比 `outputs[i]`。

## 算子覆盖

默认 `--operators sample`（代表性采样，深/中/浅三层 × 各算子类型 + lm_head）：

| 层 | 采样的算子类型 |
|---|---|
| layer 0 / 21 / 41 | `self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` |
| 全局 | `lm_head` |

约 20 个算子 × 4 条固定 prompt。`--operators all` 可捕获全部 `QLinear`（344+，体积更大）。

## 用法

### 环境

```bash
export DOPT_PYTHON=<含 torch_npu 的 python>          # 默认 python3
export GEMMA4_MODEL_PATH=<gemma4 模型目录>
export DOPT_QUANT_DIR=<dopt 量化输出目录>            # 含 trained_quant_weight.pth + dopt_config.json + runtime_config.yaml
export DEVICE=cpu                                     # 或 npu（注意 64GB HBM 可能 OOM，建议 CPU）
```

`run.sh` 会自动检测系统 python 的 torchaudio 是否损坏（torch/torchaudio 版本错配），损坏时自动启用内置 `shims/torchaudio` stub——GPTQ 文本路径不需要音频。

### 生成/重建 golden

```bash
bash examples/test_converter/dopt_op_benchmark/run.sh gen
# 可选: --output-dir <dir> --device cpu --operators sample|all|<comma-list> --prompts a b c
```

### 自洽校验 / 候选算子对齐

```bash
# 自洽校验：参考算子重现自己的 golden（必为 0，验证标杆契约有效）
bash examples/test_converter/dopt_op_benchmark/run.sh compare \
  --op-name model.layers.0.self_attn.q_proj
```

`compare.py` 当前是自洽校验模式（重载参考算子、喂保存的输入、比保存的输出）。接候选算子时：把 `build_ref_model` + `op(xa)` 段替换为候选算子实现（同样的权重 + 同样的输入），保留对 `golden` 的对比逻辑即可。

## 边界

- `golden/` 是**生成产物**（gitignored，见 `.gitignore`），由 `gen` 从 `DOPT_QUANT_DIR` 派生；换机器/换量化产物用 `gen` 重建。
- 脚本无机器路径硬编码，路径全部经 env 变量 / CLI 参数传入（`GEMMA4_MODEL_PATH` / `DOPT_QUANT_DIR` / `DOPT_PYTHON` / `DEVICE`）。
- 参考算子是 **fakequant 仿真**（`quant→dequant→fp matmul`），非真 int4 内核；和 OM 真 int4 比对时，差异同时反映"算子实现差异"与"fakequant vs 真 int4 的数值路径差异"——若要纯粹校验 OM 内核实现，需另建匹配内核语义的 reference。
```
