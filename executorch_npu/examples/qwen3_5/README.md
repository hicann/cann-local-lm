# Qwen3.5示例

简体中文 | [English](README_en.md)

Full/linear attention；导出prefill提供full_mask，linear层维护状态。

## 目录

- `model/`：模型配置与前向实现；两个模型的Cache实现在 `examples/common/cache.py`。
- `export/`：`adapter.py` 提供模型专属wrapper，`pt2.py` 和 `pte.py` 是导出入口。
- `inference/`：`eager.py` 运行单次前向，`session.py` 执行prefill/decode。
- `mmlu/`：`prepare.py`、`evaluate.py`、`report.py` 分别负责数据、评分、报告。

两个模型目录具有完全相同的相对文件路径。公共实现位于 `examples/common`。

## 环境

从仓库根目录执行命令，先加载CANN与delegate环境：

```bash
conda activate executorch_npu
# 先加载已安装的 CANN/delegate 环境，或设置对应 setup 脚本的环境变量。
source examples/scripts/env.sh
```

环境脚本使用当前Python环境；可配置项见 [scripts/README.md](../scripts/README.md)。

## 导出

```bash
MODEL_PATH=/mnt/workspace/models/Qwen3.5-0.8B
RUN_DIR=/mnt/workspace/qwen3_5_run
python -m examples.qwen3_5.export.pt2 --model-path "$MODEL_PATH" \
  --output-dir "$RUN_DIR/pt2" --max-seq-len 5120 --trace-seq-len 8 --device cpu
python -m examples.qwen3_5.export.pte --pt2-dir "$RUN_DIR/pt2" \
  --output-dir "$RUN_DIR/pte" --acl-soc-name Ascend910B3 --host-only
```

默认导出 `prefill` 和 `decode` 两个method。PT2/PTE产物位于运行目录。

## 推理

```bash
python -m examples.qwen3_5.inference.eager
python -m examples.qwen3_5.inference.session --model-path "$MODEL_PATH" \
  --pte-dir "$RUN_DIR/pte" --max-seq-len 5120 --prompt "Hello"
```

## MMLU

```bash
python -m examples.qwen3_5.mmlu.prepare --model-path "$MODEL_PATH" \
  --output-dir "$RUN_DIR/mmlu/inputs" --max-length 5119 --source parquet
CHOICE_IDS="$(cat "$RUN_DIR/mmlu/choice_token_ids.txt")"
python -m examples.qwen3_5.mmlu.evaluate --backend pte \
  --manifest "$RUN_DIR/mmlu/inputs/manifest.tsv" \
  --choice-token-ids "$CHOICE_IDS" --pte "$RUN_DIR/pte/model.pte" \
  --ptd "$RUN_DIR/pte/ascend.weights.package.ge_aot.ptd" \
  --max-seq-len 5120 --output "$RUN_DIR/mmlu/pte.log"
python -m examples.qwen3_5.mmlu.evaluate --backend npu-eager \
  --model-path "$MODEL_PATH" --manifest "$RUN_DIR/mmlu/inputs/manifest.tsv" \
  --choice-token-ids "$CHOICE_IDS" --output "$RUN_DIR/mmlu/eager.log"
python -m examples.qwen3_5.mmlu.report \
  --samples "$RUN_DIR/mmlu/inputs/samples.jsonl" \
  --logs "$RUN_DIR/mmlu/pte.log" --output "$RUN_DIR/mmlu/pte-summary.json"
python -m examples.qwen3_5.mmlu.report \
  --samples "$RUN_DIR/mmlu/inputs/samples.jsonl" \
  --logs "$RUN_DIR/mmlu/eager.log" --output "$RUN_DIR/mmlu/eager-summary.json" \
  --reference "$RUN_DIR/mmlu/pte-summary.jsonl"
```

`--max-seq-len` 是导出容量；报告里的实际输入长度来自样本。历史完整MMLU的最长输入约3.1K。

## 本地完整MMLU验证

模型为 `Qwen3.5-0.8B`，在Ascend 910B3、`executorch_npu` 环境下使用BF16、batch size 1验证。软件版本为CANN 9.1.0、PyTorch 2.11.0+cpu、torch_npu 2.11.0.post3.dev20260924、ExecuTorch 1.5.0、Transformers 5.17.0。

英文MMLU共 **14,042题**，采用5-shot和A/B/C/D候选token logits评分；导出容量为 **5120 tokens**，实际输入长度为 **240～3133 tokens**（P50=532，P95=1721），截断题数为 **0**。

| 模式 | 正确 / 总题数 | 准确率 | 相对eager（百分点） | 与eager预测一致率 | 完整评分墙钟耗时 |
|---|---:|---:|---:|---:|---:|
| NPU eager | 5767 / 14042 | 41.0696% | +0.0000 | 100.0000% | 04:49:32 |
| export PTE | 5773 / 14042 | 41.1124% | +0.0427 | 98.4333% | 08:24:42 |
| lite export PTE | 5780 / 14042 | 41.1622% | +0.0926 | 98.4974% | 08:22:22 |

`export` 使用NPU捕获PT2；`lite export` 使用CPU捕获PT2，并通过host-only路径编译PTE。CPU捕获时屏蔽NPU，KV Cache图中保留 `torch_npu.scatter_update_`，不注册CPU kernel。NPU eager基线使用 `use_cache=False`；PTE使用固定容量Cache。

表中耗时包括评分进程加载和预热，不含PT2/PTE导出。各组并非全部在相同资源占用下运行：eager评分存在任务重叠，标准export评分期间还并行进行了lite PTE的CPU编译。这些耗时用于记录本次运行，不作为独占性能加速比。

标准及lite PTE均通过240 / 3133 token输入的prefill/decode检查；`model/modeling.py` 使用int64索引构造eye，避免PyTorch 2.11的uint8 arange分解生成CANN不支持的Range。

**MMMLU尚未完成验证**，本节结果仅代表英文MMLU。

## 多语言MMMLU

MMMLU使用 `openai/MMMLU` 的语言CSV和lm-eval `openai-mmmlu` 任务描述。
从同语言、同科目的test数据选择5-shot示范，并排除当前待评估题。
先将 `mmlu_ZH-CN.csv`、`mmlu_ES-LA.csv` 等文件放到独立的数据目录。

```bash
MMMLU_DATA_DIR=/mnt/workspace/mmmlu_data/test
MMMLU_RUN_DIR="$RUN_DIR/mmmlu"
python -m examples.qwen3_5.mmlu.prepare --benchmark mmmlu \
  --model-path "$MODEL_PATH" --data-dir "$MMMLU_DATA_DIR" \
  --languages ZH_CN,ES_LA --num-fewshot 5 \
  --output-dir "$MMMLU_RUN_DIR/inputs" --max-length 5119
CHOICE_IDS="$(cat "$MMMLU_RUN_DIR/choice_token_ids.txt")"
python -m examples.qwen3_5.mmlu.evaluate --backend pte \
  --manifest "$MMMLU_RUN_DIR/inputs/manifest.tsv" \
  --choice-token-ids "$CHOICE_IDS" --pte "$RUN_DIR/pte/model.pte" \
  --ptd "$RUN_DIR/pte/ascend.weights.package.ge_aot.ptd" \
  --max-seq-len 5120 --output "$MMMLU_RUN_DIR/pte.log"
python -m examples.qwen3_5.mmlu.report --benchmark mmmlu \
  --samples "$MMMLU_RUN_DIR/inputs/samples.jsonl" \
  --logs "$MMMLU_RUN_DIR/pte.log" \
  --output "$MMMLU_RUN_DIR/pte-summary.json"
```

MMMLU报告包含总准确率、各语言准确率、语言×科目准确率和截断题数。
MMMLU与英文MMLU的样本及报告应放在不同运行目录。

## Ascend融合推理实现

文本模型直接使用融合实现：

- Prefill使用`npu_chunk_gated_delta_rule`，按实际输入长度计算；decode使用functional recurrent GDN。
- Recurrent state固定为BF16；保持Q/K L2归一化和scale，并转换模型与CANN的state布局。
- 使用连续BNSD FIA v2、合并的QKVZ/BA和Attention QKV投影、SwiGLU及RMS/GatedRMS/AddRMS。
- 原始checkpoint严格加载后，模型自动合并投影一次；forward直接使用融合计算。

当前融合GDN要求batch=1、BF16及key/value head_dim=128，验证环境为Ascend910B3。由于安装版本缺少ChunkGatedDeltaRule AscendIR注册，导出前需要构建独立注册so；PTE入口自动加载它。这是编译适配，不修改CANN安装和runner。

```bash
bash examples/scripts/build_chunk_gdn_proto.sh
python -m examples.qwen3_5.export.pt2 \
  --model-path /mnt/workspace/models/Qwen3.5-0.8B \
  --output-dir /tmp/qwen/pt2 --max-seq-len 5120 --trace-seq-len 8 --device npu
python -m examples.qwen3_5.export.pte \
  --pt2-dir /tmp/qwen/pt2 --output-dir /tmp/qwen/pte --acl-soc-name Ascend910B3
```

Prefill/decode必须使用同一BF16状态ABI重新导出。PT2/PTE分项耗时分别记录在产物目录的`pt2_timings.json`和`pte_timings.json`。历史性能与精度记录见`docs/performance_log.md`；早期记录中的开关仅描述当时实现，当前入口已移除这些开关。BF16 state相对原始FP32 state属于数值策略变化，历史WARN仍保留。
