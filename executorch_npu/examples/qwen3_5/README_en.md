# Qwen3.5 Example

English | [简体中文](README.md)

Full/linear attention; the exported prefill provides full_mask, and the linear layers maintain state.

## Directories

- `model/`: model configuration and forward implementation; the Cache implementation for both models is in `examples/common/cache.py`.
- `export/`: `adapter.py` provides the model-specific wrapper, and `pt2.py` and `pte.py` are the export entry points.
- `inference/`: `eager.py` runs a single forward pass, and `session.py` performs prefill/decode.
- `mmlu/`: `prepare.py`, `evaluate.py`, and `report.py` are responsible for data, scoring, and reporting respectively.

The two model directories have exactly the same relative file paths. Shared implementations are located in `examples/common`.

## Environment

Run the commands from the repository root, and load the CANN and delegate environment first:

```bash
conda activate executorch_npu
# Load the installed CANN/delegate environment first, or set the environment variables of the corresponding setup script.
source examples/scripts/env.sh
```

The environment script uses the current Python environment; for the configurable options, see [scripts/README.md](../scripts/README_en.md).

## Export

```bash
MODEL_PATH=/mnt/workspace/models/Qwen3.5-0.8B
RUN_DIR=/mnt/workspace/qwen3_5_run
python -m examples.qwen3_5.export.pt2 --model-path "$MODEL_PATH" \
  --output-dir "$RUN_DIR/pt2" --max-seq-len 5120 --trace-seq-len 8 --device cpu
python -m examples.qwen3_5.export.pte --pt2-dir "$RUN_DIR/pt2" \
  --output-dir "$RUN_DIR/pte" --acl-soc-name Ascend910B3 --host-only
```

By default, the two methods `prefill` and `decode` are exported. The PT2/PTE artifacts are located in the run directory.

## Inference

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

`--max-seq-len` is the export capacity; the actual input lengths in the report come from the samples. The longest input in historical full MMLU runs was about 3.1K.

## Local Full MMLU Validation

The model is `Qwen3.5-0.8B`, validated with BF16 and batch size 1 on Ascend 910B3 in the `executorch_npu` environment. The software versions are CANN 9.1.0, PyTorch 2.11.0+cpu, torch_npu 2.11.0.post3.dev20260924, ExecuTorch 1.5.0, and Transformers 5.17.0.

The English MMLU contains **14,042 questions**, scored with 5-shot prompting and A/B/C/D candidate token logits; the export capacity is **5120 tokens**, the actual input lengths are **240–3133 tokens** (P50=532, P95=1721), and the number of truncated questions is **0**.

| Mode | Correct / Total Questions | Accuracy | Relative to eager (percentage points) | Agreement with eager predictions | Full scoring wall-clock time |
|---|---:|---:|---:|---:|---:|
| NPU eager | 5767 / 14042 | 41.0696% | +0.0000 | 100.0000% | 04:49:32 |
| export PTE | 5773 / 14042 | 41.1124% | +0.0427 | 98.4333% | 08:24:42 |
| lite export PTE | 5780 / 14042 | 41.1622% | +0.0926 | 98.4974% | 08:22:22 |

`export` captures PT2 on the NPU; `lite export` captures PT2 on the CPU and compiles the PTE through the host-only path. During CPU capture, the NPU is masked, `torch_npu.scatter_update_` is retained in the KV Cache graph, and no CPU kernel is registered. The NPU eager baseline uses `use_cache=False`; the PTE uses a fixed-capacity Cache.

The times in the table include scoring process loading and warm-up, and exclude PT2/PTE export. Not all groups ran under the same resource occupancy: the eager scoring had overlapping tasks, and during the standard export scoring the CPU compilation of the lite PTE also ran in parallel. These times document this run and are not intended as exclusive performance speedups.

Both the standard and the lite PTE passed the prefill/decode checks with 240 / 3133 token inputs; `model/modeling.py` constructs eye with int64 indices to avoid the uint8 arange decomposition in PyTorch 2.11 generating a Range that CANN does not support.

**MMMLU validation is not yet complete**; the results in this section only represent the English MMLU.

## Multilingual MMMLU

MMMLU uses the language CSVs from `openai/MMMLU` and the lm-eval `openai-mmmlu` task descriptions.
The 5-shot demonstrations are selected from the test data of the same language and subject, and the question currently being evaluated is excluded.
First, place files such as `mmlu_ZH-CN.csv` and `mmlu_ES-LA.csv` in a separate data directory.

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

The MMMLU report includes the overall accuracy, per-language accuracy, language×subject accuracy, and the number of truncated questions.
The samples and reports of MMMLU and of the English MMLU should be placed in different run directories.

## Ascend Fused Inference Implementation

The text model uses the fused implementation directly:

- Prefill uses `npu_chunk_gated_delta_rule`, computed according to the actual input length; decode uses the functional recurrent GDN.
- The recurrent state is fixed to BF16; Q/K L2 normalization and scaling are preserved, and the state layout is converted between the model and CANN.
- It uses contiguous BNSD FIA v2, merged QKVZ/BA and Attention QKV projections, SwiGLU, and RMS/GatedRMS/AddRMS.
- After the original checkpoint is loaded in strict mode, the model merges the projections once automatically; forward directly uses the fused computation.

The current fused GDN requires batch=1, BF16, and key/value head_dim=128, and the validation environment is Ascend910B3. Because the installed version lacks the ChunkGatedDeltaRule AscendIR registration, a standalone registration so must be built before export; the PTE entry point loads it automatically. This is a compilation adaptation and does not modify the CANN installation or the runner.

```bash
bash examples/scripts/build_chunk_gdn_proto.sh
python -m examples.qwen3_5.export.pt2 \
  --model-path /mnt/workspace/models/Qwen3.5-0.8B \
  --output-dir /tmp/qwen/pt2 --max-seq-len 5120 --trace-seq-len 8 --device npu
python -m examples.qwen3_5.export.pte \
  --pt2-dir /tmp/qwen/pt2 --output-dir /tmp/qwen/pte --acl-soc-name Ascend910B3
```

Prefill/decode must be re-exported using the same BF16 state ABI. The per-item PT2/PTE timings are recorded in `pt2_timings.json` and `pte_timings.json` in the artifact directory. For historical performance and accuracy records, see `docs/performance_log.md`; the switches in the early records only describe the implementation at that time, and the current entry points have removed them. The BF16 state is a numerical strategy change relative to the original FP32 state, and the historical WARN is retained.
