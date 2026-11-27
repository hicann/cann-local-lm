# CANN Lightweight Tool Usage Manual (vendored GPTQ Weight Quantization Framework)

English | [简体中文](../../../zh/cann_lm_converter/quant/quant_manual.md)

> Introduction and usage parts (architecture analysis / historical debugging archives / converter export details are out of scope for this manual).
>
> Status: Currently adopted | Last reviewed: 2026-10-10

## Table of Contents

- [What is the CANN Lightweight Tool](#what-is-the-cann-lightweight-tool)
- [Environment Preparation](#environment-preparation)
- [Three-Step Usage](#three-step-usage)
- [Artifacts](#artifacts-dir-specified-by---output-dir)
- [Configurable Parameters](#configurable-parameters)
- [Known Constraints](#known-constraints)
- [Golden Generation and Usage (Operator Debugging)](#golden-generation-and-usage)
- [Related Documentation](#related-documentation)

---

<a id="what-is-the-cann-lightweight-tool"></a>
## What is the CANN Lightweight Tool

The CANN Lightweight Tool (hereafter "lightweight tool") core `dopt_lm/` is Huawei's vendored **GPTQ-style weight PTQ framework** (named `dopt_lm` = "do optimize", Quantization Optimization). It is a self-contained, independently runnable Python package (entry `opt_main.py`, CLI mode) that performs **GPTQ weight quantization** on HuggingFace models on **NPU / GPU / CPU**, producing per-layer `group_scale` (per-channel × per-group symmetric, offset=0) + GPTQ-corrected weights, finally serialized into `trained_quant_weight.pth` (or an HF dir via `--hf_type_save`).

**Invocation relationship**: dopt_lm is **not `import`ed** in the main repo code; it runs as an independent CLI tool, and its artifacts (.pth/safetensors) are loaded by the converter's `--quant-path` at export time. dopt_lm internally self-references via relative imports `from dopt_lm.xxx import yyy` (requires putting `cann_local_lm/quant` into `PYTHONPATH`).

> **Note**: The algorithm is essentially GPTQ (Frantar et al. ICLR 2023, Hessian-matrix-based PTQ).

<a id="environment-preparation"></a>
## Environment Preparation

The lightweight tool GPTQ selects the execution device via `DEVICE=npu|cuda|cpu`. NPU needs `torch_npu`; CUDA needs a working CUDA PyTorch; CPU has no device extension dependency (fp16 config auto-falls-back to FP32, suitable for functional validation and small models, extremely slow for large models).

```bash
# NPU environment
source /home/developer/Ascend/cann/set_env.sh
# dopt_lm internal relative import `from dopt_lm.xxx` → needs cann_local_lm/quant in PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/cann_local_lm/quant:${PYTHONPATH:-}

# CPU example (if fp16=true in config, runtime auto-falls-back to FP32)
DEVICE=cpu python -m dopt_lm.opt_main --help
```

> Environment setup details in `AGENTS.md` "NPU Environment (torch_npu, verified)".

<a id="three-step-usage"></a>
## Three-Step Usage

### Step 1: Generate dopt_config Scaffold

On first run, the file `--dopt-config` points to doesn't exist; the lightweight tool generates a scaffold JSON (all layers `quant_strategy="float"`) and `exit()`; you must manually edit the strategy and re-run.

```bash
cd cann_local_lm/quant
python -m dopt_lm.opt_main \
    --model-path /path/to/hf-model \
    --dopt-config dopt_config.json
# → generates dopt_config.json (all layers float) then exit
```

### Step 2: Edit dopt_config.json to Set Layer Strategy

Edit `layer_strategy`, choosing a `quant_strategy` for each layer (strategy names in [Configurable Parameters](#configurable-parameters) ②):

| Module type | Recommended `quant_strategy` | Description |
|---------|----------------------|------|
| `nn.Linear` (q/k/v/o/gate/up/down_proj, etc.) | `Quant_INT8-4_ECO` | W4A8 main path (default group 128); use `;group_size` to override per layer |
| `nn.Linear` (lm_head) | `Quant_INT8-4_ECO` | W4A8 main path |
| `nn.Embedding` | `Quant_Embed_MinMax` | embed int8 per-channel, maps to `QEmbedding` |
| Layers not to quantize | `float` | Removed from the GPTQ task, keeps the original module |

Example (`dopt_config.json` snippet):

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

### Step 3: Run GPTQ

Recommended to use `dopt_wrapper.py` (auto-generates `dopt_config.json` into `--output-dir`, skipping manual Step 1/2):

```bash
python cann_local_lm/quant/dopt_wrapper.py \
    --model-path /path/to/hf-model \
    --optimize-config config.yaml \
    --output-dir /path/to/quant_out \
    --hf-type-save
```

Or manually `python -m dopt_lm.opt_main` (provide your own `--dopt-config`):

```bash
python -m dopt_lm.opt_main \
    --model-path /path/to/hf-model \
    --dopt-config dopt_config.json \
    --optimize-config config.yaml \
    --output-dir /path/to/quant_out \
    --hf_type_save
```

- `--optimize-config config.yaml`: calibration config (see [Configurable Parameters](#configurable-parameters) ③)
- `calibrate_sets/*.jsonl`: calibration corpus (generated by `download_dataset.py`, resolved relative to CWD, must run in the `cann_local_lm/quant/` dir)
- `--hf_type_save`: save in HF format (`save_pretrained`, produces `model.safetensors`); without it, stores `trained_quant_weight.pth`
- Device: `DEVICE=npu|cuda|cpu` env var

<a id="artifacts-dir-specified-by---output-dir"></a>
## Artifacts (dir specified by --output-dir)

| File | Description |
|------|------|
| `model.safetensors` | GPTQ weights + per-layer `quant_op.weight_quantizer.s` (group_scale) + `{bit, unsigned_quant, offset}` meta info (size varies with model/dtype) |
| `config.json` | HF model config copy |
| `config.yaml` | Training config replay written by the lightweight tool |

Artifact state_dict keys example:

```
model.layers.N.self_attn.q_proj.weight                    (GPTQ weight)
model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.s   (group_scale, flat [out*in//group])
model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.{bit,unsigned_quant,offset}
model.embed_tokens.weight                                 (fakequant fp)
model.embed_tokens.quant_op.weight_quantizer.s            (per-channel scale)
```

> Artifacts are consumed by the converter `--quant-path` (materialize int4 + swap `QuantLinear`), see [quant_design.md](./quant_design.md). PPL comparison evaluation can use [ppl_compare.py](../../../../cann_local_lm/quant/ppl_compare.py) (usage in [ppl_compare_usage.md](./ppl_compare_usage.md)).

<a id="configurable-parameters"></a>
## Configurable Parameters

The lightweight tool's configurable quantization params span 4 levels; **users can actually set only ①②③, ④ GPTQ algorithm params are mostly hardcoded and not exposed**.

### ① Lightweight Tool CLI Params (`opt_main.py`)

| Param | Default | Description |
|------|------|------|
| `--model-path` | (required) | HF model path |
| `--dopt-config` | (required) | Layer strategy config file (`dopt_config.json`) |
| `--optimize-config` | `None` | Training config (`config.yaml`) |
| `--output-dir` | `None` | Output dir |
| `--hf_type_save` | `False` | Save in HF format (`save_pretrained`) |

### ② `dopt_config.json` Layer Strategy

`quant_strategy` is selected from `strategy_dict` (6 presets), supports `"name;group_size"` syntax to override `group_size` per layer:

| Strategy name | Use case | weight | act | Notes |
|--------|------|--------|-----|------|
| `float` | No quantization | — | — | |
| `Quant_aigc_ptq` | W8A8 | 8(per-channel min_max) | 8(ema_min_max, unsigned) | |
| `Quant_aigc_qat` | W8A8 | 8(omni_min_max) | 8(omni, unsigned) | For QAT |
| `Quant_INT8-4_ECO` | **W4A8 (main path)** | 4(group 128, block 128, omni_group) | 8(dyn_s8, signed) | converter default |
| `Quant_Embed_MinMax` | embed int8 | 8(per-channel min_max) | — | host gather |
| `Quant_Dummy` | Placeholder | — | — | |

### ③ `config.yaml` Training Config (GPTQ Calibration)

| Field | Current repo value | Description |
|------|-----------|------|
| `dataset.train_files` | `["calibrate_sets/calibrate_coig_cqia.jsonl"]` | Calibration corpus (downloaded by `download_dataset.py`, replaceable/multi-select, supports `path:ratio` ratio) |
| `dataset.ptq_samples` | `256` | Sample count |
| `cutoff_len` | `1024` | Truncation length |
| `extra_training_config.fp16` | `true` | fp16 training + save (CPU dispatch auto-falls-back to float32) |

### ④ GPTQ Algorithm Params — Hardcoded, Not Exposed via CLI

`percdamp`(0.01) / `act_order`(False) / `static_groups`(False) / `sym`(True) / `blocksize`(128) / `seed`(1234) and other defaults are hardcoded in `QuantArgs` / `fasterquant` / `configure`. To change them you must edit source (`ptq_loop.py`'s `QuantArgs` defaults or `weight_quant_loop` call args); `blocksize` does not support `;` syntax override, changing it requires touching `strategy_dict`.

<a id="known-constraints"></a>
## Known Constraints

1. **Multimodal `ForConditionalGeneration` checkpoints can run directly**: `train.py`'s `model_define` detects `model.model.language_model` then auto-extracts `text_config` and re-mounts as a text-only ForCausalLM (lm_head/tie_word_embeddings handled together); `ptq_loop` has gemma4/PLE structural support (`unique_layer_types` / `get_per_layer_inputs` / `per_layer_projection` / `embed_tokens_per_layer`). Measured: `gemma-4-E4B-it` multimodal original weights (`Gemma4ForConditionalGeneration`) full GPTQ run on CPU passed (2026-10-12, 42 layers + PLE + lm_head). Prerequisite is that the transformers version supports the model (gemma4 needs a new version like 5.x; Qwen3 4.51.3 suffices).
2. **CPU mode is for functional validation**: fp32 is slow and memory-heavy; for large models use NPU/GPU.
3. **Calibration corpus relative paths** are resolved against CWD; must run in the `cann_local_lm/quant/` dir.
4. **Artifacts have no tokenizer**: when downstream consumers like converter/PPL evaluation use them, use the original HF model dir's tokenizer.

<a id="golden-generation-and-usage"></a>
## Golden Generation and Usage (Operator Debugging)

> **A quantized artifact must pass two distinct validations before deployment**: ① accuracy (how much quantization cost — fp16 is a legitimate reference here, use [ppl_compare.py](./ppl_compare_usage.md)); ② **operator correctness** (bugs in the quant-op implementation — **neither fp16 nor another simulation can serve as the reference**). ② relies on golden data: freeze the reference operator's input/output tensors under fixed inputs, then the candidate operator (converter-materialized `QuantLinear` → `DynamicQuant` + `QuantBatchMatmulV4` OM kernel) recomputes with **the same quantized weights + the same inputs** and is compared layer by layer. Verification methodology in [VERIFICATION.md](../../../../examples/test_converter/dopt_op_benchmark/VERIFICATION.md).

### Why golden is mandatory

- Debugging NPU quant operators needs an **implementation-independent** ground truth: fp16 differs semantically (W4A16/W4A8 vs fp32), and a dopt simulation may share the same spec-level misunderstanding with in-repo code (e.g. int4 range [-7,7] vs [-8,7]) — agreement does not imply correctness.
- Golden isolates **operator implementation** from **quantization accuracy**: reference and candidate operators use the same `trained_quant_weight.pth` + the same fixed activations, so output differences can only come from the operator implementation itself.

### Two levels of golden tooling

| Level | Tool | Artifacts | Use case |
|-------|------|-----------|----------|
| **Per-operator** (main path) | [`dopt_op_benchmark`](../../../../examples/test_converter/dopt_op_benchmark/README.md) `run.sh gen` | `golden/op_io/<op>.pt` (per-op inputs/outputs) + `manifest.json` (data contract) + `e2e_logits.pt` | Numerical consistency of candidate operators (NPU real kernel vs golden) |
| **Per-FX-node** | [`gen_golden_with_dump.py`](../../../../examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py) `--quant-path` | Per-FX-node input/output `.bin` + `golden_meta.json` | Per-op intermediate tensor comparison before/after quantization; locate the loss source layer |

### Per-operator golden usage

```bash
export GEMMA4_MODEL_PATH=<HF model dir>          # original weights + tokenizer source
export DOPT_QUANT_DIR=<Step 3's --output-dir>   # quant artifact dir (see note ① below)
export DEVICE=cpu                                # or npu

# 1. Generate golden (~20 ops sampled: deep/mid/shallow layers × 7 proj types + lm_head; --operators all for full coverage)
bash examples/test_converter/dopt_op_benchmark/run.sh gen

# 2. Candidate-operator chain verification (materialize same weights → build_epair → compile OM → NPU run → compare)
cd examples/test_converter/dopt_op_benchmark/op_npu_verify
bash compile_om.sh && python run_npu.py && python verify_op.py
```

**Acceptance criteria** (`verify_op.py`, weights/inputs identical to golden):

| Comparison axis | Reference | Threshold | Semantics |
|-----------------|-----------|-----------|-----------|
| NPU vs **Tier 0 [`ref_w4a8.py`](../../../../examples/test_converter/dopt_op_benchmark/ref_w4a8.py)** (mathematical-definition reference, one step per line, pinned by hand-computed fixtures) | same semantics | `maxΔ ≤ 1e-2 + 1e-2·|golden|`, cos≥0.999 | only int accumulation/implementation diff |
| NPU vs dopt fakequant golden (W4A16) | cross semantics | cos≥0.99 | additionally includes activation int8 quantization diff |

### Per-FX-node golden usage (locate loss layers)

```bash
# Dump both fp16 baseline and W4A8 fakequant (same model, same inputs), compare per node via cosine/MSE
python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
    --model-type gemma4 --hf-path $GEMMA4_MODEL_PATH \
    --seq-len 8 --num-layers 3 --dtype float16 --output-dir out/golden_fp16
python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
    --model-type gemma4 --hf-path $GEMMA4_MODEL_PATH --quant-path $DOPT_QUANT_DIR \
    --seq-len 8 --num-layers 3 --dtype float16 --output-dir out/golden_quant
```

### Caveats

1. **Artifact form**: per-operator golden consumes the `.pth` form (`trained_quant_weight.pth` + `dopt_config.json` + `runtime_config.yaml`) — run GPTQ **without** `--hf-type-save` (the safetensors form is for converter `--quant-path`); if `runtime_config.yaml` is missing, rename the `config.yaml` replay-written by the lightweight tool in the artifact dir.
2. **Same weights, same inputs**: golden and candidate operator must use the same quantized weights + the same fixed activations (tool has built-in fixed prompts + seed, reproducible across runs).
3. **packed vs unpacked**: CPU simulation consumes unpacked `[K,N]` int8, the real NPU operator consumes pre-transposed nibble-packed `[K//2,N]` — the tool re-materializes automatically; do not mix layouts in custom scripts.
4. **scale truncation**: on-disk scale is mantissa-truncated (low 13 bits) by `transdata_scale`; align with `restore_scale` before comparing, so format-conversion loss is not misattributed as an operator bug.
5. **Switching models**: the per-operator tool's sampled layer-name patterns hardcode the llama family (`self_attn.q_proj` etc.); MoE/heterogeneous naming requires editing `gen_golden.py:default_op_sample`; end-to-end simulation (`sim_inference.py`) uses token-by-token argmax (no `model.generate` dependency) and is reusable across models.

<a id="related-documentation"></a>
## Related Documentation

| Document | Related content |
|------|---------|
| [quant_design.md](./quant_design.md) | Converter-side quantization integration (`materialize` / `swap` / `quant_linear` / end-to-end flow) |
| [ppl_compare_usage.md](./ppl_compare_usage.md) | Float vs quantized artifact PPL comparison script usage |
| [feature_dump_design.md](../feature_dump_design.md) | Dump diagnostics (FX node-level `gen_golden_with_dump.py` quantized golden usage) |
| [dopt_op_benchmark README](../../../../examples/test_converter/dopt_op_benchmark/README.md) | Per-operator golden generation and verification (`run.sh gen` / manifest data contract / acceptance criteria) |
| [VERIFICATION.md](../../../../examples/test_converter/dopt_op_benchmark/VERIFICATION.md) | Quantization verification methodology (accuracy vs operator correctness, Tier 0 reference, trust tiers) |
