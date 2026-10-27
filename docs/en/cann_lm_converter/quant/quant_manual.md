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
- [Related Documentation](#related-documentation)

---

<a id="what-is-the-cann-lightweight-tool"></a>
## What is the CANN Lightweight Tool

The CANN Lightweight Tool (hereafter "lightweight tool") core `dopt_lm/` is Huawei's vendored **GPTQ-style weight PTQ framework** (named `dopt_lm` = "do optimize", Quantization Optimization). It is a self-contained, independently runnable Python package (entry `opt_main.py`, CLI mode) that performs **GPTQ weight quantization** on HuggingFace models on **NPU / GPU / CPU**, producing per-layer `group_scale` (per-channel × per-group symmetric, offset=0) + GPTQ-corrected weights, finally serialized into `trained_quant_weight.pth` (or an HF dir via `--hf_type_save`).

**Invocation relationship**: dopt_lm is **not `import`ed** in the main repo code; it runs as an independent CLI tool, and its artifacts (.pth/safetensors) are loaded by the converter's `--quant-path` at export time. dopt_lm internally self-references via relative imports `from dopt_lm.xxx import yyy` (requires putting `cann_lm/converter/quant` into `PYTHONPATH`).

> **Note**: The algorithm is essentially GPTQ (Frantar et al. ICLR 2023, Hessian-matrix-based PTQ).

<a id="environment-preparation"></a>
## Environment Preparation

The lightweight tool GPTQ selects the execution device via `DEVICE=npu|cuda|cpu`. NPU needs `torch_npu`; CUDA needs a working CUDA PyTorch; CPU has no device extension dependency (fp16 config auto-falls-back to FP32, suitable for functional validation and small models, extremely slow for large models).

```bash
# NPU environment
source /home/developer/Ascend/cann/set_env.sh
# dopt_lm internal relative import `from dopt_lm.xxx` → needs cann_lm/converter/quant in PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/cann_lm/converter/quant:${PYTHONPATH:-}

# CPU example (if fp16=true in config, runtime auto-falls-back to FP32)
DEVICE=cpu python -m dopt_lm.opt_main --help
```

> Environment setup details in `AGENTS.md` "NPU Environment (torch_npu, verified)".

<a id="three-step-usage"></a>
## Three-Step Usage

### Step 1: Generate dopt_config Scaffold

On first run, the file `--dopt-config` points to doesn't exist; the lightweight tool generates a scaffold JSON (all layers `quant_strategy="float"`) and `exit()`; you must manually edit the strategy and re-run.

```bash
cd cann_lm/converter/quant
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
python cann_lm/converter/quant/dopt_wrapper.py \
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
- `calibrate_sets/*.jsonl`: calibration corpus (generated by `download_dataset.py`, resolved relative to CWD, must run in the `cann_lm/converter/quant/` dir)
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

> Artifacts are consumed by the converter `--quant-path` (materialize int4 + swap `QuantLinear`), see [quant_design.md](./quant_design.md). PPL comparison evaluation can use [ppl_compare.py](../../../../cann_lm/converter/quant/ppl_compare.py) (usage in [ppl_compare_usage.md](./ppl_compare_usage.md)).

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
3. **Calibration corpus relative paths** are resolved against CWD; must run in the `cann_lm/converter/quant/` dir.
4. **Artifacts have no tokenizer**: when downstream consumers like converter/PPL evaluation use them, use the original HF model dir's tokenizer.

<a id="related-documentation"></a>
## Related Documentation

| Document | Related content |
|------|---------|
| [quant_design.md](./quant_design.md) | Converter-side quantization integration (`materialize` / `swap` / `quant_linear` / end-to-end flow) |
| [ppl_compare_usage.md](./ppl_compare_usage.md) | Float vs quantized artifact PPL comparison script usage |
