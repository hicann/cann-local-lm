# CLI Module Design Document

English | [简体中文](../../../zh/cann_lm_converter/cli/cli_design.md)

> The sole external entry point `cann_lm_converter.py`: converts HuggingFace models to the epair format.
>
> Model type is auto-detected from the `model_type` field in `config.json`; users do not need to specify `--model-type`. A programmatic entry `convert()` is also provided for script/test reuse.
>
> Status: Current | Last verified: 2026-08-25 | ADR: [ADR-0001](../../adr/0001-load-model-cpu-quant-subprocess-device.md) (load_model CPU vs lightweight tool subprocess device)

## Table of Contents

- [Overview](#overview)
- [convert Programmatic Entry](#convert-programmatic-entry)
- [Command-Line Parameters](#command-line-parameters)
- [Complete Call Chain](#complete-call-chain)
- [prefill/decode Separation Mode CLI](#prefilldecode-separation-mode-cli)
- [Environment Variables](#environment-variables)
- [Usage Examples](#usage-examples)
- [Runtime Environment Support (NPU / GPU / Pure CPU)](#runtime-environment-support-npu-gpu-pure-cpu)
- [converter and torch_npu Relationship (Design Decision: No Real torch_npu Dependency)](#converter-and-torch_npu-relationship-design-decision-no-real-torch_npu-dependency)
- [Document References](#table-of-contents)

---

<a id="overview"></a>
## Overview

`cann_lm_converter.py` is the sole external entry point exposed by the converter; its responsibility is to orchestrate three stages:

1. **Load**: `loader.load_model(hf_path)` → `ModelArtifact` (see [export_design.md](../export/export_design.md) "loader")
2. **Save**: `saver.save_epair` / `save_epair_prefill_decode` → epair directory (see [export_design.md](../export/export_design.md) "saver")
3. **Verify artifacts**: check that output files are complete

The CLI only does orchestration; it does not implement loading/saving logic (both are in [export_design.md](../export/export_design.md)). Quantization integration is triggered by `--quant-path` / `--quant-method`, calling the [quant module](../quant/quant_design.md)'s swap function (see [quant_design.md](../quant/quant_design.md)).

<a id="convert-programmatic-entry"></a>
## convert Programmatic Entry

```python
def convert(
    hf_path: str,
    output_dir: str,
    dtype: torch.dtype,
    max_file_size: int = 50 * 1024 * 1024 * 1024,
    use_embeddings: bool = True,   # Default pure embeddings (lookup outside graph); False goes in-graph lookup
    mode: str = "prefill-decode",   # "single" | "prefill-decode"
    num_layers: int | None = None,  # Truncate layers for debugging; None = all
    quant_path: str | None = None,  # Scenario 1: CANN lightweight tool GPTQ output; mutually exclusive with quant_method
    quant_method: str | None = None,  # Scenario 2: "gptq" auto-spawns lightweight tool; mutually exclusive with quant_path
    target_soc: str = "KirinX90",  # Target SoC (determines derived dtype + swap packing format)
) -> None:
    """Programmatic entry: HF model → epair

    load_model is hardwired to CPU (fused ops cann_lm/models/fused_ops/ only implemented on CPU),
    does not accept a device parameter. NPU GPTQ is isolated via quant_method + subprocess.

    mode:
      - "single": one shape-specialized graph
      - "prefill-decode": prefill + decode two graphs, shared weights
    """
    # Scenario 2 (--quant-method gptq): first spawn lightweight tool subprocess to run GPTQ (main process does not pre-load, avoiding 2x memory)
    # Scenario 1 (--quant-path): lightweight tool already ran offline, directly consume output
    # No quantization: dopt_out_path = None
    dopt_out_path = None
    dopt_out_is_temp = False  # True = converter temp dir (scenario 2), cleaned after swap; False = user path (scenario 1)
    if quant_method == "gptq":
        from cann_lm.converter.quant_entry import run_auto_quant
        dopt_out_path = run_auto_quant(hf_path)
        dopt_out_is_temp = True
    elif quant_path:
        dopt_out_path = quant_path

    print(f"[1/2] Loading model from {hf_path}")
    artifact = load_model(hf_path, dtype=dtype, device="cpu")  # After subprocess in scenario 2
    print(f"      Model: {type(artifact.model).__name__}")
    print(f"      Parameters: {sum(p.numel() for p in artifact.model.parameters()):,}")
    print(f"      Tokenizer files: {len(artifact.tokenizer_files)}")
    print(f"      Wrapper: {artifact.wrapper_cls.__name__}")
    print(f"      Embedding mode: {'external' if use_embeddings else 'internal'}")
    print(f"      Export mode: {mode}")

    if num_layers is not None:
        text_model = artifact.model.model
        total = len(text_model.layers)
        if num_layers > total:
            raise ValueError(f"--num-layers {num_layers} exceeds total {total} layers")
        if num_layers < 1:
            raise ValueError(f"--num-layers must be >= 1, got {num_layers}")
        text_model.layers = text_model.layers[:num_layers]
        print(f"      Truncated to first {num_layers}/{total} layers (config.num_hidden_layers unchanged)")

    # Quantization swap: scenario 1 + scenario 2 share unified path (one load → Linear swap + Embedding extraction)
    embed_quant = None
    if dopt_out_path:
        from cann_lm.converter.quant_entry import swap_from_dopt
        info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path, target_soc=target_soc)
        artifact.quant_info = info
        print(f"      Quantized {info['n_quantized']} linears (skipped {info['n_skipped_no_dopt_key']}, group_size={info['group_size']})")
        if embed_quant:
            print(f"      Embedding int8 (auto-detected from dopt): {list(embed_quant)}")
        if dopt_out_is_temp:
            import shutil
            shutil.rmtree(dopt_out_path, ignore_errors=True)
            print(f"      Cleaned temp dopt output: {dopt_out_path}")
            dopt_out_path = None

    print(f"[2/2] Saving epair to {output_dir} (target_soc={target_soc})")
    if mode == "prefill-decode":
        save_epair_prefill_decode(
            artifact, output_dir,
            max_file_size=max_file_size,
            use_embeddings=use_embeddings,
            embed_quant=embed_quant,
            target_soc=target_soc,
        )
    else:
        save_epair(
            artifact, output_dir,
            max_file_size=max_file_size,
            use_embeddings=use_embeddings,
            embed_quant=embed_quant,
            target_soc=target_soc,
        )

    print(f"\nDone. Output: {output_dir}")
```

<a id="command-line-parameters"></a>
## Command-Line Parameters

```python
def parse_args():
    parser = argparse.ArgumentParser(description="CANN-LM Converter: HF model → epair format")
    parser.add_argument("--hf-path", required=True, help="HuggingFace model path (local directory)")
    parser.add_argument("--output-dir", required=True, help="epair output directory")
    parser.add_argument("--target-soc", default="KirinX90",
                        help="Target SoC for epair export (default: KirinX90). Determines dtype: "
                             "Ascend→bfloat16, Kirin→float16.")
    parser.add_argument("--max-file-size", type=int, default=50*1024*1024*1024)
    parser.add_argument("--mode", default="prefill-decode", choices=["single", "prefill-decode"],
                        help="single: one shape-specialized graph; prefill-decode: prefill+decode two graphs + shared weights (default)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Export only first N layers (truncate for debugging). None = all. Truncates model.model.layers, config.num_hidden_layers unchanged")
    # Quantization: --quant-path (scenario 1) and --quant-method (scenario 2) are mutually exclusive
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument("--quant-path", default=None,
                             help="Lightweight tool GPTQ output path (scenario 1). Mutually exclusive with --quant-method")
    quant_group.add_argument("--quant-method", default=None, choices=["gptq"],
                             help="Quantization algorithm (scenario 2, auto-spawns lightweight tool, NPU/GPU/CPU). Mutually exclusive with --quant-path")
    return parser.parse_args()
```

> dtype is derived from `--target-soc` (`_derive_dtype`): Ascend→bfloat16 (`ChunkGatedDeltaRule` op-info only declares bfloat16), Kirin→float16. No CLI override.

| Parameter | Default | Description |
|------|------|------|
| `--hf-path` | (required) | HuggingFace model local directory |
| `--output-dir` | (required) | epair output directory |
| `--target-soc` | `KirinX90` | Target SoC (determines dtype derivation + swap packing format; Ascend→bfloat16, Kirin→float16) |
| `--max-file-size` | 50GB | Max bytes per single `weights_*.bin` |
| `--mode` | `prefill-decode` | `single` / `prefill-decode` |
| `--num-layers` | `None` | Export only the first N layers (truncate for debugging); `None` = all; truncates `model.model.layers`, `config.num_hidden_layers` unchanged |
| `--quant-path` | `None` | Lightweight tool GPTQ output path (scenario 1, auto-infers group_size/embed; mutually exclusive with `--quant-method gptq`) |
| `--quant-method` | `None` | Quantization algorithm (currently only gptq; auto-spawns lightweight tool subprocess for GPTQ, scenario 2; mutually exclusive with `--quant-path`) |

> **load_model is hardwired to CPU** (fused ops only implemented on CPU), so **no `--device` CLI parameter is exposed**. GPTQ runs in an NPU/GPU/CPU subprocess (dopt_lm supports CPU float32), isolated via `--quant-method gptq` + subprocess (subprocess device is auto-detected by `_detect_device_type`).

> **Two quantization flows**: `--quant-path <lightweight_tool_output>` (scenario 1, manually run lightweight tool GPTQ output to `--output-dir` directory, converter consumes it) or `--quant-method gptq` (scenario 2, converter auto-spawns lightweight tool subprocess to run GPTQ on NPU/GPU/CPU → swap → export, one-click). Mutually exclusive. Fine-tuning goes through the lightweight tool step-by-step. See [quant_design.md](../quant/quant_design.md).

<a id="complete-call-chain"></a>
## Complete Call Chain

> **Refactoring (2026-08-18)**: In scenario 2 (`--quant-method gptq`), `load_model` is deferred until after `run_auto_quant`, avoiding the main process cann_lm model and subprocess transformers model being held simultaneously (2× memory peak). Strategy assignment logic runs inside the subprocess using the transformers model (`dopt_wrapper.build_layer_strategy`), and swap is moved back to the main process sharing a unified path with scenario 1. **The current subprocess entry is a standalone script `quant/dopt_wrapper.py`** (`python dopt_wrapper.py ...`, system Python + torch_npu + NPU env).

```
cann_lm_converter.py (CLI)
    │
    ├── parse_args()  ← --hf-path, --output-dir, --target-soc, --max-file-size,
    │                    --mode, --num-layers,
    │                    [--quant-path | --quant-method gptq]
    │
    ▼
convert(hf_path, output_dir, dtype, ...)   # device hardwired to CPU, not exposed in CLI
    │
    ▼
(Optional) Scenario 2 pre-quantization: spawn lightweight tool subprocess      # ① GPTQ (see quant_design.md "Option C")
    │   Scenario 2 (--quant-method gptq, main process does not pre-load cann_lm):
    │     auto_quant.run_auto_quant(hf_path) → returns dopt_out_dir
    │       └─ spawn 'python dopt_wrapper.py --model-path <hf> --optimize-config config.yaml --output-dir <temp disk> --hf-type-save' (system python + torch_npu + NPU)
    │            internal dopt_wrapper.main() runs (parse_args reads --model-path / --optimize-config / --output-dir / --hf-type-save from command line)
    │            ├─ model_define(hf_path) loads transformers model
    │            ├─ build_layer_strategy(model) → dopt_config.json (with strategy)
    │            │   Linear (incl. lm_head)→Quant_INT8-4_ECO,
    │            │   Embedding→Quant_Embed_MinMax
    │            ├─ del model to free
    │            └─ weight_quant_pipline(...) (lightweight tool loads internally twice, the cost of not modifying the lightweight tool)
    │               → output → temp disk model.safetensors + dopt_config.json
    │   Scenario 1 (--quant-path): dopt_out_path = quant_path (offline output, no subprocess)
    │   No quantization: dopt_out_path = None
    │
    ▼
loader.load_model(hf_path, dtype, device)            # ② Load cann_lm model (after subprocess in scenario 2)
    │
    ├── Read config.json
    ├── model_registry.detect_model_type(hf_config)
    │     → hf_config["model_type"] → get_entry → ModelEntry(config_cls, model_cls, ...)
    ├── build_config(hf_config, entry)
    ├── entry.model_cls(config).to(dtype).eval()
    ├── entry.load_weights_fn(model, hf_path, ...)   # safetensors → key remapping → param.copy_()
    └── collect_tokenizer_files(hf_path)

    ▼
(Optional) Quantization integration (scenario 1 + scenario 2 share unified swap)        # ③ Quantization swap (see quant_design.md)
    │   if dopt_out_path (scenario 1 quant_path or scenario 2 temp disk):
    │     info, embed_quant = swap_from_dopt(artifact.model, dopt_out_path)  # one load → (quant_info, embed_quant)
    │       ├─ _swap_linear_from_sd → QuantLinear replacement
    │       └─ swap_embeddings_from_dopt → embed auto-detection

    ▼
saver.save_epair(...) / save_epair_prefill_decode(...)  # ④ Save (see export_design.md "saver")
    │
    ├── register_ge_converters()
    ├── wrapper = artifact.wrapper_cls(model, ...).eval()
    ├── wrapper.get_sample_inputs(use_embeddings)
    ├── (use_embeddings) _extract_host_side_weights + detach_externalized_lookups
    ├── torch.export.export(wrapper, sample, strict=False)
    ├── torch_delegate.export_to_epair(ep) → epair
    ├── del ep, wrapper, model; gc.collect()           # OOM protection
    ├── consolidated_weights_save(epair, output_dir, ...)
    ├── copy_tokenizer_files(...)
    └── save_model_config(artifact, output_dir, use_embeddings)
```

<a id="prefilldecode-separation-mode-cli"></a>
## prefill/decode Separation Mode CLI

Real LLM inference requires two stages: prefill (process prompt) + decode (per-token generation). After `torch.export` shape specialization, a single graph cannot cover both stages, so the CLI supports `--mode prefill-decode` to export two graphs.

### CLI Usage

```bash
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/gemma-4-E2B \
    --output-dir /path/to/output \
    --mode prefill-decode \
    --target-soc Ascend910B
```

> dtype is derived from `--target-soc` (Ascend→bfloat16, Kirin→float16).

### Interface (saver side, see export_design.md)

```python
from cann_lm.converter.saver import save_epair_prefill_decode

save_epair_prefill_decode(
    artifact, output_dir,
    *,
    max_file_size=50*1024**3,      # Max bytes per single bin
    use_embeddings=True,          # Fixed True (pure embeddings); not a CLI parameter
    save_weights=True,             # Whether to write weights_*.bin (False only writes ep_json)
    whitelist_decompositions=None, # Restrict decomposition whitelist (None = run all)
    embed_quant=None,              # Quantized embed int8+scale (returned by swap_from_dopt; None means no quantized embed)
    target_soc="KirinX90",       # Target SoC (determines swap packing format)
)
```

### Artifact Layout

```
output_dir/
├── prefill_ep_json_with_ascend_ir.json   # prefill graph
├── decode_ep_json_with_ascend_ir.json    # decode graph
├── weight_info.json                       # Shared weight index (one copy)
├── weights_0.bin                          # Shared weights (one copy)
├── model_config.json
└── tokenizer files
```

> For design details (prefill/decode two graphs sharing weights, internal flow of `save_epair_prefill_decode`, C++ loading two graphs), see [export_design.md](../export/export_design.md) "prefill/decode Separated Export".

<a id="environment-variables"></a>
## Environment Variables

> **Each env is sourced directly from a named script**: `scripts/set_env_<env>.sh` (export/compile/generate/npu four sets, non-overlapping; common part in `_env_common.sh`). Path defaults can be overridden by environment variables (`TORCH_DELEGATE_HOME` / `TORCHAIR_ROOT` / `DOPT_PYTHON` / `PROJECT_ROOT` / `CANN_SET_ENV` / `SOC_VERSION` / `GE_FUSION_SWITCH_FILE`). The table below lists variables and defaults.

| Variable | Description | Default |
|------|------|--------|
| `ASCEND_HOME_PATH` | CANN install path (set by CANN `set_env.sh`) | (Need to source CANN set_env.sh) |
| `PYTHONPATH` | Project root (`$PROJECT_ROOT`) | (Need to set) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | backend autoload switch (set to 0 to disable) | `0` |
| `TORCH_DELEGATE_HOME` | torch-delegate install path | `$TORCH_DELEGATE_HOME` |
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | fallback codegen switch (set to 0 to disable) | `0` (converter mode) |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | GE detailed logs (locate not registered etc.) | `1` (converter mode) |
| `LD_LIBRARY_PATH` | converter mode: stubs before CANN lib64; NPU/OM mode: only CANN lib64 | (per mode) |
| `DOPT_PYTHON` | System Python used by lightweight tool subprocess (triggered by `--quant-method gptq`) | _(auto-detected; `_env_common.sh` finds a python that can `import torch_npu`, no need to change when switching machines)_ |
| `CONVERTER_PYTHON` | converter venv python (set after export for developer use) | `${TORCH_DELEGATE_HOME}/.venv/bin/python` |
| `SOC_VERSION` | OM compile target SOC (om mode) | `Ascend910_9362` |
| `GE_FUSION_SWITCH_FILE` | OM compile fusion switch config (disable AUB fusion) | `/tmp/fusion_autoub_off.cfg` |

> For full environment configuration instructions, see `AGENTS.md` "Runtime Environment Setup (converter, verified)".

<a id="usage-examples"></a>
## Usage Examples

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$TORCHAIR_ROOT/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0

# Default: pure embeddings (use_embeddings=True default, dtype derived from --target-soc)
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3.5-0.8B \
    --output-dir /path/to/output

# Numerical verification: OM runner (superset of ep loop, OM vs eager/HF)
python3 examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_om.py   # gemma4: OM vs eager, maxdiff<1.0
python3 examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_om.py  # qwen3.5: OM vs EP/HF

# C++ OM compile: pure embeddings epair → .om
./examples/test_converter/export_llm_model/build/cann_lm_compile \
    output/ep_json_with_ascend_ir.json output/weight_info.json output

# prefill/decode separation (for inference, prefill+decode two graphs + shared weights; default mode)
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/gemma-4-E2B \
    --output-dir /path/to/output \
    --mode prefill-decode \
    --target-soc Ascend910B
```

> For pure embeddings design (`detach_externalized_lookups` / host_weights / `model_config.json`'s `use_embeddings` field), see [export_design.md](../export/export_design.md) "Pure Embeddings Mode".

<a id="runtime-environment-support-npu-gpu-pure-cpu"></a>
## Runtime Environment Support (NPU / GPU / Pure CPU)

The converter has three full-pipeline stages: **Load** (torch CPU) → **Quantization** (GPTQ requires compute device + swap CPU) → **Export** (torchair IR conversion).

### Environment Matrix

| Stage | NPU (Ascend910 + CANN + torch_npu) | GPU (CUDA + torch) | Pure CPU |
| ----------------------------------------------| -------------------------------------| ---------------------------------| ----------------------------------------------------|
| Load `load_model` | ✅ torch CPU | ✅ | ✅ |
| swap (`--quant-path` consumes lightweight tool output) | ✅ CPU | ✅ | ✅ |
| GPTQ (`--quant-method gptq`) | ✅ Option C subprocess | ⚠️ Lightweight tool `DEVICE=cuda` adaptation pending verification | ✅ Falls back to `cpu` (dopt_lm supports CPU float32, 7B+ extremely slow) |
| Export epair (torch.export + export_to_epair) | ✅ stubs torchair | ✅ stubs torchair (CPU) | ✅ stubs torchair |

### Current Per-Environment Availability

**NPU** (verified):
- `--quant-path` (consume output + swap + export stubs) ✅ full pipeline
- `--quant-method gptq` Option C (subprocess system Python torch_npu + stubs export) ✅ full pipeline

**GPU** (unverified):
- `--quant-path` ✅ (swap + export stubs, CPU)
- `--quant-method gptq`: lightweight tool GPTQ requires GPU (torch CUDA). **Lightweight tool's `DEVICE=cuda` branch unverified** (lightweight tool has `if DEVICE==npu` NPU-specific logic, GPU goes non-npu branch, whether Hessian cholesky etc. are GPU-compatible pending verification)
- Export ✅ (converter venv stubs torchair, CPU, no GPU needed)

**Pure CPU**:
- `--quant-path` ✅ (swap + export stubs)
- `--quant-method gptq` ✅ (**falls back to `cpu`**, dopt_lm `SUPPORTED_DEVICE_TYPES` includes `cpu`, goes `dispatch_dtype=float32`). **Cost**: 7B+ Hessian extremely slow, only suitable for small models/debugging; production recommends NPU/GPU or `--quant-path`
- Export ✅ (stubs torchair)

### What's Still Missing

**Common (`auto_quant.py`)**:
1. ✅ **Device detection**: `_detect_device_type` auto-detects NPU→`npu` / GPU→`cuda` / no accelerator→**falls back to `cpu`** (no longer errors; dopt_lm supports CPU float32)
2. ✅ **CPU fallback**: no NPU/GPU → `_detect_device_type` auto-falls back to `cpu`, warns 7B+ extremely slow

**NPU**: Option C available (subprocess + stubs export), no gaps (Option A abandoned, torchair with CANN SDK incomplete)

**GPU**:
1. ⏳ **Lightweight tool GPU adaptation verification**: lightweight tool's `if DEVICE==npu` branch (NPU-specific: Hessian moved to CPU cholesky). GPU (`DEVICE=cuda`) goes non-npu branch, verify lightweight tool GPU running GPTQ

**Pure CPU**: ✅ `--quant-method gptq` falls back to `cpu` implemented (warns slow)

### Priority

| What's missing | Priority | Status |
|--------|--------|------|
| auto_quant device detection (NPU/GPU/CPU fallback) | High (common across three environments) | ✅ Done |
| CPU GPTQ fallback + warn | High (CPU usable) | ✅ Done |
| GPU lightweight tool adaptation verification | Medium (GPU support) | ⏳ TODO |

<a id="converter-and-torch_npu-relationship-design-decision-no-real-torch_npu-dependency"></a>
## converter and torch_npu Relationship (Design Decision: No Real torch_npu Dependency)

> converter (① export epair) **does not depend on the real torch_npu python package**. It goes the stub path: converter venv does not install `torch_npu`; `cann_lm/models/fused_ops/_library.py` custom-defines `npu::` op schema+fake-meta (incl. `npu_chunk_gated_delta_rule`/`npu_dynamic_quant`/`npu_quant_matmul` etc.), so torch.export can trace; `cann_lm::` ops are already custom-defined by `_library.py`. Lib-layer stubs (`.so` in `LD_LIBRARY_PATH`) provide C symbols for the torchair GE engine.

### Why No Real torch_npu Dependency (Background and Pitfalls)

converter venv (CPU-only) components: `torch 2.9.0+cpu` + `torchair` (source-compiled `NO_ASCEND_SDK=1`, CPU+stubs usable) + lib stubs (`/mnt/workspace/torchair/build/stubs`'s `libtorch_npu.so` etc., providing C symbols for the torchair GE engine, avoiding real NPU `aclrtSetDevice→107001`).

`_library.py`'s top-level `try: import torch_npu`: import success → real path (`npu::` provided by torch_npu); `ImportError` → stub path (`_library.py` custom-defines `npu::` schema+fake). Both paths let torch.export trace `npu::` ops.

If real torch_npu is installed, it hits three coupling problems:

1. **`npu_chunk_gated_delta_rule` python version**: CANN kernel 8.5.0+ supports it (this machine's 9.2.0 satisfies), but torch_npu **python schema/fake-meta only available from ≥2.7.1.post8**; 2.7.1.post4 / 2.9.0 / 2.9.1 etc. non-post versions **lack this op** (`torch.ops.npu.npu_chunk_gated_delta_rule` AttributeError). Only 2.9.0.post6 (matches torch 2.9.0, no torch upgrade) has it.
2. **`_LazyTorchair` proxy overrides source torchair (core pitfall)**: `torch_npu/dynamo/__init__.py:113` (external dependency, torch_npu 2.9.0) at `import torch_npu` time **unconditionally** installs `sys.modules['torchair'] = _LazyTorchair()`, hijacking `import torchair` to torch_npu's bundled wrapper (`torch_npu/dynamo/torchair/`), **ignoring** the source torchair in venv. But the wrapper's `_ge_concrete_graph/fx2ge_converter.py:1336` (external dependency, torchair source) has `parse_input` with `AssertionError("meta_outputs must be instance of torch.Tensor")` (fails on `input_ids` placeholder), **source torchair has fixed this**. That is, "install real torch_npu → proxy shadows source torchair (fixed) → uses wrapper (unfixed) → GE conversion breaks".
3. **`_get_deterministic_level` API**: source torchair's `_normalize_ge_option` (`fx2ge_converter.py:1901`, external dependency, torchair source) calls `sys.modules['torch_npu'].npu._get_deterministic_level()`; torch_npu 2.9.0 lacks this API (2.9.0.post6/2.9.1 have it). Source torchair has `if 'torch_npu' not in sys.modules: → "0"` fallback, but as long as torch_npu is in sys.modules it goes to else and must call it.

### Key Conclusion: Bug Only Triggered in ① Export epair

The above bug (wrapper's `fx2ge_converter.parse_input` AssertionError) is only triggered on the **FX graph → GE IR conversion** path, i.e. ① `torch_delegate.export_to_epair → torchair._optimize_and_convert → npu_fx_compiler`. Other stages do not run FX→GE conversion and do not hit it:

| Stage | `import torch_npu`? | Runs FX→GE conversion? | Hits bug? |
|------|---|---|---|
| ① converter export epair | ✓ | ✓ (`export_to_epair`) | ✓ (`input_ids` placeholder triggers `parse_input`) |
| Step A lightweight tool GPTQ | ✓ (system python, real NPU) | ✗ (transformers eager runs GPTQ) | ✗ |
| ③ OM compile | ✗ (C++ binary, no python) | ✗ (reads pre-converted `ascend_ir` in epair) | ✗ |
| ④ NPU text generation | ✓ (system python, acl runs OM) | ✗ (`acl.mdl.execute`; CPU-side model loading only takes mask/kv in eager) | ✗ |

The proxy is installed at `import torch_npu` time, but the proxy itself does not error; only when actually calling torchair for GE conversion (`fx2ge_converter` processing placeholder) does the wrapper's bug surface. Lightweight tool / text generation / OM compile also import torch_npu (proxy also installed), but do not run FX→GE conversion → `fx2ge_converter.parse_input` not called → bug not hit.

### Design Decision: converter Goes Stub Path (No torch_npu Installed)

converter only needs `npu::` op **fake-meta** (for torch.export trace); `_library.py` can self-provide, **does not need real torch_npu's ops**. Therefore:

- converter venv **does not install `torch_npu` python package** → `import torch_npu` → `ImportError` → `_HAS_TORCH_NPU=False` (stub path) → `_library.py` custom-defines `npu::` schema+fake (incl. `npu_chunk_gated_delta_rule`/`npu_dynamic_quant`/`npu_quant_matmul` etc.).
- No torch_npu → `_LazyTorchair` proxy not installed → **source torchair used directly** (version where GE conversion works); `_normalize_ge_option` goes `torch_npu not in sys.modules → "0"` fallback, does not call `_get_deterministic_level`.
- Lib-layer stubs retained (`LD_LIBRARY_PATH`), providing C symbols for the torchair GE engine.

Benefits: bypasses P1 (torch_npu version dependency) / P2 (proxy override) / P3 (API missing) issues; no torch_npu version drift; no need to rename wrapper. **converter correctness does not depend on any specific torch_npu version**.

Side effect: after the converter exports valid artifacts, the process **hangs without exiting** at GE finalize/stub teardown (artifacts are written to disk before the hang) → worked around by "run in background + poll artifacts + kill".

### Alternative (Not Adopted, Recorded Only)

- **Real torch_npu 2.9.0.post6 + rename `dynamo/torchair`**: install torch_npu 2.9.0.post6 (has `npu_chunk_gated_delta_rule`, real path, torch_npu provides op fake) + rename venv's `torch_npu/dynamo/torchair` → `torchair.disabled` (lets `dynamo/__init__.py`'s `if not exists(...)` hit (external dependency, torch_npu 2.9.0.post6) → does not install proxy → source torchair used directly). Works (uses official ops + source torchair), but depends on torch_npu version (must be ≥2.7.1.post8 and match torch) + requires one-time wrapper rename, high complexity. Not adopted by design — converter does not need real torch_npu ops (fake-meta self-provided by `_library.py` suffices).



| Document | Related Content |
| -----------------------------------------------------------------------------------------| -------------------------------------------------------------------------------------------------------|
| [export_design.md](../export/export_design.md) | `loader.load_model` / `saver.save_epair` / `ExportWrapper` / `model_registry` / artifact layout / C++ loading side |
| [quant_design.md](../quant/quant_design.md) | `--quant-path` / `--quant-method` quantization integration / `swap_from_dopt` (main path) |
| [README.md](../README.md) | Cross-module call chain overview / model differences overview / accuracy summary |
