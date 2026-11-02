# Quick Start

English | [简体中文](../zh/QUICKSTART.md)

> End-to-end example: HF model → epair → omc → deployment. Environment dependencies are described in [env_dependencies.md](env_dependencies.md), and build commands in [Build Guide](build_guide.md).

> **Prerequisites**: CANN + P1 venv are ready (see [Build Guide](build_guide.md) Part 1 + runtime prerequisites), and `cann_local_lm_converter-*.run` has been obtained (build instructions in [Build Guide](build_guide.md) §2.1). HarmonyOS deployment also requires the signed engine wheel `cann_local_lm_engine-*+signed*.whl` (produced by `--pkg` or `bash scripts/cann_lm_engine/build_package_ohos.sh`).

---

## Installation

```bash
# 1. Extract .run package (automatically runs set_env.sh to print environment hints)
bash /path/to/cann_local_lm_converter-0.1.0-aarch64.run --extract=/opt/cann-local-lm

# 2. Install cann_lm whl into P1 venv (used by converter, provides cllm-converter CLI)
export CONVERTER_PYTHON=/path/to/venv/bin/python
$CONVERTER_PYTHON -m pip install /opt/cann-local-lm/wheels/cann_local_lm-*.whl

# 3. (Only needed for quantization) Set DOPT_PYTHON to the system python that has torch_npu
export DOPT_PYTHON=/path/to/python3
# Or auto-detect (requires CANN sourced first): source $HOME/Ascend/cann/set_env.sh && source /opt/cann-local-lm/set_env.sh --npu
# First deployment only — if system python lacks cann_lm/dopt_lm: $DOPT_PYTHON -m pip install /opt/cann-local-lm/wheels/cann_local_lm-*.whl --no-deps --user
```

---

## End-to-end Example: HF → epair → omc → Deployment

### 0. Prepare env

```bash
source $HOME/Ascend/cann/set_env.sh    # CANN
source /opt/cann-local-lm/set_env.sh         # cann-local-lm (LD_LIBRARY_PATH + GE configuration)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 # Prevent torch from autoloading npu backend (P1 venv has no torch_npu)
export CONVERTER_PYTHON=/path/to/venv/bin/python   # P1 venv python
```

### ① HF → epair (export prefill + decode)

> You need to download the model from [HuggingFace](https://huggingface.co) or [ModelScope](https://modelscope.cn) to a local directory in advance, and point `--hf-path` to the download directory. Download methods:
> ```bash
> # HuggingFace
> pip install huggingface_hub
> huggingface-cli download <org>/<model-name> --local-dir /path/to/model
>
> # ModelScope (faster in China)
> pip install modelscope
> modelscope download --model <org>/<model-name> --local_dir /path/to/model
> ```

**Standard export:**

```bash
cllm-converter \
    --hf-path <your-model-path> \
    --output-dir ./output/e2e \
    --mode prefill-decode --target-soc KirinX90
```

> The converter will hang at GE finalize during background execution (known issue), but the artifacts have already been written to disk by then. You can run it in the background + poll for artifact files + kill to work around this.

**Quantized export (optional):**

> **Quantization special requirements**:
> - GPTQ quantization requires **NPU device acceleration** (the dopt subprocess runs on the NPU), requiring an additional **P2 environment** (system python + torch_npu, isolated from P1 venv, cannot be shared).
> - Must `source $HOME/Ascend/cann/set_env.sh` first, then `source /opt/cann-local-lm/set_env.sh --npu` (auto-detects `DOPT_PYTHON` + sets `DEVICE=npu`); if detection fails, manually `export DOPT_PYTHON=/path/to/python3`.
> - The torchaudio version must match the system python's torch, otherwise an `undefined symbol` error will be reported.
> - Without an NPU, you can skip GPTQ and use `--quant-path` to consume existing dopt artifacts for offline quantization.

```bash
# NPU quantization env (CANN first, then --npu which needs ASCEND_HOME_PATH for DOPT_PYTHON detection)
source $HOME/Ascend/cann/set_env.sh
source /opt/cann-local-lm/set_env.sh --npu

# Method 1: converter automatically invokes dopt (one-step, requires NPU)
cllm-converter \
    --hf-path <your-model-path> \
    --output-dir ./output/quant \
    --target-soc KirinX90 \
    --quant-method gptq

# Method 2: manually run dopt + converter consumes the output (two-step; dopt requires NPU, converter does not)
# Preparation: copy calibration data and config from cann_lm package directory to working directory
PKG_QUANT=$($DOPT_PYTHON -c "import cann_lm.converter.quant as q; print(q.__path__[0])")
mkdir -p ./output/dopt_quant
cp -r "$PKG_QUANT/calibrate_sets" ./output/dopt_quant/
cp "$PKG_QUANT/config.yaml" ./output/dopt_quant/
# config.yaml: can configure calibration dataset (train_files), sample count (ptq_samples), sequence length (cutoff_len), etc.
# dopt_config.json: per-layer quantization strategy (layer_strategy); each layer can be configured with quant_strategy individually (e.g. Quant_INT8-4_ECO / Quant_Embed_MinMax / float to skip)

# Step 1a: First run (generates dopt_config.json and exits)
cd ./output/dopt_quant
$DOPT_PYTHON -m dopt_lm.opt_main \
    --model-path <your-model-path> \
    --dopt-config ./dopt_config.json \
    --optimize-config ./config.yaml \
    --output-dir .
# Step 1b: Edit dopt_config.json quantization strategy (Linear→Quant_INT8-4_ECO, Embedding→Quant_Embed_MinMax, can be adjusted per layer)
# Step 1c: Re-run (executes GPTQ quantization)
$DOPT_PYTHON -m dopt_lm.opt_main \
    --model-path <your-model-path> \
    --dopt-config ./dopt_config.json \
    --optimize-config ./config.yaml \
    --output-dir .
cd ../..    # Back to repo root

# Step 2: converter consumes dopt artifacts (offline quantization, no NPU required)
cllm-converter \
    --hf-path <your-model-path> \
    --output-dir ./output/quant \
    --target-soc KirinX90 \
    --quant-path ./output/dopt_quant
```

> Quantization scheme (W4A8) is described in [Quantization Design](../zh/cann_lm_converter/quant/quant_design.md).

### ② epair → omc (compilation)

```bash
BIN=/opt/cann-local-lm/cann_lm_compile
OUT=./output/e2e       # Standard export; for quantized export use ./output/quant
for st in prefill decode; do
  mkdir -p "$OUT/$st"
  "$BIN" "$OUT/${st}_ep_json_with_ascend_ir.json" "$OUT/weight_info.json" "$OUT/$st/$st"
done
```

---

## Artifact Overview

Complete artifacts after ① export + ② compilation (`$OUT/` = `./output/e2e` or `./output/quant`):

| File | Source | Description |
|------|--------|-------------|
| `prefill_ep_json_with_ascend_ir.json` | ① export | prefill segment EP JSON (② compilation input) |
| `decode_ep_json_with_ascend_ir.json` | ① export | decode segment EP JSON (② compilation input) |
| `weight_info.json` | ① export | Weight index (② compilation input) |
| `weights_0.bin` | ① export | Weight data (loaded during ② compilation) |
| `host_weights_info.json` | ① export | Host-side weight index (weights not in the graph, such as embeddings; loaded during ② compilation) |
| `host_weights_0.bin` | ① export | Host-side weight data (such as embeddings; loaded during ② compilation) |
| `model_config.json` | ① export | Model configuration (for inference) |
| `tokenizer.json` / `tokenizer_config.json` | ① export | tokenizer (for inference) |
| `prefill/prefill.omc` | ② compilation | prefill segment omc (deployed to device for execution) |
| `decode/decode.omc` | ② compilation | decode segment omc |
| `prefill\|decode/weight/*.bin` | ② compilation | External weights when weight separation is enabled |

---

## Deploy to HarmonyOS Device

### 1. Push engine + model artifacts

```bash
# Signed engine wheel (.so embedded in cann_lm/lib/, from --pkg or build_package_ohos.sh)
hdc file send build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl /data/local/tmp/cann_lm_engine/

# Model artifacts (omc + weights + tokenizer + model_config)
hdc file send ./output/e2e /data/local/tmp/cann_lm_engine/
```

### 2. Install engine + write config on device

```bash
hdc shell
cd /data/local/tmp/cann_lm_engine

# Install engine wheel (.so auto-discovered from cann_lm/lib/)
pip3 install cann_local_lm_engine-0.1.0+signed-py3-none-any.whl

# Write testcase.json (model paths + tokenizer + sampling params)
cat > testcase.json <<'EOF'
{
    "tokenizer_type": "BPE",
    "tokenizer_path": "/data/local/tmp/cann_lm_engine/output/e2e/tokenizer.json",
    "model_path_prefill": "/data/local/tmp/cann_lm_engine/output/e2e/prefill/prefill.omc",
    "model_path_decode": "/data/local/tmp/cann_lm_engine/output/e2e/decode/decode.omc",
    "model_meta_config": "/data/local/tmp/cann_lm_engine/output/e2e/model_config.json",
    "model_weight_dir_prefill": "/data/local/tmp/cann_lm_engine/output/e2e/prefill/weight",
    "model_weight_dir_decode": "/data/local/tmp/cann_lm_engine/output/e2e/decode/weight",
    "sentences": [{
        "prompt": "Hello",
        "seed": 99, "topK": 1, "topP": 1.0,
        "temperature": 1.0, "maxGenTokens": 128
    }]
}
EOF
```

> **No pip available**: `unzip cann_local_lm_engine-*.whl -d . && export PYTHONPATH=.:$PYTHONPATH && export CANN_LM_LIB=./cann_lm/lib/libcann_lm_engine.so`
>
> **Weight directories**: if `GE_EXTERNAL_WEIGHT=2` was not set during compilation (embedded weights), `model_weight_dir_*` can point to the omc's directory.

### 3. Run inference

```bash
# Single-shot generation
cllm generate --config testcase.json --prompt "Hello"

# Interactive multi-turn chat
cllm chat --config testcase.json

# Diagnose paths (no .so load, just checks file existence)
cllm generate --config testcase.json --show-paths
```

> `.so` discovery order: `--lib PATH` > `$CANN_LM_LIB` > wheel's `cann_lm/lib/`. If loading fails, add the `.so` directory to `LD_LIBRARY_PATH` (`libc++_shared.so` must be in the same directory).
>
> C++ black-box tests (bbit/smoke) can be built separately via `bash build.sh --smoke`, pushed to device, and run with `./bbit --config testcase.json`. See [engine Python README](../../cann_lm/engine/python/README.md).
