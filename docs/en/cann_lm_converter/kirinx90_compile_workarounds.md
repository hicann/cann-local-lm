# kirinx90 .omc Compile Workaround List (Operation Log)

English | [简体中文](../../zh/cann_lm_converter/kirinx90_compile_workarounds.md)

> Purpose: Records all blocking points and workarounds encountered when compiling CANN-LM small networks (`examples/test_converter/export_kirin_ops_net`, covering 8 kirinx90 ops) into kirinx90 `.omc`, for reproduction/future consolidation.
> Scope: aarch64 HarmonyOS host + cann-9.2.0 (kirinx90 mobile model path).
> Conclusion: `.omc` successfully generated (`kirin_ops_kirinx90.omc`, `atc --mode=6` confirms `soc_version[KirinX90]` + weights embedded). **However this .omc contains aicpu ops (MatMul/BatchMatMul/SoftmaxV2/SplitV, see aicpu heterogeneous analysis), kirinx90 doesn't support AICPU → runtime invalid**; `aicore_only_kirinx90.omc` (zero aicpu) is the truly usable artifact.

## Background

- Target SOC: `KirinX90` (HarmonyOS Kirin chip, CANN 9.2.0; artifact extension `.omc`, not Ascend's `.om`).
- Host: aarch64 (HarmonyOS kernel `hce2.aarch64`, not x86; so kirinx90 OM compile is **native**, no cross-compile).
- kirinx90 op library: `aic-kirinx90-ops-info-nn.json` (118 ops) + `kernel/kirinx90/ops_nn`+`ops_math` (pre-compiled kernels).
- Original pain point: kirinx90's kernel/engine coverage is sparser than Ascend910, and several ops (Gelu/L2Normalize forward entry) are **missing from packaging** in ops-nn package.

> ⚠️ **Hard constraint: kirinx90 doesn't support AICPU; any op heterogeneously dispatched to AICPU is definitely problematic**.
> kirinx90 is mobile path (`MobileModelHelper`), **no aicpu execution capability** (unlike Ascend910's `DNN_VM_AICPU`/`DNN_VM_AICPU_ASCEND`). Any op assigned by `EnginePlacer` to aicpu engine (`DNN_VM_AICPU*`) or converter fallback to aicpu op (e.g. `ConcatV2` aicpu variant) → compile-time mobile path doesn't recognize aicpu subgraph / runtime no aicpu kernel execution → **certain failure**.
> **Corollary**: all workarounds must keep ops in aicore (or host/GE_LOCAL/RTS), **cannot fallback to aicpu**. B4's originally planned `cat→ConcatV2 (lands aicpu)` scheme **invalid** — need to change to aicore-available ops or host-side preprocessing (e.g. scatter indices fixed/constantized at converter stage, not in runtime graph).

## Environment Preparation (toolkit isolation, doesn't pollute original A3 env)

Original A3 op environment at `/home/developer/Ascend` (untouched); build independent toolkit:

```
mkdir -p /mnt/workspace/cann-kirinx90
# ① toolkit (lib64/GE/platform ini/opp config)
bash Ascend-cann-toolkit_9.2.0~weekly.20260909.01_linux-aarch64.run --full \
    --install-path=/mnt/workspace/cann-kirinx90 --force --quiet
# ② A3-ops (completes opp structure op_tiling/op_api/... + lib/linux/aarch64, for GE OS validation)
bash Ascend-cann-A3-ops_9.2.0~weekly.20260909.01_linux-aarch64.run --install \
    --type=toolkit --install-path=/mnt/workspace/cann-kirinx90 --force --quiet
# ③ kirinx90 ops-nn + ops-math + ops-transformer (kirinx90 pre-compiled kernels + config)
#    ⚠️ Must use --upgrade --quiet, not --full --force!
#    --full --force doesn't overwrite toolkit's built-in opp/built-in ops-info (read-only), causing ops-info missing ops
#    --upgrade --quiet correctly overwrites opp/built-in ops-info + kernel binary
bash cann-kirinx90-ops-nn_9.1.0_linux-aarch64.run --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
bash cann-kirinx90-ops-math_9.1.0_linux-aarch64.run --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
bash cann-kirinx90-ops-transformer_9.1.0_linux-aarch64.run --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
```

Pre-run env (must unset old CANN vars to prevent pollution, general principle see AGENTS.md "Multiple CANN coexistence must unset first"):
```bash
# Clear multi-CANN pollution (only unset CANN-related, keep system Python/numpy/HOME)
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH TOOLKIT_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH PYTHONPATH SOC_VERSION GE_FUSION_SWITCH_FILE GE_EXTERNAL_WEIGHT ASCEND_VISIBLE_DEVICES DEVICE
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/buildtools/python-3.12.9/bin:/usr/local/cmake/bin
source /mnt/workspace/cann-kirinx90/cann-9.2.0/set_env.sh
export SOC_VERSION=KirinX90                 # CamelCase, matches KirinX90.ini's SoC_version=KirinX90
unset  GE_FUSION_SWITCH_FILE                # First compile use GE default fusion
export GE_EXTERNAL_WEIGHT=0                 # Embed weights
# libpython prepend (fix _ctypes undefined symbol _PyErr_SetLocaleString, see A5) + build_kx + CANN lib64
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:$PROJECT_ROOT/build_kx:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
# A7 fix: sitecustomize shields torch_npu (avoid GE embedded python loading system torch_npu → cann-9.0.0 GE double-load heap corruption)
#   sitecustomize.py content: import sys; [sys.modules.__setitem__(m, None) for m in ('torch_npu','torch','torchair','torch_delegate')]
#   PYTHONPATH prepend directory containing sitecustomize.py; GE embedded libpython loads site, blocking torch_npu from import
export PYTHONPATH="<sitecustomize_dir>:${PYTHONPATH:-}"
# ASAN build (for locating heap corruption, D section): cann_lm_compile built into build_kx_asan/, GE lib not instrumented but cann_lm code instrumented
#   cmake -S $PROJECT_ROOT -B build_kx_asan -DASCEND_INSTALL_PATH=<cann-kirinx90>/aarch64-linux -DENABLE_ASAN=ON ...
# cd to output dir to run (onlinec→.omc rename same dir same device, see A6)
```

## Op Package Source Build (ops-nn / ops-math / ops-transformer)

> Scenario: pre-compiled `.run` packager missing ops (e.g. l2_normalize forward apt entry, see T2/B3) or need to rebuild from source with changes, build from gitCode three repos `build.sh --pkg --soc=kirinx90`, then `--upgrade --quiet` install back to cann-kirinx90. Complementary to above "Environment Preparation" installing pre-compiled packages; artifacts are the `.run` installed by above `--upgrade`.

### Source Repos

| Repo | Path | build.sh |
|----|------|----------|
| ops-nn | `/mnt/workspace/gitCode/ops-nn` | Contains `norm/l2_normalize` forward apt entry (T2 landed) |
| ops-math | `/mnt/workspace/gitCode/ops-math` | — |
| ops-transformer | `/mnt/workspace/gitCode/ops-transformer` | — |

Three repos build.sh same framework: `--soc=kirinx90` → `COMPUTE_UNIT_SHORT=kirinx90` → CMake `-DASCEND_COMPUTE_UNIT=kirinx90`; `--pkg` → `build_binary` (ccec compiles kernels) + `build_pkg` (CPack makeself packs `.run`). `build_torch_extension_whl` gracefully skips without `build` module (doesn't depend on torch). `kirinx90` outside `SOC_TO_ARCH` but doesn't affect `--pkg` (that mapping only for `--run_example` simulation path).

### env (A8 + A5 are key)

> ⚠️ **A8**: System `~/.bashrc` sources old CANN setting `TOOLCHAIN_HOME` (ccec/asc_opc resolves asc headers via it) → without unset uses wrong CANN headers, `cast` etc. AIV_ONLY kernels all fail. **Must unset first (incl. `TOOLCHAIN_HOME`; AGENTS.md unset list originally wrote `TOOLKIT_HOME`, actual system var is `TOOLCHAIN_HOME`, unset both for safety) then source cann-kirinx90**.

```bash
# kx_env.sh (reusable)
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH TOOLKIT_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH PYTHONPATH SOC_VERSION GE_FUSION_SWITCH_FILE GE_EXTERNAL_WEIGHT ASCEND_VISIBLE_DEVICES DEVICE ASCEND_RUNTIME_OPTIONS
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/buildtools/python-3.12.9/bin:/usr/local/cmake/bin
source /mnt/workspace/cann-kirinx90/cann-9.2.0/set_env.sh
export SOC_VERSION=KirinX90
# A5: libpython prepend (fix _ctypes undefined symbol _PyErr_SetLocaleString)
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
```

Verify: `which ccec` → `/mnt/workspace/cann-kirinx90/cann-9.2.0/bin/ccec`, `TOOLCHAIN_HOME=/mnt/workspace/cann-kirinx90/cann-9.2.0/toolkit`, `python3 -c "import ctypes"` OK.

### Build (three repos, one by one)

```bash
source /tmp/opencode/kx_env.sh   # or inline above env
for pkg in ops-nn ops-math ops-transformer; do
  cd /mnt/workspace/gitCode/$pkg
  rm -rf build build_out          # Clean old artifacts prevent cross-SOC pollution (cmake_init also deletes CMakeCache)
  bash build.sh --pkg --soc=kirinx90 -j16
done
# Artifacts: each repo build_out/cann-kirinx90-<pkg>_9.1.0_linux-aarch64.run
```

### Install (--upgrade --quiet, not --full --force)

> ⚠️ Must use `--upgrade --quiet`, not `--full --force`: `--full --force` doesn't overwrite toolkit's built-in opp/built-in ops-info (read-only) → ops-info missing ops; `--upgrade --quiet` correctly overwrites opp/built-in ops-info + kernel binary (consistent with above "Environment Preparation" step ③).

```bash
source /tmp/opencode/kx_env.sh
for pkg in ops-nn ops-math ops-transformer; do
  bash /mnt/workspace/gitCode/$pkg/build_out/cann-kirinx90-${pkg}_9.1.0_linux-aarch64.run \
    --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
done
```

> Before install can backup manual workaround files (A3 `scene.info`/`version.info`, B3 hand-written `l2_normalize.py`, current ops-info json) for rollback. Measured `--upgrade` doesn't overwrite opp-level A3 files (package only writes `share/info/ops_<pkg>/`, doesn't touch `ops_legacy/` symlink targets), B3 `l2_normalize.py` correctly replaced by new package's same-name file (md5 identical to hand-written version).

### Verification (verified 2026-09-17)

| Item | Result |
|----|------|
| Three repos `.run` | ops-nn 41M / ops-math 19M / ops-transformer 24M, `KIRINX90_OPS-*_RUN_PACKAGE` makeself identifier ✓ |
| Kernel on disk | `tbe/kernel/kirinx90/{ops_nn,ops_math,ops_transformer}/` total **647 `.o`** (ops_nn 422 / ops_math 180 / ops_transformer 45) ✓ |
| l2_normalize (B3/T2) | **New package includes** `impl/ops_nn/dynamic/l2_normalize.py` (9688B, md5 `821b3bef…` identical to original hand-written wrapper bytes → source repo T2 landed) + kernel `L2Normalize_1_high_performance.o`(26KB) + `kernel/config/kirinx90/ops_nn/l2_normalize.json` ✓ |
| ops-info-nn.json | fresh 229300B (replaced old 232183B), dict 120 ops, includes L2Normalize/Gelu/Sigmoid/RmsNorm/ScatterNdUpdate/GatherV2/MatMulV3/BatchMatMulV3/ScaledMaskedSoftmaxV2 ✓ |
| A3 scene.info/version.info | `--upgrade` didn't overwrite opp-level workaround, md5 identical to pre-install ✓ |
| Non-fatal | ops-nn log `On [kirinx90], [xxx] not supported` (matmul/conv variants kirinx90 doesn't support, INFO skip); ops-math `gen_esb` stage `... source file does not found` (add/mul/transpose etc. kernel source not in ops-math repo, provided by toolkit built-in/other packages, build continues) |

### End-to-End (simple_network)

After installing three packages, use `examples/test_converter/export_epair_and_compile/simple_network.py` (rmsnorm+matmul+mul+add) to verify full chain:

```bash
# 1. Export epair (converter env + --target-soc KirinX90, auto patches MatMul→MatMulV3)
source /tmp/opencode/kx_env.sh
export SOC_VERSION=KirinX90
cd $PROJECT_ROOT/examples/test_converter/export_epair_and_compile
source ../common/check_env.sh               # ASCEND_HOME_PATH set then don't re-source; CONVERTER_PYTHON=deps/torch-delegate/.venv
python export_and_save_epair.py             # → epair_model/ep_json_with_ascend_ir.json (contains 3× MatMulV3, 0× plain MatMul)

# 2. Compile .omc (build/cann_lm_compile 0715 link + cann-kirinx90 runtime lib override)
cd epair_model
export SOC_VERSION=KirinX90; unset GE_FUSION_SWITCH_FILE; export GE_EXTERNAL_WEIGHT=0
export LD_LIBRARY_PATH=$PROJECT_ROOT/build:${LD_LIBRARY_PATH:-}
$PROJECT_ROOT/build/cann_lm_compile ep_json_with_ascend_ir.json weight_info.json simple
# → simple.omc (mobile path, T1 auto onlinec→.omc; rc=1 is GE teardown abort, .omc already produced, tolerated)
```

`.omc` verify: `atc --mode=6` → `soc_version[KirinX90]`/`weight_size=36352B` embedded; `atc --mode=1` → `graph[1]` SubGraph_0 27 ops contains **2× MatMulV3 (aicore)**+ rmsnorm decomposition(Square/ReduceMean/Rsqrt/Mul)+ Add, **zero aicpu ops**.

> Simplification findings: ① **No need to rebuild `build_kx`** — `build/cann_lm_compile` (0715 link) + `LD_LIBRARY_PATH` override to cann-kirinx90/lib64, runtime loads new weekly (≥20260909) libruntime accepting `rtSetSocVersion(KirinX90)`, 0715 binary and cann-kirinx90 lib ABI compatible direct compile; ② **No need for A7 sitecustomize** — this machine's torch_npu on Python 3.11.4, while GE embedded libpython is 3.12.9, won't load 3.11.4's torch_npu, no double GE heap corruption (teardown's ASAN leak summary is after `.omc` written, benign).

## Golden Reference Data Generation (`gen_golden_with_dump.py`)

`examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py`: uses `torch.export` + `DumpInterpreter` (`torch.fx.Interpreter`) to run wrapper in CPU eager, dumps **each FX node's output** as `.bin`, as the "standard answer" (golden) for per-op accuracy comparison. OM after compile is binary, can't read intermediate values; need to cooperate with `cann_lm_compile --tap` (see next section) to expose specified intermediate values as OM outputs, then compare maxdiff with golden's `node_<name>.bin`.

> Goes through **wrapper entry** (`Gemma4ExportWrapper`/`Qwen3_5ExportWrapper`'s forward → `_drive_layers` → directly drives layers), same graph as converter export + OM compile (not raw `model.forward`), so golden's `node_*.bin` and OM per-op comparable.

### Usage

```bash
# converter env (① CPU venv + torchair, no NPU)
source scripts/set_env_export.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
  --model-type {gemma4|qwen3_5} \
  --hf-path <HF model dir> \
  --output-dir <output dir> \
  [--seq-len N] [--dtype D] [--num-layers N] [--no-embeddings] [--no-intermediates] [--seed N]
```

| Option | Default | Description |
|---|---|---|
| `--model-type` | required | `gemma4` / `qwen3_5` (determines `load_model` → `wrapper_cls`)|
| `--hf-path` | required | HF model dir (safetensors weights + config)|
| `--output-dir` | required | golden output dir |
| `--seq-len` | 128 | prefill seq length (**must match OM contract**)|
| `--dtype` | float16 | `float16`/`fp16`/`bfloat16`/`bf16`/`float32`/`fp32` (**must match OM**, see below)|
| `--num-layers` | all | Truncate to N layers (**must match OM's layer count**)|
| `--no-embeddings` | off | input_ids into graph (default embeddings externalized, matches converter export flow)|
| `--no-intermediates` | off | Only store input+final output, don't dump intermediate `node_*.bin` (large seq saves memory, see below)|
| `--seed` | 0 | Global `torch.manual_seed`, fixes `get_sample_inputs`'s random input_ids, making golden reproducible across runs |

### Artifact Layout (under `--output-dir`)

```
input_<idx>_<name>[_<j>].bin   Each wrapper input tensor (FX output name, single-output node uses node.name)
output_<idx>.bin               Each graph output tensor (logits + KV etc.)
node_<name>[_<j>].bin          Each FX node output (skipped with --no-intermediates)
golden_meta.json               Meta info (model_type/seq_len/num_layers/dtype/seed/inputs/outputs/intermediates, incl. each .bin's name/shape/dtype)
```

### Key Constraints

1. **dtype must match OM**: OM is `DT_FLOAT16` then use `--dtype float16`; OM is `DT_BFLOAT16` then use `--dtype bfloat16`. Use `atc --mode=1` to check OM's `input_desc.dtype`. gemma4 model native bf16, but some OM exported fp16 (e.g. `gemma4_1layer_prefill.omc` is `DT_FLOAT16`) → golden must `--dtype float16` for numerical path consistency. bfloat16 tensor losslessly upcast to float32 on disk write (numpy doesn't support bf16), meta top-level `dtype` records model dtype (`torch.bfloat16`), per-tensor `dtype` records disk dtype (`float32`), compare both sides upcast to float32.

2. **seq_len / num_layers must match OM**: golden's input shape must align per-dim with OM's `input_desc.shape` (mask/pos/embeddings/PLE/KV seq dim + KV layer count). Use `atc --mode=1` to get OM's 9 Data input shapes and compare against generation params.

3. **`--seed` reproducible**: `get_sample_inputs` uses `torch.randint` to generate input_ids (no seed differs each time), `--seed 0` fixes for cross-run golden bit-identical (logits + all node `.bin` all equal).

4. **`--no-intermediates` saves memory**: default `DumpInterpreter` captures **all** node outputs to memory (`self.dump`), large seq (e.g. 128) OOM; with `--no-intermediates` runs `gm(*sample_inputs)` directly (no intermediate capture), only stores input+final output, large seq doesn't OOM. For per-op comparison still need intermediates, use small seq (e.g. 8) for full golden, or batched tap.

### Examples

```bash
# qwen3.5 5-layer fp16 seq8 (full golden, incl. intermediates, small seq no OOM)
python gen_golden_with_dump.py --model-type qwen3_5 \
  --hf-path /mnt/workspace/models/Qwen/Qwen3.5-0.8B \
  --seq-len 8 --num-layers 5 --dtype float16 --seed 0 \
  --output-dir out/golden_qwen35_5l

# gemma4 6-layer bf16 seq8 (full golden; bf16 lossless upcast float32 on disk)
python gen_golden_with_dump.py --model-type gemma4 \
  --hf-path /mnt/workspace/models/unsloth/gemma-4-E2B-it \
  --seq-len 8 --num-layers 6 --dtype bfloat16 --seed 0 \
  --output-dir out/golden_gemma4_6l

# gemma4 1-layer fp16 seq128 aligning gemma4_1layer_prefill.omc (only input+output, large seq use --no-intermediates avoid OOM)
python gen_golden_with_dump.py --model-type gemma4 \
  --hf-path /mnt/workspace/models/unsloth/gemma-4-E2B-it \
  --seq-len 128 --num-layers 1 --dtype float16 --seed 0 --no-intermediates \
  --output-dir out/golden_gemma4_1l_seq128
```

> golden's `node_<name>.bin` filename = FX node name (single-output node same as output tensor name), directly used as `cann_lm_compile --tap`'s tap name (strip `node_` prefix and `.bin` suffix), see next section.

## Tapping Intermediate Ops to OM/OMC Output (Per-Op Accuracy Comparison)

`cann_lm_compile` supports `--tap`: expose any intermediate FX output tensor as additional graph output, compiled into OM/OMC. Use case: OM after compile is binary, can't read intermediate values; `--tap` makes specified intermediate values OM outputs → run OM to get tapped output → compare maxdiff with golden's `node_<name>.bin` (`gen_golden_with_dump.py`'s per-FX-node output), locate accuracy deviation points. Applies to both Ascend `.om` and Kirin `.omc`.

### Usage

```bash
cann_lm_compile <model_json> <weights_index> [output_name] --tap "<name1>;<name2>;..."
# or --tap=<name1>;<name2>  (= form)
#   - Single --tap, multiple names separated by ';' or ',' (auto-trim spaces)
#   - Multiple --tap flags error ("multiple --tap not allowed"), only one allowed
#   - Empty list (--tap ";") errors "no valid names"
```

### What is a tap name

**FX output tensor name** (not FX op type like `aten.mm`, not GE op name like `MatMul`):
- Single-output FX node: use **node.name** (FX auto-named, e.g. `mm`, `add_1`, `add_2`).
- Multi-output node (returns tuple): use `name_0`/`name_1` (not `node.name`).

Parse path: `EpairModelLoader` 3-arg constructor takes tap list → `GeGraphBuilder::SetTapOutputs` → `FinalizeGraph` (`ge_graph_builder.cc`) for each name calls `ValueRegistry.Resolve(name)`, via `ascend_ir.mapping.outputs` (FX→GE boundary mapping computed by converter) alias resolution to `{GE GNode, output_index}`, appended to graph outputs (after original outputs), then `graph->SetOutputs`. Not found reports `TAP_NOT_FOUND` (confirm name is FX output tensor name; multi-output use `name_0`).

> Name source: after exporting epair view `ep_json_with_ascend_ir.json`'s `graph_module.graph.nodes[i].outputs[0].as_tensor.name` (FX node output name); or directly view golden's `out/golden_*/node_<name>.bin` filename (`gen_golden_with_dump.py`'s DumpInterpreter-captured FX node name, single-output node same as output tensor name).

### Verification (simple_network)

```bash
# 1 original output add_2 + tap mm + tap add_1 → OM should have 3 outputs
cann_lm_compile epair_model/ep_json_with_ascend_ir.json epair_model/weight_info.json \
  epair_model/simple_tap --tap "mm;add_1"
# → simple_tap.om, [PASS] aclgrphBuildModel succeeded (GE accepts graph with tap)

# Verify OM output count (atc --mode=1 check NetOutput node's input count = graph output count)
atc --mode=1 --om=epair_model/simple_tap.om --json=/tmp/simple_tap.json
python -c "import json; g=json.load(open('/tmp/simple_tap.json'))['graph'][0]; \
  no=[o for o in g['op'] if o.get('type')=='NetOutput'][0]; \
  print('outputs:', len(no['input']))"   # → 3 (add_2 + mm + add_1 two taps)
```

No-tap control: `simple.om`'s `NetOutput` input count = 1 (only `add_2`). With `--tap "mm;add_1"` = 3, exactly +2 = two taps. GE auto-adds TransData/Cast for tap output format conversion (output name becomes `trans_Cast_*`), but tensor is the tap's intermediate value; tapped intermediate values won't be dead-code-eliminated by GE (kept alive), this is the prerequisite for per-op comparison.

### Verification (real model gemma4_1layer prefill, kirinx90 .omc)

`examples/test_converter/export_llm_model/output/gemma4_1layer/`'s 1-layer gemma4 prefill, tap after `scatter_nd_update` (KV cache update ×2) + `FIA` (FusedInferAttentionScore) ops:

```bash
# tap names (FX output tensor names) → GE ops (FX→GE mapping in ascend_ir.mapping.outputs):
#   npu_scatter_nd_update   → ScatterNdUpdate:0            (KV cache K update result)
#   npu_scatter_nd_update_1 → ScatterNdUpdate_1:0          (KV cache V update result)
#   getitem_10              → FusedInferAttentionScore:0   (FIA attention output)

# kirinx90 env (clear multi-CANN pollution + source cann-kirinx90 + LD override + A5 libpython + cd output dir avoid onlinec cross-device)
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH TOOLKIT_HOME \
      TOOLCHAIN_HOME CMAKE_PREFIX_PATH LD_LIBRARY_PATH SOC_VERSION GE_FUSION_SWITCH_FILE \
      GE_EXTERNAL_WEIGHT ASCEND_VISIBLE_DEVICES DEVICE PYTHONPATH
source /mnt/workspace/cann-kirinx90/cann-9.3.0/set_env.sh
export SOC_VERSION=KirinX90; unset GE_FUSION_SWITCH_FILE; export GE_EXTERNAL_WEIGHT=0
export LD_LIBRARY_PATH=$PROJECT_ROOT/build:/opt/buildtools/python-3.12.9/lib:${LD_LIBRARY_PATH}
cd examples/test_converter/export_llm_model/output/gemma4_1layer   # A6: onlinec→.omc rename same dir same device

$PROJECT_ROOT/build/cann_lm_compile prefill_ep_json_with_ascend_ir.json weight_info.json \
  gemma4_1layer_prefill_tap --tap "npu_scatter_nd_update;npu_scatter_nd_update_1;getitem_10"
# → [INFO] tap outputs (3): npu_scatter_nd_update npu_scatter_nd_update_1 getitem_10
#   [PASS] aclgrphBuildModel succeeded  [PASS] mobile model path: renamed onlinec -> gemma4_1layer_prefill_tap.omc
#   All 3 stages passed  →  gemma4_1layer_prefill_tap.omc (1.1 GB, embedded weights)

# Verify OM output count (atc --mode=1 check NetOutput input count = graph output count)
atc --mode=1 --om=gemma4_1layer_prefill_tap.omc --json=/tmp/opencode/tap.json
python -c "import json; g=json.load(open('/tmp/opencode/tap.json'))['graph'][0]; \
  no=[o for o in g['op'] if o.get('type')=='NetOutput'][0]; print('outputs:', len(no['input']))"
# → 6  (original 3: logits+2KV  +  3 tap: 2×scatter_nd_update + FIA)
```

No-tap control: `gemma4_1layer_prefill.omc`'s `NetOutput` input count = 3 (logits + 2 KV). With `--tap "npu_scatter_nd_update;npu_scatter_nd_update_1;getitem_10"` = 6, exactly +3 = three taps. Run this omc to get original 3 outputs + KV update results ×2 + FIA attention output, for per-op comparison with golden.

> tap name source: after exporting epair `python -c "import json; j=json.load(open('prefill_ep_json_with_ascend_ir.json')); [print(n['outputs'][0]['as_tensor']['name'], n['target']) for n in j['graph_module']['graph']['nodes']]"` lists FX node output names + targets, pick scatter/FIA node output names. Or directly view golden `out/golden_*/node_*.bin` filenames.

### Coordination with golden

Run `gen_golden_with_dump.py` to generate golden (`out/golden_<model>/`, incl. `node_<name>.bin` per-FX-node output) → use `--tap` to compile same-name intermediate values into OM → run OM to get tapped output → `np.fromfile` read both sides' `.bin` compare maxdiff. tap names directly from golden dir's `node_*.bin` filenames (strip `node_` prefix and `.bin` suffix), i.e. FX node name, one-to-one correspondence.

> `--tap` full implementation see `cann_lm/converter/export/csrc/`: `tools/compile_main.cc` (CLI parsing + `SplitTapList`), `backend/ge_graph_builder.cc` (`FinalizeGraph` tap resolution loop), `inc/external/torch_delegate/epair_model_loader.h` (3-arg constructor overload). `csrc/tools/README.md` has cann_lm_compile's full options (env vars, weight separation, `.air` save etc.).

## Workaround List

### A. toolkit/env layer

| # | Blocker | Workaround |
|---|---|---|
| A1 | kirinx90 ops scattered, original A3 env can't be touched | Isolated install independent toolkit `/mnt/workspace/cann-kirinx90` (three steps above), original `/home/Developer/Ascend` zero changes |
| A2 | OS validation fails `The OS linux is not within the support list of {}` | Install **A3-ops** to complete `tbe/{op_tiling,op_api,op_master_device,op_tiling_device,impl_mode}` + `lib/linux/aarch64` (GE's `GetOppSupportedOsAndCpuType` scans these os/cpu subdirs to build support list; toolkit alone only has `config/impl/kernel/op_host`) |
| A3 | `scene.info`/`version.info` missing (install interrupted by timeout, didn't generate) | `chmod u+w` read-only `tbe/kernel/` dir, copy `scene.info` (`os=linux`) + `version.info` from existing `cann-9.2.0/cann-9.2.0` to new opp; fixes OS validation |
| A4 | SOC string case | `SOC_VERSION=KirinX90` (CamelCase, matches `KirinX90.ini`); lowercase `kirinx90` is only opp config dir name, not socVersion, GE can't find `kirinx90.ini` |
| A5 | **New weekly (≥20260909)** source CANN后 `_ctypes` import fails (`undefined symbol: _PyErr_SetLocaleString`) → TBE/tvm unusable → `aclgrphBuildModel failed` | `python3` NEEDED `libpython3.12.so.1.0`, after source CANN `LD_LIBRARY_PATH` changes to resolve to **system old version** `/lib/aarch64-linux-gnu/libpython3.12.so.1.0` (missing `_PyErr_SetLocaleString`, that symbol only in newer patches; `/opt/buildtools/python-3.12.9/lib/libpython3.12.so.1.0` has it). Standalone `python3 -c "import ctypes"` (no source) normal, only reproduces after source. **Fix**: `LD_LIBRARY_PATH` prepend `/opt/buildtools/python-3.12.9/lib` (has correct libpython), or `LD_PRELOAD=/opt/buildtools/python-3.12.9/lib/libpython3.12.so.1.0` |
| A6 | **New weekly (≥20260909)** `compile_main.cc`'s `onlinec→{output}.omc` rename fails: `std::filesystem::rename: Invalid cross-device link` | mobile path writes model to **CWD**'s `onlinec`, while `compile_main.cc` T1 uses `std::filesystem::rename("onlinec", output+".omc")` to rename; when CWD and output dir cross partitions (e.g. CWD=`/mnt/workspace`, output=`/tmp`) rename fails → process exits abnormally + TBE subprocess cascade errors. **Fix**: `cd` to output dir to run (`onlinec` and target `.omc` same dir same device); or output dir same partition as CANN-LM repo |
| A7 | **torch_npu double GE heap corruption** (masks real error): GE BuildModel stage embeds libpython (`ge_python_runtime_manager.cc:228 EnsureLibphanLoaded`, loads `/opt/buildtools/python-3.12.9/lib/libpython3.12.so.1.0`), embedded python uses **system sys.path** → loads system site-packages' `torch_npu` → `libtorch_npu.so` NEEDED links **cann-9.0.0**'s `libge_runner`/`libge_compiler`/`libascendcl` (`ldd libtorch_npu.so` visible), double-loaded with cann_lm_compile's already-loaded **kirinx90 cann-9.2.0** GE lib → global GE state/heap conflict → glibc `corrupted size vs. prev_size` heap corruption (not ASAN detectable, GE lib not instrumented). **This heap corruption masks real compile errors** (e.g. B4 `get_kernel_type_enum` exception), must eliminate first to locate. **Fix**: `sitecustomize.py` sets `sys.modules['torch_npu']=None` (+ `torch`/`torchair`/`torch_delegate`), `PYTHONPATH` prepend dir containing `sitecustomize.py` → GE embedded python loads site, blocks torch_npu from subsequent import. Note: TBE fusion `fusion_util.py:get_torch_npu_version` will `import torch_npu` fail (`import of torch_npu halted; None in sys.modules`), but DEBUG-level fallback, non-fatal | `sitecustomize.py` + `PYTHONPATH` (see Environment Preparation section); venv python can't directly as PYTHONHOME (venv lacks stdlib `encodings`, will crash); `PYTHONHOME=synthetic dir` (system stdlib symlink + venv/tbe_pylibs site-packages) can isolate but crashes at BuildInitialize, not adopted |
| A8 | **`TOOLCHAIN_HOME` pollution causes ops-math/ops-nn source repo compile ccec uses old CANN asc headers** (distinct from A1-A7's OM compile scenario, this is ops source repo `build.sh --pkg` compile scenario): System `~/.bashrc` at startup sources `/home/developer/Ascend/cann/set_env.sh` (cann-9.0.0) setting `TOOLCHAIN_HOME=/home/developer/Ascend/cann-9.0.0/toolkit`. Compiling ops-math (`/mnt/workspace/gitCode/ops-math`, `bash build.sh --pkg --soc=kirinx90`) ccec/asc_opc resolves via `TOOLCHAIN_HOME` to **cann-9.0.0**'s `aarch64-linux/asc` headers (not cann-kirinx90 cann-9.2.0), causing `int4x2_t` redefinition (cann-9.0.0 `kernel_utils_macros.h` vs bisheng `__clang_cce_types.h`) + `GetBlockIdxImpl`/`GetSubBlockIdxImpl` etc. undeclared + `recursive aicore function call is not allowed` → ccec compile cast etc. AIV_ONLY kernels all fail (`ld.lld: cannot open ..._0.o`, .o not generated) → entire package `binary` stage fails. AGENTS.md unset list originally wrote `TOOLKIT_HOME` (old name), **actual system var is `TOOLCHAIN_HOME`** (multiple CHAIN), missing unset hits it. **Fix**: unset list add `TOOLCHAIN_HOME` (and `TOOLKIT_HOME` both unset for safety), after source cann-kirinx90 set_env ccec uses cann-9.2.0's asc headers, cast etc. compiles | env unset add `TOOLCHAIN_HOME` (see Environment Preparation section + AGENTS.md "Multiple CANN coexistence must unset first") |

### B. network/converter layer (workaround kirinx90 no engine ops)

> 8 target ops: `MatMul`/`BatchMatMul`/`RmsNorm`/`Gelu`/`L2Normalize`/`Sigmoid`/`GatherV2`/`ScatterNdUpdate`.
> Of these 6 (MatMul/BatchMatMul/RmsNorm/Gelu/Sigmoid/GatherV2/ScatterNdUpdate) have engine after installing kirinx90 ops packages; need workaround 3 places:

| # | Blocker | Workaround | Landing |
|---|---|---|---|
| B1 | `Muls` (`scores*1/√d` scalar mul) no engine | `self.scale` from python `float` to **tensor buffer** → `scores*scale` becomes `ge.Mul` (has engine), not `ge.Muls` | `kirin_ops_net.py` |
| B2 | `TensorMove` (scatter out-of-place converter's const→mutable copy) no engine | ① `kv_cache` from `nn.Parameter` to **forward input** (placeholder→Data mutable variable; `ge.ScatterNdUpdate` is ref node, ref input can't reference Const, so cache must be input not Parameter); ② `compile_to_epair` **monkeypatch** `npu_scatter_nd_update.default` converter, direct emit `ge.ScatterNdUpdate` (skip `ge.TensorMove`) | `kirin_ops_net.py` + `kirin_ops_net.py::compile_to_epair` |
| B3 | `L2Normalize` forward impl missing + tvm CCE codegen crashes | ops-nn package packaging gap (missing forward `ops_nn/dynamic/l2_normalize.py` dynamic entry, only `_grad`; legacy `ops_legacy/dynamic/l2_normalize.py` is tvm/tbe-DSL, kirinx90 `tvm CCECodegenImpl` crashes EB9999) → **hand-written AscendC-apt wrapper** `l2_normalize.py` (mimic `rms_norm.py`/`gelu_apt.py`), uses `asc_op_compiler.compile_op`→ccec compiles pre-packaged `ascendc/l2_norm/l2_normalize.cpp`, bypass tvm CCE path | Written to kirinx90 opp `ops_nn/dynamic/l2_normalize.py` |

> Above B1–B3 are `kirin_ops_net.py` small network's **in-place workarounds** (modify modeling/monkeypatch). Real model layers (gemma4/qwen3.5) go through **Kirin converter patches** mechanism (converter export patches GE converter, see B4–B6).

### B.2 Kirin converter patches mechanism (real model layer workaround)

> When `--target-soc` is Kirin series (`KirinX90` etc.), `cann_lm/converter/export/ge_converters/ge_converter.py` detects Kirin soc via `active_config().target_soc` (injected by saver via `lite_export_context(LiteConfig(target_soc=...))`), loads `_kirinx90_converter_patches.py`; `KIRINX90_CONVERTER_PATCHES=1` can force-on override (backward compat). After `register_ge_converters()` then `register_kirinx90_patches()` overrides (last-write-wins) 6 classes of op converters:
> 1. `aten.mm`/`matmul`/`bmm` → `MatMulV3`/`BatchMatMulV3` (kirinx90 aicore only V3, no plain MatMul)
> 2. `npu_rotary_mul` → `RotaryPositionEmbedding` (kirinx90 has this op, goes arch35)
> 3. `aten.mul` (scalar `Muls`) → `Mul` (scalar to `Const` tensor)
> 4. `npu_scatter_nd_update` → direct emit `ScatterNdUpdate` (skip `TensorMove`)
> 5. `aten.slice` → `StridedSlice` (non-V2, int32; kirinx90's `ConstToAttrStridedSliceV2Fusion` is forbidden closed, would const-to-attr V2 → `StridedSliceD` no aicore kernel → lands host CPU → unknown shape → mobile fails)
> 6. `aten.cat` → `ConcatD` (non-V2) — **this fails in gemma4 real layer, see B4**

| # | Blocker | Workaround | Landing |
|---|---|---|---|
| B4 | **`cat→ConcatD` apt compile fails**: `concat_d_apt.py:216` hardcodes `ascendc_src_file="arch31/concat_d.cpp"`, kernel uses `KERNEL_TYPE_AIV_ONLY` (`arch31/concat_d.cpp:17`). `compile_op`→`KernelInfoInfer.get_kernel_type_enum("AIV_ONLY")` (`asc_op_compile_base/asc_op_compiler/kernel_info_infer.py:195`) goes **else branch `raise Exception("...not support in current core version")`** — because kirinx90's soc (`KirinX90`) not in `is_v200`/`is_v220`/`is_c310` whitelist (`ascendc_common_utility.py`: `is_v200` only recognizes `Ascend310P/610`, `is_v220` only `Ascend910B/910_93`, `is_c310` only `Ascend950/350`), all three False → else raises exception → `opRes.result=exception` → `second gen tbe op_info not found` → `BuildSingleOp ConcatD failed`. **Note**: ① Not int32 unsupported (ConcatD op_info dtype includes int32, `concat_d_apt.py` DTYPE_MAP has int32, kernel copies by byte doesn't depend on dtype); ② Not arch31/arch32 mismatch (`get_kernel_type_enum` raises before ccec compile, unrelated to arch); ③ `GatherV2` also `AIV_ONLY` but GE compile succeeds (goes different path not triggering `get_kernel_type_enum`, pending re-test); ④ Source comment `KERNEL_TYPE_AIV_ONLY` (`conversion/concat_d/op_kernel/arch31/concat_d.cpp` uncommitted change) can pass `get_kernel_type_enum`, but **introduces new errors** — tiling info missing `_pattern` + `dsl_compile_info is nullptr` (`TilingPrepare4ConcatD`), comment not complete fix. Trigger scenario: gemma4 real layer KV cache scatter indices construction (`modeling_gemma4.py:68` `torch.cat([col_b,col_n,col_pos],dim=0)`, int32, K/V cache each once) | ⚠️ **Cannot fallback `cat→ConcatV2` (lands aicpu) — kirinx90 doesn't support AICPU (see background hard constraint)**. Correct direction: scatter indices' int32 cat is **coordinate concatenation**, doesn't enter aicore compute — at converter stage constantize/fix indices (K/V cache scatter's b/n/pos coordinates can be precomputed when seq_len fixed), or use host-side (`DNN_VM_GE_LOCAL`/RTS) ops instead, **don't leave aicpu ops in graph** | `cann_lm/converter/export/ge_converters/_kirinx90_converter_patches.py::_kirinx90_aten_cat` + `cann_lm/models/gemma4/modeling_gemma4.py::_scatter_kv_cache` |

### C. artifact layer (.omc actually already generated)

| # | Blocker | Workaround |
|---|---|---|
| C1 | `aclgrphSaveModel` fails "model is illegal" (`model.data==nullptr`) — red herring | kirinx90 goes `mobile_model_helper.cc` mobile model path, **`aclgrphBuildModel` directly writes model to file `onlinec`** (default name, CWD), doesn't fill in-memory `ModelBufferData.data`; so `aclgrphSaveModel`'s null-data validation fails, but model already saved → **rename `onlinec → kirin_ops_kirinx90.omc`** |

### D. Diagnostic means (non-resident, for locating)

- `--asan` rebuild `cann_lm_compile` (targeting cann-kirinx90, `build_kx_asan/`): confirms not our heap corruption — no ASAN errors, `model.length` from garbage value (~2^48 pointer) to sane (6778732); located `model.data null` is mobile path behavior, not memory corruption.
- debug logs `ASCEND_GLOBAL_LOG_LEVEL=0`: located to `mobile_model_helper.cc::SaveToOmModel → model_file_saver.cc::SaveCompiledModelToFile output_model_file: onlinec`.
- **ASAN + sitecustomize locates torch_npu double GE (A7) + reveals B4**: non-ASAN build + system python directly crashes `corrupted size vs. prev_size` (A7 heap corruption), crash point non-deterministic (BuildInitialize/BuildModel/Finalize varies with heap layout), can't locate real error. **Key three steps**: ① `sitecustomize.py` sets `sys.modules['torch_npu']=None` to shield torch_npu (eliminate A7 heap corruption); ② ASAN build (`build_kx_asan`, `-DENABLE_ASAN=ON`) changes memory layout so glibc no longer crashes (GE lib not instrumented but cann_lm code instrumented, exit 255 not crash); ③ `ASCEND_GLOBAL_LOG_LEVEL=0` SLOG redirect then grep ERROR locates `get_kernel_type_enum`/`Cannot find engine`/`failed to recompile` etc. real compile errors (B4's `AIV_ONLY not support` located this way)
- **Manual apt compile_op reproduce single op**: `python3 -c "from impl.ops_math.dynamic import concat_d_apt; concat_d_apt.concat_d(...)"` (PYTHONPATH includes cann-kirinx90 python site-packages + tbe opp), directly see `compile_op`'s raised Exception (B4's `current kernel type: KERNEL_TYPE_AIV_ONLY is not support in current core version`), faster than GE full flow and no A7 interference

## Verification

```bash
atc --mode=6 --om=.../kirin_ops_kirinx90.omc
# system   info: atc_version[9.2.0], soc_version[KirinX90], framework_type[].
# resource info: memory_size[16801792 B], weight_size[78848 B], stream_num[2], event_num[3].
```
- `soc_version[KirinX90]` ✓
- `weight_size[78848 B]` = weights embedded (matches epair `weights_0.bin` 77KB) ✓
- File 677K, magic `494d4f44`("IMOD") = OM/OMC format ✓
- 8 target ops all appear in exported `ep_json_with_ascend_ir.json` ascend_ir (`check_op_coverage.py` verified 8/8 hit kirinx90 ops-info-nn.json)

## End-to-End Flow (reproduction)

```bash
# 1. Export epair (CPU venv + torchair stubs, SOC-agnostic)
source scripts/set_env_export.sh
python examples/test_converter/export_kirin_ops_net/kirin_ops_net.py export  --output .../kirin_ops.pt2
python examples/test_converter/export_kirin_ops_net/kirin_ops_net.py compile --ep-path .../kirin_ops.pt2 --output-dir .../kirin_ops_epair
# 2. Compile OM/OMC (kirinx90)
source /mnt/workspace/cann-kirinx90/cann-9.2.0/set_env.sh
export SOC_VERSION=KirinX90; unset GE_FUSION_SWITCH_FILE; export GE_EXTERNAL_WEIGHT=0
export LD_LIBRARY_PATH=$PWD/build_kx:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
./build_kx/cann_lm_compile .../kirin_ops_epair/ep_json_with_ascend_ir.json .../kirin_ops_epair/weight_info.json kirin_ops_kirinx90
# 3. mobile path artifact rename (aclgrphSaveModel will report red herring failure, ignore)
mv onlinec .../kirin_ops_kirinx90.omc
```

## aicpu Heterogeneous Analysis (⚠️ kirinx90 doesn't support AICPU, heterogeneous to aicpu definitely problematic)

> **Restate hard constraint**: kirinx90 mobile path has no aicpu execution capability. Below table's "8-op network compiled .omc success" is **early misjudgment** — aicpu ops in mobile path actually can't run (compile-time may pass, but runtime/artifact invalid). Any `EnginePlacer` assigned to `DNN_VM_AICPU*` or converter fallback to aicpu op, **all considered failure, must workaround to aicore**.

8-op network (`kirin_ops_net.py`) build log `IsUnknownShapeNode` shows **ops landed in aicpu** (`DNN_VM_AICPU*`) — these **definitely problematic**, not dependable:

| aicpu op | engine | count | Root cause |
|---|---|---|---|
| `MatMul` | DNN_VM_AICPU | 5 | kirinx90 aicore only has `MatMulV3` (ops-info-nn.json no plain MatMul), torchair emits plain `ge.MatMul`, no auto-rewrite → aicpu |
| `BatchMatMul` | DNN_VM_AICPU | 2 | Same (only `BatchMatMulV3`) |
| `SoftmaxV2` | DNN_VM_AICPU_ASCEND | 1 | kirinx90 only has `MaskedSoftmaxWithRelPosBias`/`ScaledMaskedSoftmaxV2`, no plain SoftmaxV2 |
| `SplitV` | DNN_VM_AICPU_ASCEND | 1 | From `qkv.chunk(3)`, kirinx90 aicpu承接 |

On aicore (`AIcoreEngine`): Add / Cast / Gelu / L2Normalize / Mul / RmsNorm / ScatterNdUpdate / Sigmoid / Transpose. `DNN_VM_GE_LOCAL` (Const/Reshape), `DNN_VM_RTS` (Identity/Output) not aicpu (host/RTS, usable).

### Why MatMul Can't Map to MatMulV3 (root cause)
1. torchair converter emits **plain `MatMul`**: `aten/mm.py → ge.MatMul(self, mat2, None)`, op type is `"MatMul"`; `aten/bmm.py → ge.BatchMatMul`.
2. **kirinx90 aicore only accepts V3 variants**: `aic-kirinx90-ops-info-nn.json`'s matmul class only has `MatMulV3`/`BatchMatMulV3`/`GemmV2`/`TransposeBatchMatMul`, **no plain MatMul/BatchMatMul**.
3. **GE doesn't do MatMul→MatMulV3 rewrite**: GE source `MatMulV3` only has `InferSymbolShape` + `REGISTER_LOWERING_WITH_EXISTED(MatMulV3, LowerMatMul)` (lowering for **already-MatMulV3** nodes, not rewriting from plain MatMul). Two different op types, no rewrite pass.
4. **Ascend910 comparison**: `kirin_ops_910.om` (atc --mode=1) op type is plain `MatMul`(5)/`BatchMatMul`(2), **not mapped to V3** — 910 aicore natively supports plain MatMul; kirinx90 cut plain MatMul keeping only V3, torchair still emits plain → mismatch → aicpu.
5. `ge.MatMulV3` in this torchair's `ge_apis` (auto_generated) **not generated** (`hasattr(ge,'MatMulV3')=False`; `BatchMatMulV3`/`ScaledMaskedSoftmaxV2` same; `MaskedSoftmaxWithRelPosBias` callable), can't directly `ge.MatMulV3(...)`, must `torchair.ge.custom_op("MatMulV3",...)` manually build V3 node (inputs x1/x2/bias/offset_w, attrs transpose_x1/x2/offset_x/op_impl_mode).

> Conclusion: **Not "can't map", but this torchair version's converter doesn't emit MatMulV3, ge_apis doesn't expose MatMulV3 callable function**; kirinx90 aicore only accepts V3. Both sides version/registration not aligned. Cleanest fix: add SOC-based `MatMulV3`/`BatchMatMulV3` branch to torchair `aten.mm`/`aten.bmm` converter (or monkeypatch + `custom_op`).

## Pure aicore Network (zero aicpu) — `aicore_only_net.py`

Network rewritten for "zero aicpu heterogeneity" (per "even if not fully covered" trade-off): only uses verified aicore ops, **avoids MatMul/BatchMatMul/SoftmaxV2/SplitV** (no usable aicore variant).

| Use (all aicore) | Avoid (kirinx90 lands aicpu) |
|---|---|
| GatherV2 · Add · RmsNorm · Sigmoid · Gelu · Mul · L2Normalize · ScatterNdUpdate | MatMul/BatchMatMul (aicore only V3, ge_apis not callable) · SoftmaxV2 (only Masked variant) · SplitV (chunk) |

Structure (no linear layer/no attention, pure norm+activation+residual+scatter):
`x → GatherV2(pos) → Add → RmsNorm → Sigmoid(gate)/Gelu(ffn) → Mul → Add(residual) → L2Normalize → ScatterNdUpdate(kv_cache) → out, new_cache`

### Verification (IsUnknownShapeNode engine assignment)
- **DNN_VM_AICPU: 0** ✓✓✓ (zero aicpu)
- AIcoreEngine: 8 (Add/Add_1/Gelu/L2Normalize/Mul/RmsNorm/ScatterNdUpdate/Sigmoid)
- DNN_VM_GE_LOCAL: 7 (Const/Reshape) · DNN_VM_RTS: 2 (Identity/Output)

> Note: log `runtime.cc InitAiCpuCnt:aicpuNum=6` is runtime init fixed value (not graph's aicpu op count; 8-op network with 9 aicpu ops also 6). Actual aicpu op count see `IsUnknownShapeNode`'s `DNN_VM_AICPU*` count.

### Artifacts
- `out/aicore_only_kirinx90.omc` (350K, `soc_version[KirinX90]`, `weight_size[2560 B]`, `stream_num[1]`, weights embedded, atc verified valid)
- `out/golden_aicore/`: `input_x/input_kv_cache/output_out/output_new_cache.bin` + `golden_meta.json` (input x[2,4,64]/kv_cache[2,8,64] fp16)
- CLI same as `kirin_ops_net.py`: `eager/export/compile/golden`

## Weight-Separated .omc (externalWeight=2 + ge.externalWeightDir fix)

mobile path defaults externalized weights to `tmp_weight_<pid>_<sid>/` temp dir, cleaned after build → .omc references lost weights (see earlier =2 incomplete analysis). Fix: set `ge.externalWeightDir` specifying weights land in output's same dir `weight/`.

### Three Mode Conclusions (post-fix)

| GE_EXTERNAL_WEIGHT | Result | Details |
|---|---|---|
| `=0` embedded | ✅ Usable | .omc 350K, `weight_size=2560B` (weights fully embedded), self-contained |
| `=1` per-file | ❌ Fails | Weights externalized to `weight/` (3 `weight_<hash>` + meta.json), but `aclgrphBuildModel failed 1343225857` (per-file mode mobile path unsupported) |
| `=2` all-in-one | ✅ **Weight separation success** | .omc 349K + `weight/delegate_graph_weight_combined` (1.5K externalized weights) + `weight/meta.json` (hash→offset mapping) |

### =2 Weight-Separated Artifacts (three-piece set)
```
out/aicore_only_kirinx90.omc              349K  (structure+kernel+1024B embedded weights)
out/weight/delegate_graph_weight_combined  1.5K  (3 externalized weights, offset 0/512/1024)
out/weight/meta.json                       800B  (hash→file/offset mapping)
```
atc: `weight_size[1024 B]` (partial embedded) + `memory_size[16789504 B]`. .omc references externalized `delegate_graph_weight_combined`, runtime .omc + `weight/` loaded together.

### Fix Points (compile_main.cc)
- **`ge.externalWeightDir` must go through `init_options` (`aclgrphBuildInitialize`, global context), not `build_options`** — in build_options rejected (`50331649 options illegal`, not in legal build option whitelist)
- Set it = output's same dir `weight/` → build time `graph_external_weight_manager.cc:104 GetThreadLocalContext().GetOption(OPTION_EXTERNAL_WEIGHT_DIR)` (`"ge.externalWeightDir"`, defined in `ge_common_api_types.h:180`) reads → weights directly land `weight/` (fallback `tmp_weight_<pid>_<sid>/`, `file_constant_utils.cc:748`)
- `ge.externalWeight` (0/1/2 value) still goes `build_options` (legal)
- Change: `init_options` section (when `GE_EXTERNAL_WEIGHT≠0` create `weight/` dir + `emplace("ge.externalWeightDir", weight/)`); `build_options` section only keeps `ge.externalWeight`

## AIR Export (SAVE_AIR option)

Optional: dump GE graph IR (AIR) to readable file before compile, for viewing/debugging.

Usage (parallel to OM compile, env-triggered):
```bash
SAVE_AIR=1 ./build_kx/cann_lm_compile <epair>/ep_json_with_ascend_ir.json <epair>/weight_info.json <output_name>
```
Artifacts (output's same dir):
- `ge_proto_00000_graph_0_<output>_graph.txt` — GE graph proto text (graph structure/ops/attr, readable)
- `ge_onnx_00000_graph_0_<output>_graph.pbtxt` — ONNX pbtxt

Implementation: after GetGEGraph, before BuildModel, `SAVE_AIR=1` → `aclgrphDumpGraph(graph, "<output>_graph", len)` (`ge_ir_build.h:170`) → `GraphUtils::DumpGEGrph`(.txt) + `DumpGrphToOnnx`(.pbtxt), subgraphs also dumped. This is **pre-compile** GE graph IR; post-compile fused graph IR already in `ep_json_with_ascend_ir.json` (JSON). Independent of OM/weight separation (`GE_EXTERNAL_WEIGHT`/`SOC_VERSION` etc.), can run simultaneously.

## gemma4 Real Model .omc Compile (2026-09-20 empirical)

> Use gemma4-E2B (5.03B, 35 layers) first N layers truncated export prefill+decode epair, compile kirinx90 `.omc`. **First 1 layer and first 5 layers both compile success (exit=0)**.

### Environment

```bash
# Export env (converter venv, CPU, no torch_npu)
source scripts/set_env.sh
# --target-soc KirinX90 → converter detects Kirin soc via active_config(), auto loads kirin patches
# (KIRINX90_CONVERTER_PATCHES=1 can force-on override, generally no need to set manually)
# converter --target-soc KirinX90 → dtype fp16 (_derive_dtype derived, no Cast)

# Compile env (kirinx90 CANN, non-ASAN build)
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH TOOLKIT_HOME CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH ASCEND_VISIBLE_DEVICES DEVICE SOC_VERSION GE_FUSION_SWITCH_FILE GE_EXTERNAL_WEIGHT
source /mnt/workspace/cann-kirinx90/cann-9.2.0/set_env.sh
export SOC_VERSION=KirinX90 GE_EXTERNAL_WEIGHT=0
export LD_LIBRARY_PATH="/opt/buildtools/python-3.12.9/lib:${LD_LIBRARY_PATH:-}"
```

### Export

```bash
$CONVERTER_PYTHON cann_lm/converter/cann_lm_converter.py \
  --hf-path <gemma-4-E2B-it> --output-dir <out> \
  --mode prefill-decode --seq-len 128 --num-layers <N> \
  --target-soc KirinX90
# --target-soc KirinX90 detected via active_config() → auto loads kirin patches
```

Patch effects (epair ascend_ir verification):

| Original op | After patch | kirinx90 reason |
|---|---|---|
| `RotaryMul` | `RotaryPositionEmbedding` | kirinx90 no RotaryMul kernel, has RotaryPositionEmbedding |
| plain `MatMul` | `MatMulV3` | kirinx90 aicore only V3 |
| `TensorMove` + `ScatterNdUpdate` | direct emit `ScatterNdUpdate` | kirinx90 no TensorMove kernel |
| `Muls` (scalar mul) | `Mul` (scalar to Const tensor) | kirinx90 no Muls engine |
| `StridedSliceV2` | `StridedSlice` (non-V2, int32) | V2's forbidden closed lands host CPU → mobile fails |
| `cat`→`ConcatD` | patch exists but **gemma4 scenario not triggered** (scatter indices already constantized at converter stage) | B4's apt compile failure not reproduced |

### Compile

```bash
build/cann_lm_compile <out>/prefill_ep_json_with_ascend_ir.json <out>/weight_info.json <prefix>_pf
build/cann_lm_compile <out>/decode_ep_json_with_ascend_ir.json  <out>/weight_info.json <prefix>_dc
# mobile path directly produces .omc, exit=0 (no ASAN leak, non-ASAN build)
```

### Artifacts

| Model | prefill .omc | decode .omc | memory | weight |
|---|---|---|---|---|
| 1 layer | 1.03 GB | 1.03 GB | 85.6 MB / 19.0 MB | 1.03 GB |
| 5 layers | 1.32 GB | 1.32 GB | 95.6 MB / 29.0 MB | 1.31 GB |

> weight_size ~1.03-1.31 GB = embed_tokens(262144×1536 fp16 ~786MB) + lm_head + per_layer_model_projection + layer weights. 1 layer and 5 layer weight diff ~280MB = 4 layers × ~70MB/layer (gemma4 ~70MB fp16 per layer).

### .omc I/O (1 layer prefill, 13 inputs)

| Input | dtype | shape |
|---|---|---|
| mask_full | BOOL | [1,1,128,2048] |
| mask_sliding | BOOL | [1,1,128,2048] |
| position_ids | INT32 | [1,128] |
| past_key_values_0_0 | FP16 | [1,1,2048,256] |
| past_key_values_0_1 | FP16 | [1,1,2048,256] |
| new_kv_cache_pos | INT32 | [128] |
| embeddings | FP16 | [1,128,1536] |
| output_pos | INT32 | [1] |
| per_layer_token_identity | FP16 | [1,128,35,256] |

> `per_layer_token_identity` shape `[1,128,35,256]` — 35 = full `num_hidden_layers` (truncation only changes `model.model.layers[:N]`, doesn't change config.num_hidden_layers). Layer 0 only takes `[:,:,0:1,:]`, remaining 34 layers PLE data in input but model doesn't access, **non-random** (weights/KV/PLE layer 0 data all real).

### Key Experience

1. **Must use non-ASAN build cann_lm_compile**: ASAN build (`build.sh cpp --asan`) conflicts with kirinx90 CANN's TVM/Python path → ASAN leak → `aclgrphBuildModel failed`. Non-ASAN build (`build.sh cpp`, built with kirinx90 CANN env) → exit=0.
2. **cann_lm_compile must be built with kirinx90 CANN env** (not 0715): new GE's `libgert.so` references `aclrtHostGetDevicePointerAddrRange` (new runtime API), 0715's `libascendcl.so` lacks this symbol → link failure. Build with kirinx90 CANN (GE + runtime match) → passes.
3. **Kirin converter patches are export-time SOC-triggered** (not compile-time): `--target-soc KirinX90` (or `SOC_VERSION=KirinX90` via `export_and_save_epair.py`) passed to `LiteConfig.target_soc`, converter detects Kirin soc via `active_config()` then loads `_kirinx90_converter_patches.py` to override converters (last-write-wins). `KIRINX90_CONVERTER_PATCHES=1` can force-on override (generally no need to set manually).
4. **`--target-soc KirinX90`** triggers `_derive_dtype` → fp16 (no need for `--dtype float16`, also doesn't support `--dtype bfloat16`).
5. **gemma4's `scatter_kv_to_cache` uses `torch.cat`** (B4/ConcatD failure point), but 1-layer/5-layer scenarios **didn't trigger** B4 — scatter indices already constantized at converter stage (`inline_fused_attn_seq_constants` makes lifted tensors shared by name across graphs → GE no longer generates ConcatD node).

### New GE Install (build+install from GE source repo)

GE source repo `/mnt/workspace/gitCode/cann/ge/`'s compiled `.run` packages installed to two CANN:

```bash
# Install from GE repo build_out
bash /mnt/workspace/gitCode/cann/ge/build_out/cann-ge-compiler_9.2.0_linux-aarch64.run --run --quiet --install-path=/mnt/workspace/cann-kirinx90
bash /mnt/workspace/gitCode/cann/ge/build_out/cann-ge-executor_9.2.0_linux-aarch64.run --run --quiet --install-path=/mnt/workspace/cann-kirinx90
# Also install to 0715 (if needed)
bash /mnt/workspace/gitCode/cann/ge/build_out/cann-ge-compiler_9.2.0_linux-aarch64.run --run --quiet --install-path=/home/developer/Ascend/cann-9.2.0-0715
bash /mnt/workspace/gitCode/cann/ge/build_out/cann-ge-executor_9.2.0_linux-aarch64.run --run --quiet --install-path=/home/developer/Ascend/cann-9.2.0-0715
```

Note: new GE's `libgert.so` references `aclrtHostGetDevicePointerAddrRange`, 0715 CANN's `libascendcl.so` lacks this symbol → cann_lm_compile built with 0715 link failure. **Must build with kirinx90 CANN** (GE + runtime version match).

### ge_graph_builder.cc Adapt New GE Headers

New GE headers mark `GetRegisteredIrDef` deprecated (suggest `GetRegisteredIrDefV2`), but `GetRegisteredIrDefV2` not actually declared in installed GE (only mentioned in deprecation message). Use `#pragma GCC diagnostic` to suppress:

```cpp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  if (::GetRegisteredIrDef(op_type.c_str(), inputs, outputs, attrs) != ge::GRAPH_SUCCESS) {
    ...
#pragma GCC diagnostic pop
```

## TODO (consolidation, doesn't affect already-produced .omc)

| # | Item | Location | Status |
|---|---|---|---|
| T1 | `compile_main.cc` adapt mobile path: after BuildModel detect `onlinec` → rename `{output_name}.omc` (Kirin SOC)/`.om` (else), skip redundant `aclgrphSaveModel`, eliminate red herring error + auto-correct naming | `cann_lm/converter/export/csrc/tools/compile_main.cc` | ✅ Landed (`All 3 stages passed`, directly produces `kirin_ops_kirinx90.omc`, no `onlinec` residual) |
| T2 | ops-nn build add l2_normalize forward apt entry packaging (`norm/l2_norm` add `gelu_src_copy`-same CMake target), completely replace hand-written wrapper | `/mnt/workspace/gitCode/ops-nn` (external repo) | ✅ Landed (source repo `norm/l2_normalize` contains forward apt entry; `build.sh --pkg --soc=kirinx90` produces `.run` containing `impl/ops_nn/dynamic/l2_normalize.py`(9688B, md5 identical to B3 hand-written wrapper)+ kernel `L2Normalize_1_high_performance.o` + `kernel/config/kirinx90/ops_nn/l2_normalize.json`; `--upgrade` install back ops-info-nn.json contains L2Normalize. 2026-09-17 verified, see "Op Package Source Build" section) |
| T3 | Network's `Muls→Mul`, `kv_cache as input`, scatter converter monkeypatch consolidate as kirinx90-target formal change (or converter layer SOC-conditional don't emit TensorMove) | `examples/.../kirin_ops_net.py` + `cann_lm/converter/export/ge_converters/_scatter_nd_update_converter.py` | ⬜ TODO |
| T4 | Confirm 6 ops other than Gelu/L2Normalize kernels on kirinx90 have no tvm-CCE residual issues (all compiled, re-test retained) | — | ⬜ TODO |
| T5 | Move MatMul/BatchMatMul back to aicore: monkeypatch `aten.mm`/`aten.bmm` converter, use `torchair.ge.custom_op("MatMulV3"/"BatchMatMulV3",...)` to manually build V3 node (must first verify converter stub env `get_registered_ir_def("MatMulV3")` passes); SoftmaxV2→`MaskedSoftmaxWithRelPosBias` (callable, needs rel_pos_bias) | `examples/.../kirin_ops_net.py` + `compile_to_epair` | ✅ MatMulV3 landed — Kirin converter patches mechanism (`--target-soc KirinX90` triggers; `_kirinx90_converter_patches.py::_matmul_v3` uses `ge_op("MatMulV3",...)` to override `aten.mm`/`matmul`, last-write-wins), 2026-09-17 simple_network e2e verified: ep_json 3× MatMulV3/0× plain MatMul, .omc aicore zero aicpu; ⬜ SoftmaxV2→`MaskedSoftmaxWithRelPosBias` still not done (simple_network doesn't include softmax, pending real model layer verification) |
| T6 | Weight separation: `compile_main.cc` set `ge.externalWeightDir` (via init_options not build_options) → =2 produces weight-separated .omc (.omc + `weight/delegate_graph_weight_combined` + meta.json) | `cann_lm/converter/export/csrc/tools/compile_main.cc` | ✅ Landed (=2 verified effective; =1 still fails per-file unsupported) |
| T7 | ~~`cat→ConcatV2` (lands aicpu)~~ **Plan abandoned** — kirinx90 doesn't support AICPU (see background hard constraint), ConcatV2 lands aicpu also fails. Correct direction: scatter indices' int32 cat (`modeling_gemma4.py:68` `[col_b,col_n,col_pos]`) is coordinate concatenation not aicore, constantize at converter stage (b/n/pos precomputable when seq_len fixed) or use host-side ops, don't leave aicpu ops. B4 root cause (`get_kernel_type_enum` AIV_ONLY + kirinx90 not in whitelist; commenting KERNEL_TYPE introduces tiling `_pattern`/`dsl_compile_info` missing) shows concat_d apt itself unusable on kirinx90, need to bypass apt path | `cann_lm/converter/export/ge_converters/_kirinx90_converter_patches.py::_kirinx90_aten_cat` + `cann_lm/models/gemma4/modeling_gemma4.py::_scatter_kv_cache` | ⬜ TODO |

## Legacy bug: mobile path `aclgrphBuildModel` doesn't fill `ModelBufferData` and hardcodes output filename

**Problem**: `aclgrphBuildModel` in mobile (kirinx90) path goes through `MobileModelHelper::SaveToOmModel` (`mobile_model_helper.cc:681`), this method:
1. **`model` param ignored by `(void)model`** (line 684) — doesn't fill `ModelBufferData.data` (left null), causing subsequent `aclgrphSaveModel`'s `model.data == nullptr` validation fail ("model is illegal").
2. **Hardcodes output filename `"onlinec"`** — `output_file` uses GE internal default `"online"` (because `output_name` only passed to `aclgrphSaveModel`, not to `aclgrphBuildModel`), filename = `"online" + "c"` = `"onlinec"`, doesn't reflect user-specified output_name.

**Impact**:
- `aclgrphSaveModel`必然 fails (red herring error `50331649 "model is illegal"`), but model file `onlinec` already on disk.
- Artifact filename non-intuitive (`onlinec` instead of `{output_name}.omc`), needs external rename.
- `ModelBufferData.length` in mobile path is garbage value (heap corruption/uninitialized), printed "model size" unreliable.

**Root cause**: `MobileModelHelper`'s design is mobile scenario compile-then-direct-write-to-disk (`SaveCompiledModelToFile` writes disk), bypassing in-memory buffer; but `output_file` uses GE internal default `"online"` not user's `output_name` (`aclgrphBuildModel`'s signature doesn't accept output_file param).

**Temporary workaround** (landed, T1): `compile_main.cc` detects `onlinec` exists then rename → `{output_name}.omc`, skip `aclgrphSaveModel`.

**Correct fix (GE side)**:
- Option A: `aclgrphBuildModel` accepts `output_file` param, passes to `MobileModelHelper`, lets mobile path directly write `{output_name}.omc` (avoid hardcoded `onlinec`).
- Option B: `MobileModelHelper::SaveToOmModel` fills `ModelBufferData` (after writing file backfill data/length), lets `aclgrphSaveModel` work normally (but mobile scenario's design intent to avoid large memory copy contradicts this).
- Option C: GE after `aclgrphBuildModel` returns, if detects mobile path (model.data==null but onlinec file exists), auto rename + return success (instead of letting `aclgrphSaveModel` error).

**GE source involved**:
- `base/common/helper/mobile_model_helper.cc:681-718` (`SaveToOmModel`, `(void)model` + hardcoded `onlinec`)
- `base/common/helper/mobile_model_helper.cc:706-711` (`output_weight_dir` derived from `output_file`'s dir part, `output_file="online"` no dir → empty)
- `compiler/api/aclgrph/ge_ir_build.cc:1640-1655` (`aclgrphSaveModelInner`, `model.data==nullptr` validation)

## Files/Artifacts Involved

| Path | Description |
|---|---|
| `examples/test_converter/export_kirin_ops_net/kirin_ops_net.py` | Small network + export/compile/eager CLI; incl. B1/B2 workarounds |
| `examples/test_converter/export_kirin_ops_net/aicore_only_net.py` | Pure aicore network (zero aicpu, avoids MatMul/BatchMatMul/SoftmaxV2/SplitV) + golden CLI |
| `examples/test_converter/export_kirin_ops_net/check_op_coverage.py` | Static op coverage check (epair ascend_ir ∩ SOC ops-info json) |
| `cann_lm/converter/export/csrc/tools/compile_main.cc` | mobile path wrap-up (T1: onlinec→.omc rename) + weight separation (T6: ge.externalWeightDir via init_options) |
| `cann_lm/converter/export/ge_converters/_kirinx90_converter_patches.py` | Kirin converter patches mechanism (B.2): 6 classes op converter override; B4's cat→ConcatD failure pending change to ConcatV2 (T7) |
| `cann_lm/converter/export/ge_converters/ge_converter.py:87` | `active_config().target_soc` contains "kirin" detection (or `KIRINX90_CONVERTER_PATCHES=1` force-on) + load patches |
| `asc_op_compile_base/asc_op_compiler/kernel_info_infer.py:195` | B4 root cause: `get_kernel_type_enum` else branch raises `AIV_ONLY not support` (kirinx90 soc not in `is_v200/v220/c310` whitelist) |
| `asc_op_compile_base/asc_op_compiler/ascendc_common_utility.py:516-609` | soc whitelist: `is_v220`(Ascend910B/910_93) / `is_v200`(Ascend310P/610) / `is_c310`(Ascend950/350), all exclude KirinX90 |
| `/mnt/workspace/cann-kirinx90/cann-9.2.0/opp/.../tbe/impl/ops_math/dynamic/concat_d_apt.py:216` | Hardcoded `arch31/concat_d.cpp` (AIV_ONLY), triggers B4 |
| `sitecustomize.py` + `PYTHONPATH` | A7 workaround: shield torch_npu (avoid GE embedded python double-load cann-9.0.0 GE) |
| `/mnt/workspace/cann-kirinx90/` | Isolated toolkit (toolkit + A3-ops + kirinx90 ops) |
| `/mnt/workspace/cann-kirinx90/cann-9.2.0/opp/.../tbe/impl/ops_nn/dynamic/l2_normalize.py` | AscendC-apt wrapper (B3; T2 onwards from ops-nn source repo `norm/l2_normalize` officially packaged, `--upgrade` install replaces hand-written, md5 identical) |
| `/mnt/workspace/cann-kirinx90/cann-9.2.0/opp/.../tbe/kernel/scene.info` | Manually added `os=linux` (A3 workaround) |
| `examples/.../out/kirin_ops_kirinx90.omc` | 8-op network artifact (677K, soc=KirinX90, weights embedded, has aicpu ops) |
| `examples/.../out/aicore_only_kirinx90.omc` | Pure aicore network artifact (350K, zero aicpu) |
| `examples/.../out/aicore_only_kirinx90.omc` + `weight/` | Weight-separated artifacts (=2: .omc 349K + `delegate_graph_weight_combined` 1.5K + meta.json) |
| `/tmp/opencode/kx_env.sh` | kirinx90 compile env script (unset pollution + source cann-kirinx90 + A5 libpython; op package build/install/OM compile reuse) |
| `/mnt/workspace/gitCode/{ops-nn,ops-math,ops-transformer}/build_out/cann-kirinx90-<pkg>_9.1.0_linux-aarch64.run` | Op package source build artifacts (`build.sh --pkg --soc=kirinx90 -j16`, see "Op Package Source Build" section) |
| `examples/test_converter/export_epair_and_compile/epair_model/simple.omc` | simple_network end-to-end artifact (686K, soc=KirinX90, weights embedded, 2× MatMulV3 aicore zero aicpu; `--target-soc KirinX90` triggers kirin patches export) |
