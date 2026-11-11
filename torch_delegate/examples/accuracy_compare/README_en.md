# Accuracy Comparison Example

English | [简体中文](README.md)

This example demonstrates the end-to-end accuracy comparison feature of TorchDelegate, verifying numerical consistency between two execution paths:

- **Reference path**: Python `torch.compile(backend='npu')` using the GE backend
- **Delegate path**: C++ `td::EpairModelLoader` (`CompileAndLoad` / `RunModelWithStreamAsync`) executing directly from the epair file

---

## Directory Structure

```text
accuracy_compare/
├── README.md                 # This document
├── run.sh                    # Run script
├── __init__.py               # Python package
├── accuracy_compare.py       # Main accuracy comparison script
├── compile_to_air.py         # Export epair
├── export_program.py         # ExportedProgram export
├── ge_converters.py          # GE converter
├── backend_ops.py            # backend ops
├── placeholder_to_custom_pass.py  # rewrite pass
├── ops.py                    # placeholder ops
├── util.py                   # Utility functions
├── custom_op/                # Custom op build
│   ├── custom_op.cpp
│   └── CMakeLists.txt
├── epair_runner/             # C++ runner
│   ├── CMakeLists.txt
│   ├── epair_runner.cpp
│   ├── tensor_io.cpp
│   └── tensor_io.h
└── generate/                 # Runtime-generated directory
```

---

## Requirements

1. **torch 2.9.0+cpu** (the torch_delegate venv is recommended)
2. **torch_npu** + **torchair**
3. **torch-delegate** built
4. **CANN 9.2.0+**
5. **libtorch_npu** source available

---

## Environment Variables

### Required environment variables

| Environment variable | Description | Example |
|----------|------|------|
| `TORCH_NPU_DIR` | torch_npu source directory | `/path/to/pytorch/libtorch_npu` |

### Optional environment variables

| Environment variable | Description | Default |
|----------|------|--------|
| `ASCEND_HOME_PATH` | CANN environment | ASCEND_HOME_PATH is checked first |
| `TORCH_PKG_DIR` | Python torch package directory | Detected automatically |
| `DEVICE` | Export device | `cpu` |
| `COMPILE_DEVICE` | Compile device | `npu:0` |

---

## Quick Start

```bash
# 1. Load the CANN environment
source ~/Ascend/cann/set_env.sh

# 2. Activate the Python environment with torch 2.9.0
source /path/to/torch_delegate/bin/activate

# 3. Set the only required environment variable
export TORCH_NPU_DIR=/path/to/libtorch_npu

# 4. Run the accuracy comparison
cd /path/to/torch-delegate/examples/accuracy_compare
bash run.sh compare
```

---

## Detailed Steps

### Export the ExportedProgram

```bash
bash run.sh step1
```

Generated files:
- `generate/sample.pt2` - ExportedProgram

### Compile AIR and export the Epair

```bash
source ~/Ascend/cann/set_env.sh
bash run.sh step2
```

Generated files:
- `generate/sample.rewritten.pt2` - Rewritten ExportedProgram
- `generate/rewritten.epair` - Epair file

### compare run the accuracy comparison

```bash
source ~/Ascend/cann/set_env.sh
export TORCH_NPU_DIR=/path/to/pytorch/libtorch_npu

bash run.sh compare
```

Output:
- Accuracy comparison results printed to the console
- `generate/accuracy_result.json` - Results in JSON format

---

## Sample Accuracy Comparison Output

```text
======================================================================
Accuracy Comparison Summary
======================================================================

[PASS] token_index
  Shape: [128, 1], Dtype: torch.int32
  Max Abs Diff: 0.000000

[PASS] token_ids
  Shape: [128, 1], Dtype: torch.int32
  Max Abs Diff: 0.000000

[PASS] log_probs
  Shape: [128, 1], Dtype: torch.bfloat16
  Max Abs Diff: 0.000000

======================================================================
Overall Result: PASS
======================================================================
```

---

## Technical Notes

### Separate converter registration timing

- When exporting the epair, only the BeamSearch converter is registered (which triggers CANN IR preregistration)
- The Reference path dynamically registers all converters (BeamSearch + RopeInPlace)
- RopeInPlace has `source=""` in the epair, which triggers fallback codegen

### torch.compile configuration

The Reference path must use `fullgraph=True` + `torchair.CompilerConfig()`:

```python
import torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend, fullgraph=True)
```

### BeamSearch input layout

See `xllm_ops/torch_compile_example/python/compare_beam_search.py`:
- `REQUEST_NUM=32, BEAM_WIDTH=4, VOCAB_SIZE=32000`
- `top_k = beam_width` (not vocab_size)

---

## Troubleshooting

### Link error undefined reference to `PyTorchStreamReader::getRecord`

**Cause**: torch version mismatch. torch-delegate requires torch 2.9.0+cpu.

**Solution**: Use the torch_delegate venv:
```bash
source /path/to/torch_delegate/bin/activate
```

### Error "environment variable XXX is not set"

**Solution**:
1. Load the CANN environment:
```bash
source ~/Ascend/cann/set_env.sh
```

2. Set the only required environment variable:
```bash
export TORCH_NPU_DIR=/path/to/pytorch/libtorch_npu
```

### Error "TopK does not support DT_BF16"

**Solution**: This is already handled in `export_program.py` by converting to float32 before running TopK.

---

## Known Issues and Solutions

### Handling `_assert_tensor_metadata` nodes

**Problem**:
`torch.export` inserts `_assert_tensor_metadata` metadata check nodes automatically. These nodes:
- Verify at runtime that the tensor's device/dtype/shape match those at export time
- Are not compute nodes and produce no tensors (`-> ()`)
- Are not recognized by `torch.compile` + the torchair backend as native PyTorch ops
- Cause the accuracy comparison to fail, because torchair does not reject PyTorch built-in ops

**Current solution**:
1. **Remove at export time**: `export_program.py` implements the FX Graph Pass `_remove_assert_tensor_metadata()`, which walks the graph after `torch.export.export` and erases all `_assert_tensor_metadata` nodes, preventing them from being generated in the first place
2. **Fallback converter**: `ge_converters.py` keeps `register_assert_tensor_metadata_converter()`, which registers a no-op converter as a defensive measure (if upstream fails to remove the node, torchair still will not reject it)

**Future plans**:
Generalize `_remove_assert_tensor_metadata()` into a reusable **FX Pass** utility for similar metadata check nodes:
- Support a configurable op blacklist (not limited to `_assert_tensor_metadata`)
- Clean up nodes that should not enter the compilation path after export
- Follow PyTorch best practices: use native FX APIs rather than relying on the compiler backend

**Note**:
The "export the EP, then run `torch.compile`" approach used in the current case is **not recommended**, because:
- The output of `torch.export` is not guaranteed to be fully consumable by `torch.compile` (a PyTorch limitation)
- It leads to "converter not found" errors, which is expected behavior
- The recommended workflow is: `export -> rewrite -> compile_to_air -> epair_runner`

### beam_search const-correctness compile error

**Problem**:
The source of the `beam_search` op (TBE kernel) installed in CANN 9.1.0 has a compile error:
```text
error: cannot bind non-const lvalue reference of type 'TopkTiling&' to an rvalue of type 'const TopkTiling'
```

**Root cause**:
In `beam_search.cpp`, the `TopkTiling&` parameter of the `Init()` function signature should be `const TopkTiling&`, because the struct members obtained by `GET_TILING_DATA(tiling_data, tiling)` at the kernel entry point are const.

**Current solution**:
When building dependencies, `build_accuracy_deps.sh` clones the source from the xllm_ops repository (fixed `cann-9.0` branch), automatically applies `patches/beam_search_const_tiling.patch` to fix the const-correctness issue, then builds through `xllm_ops/build.sh` and automatically installs the op package to `${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/` (the internal logic of build.sh at line 453 runs `./xllm_ops.run`). No additional install command is needed, and no runtime patch is required.

**Future plans**:
Once the upstream xllm_ops/beam_search is fixed, this logic can be simplified:
- Update the patch to point to a new commit hash or tag
- Skip patch application when a fixed version is detected

---

## References

- TorchDelegate documentation: `../../docs/en/guide/build.md`
- export_and_load_example: `../export_and_load_example/`
- run_async example: `../run_async/`
