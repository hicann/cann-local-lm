# TorchDelegate Environment Variables Reference

## Environment Variables Overview

| Variable name | Read at | Default | Summary |
|--------|---------|--------|------|
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | `csrc/backend/ge_graph_builder.cc` (`GeGraphBuilder` constructor) | `1` (enabled) | Whether to trigger Fallback Codegen during graph construction |
| `TORCH_PKG_DIR` | `csrc/fallback_codegen/fallback_codegen.cc` (`FallbackCodegen::GenerateFromModelJson()`); `cmake/third_party/modules/Findlibtorch_cpu.cmake` | — | Python torch package directory |
| `TORCH_NPU_DIR` | `csrc/fallback_codegen/fallback_codegen.cc` (`FallbackCodegen::GenerateFromModelJson()`) | — | torch_npu library directory |
| `TORCH_DELEGATE_OUTPUT_DIR` | `csrc/fallback_codegen/fallback_codegen.cc` (`FallbackCodegen::GenerateFromModelJson()`) | `./output_dir` | Directory for fallback build artifacts |
| `TORCH_DELEGATE_KEEP_ARTIFACTS` | `csrc/fallback_codegen/fallback_codegen.cc` (`FallbackCodegen::GenerateFromTargets()`) | `0` (not kept) | Whether to keep `.cpp` / `CMakeLists.txt` / `.so` after compilation |
| `TORCH_DELEGATE_DEBUG` | `torch_delegate/core/utils.py` (`_get_logger()`) | `0` (WARN) | Python-side log level switch |

---

## `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN`

**Purpose**: controls whether the Fallback Codegen feature for ATen operators is enabled.

**Values**:
- `1`: enabled (default; also enabled when not set)
- `0` or any other value: disabled

**Notes**:
- When disabled, if the epair contains unconverted nodes (`ascend_ir.source` is empty), GE graph construction fails with an error
- If all nodes are converted successfully, Fallback Codegen is not triggered even when enabled, and `TORCH_PKG_DIR` / `TORCH_NPU_DIR` are not validated

---

## `TORCH_PKG_DIR`

**Purpose**: points to the Python torch package directory. It is used by g++ to locate headers and libraries during compilation, and by CMake to locate `TorchConfig.cmake`.

**Value**: absolute path to the Python torch package (contains the `include/`, `lib/`, and `share/cmake/Torch` subdirectories).

**Example**:
```bash
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
# Example value: /path/to/site-packages/torch
```

**Notes**:
- Only required when Fallback Codegen is triggered; not validated when every node's `ascend_ir.source` is `converter`

---

## `TORCH_NPU_DIR`

**Purpose**: points to the torch_npu library directory. It is used by g++ to locate `libtorch_npu.so` and headers during compilation.

**Value**: torch_npu library directory (contains the `lib/libtorch_npu.so` and `include/` subdirectories).

**Example**:
```bash
export TORCH_NPU_DIR=/path/to/libtorch_npu
```

**Notes**:
- The generated fallback code depends on `libtorch_npu.so`
- Only required when Fallback Codegen is triggered; not validated when every node's `ascend_ir.source` is `converter`
- Getting `libtorch_npu`: download the **v2.9.0 branch** source from the [Ascend/pytorch](https://gitcode.com/Ascend/pytorch) repository, then build it following the [libtorch_npu build guide](https://gitcode.com/Ascend/pytorch/blob/master/docs/en/installation_guide/references/building_libtorch_npu.md)

---

## `TORCH_DELEGATE_OUTPUT_DIR`

**Purpose**: specifies the directory where Fallback Codegen build artifacts are kept.

**Value**: any writable directory path.

**Notes**:
- Only actually used when `TORCH_DELEGATE_KEEP_ARTIFACTS=1`
- Artifacts are kept in a subdirectory under `${TORCH_DELEGATE_OUTPUT_DIR}/`

---

## `TORCH_DELEGATE_KEEP_ARTIFACTS`

**Purpose**: controls whether the generated `.cpp`, `CMakeLists.txt`, and `.so` files are kept after online compilation completes.

**Values**:
- `1`: keep them under `${TORCH_DELEGATE_OUTPUT_DIR}/` (`./output_dir/` when `OUTPUT_DIR` is not set)
- `0` or not set: do not keep (default); the temporary directory is cleaned up after compilation

**Notes**:
- When kept, the artifacts include `custom_op.cpp` (header contains the full g++ command in comments), `CMakeLists.txt`, and `libcust_opapi.so`
- Mainly useful for debugging build issues or for cross-environment deployment

---

## `TORCH_DELEGATE_DEBUG`

**Purpose**: controls the log level of the Python-side torch_delegate library.

**Values**:
- `1`: `logging.DEBUG` level
- Not set or any other value: `logging.WARN` level (default)
