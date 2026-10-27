# stub_kernel.so - Custom Operator Schema Stub

English | [简体中文](README.md)

## Purpose

Register custom operator schemas (e.g. `xllm_ascend::rope_in_place`) with the libtorch dispatcher so that `FallbackCodegen` can query schema information for operators that are not natively registered.

**Verified**: builds successfully (485KB); schema registration passes the rope_in_place.json test.

---

## Registered Operators

### rope_in_place (inplace version)

**Namespace**: `xllm_ascend`

**Schema**:
```text
rope_in_place(Tensor(a!) input, Tensor sin_cache, Tensor cos_cache) -> ()
```

**Full Target**: `torch.ops.xllm_ascend.rope_in_place.default`

**Key characteristics**:
- `Tensor(a!)`: indicates that the input parameter is **written to** (alias annotation)
- `-> ()`: the return type is `void` (in-place operations return no new tensor)
- Kernel parameter type: `at::Tensor&` (reference, non-const)

**Dispatch Keys**:
- Meta: inplace shape inference (no output tensor; input shape unchanged)
- PrivateUse1: NPU stub (empty implementation; does not modify input)
- CPU: CPU stub (empty implementation; does not modify input)

**Source**: based on `tests/cpp/rope_in_place.json`

**Verification result**:
```bash
# Schema output
xllm_ascend::rope_in_place(Tensor(a!) input, Tensor sin_cache, Tensor cos_cache) -> ()

# Python verification
import ctypes, torch
ctypes.CDLL('libstub_kernel.so', ctypes.RTLD_GLOBAL)
op = torch.ops.xllm_ascend.rope_in_place
print(f'Operator registered: {op}')  # ✅ success
```

---

## Build Methods

### Method 1 Project CMake integration (recommended)

```bash
# Configure the project (enable BUILD_TESTS=ON)
cd torch-delegate
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -B build

# Build stub_kernel (built automatically along with the tests)
cmake --build build --target stub_kernel

# Inspect the output
ls build/tests/cpp/stub_kernel/libstub_kernel.so
```

**Advantages**: automatic dependency management and integration with the test infrastructure

---

### Method 2 Standalone CMake build (detached from the project)

```bash
# Enter the stub_kernel directory
cd tests/cpp/stub_kernel

# Create a standalone build directory
mkdir build && cd build

# Configure (TORCH_PKG_DIR and TORCH_NPU_DIR must be specified)
# TORCH_PKG_DIR points to the Python torch package directory
# TORCH_NPU_DIR points to the libtorch_npu source root
cmake .. \
    -DTORCH_PKG_DIR=/mnt/workspace/thj_delegate/lib/python3.12/site-packages/torch \
    -DTORCH_NPU_DIR=/mnt/workspace/gitCode/pytorch/libtorch_npu

# Build
cmake --build .

# Output goes to the current directory
ls libstub_kernel.so
```

**Notes**:
- `TORCH_PKG_DIR` must point to the **Python torch package directory** (e.g. `/mnt/workspace/thj_delegate/lib/python3.12/site-packages/torch`)
- `TORCH_NPU_DIR` must point to the **libtorch_npu source root** (e.g. `/mnt/workspace/gitCode/pytorch/libtorch_npu`)
- CMake automatically adds the source parent directory to the include path so that `#include "torch_npu/csrc/logging/LogContext.h"` in `NPUQueue.h` resolves
- Automatically locates and links `${TORCH_NPU_DIR}/lib/libtorch_npu.so`, providing runtime symbols such as `NPUStream`
- Replace `/path/to/libtorch` with the actual libtorch installation path

---

### Method 3 Direct g++ compilation (no CMake)

```bash
# Enter the stub_kernel directory
cd tests/cpp/stub_kernel

# Set the libtorch path
TORCH_PKG_DIR=/path/to/libtorch

# Compile directly
g++ -shared -fPIC -o libstub_kernel.so stub_kernel.cc \
    -std=c++17 \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I${TORCH_PKG_DIR}/include \
    -I${TORCH_PKG_DIR}/include/torch/csrc/api/include \
    -L${TORCH_PKG_DIR}/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,${TORCH_PKG_DIR}/lib

# Output goes to the current directory
ls libstub_kernel.so
```

**Note**: adjust the `-I` and `-L` options according to the actual libtorch installation path.

---

## Usage

stub_kernel.so must be loaded before `FallbackCodegen` calls `findSchema`, so that its schema is registered with the libtorch dispatcher. The following approaches are recommended:

### Method 1 LD_PRELOAD (recommended)

```bash
LD_PRELOAD=/path/to/libstub_kernel.so ./torch_delegate_tests
```

### Method 2 dlopen in test code

```cpp
#include <dlfcn.h>

void* handle = dlopen("/path/to/libstub_kernel.so", RTLD_NOW | RTLD_GLOBAL);
if (!handle) {
    fprintf(stderr, "Failed to load stub_kernel: %s\n", dlerror());
}
```

### Method 3 Python import (must be built as an extension)

```python
# Build stub_kernel as a Python extension (requires the PyInit_stub_kernel symbol)
import torch
import stub_kernel  # schema registration is triggered by TORCH_LIBRARY static initialization
```

---

## Verification

### Quick verification (Python)

```python
import ctypes
import torch

# Load stub_kernel.so (use RTLD_GLOBAL to register the schema)
stub_path = '/path/to/libstub_kernel.so'
handle = ctypes.CDLL(stub_path, ctypes.RTLD_GLOBAL)
print(f'Loaded: {stub_path}')

# Verify schema registration
op = torch.ops.xllm_ascend.rope_in_place
print(f'Operator registered: {op}')
print(f'Success: schema is accessible via torch.ops')
```

**Expected output**:
```text
Loaded: /path/to/libstub_kernel.so
Operator registered: xllm_ascend.rope_in_place
Success: schema is accessible via torch.ops
```

---

### Schema registration check (C++)

```cpp
#include <ATen/core/function_schema.h>

c10::OperatorName op_name("xllm_ascend::rope_in_place", "");
auto handle = torch::Dispatcher::singleton().findSchema(op_name);

if (handle.has_value()) {
    std::cout << "Schema found: " << handle->schema() << std::endl;
} else {
    std::cerr << "Schema NOT found for xllm_ascend::rope_in_place" << std::endl;
}
```

**Expected output**:
```text
Schema found: rope_in_place(Tensor input, Tensor sin_cache, Tensor cos_cache) -> Tensor
```

---

## Directory Structure

```text
tests/cpp/stub_kernel/
├── stub_kernel.cc          # Schema registration implementation
├── CMakeLists.txt          # Standalone CMake build configuration
└── README.md               # This file
```

**Project integration**:
- Built automatically with `cmake -DBUILD_TESTS=ON`
- Output: `${BUILD_DIR}/tests/cpp/stub_kernel/libstub_kernel.so`
- Integrated in `tests/CMakeLists.txt`

---

## Troubleshooting

### Schema not found error

```text
FallbackCodegen: schema not found for 'xllm_ascend::rope_in_place' with overload ''
```

**Solution**:
1. Check that the stub_kernel.so file exists
2. Load stub_kernel.so via `LD_PRELOAD` or `dlopen` in code
3. Check whether dlopen succeeded (enable debug logging):
   ```bash
   export TORCH_DELEGATE_LOG_LEVEL=DEBUG
   ./torch_delegate_tests
   ```

---

### ABI mismatch error

```text
undefined symbol: _ZNK2at6Tensor... (ABI=0 vs ABI=1)
```

**Solution**: ensure stub_kernel is compiled with `_GLIBCXX_USE_CXX11_ABI=1` (consistent with libtorch)

---

### dlopen permission error

**Solution**: check file permissions and path accessibility:
```bash
ls -la /path/to/libstub_kernel.so
```

---

## Extending with Other Operators

Edit `stub_kernel.cc` to add more custom operators:

```cpp
TORCH_LIBRARY(xllm_ascend, m) {
    m.def("rope_in_place(Tensor input, Tensor sin_cache, Tensor cos_cache) -> Tensor");
    // Add more operators here
    m.def("other_custom_op(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ascend, Meta, m) {
    m.impl("rope_in_place", [](...) { return at::empty_like(...); });
    m.impl("other_custom_op", [](...) { return at::empty_like(...); });
}
```

Then rebuild stub_kernel.so.

---

## Related Files

- `tests/cpp/rope_in_place.json`: source JSON defining the operator structure
- `csrc/fallback_codegen/fallback_codegen.cc`: fallback codegen implementation
- `tests/CMakeLists.txt`: project build integration

---

## Verification Results Summary

### Build verification
- ✅ stub_kernel.so builds successfully (485KB)
- ✅ exported symbols contain the `xllm_ascend::rope_in_place` schema registration
- ✅ ABI=1 is set correctly (consistent with libtorch)

### Schema registration verification
```bash
python3 -c "
import ctypes, torch
handle = ctypes.CDLL('tests/cpp/stub_kernel/libstub_kernel.so', ctypes.RTLD_GLOBAL)
op = torch.ops.xllm_ascend.rope_in_place
print(f'Operator: {op}')  # Output: Operator: xllm_ascend.rope_in_place
"
```

### rope_in_place.json verification
- ✅ JSON parsed successfully (214KB, 21 nodes)
- ✅ rope_in_place node confirmed:
  - target: `torch.ops.xllm_ascend.rope_in_place.default`
  - inputs: 3 (input, sin_cache, cos_cache)
  - outputs: 1
  - `ascend_ir.source`: MISSING (fallback codegen needed)
