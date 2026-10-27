# Fallback Codegen User Guide

This document is the unified user guide for Fallback Codegen, intended for end users who need to integrate custom operators into GE graph mode. It covers the complete flow from operator registration to execution, environment variable configuration, working mode selection, parameter type support, and common troubleshooting.

---

## Overview

### What Is Fallback Codegen

Fallback Codegen is the dynamic operator code generation module of TorchDelegate. When the Python-side `export_to_epair` converts a PyTorch ExportedProgram into an epair file, some operators may fail to convert to Ascend IR (GE operators). These failed nodes are marked with `ascend_ir.source = ""` and must be handled on the C++ side by Fallback Codegen.

What Fallback Codegen does:

1. **Parse the epair JSON**: identify operator nodes that failed conversion (`ascend_ir.source` is empty or missing)
2. **Query the operator schema**: obtain the operator's function signature from the PyTorch Dispatcher
3. **Generate GE custom operator code**: automatically generate C++ code and CMakeLists from the schema
4. **Compile into a shared library**: produce a dynamically loadable `.so` file

### When to Use

- The user registers custom operators through `torch.library`
- The operator has no GE graph delivery artifact (no `REG_OP` registration)
- The user's NPU implementation must be executed transparently in GE graph mode

### Comparison of the Three Operator Scenarios

TorchDelegate classifies operators into three scenarios based on whether they carry `ascend_ir` annotations and whether they have GE graph delivery artifacts:

| Scenario | Description | ascend_ir | GE deliverable | Graph construction | v1 support |
|------|------|-----------|-----------|----------|---------|
| **Scenario 1** | Operators with complete graph delivery artifacts | `source="converter"`, ops contain a full opdef | Yes | `NamedIoNodeBuilder` builds the graph directly | ✅ Supported |
| **Scenario 2** | Fallback callback operators | `source=""` or no ascend_ir | None; generated automatically by Fallback Codegen | Codegen generates GE `EagerExecuteOp` → compile → dlopen registration → runtime callback to PyTorch Dispatcher | ✅ Supported |
| **Scenario 3** | CPU operators | None | None | Requires HeteroCodegen to generate CPU execution code | ❌ Not supported yet |

Fallback Codegen handles **Scenario 2**.

### Runtime Call Chain

```text
GE graph mode
  → GE custom operator Execute
  → create NPU at::Tensor (torch::from_blob, zero copy)
  → PyTorch Dispatcher call
  → user operator NPU implementation (aclnn, AscendC, etc.)
  → write results back to the GE output tensor
```

---

## Prerequisites

### Environment Requirements

| Component | Version requirement | Notes |
|------|---------|------|
| PyTorch | >= 2.9.0 | Provides Dispatcher, torch.library, torch.export |
| torch_npu | Matching PyTorch | Provides the NPU device backend and from_blob zero copy |
| CANN | >= 9.2.0 | Provides GeSession, ACL runtime, NamedIoNodeBuilder |
| g++ | C++17 support | Required for online-mode compilation |

### What the User Must Provide

**1. PyTorch custom operator implementation (`.so` file)**

- Registered with the PyTorch Dispatcher (`PrivateUse1` key)
- Provides a Meta implementation (for shape inference in `torch.compile` graph mode)
- The operator implementation uses NPU APIs (such as aclnn, AscendC)

**2. epair file (exported by torch_delegate)**

- A JSON file describing the model
- Contains the custom operator names needed to fall back to aten

---

## Quick Start

### Complete Flow at a Glance

```text
Step 1: Register custom operators on the Python side
  ├── Define the operator schema (torch.library.Library)
  ├── Register the Meta implementation (register_fake)
  └── Register the NPU implementation (PrivateUse1 dispatch key)

Step 2: Export the epair file
  ├── torch.export.export exports an ExportedProgram
  ├── export_to_epair converts it to EpairContents
  └── epair_save saves it as an .epair file

Step 3: C++ Fallback Codegen processing (triggered automatically)
  ├── GeGraphBuilder loads the epair
  ├── FallbackCodegen scans for source="" nodes
  ├── Generate GE custom operator code
  ├── Compile into a .so (online mode)
  └── dlopen and register with GE

Step 4: Run inference
  ├── GeSession::CompileGraph compiles
  └── GeSession::RunGraphWithStreamAsync executes
```

### Register a Custom Operator

Register a custom operator through `torch.library`, including the schema definition, the Meta implementation, and the NPU implementation:

```python
import torch

# Define the operator schema
lib = torch.library.Library("my_ops", "DEF")
lib.define("add_scalar(Tensor x, float alpha) -> Tensor")

# Meta implementation (for shape inference in graph mode)
@torch.library.register_fake("my_ops::add_scalar")
def _add_scalar_meta(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.empty_like(x)

# NPU implementation (registered under the PrivateUse1 dispatch key)
@torch.library.impl("my_ops::add_scalar", "PrivateUse1")
def _add_scalar_npu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Actual NPU implementation (aclnn / AscendC, etc.); placeholder here
    return x + alpha
```

### Export the epair File

```python
import torch
from torch.export import export
from torch_delegate import export_to_epair, epair_save

class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.my_ops.add_scalar(x, 0.5)

# Export as an ExportedProgram
ep = export(MyModel(), (torch.randn(4, 4),))

# Convert to EpairContents
# The add_scalar node has no ascend_ir, so source=""
contents = export_to_epair(ep)

# Check conversion statistics
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")

# Save as an epair file
epair_save(contents, "model.epair")
```

### Load and Execute from C++

```cpp
#include "torch_delegate/epair_model_loader.h"

// Load the epair file
td::EpairModelLoader loader;
td::ModelHandle model_handle;
td::Status status = loader.LoadEpair("model.epair", &model_handle);

// FallbackCodegen is triggered automatically when GeGraphBuilder builds the graph
// (requires TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1)
loader.CompileAndLoad(options, stream);

// Run inference
std::vector<gert::Tensor> inputs, outputs;
loader.RunModelWithStreamAsync(stream, inputs, outputs);
```

### Complete Example Code

For complete end-to-end examples, see:

- **[examples/fallback_inplace_optional_case/](../../../examples/fallback_inplace_optional_case/)** — multiple inplace parameters + multiple optional parameters
- **examples/fallback_e2e_test/** — A/B testing that verifies full aten type support
- **[examples/sym_int_case/](../../../examples/sym_int_case/)** — simplified e2e that verifies non-Tensor input support

---

## Working Modes

Fallback Codegen supports two working modes: online mode and offline mode.

### Online Mode (Default)

- **Behavior**: generates code files, **compiles immediately** to produce a `.so`, and **loads it dynamically** into the process
- **On-disk artifacts**: uses a temporary directory by default (cleaned up after compilation); set `KEEP_ARTIFACTS=1` to keep them under `OUTPUT_DIR`
- **Artifacts**:
  - Default (`KEEP_ARTIFACTS=0`): only the in-memory `.so` handle (temporary files are cleaned up)
  - Debug (`KEEP_ARTIFACTS=1`): `custom_op.cpp` (with g++ command in comments) + `CMakeLists.txt` + `libcust_opapi.so`
- **Use cases**: just-in-time compilation and execution, single-machine environments, rapid prototyping

### Offline Mode

- **Behavior**: generates code files (`.cpp` + `CMakeLists.txt`) but **does not compile**
- **On-disk artifacts**: files must be written to `OUTPUT_DIR`
- **Artifacts**:
  - `custom_op.cpp` (GE custom operator implementation)
  - `CMakeLists.txt` (build script)
- **Use cases**: cross-machine deployment, version control, manual code modification, integration with an existing build system

### Behavior Matrix

| Mode | `OFFLINE` | `KEEP_ARTIFACTS` | `OUTPUT_DIR` | On-disk files | Compilation | File retention |
|------|-----------|------------------|--------------|---------|---------|---------|
| **Offline** | `1` | — (ignored) | Configured value or default `./output_dir` | cpp + CMakeLists.txt | No compilation | Always kept |
| **Online (default)** | `0` (default) | `0` (default) | — (ignored) | None (temporary directory) | g++ compile + dlopen | Temporary files cleaned up |
| **Online (debug)** | `0` | `1` | Configured value or default `./output_dir` | cpp (with g++ comments) + CMakeLists.txt + .so | g++ compile + dlopen | Files kept |

### Choosing a Mode

| Scenario | Recommended mode | Reason |
|------|---------|------|
| Single-machine development and debugging | Online mode | No manual compilation; fast iteration |
| Cross-machine deployment | Offline mode | Build and run environments are separated |
| CI/CD pipelines | Offline mode | Generated code can be version-controlled |
| Inspecting generated code | Offline mode | View the .cpp files directly |
| Debugging build issues | Online debug mode | Inspect the full g++ command |

---

## Environment Variable Reference

### Control Switches

| Environment variable | Description | Default |
|---------|------|--------|
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | Whether to enable fallback codegen (`1`/`true`/`TRUE`; `0`/`false`/`FALSE` to disable) | Enabled |

> **Important**: If all nodes are converted successfully (`ascend_ir.source = "converter"`), codegen is not triggered even when this environment variable is enabled, and `TORCH_PKG_DIR` and `TORCH_NPU_DIR` are not validated.

### Required Environment Variables (When Triggered)

When any node needs fallback codegen, the following environment variables **must be set**:

| Environment variable | Description | Example |
|---------|------|------|
| `TORCH_PKG_DIR` | libtorch CMake configuration directory (where TorchConfig.cmake resides) | `/path/to/torch/share/cmake/Torch` |
| `TORCH_NPU_DIR` | torch_npu library directory (contains `lib/libtorch_npu.so`) | `/path/to/libtorch_npu` |
| `ASCEND_HOME_PATH` | CANN installation path (required for online compilation) | `/path/to/cann` |

### Working Mode Controls

| Environment variable | Description | Default |
|---------|------|--------|
| `TORCH_DELEGATE_OFFLINE` | Working mode (`1`=offline, `0`=online) | `0` (online) |
| `TORCH_DELEGATE_OUTPUT_DIR` | Location for on-disk files | `./output_dir` |
| `TORCH_DELEGATE_KEEP_ARTIFACTS` | Whether to keep build artifacts in online mode (`1`/`true`/`TRUE`) | `0` (not kept) |

### How to Locate Environment Variables

**Find TorchConfig.cmake**:

```bash
# Python whl package
TORCH_CMAKE_DIR=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))")
export TORCH_PKG_DIR="${TORCH_CMAKE_DIR}"
```

**Find the CANN path**:

```bash
# Usually already set by the source command
source ~/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH  # Output: /path/to/cann
```

### Precedence Rules

Environment variables are read according to the following precedence:

**1. `OUTPUT_DIR`**:
```text
function argument > TORCH_DELEGATE_OUTPUT_DIR > default ./output_dir
```

**2. `OFFLINE`**:
```text
TORCH_DELEGATE_OFFLINE > default 0 (online)
```

**3. `KEEP_ARTIFACTS`** (online mode only):
```text
TORCH_DELEGATE_KEEP_ARTIFACTS > default 0 (not kept)
```

---

## Parameter Type Support

### Supported Parameter Types

Fallback Codegen supports 21 `InputKind` parameter types, covering all common parameter types in the PyTorch Schema:

| Category | InputKind | PyTorch type | GE input type |
|------|-----------|-------------|---------------|
| **Tensor** | TENSOR | `Tensor` | `TensorType::ALL()` |
| | OPTIONAL_TENSOR | `Tensor?` | `.OPTIONAL_INPUT` |
| | TENSOR_LIST | `Tensor[]` | `.DYNAMIC_INPUT` |
| **Scalar** | INT | `int` | `IntTensorType()` |
| | INT_OPT | `int?` | `IntTensorType()` |
| | FLOAT | `float` | `FloatTensorType()` |
| | FLOAT_OPT | `float?` | `FloatTensorType()` |
| | BOOL | `bool` | `BoolTensorType()` |
| | BOOL_OPT | `bool?` | `BoolTensorType()` |
| | SCALAR | `Scalar` | `TensorType::ALL()` |
| | SCALAR_OPT | `Scalar?` | `TensorType::ALL()` |
| **List** | INT_LIST | `int[]` | `IntTensorType()` |
| | INT_LIST_OPT | `int[]?` | `IntTensorType()` |
| | FLOAT_LIST | `float[]` | `FloatTensorType()` |
| | FLOAT_LIST_OPT | `float[]?` | `FloatTensorType()` |
| | BOOL_LIST | `bool[]` | `BoolTensorType()` |
| | BOOL_LIST_OPT | `bool[]?` | `BoolTensorType()` |
| | SCALAR_LIST | `Scalar[]` | `TensorType::ALL()` |
| **Special** | STRING | `str` | `StringTensorType()` |
| | STRING_OPT | `str?` | `StringTensorType()` |
| | SCALAR_TYPE | `ScalarType` | `IntTensorType()` |
| | SCALAR_TYPE_OPT | `ScalarType?` | `IntTensorType()` |

### value-dependency Mechanism

Non-Tensor parameters (scalar, list, and special categories) are passed through the GE graph using the **value-dependency** mechanism:

1. **Registration phase**: non-Tensor parameters are registered as value-dependency inputs through `InputsDataDependency`
2. **Before graph execution**: the GE framework waits for these single-element tensors to be read back from device to host
3. **During Execute**: the scalar values are read from host memory and assembled into a `c10::IValue`

### Operator Overload Handling

Fallback Codegen supports three operator overload variants:

| Overload type | Example target | Execute output handling |
|---------|------------|-----------------|
| Standard | `my_ops::add.default` | `MallocOutputTensor` allocates the output |
| out variant | `my_ops::add.out` | `GetInputTensor(out_index)` retrieves the output |
| In-place | `my_ops::add_` | `MakeOutputRefInput` makes the output reference the input |

### Optional Input Handling

Optional parameters are handled through the `as_none` placeholder mechanism:

```json
{
  "inputs": [
    {"as_tensor": "x"},
    {"as_none": ""},
    {"as_tensor": "y"}
  ]
}
```

Optional parameters that are not provided use the default values from the schema.

> For detailed parameter type mapping, Execute code generation, and the BuildStack/WriteBack implementation, see [Design of Automatic GE Custom Operator Generation Based on ATen IR Callbacks](../features/aten_ir_callback_codegen_20260715.md).

---

## Usage Examples

### All Model Nodes Convert Successfully (Codegen Not Triggered)

```bash
# Do not set any environment variables
./build.sh

# Or set the enable switch, but all model nodes convert successfully
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
./build.sh  # Codegen is not triggered; TORCH_PKG_DIR/TORCH_NPU_DIR are not validated
```

### Offline Mode (Cross-Machine Deployment)

**Use case**: the build environment and the run environment are separated, and compilation must be performed manually on the target machine.

```bash
# 1. Locate and set the required environment variables
TORCH_CMAKE_DIR=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))")
export TORCH_PKG_DIR="${TORCH_CMAKE_DIR}"
export TORCH_NPU_DIR=/path/to/libtorch_npu

# 2. Set the CANN path (optional in offline mode, but recommended for CMakeLists.txt)
source ~/Ascend/cann/set_env.sh

# 3. Configure the output directory (optional)
export TORCH_DELEGATE_OUTPUT_DIR=/path/to/output

# 4. Enable fallback codegen (offline mode)
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=1  # Default value; can be omitted

# 5. Run the program to generate the code files
./build.sh

# Artifacts: /path/to/output/custom_op.cpp + CMakeLists.txt
```

**Compile later on the target machine**:

```bash
cd /path/to/output
cmake -B build \
  -DTorch_DIR=$TORCH_PKG_DIR \
  -DTORCH_NPU_DIR=$TORCH_NPU_DIR \
  -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
cmake --build build -j4

# Artifact: build/libcustom_op.so
```

### Online Mode (Just-in-Time Compilation and Execution)

**Use case**: a single-machine environment where the build environment and the run environment are the same.

```bash
# 1. Set the required environment variables (ASCEND_HOME_PATH is mandatory in online mode)
TORCH_CMAKE_DIR=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))")
export TORCH_PKG_DIR="${TORCH_CMAKE_DIR}"
export TORCH_NPU_DIR=/path/to/libtorch_npu
source ~/Ascend/cann/set_env.sh  # Sets ASCEND_HOME_PATH

# 2. Enable fallback codegen (online mode)
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=0

# 3. Run the program; it compiles and loads automatically
./build.sh

# Artifact: in-memory .so handle (temporary files are cleaned up)
```

### Online Mode Debugging (Keep Build Artifacts)

**Use case**: debugging online compilation issues when you need to inspect the generated code and the g++ command.

```bash
# 1-2. Set the required environment variables as in Scenario 3
...

# 3. Enable online mode and keep artifacts
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=0
export TORCH_DELEGATE_KEEP_ARTIFACTS=1
export TORCH_DELEGATE_OUTPUT_DIR=/tmp/fallback_debug

# 4. Run the program
./build.sh

# Artifacts (kept in /tmp/fallback_debug):
#   - custom_op.cpp (header contains the full g++ command)
#   - CMakeLists.txt
#   - libcust_opapi.so

# 5. Inspect the generated g++ command (for debugging)
head -10 /tmp/fallback_debug/custom_op.cpp

# 6. Recompile manually (if automatic compilation fails)
cd /tmp/fallback_debug
# Copy the g++ command commented in custom_op.cpp and run it manually
g++ -shared -fPIC ... -o libcust_opapi.so custom_op.cpp
```

### Quickly Verify the Generated Code

**Use case**: you want to inspect the generated code without actually compiling it.

```bash
# Generate files using offline mode
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=1
export TORCH_DELEGATE_OUTPUT_DIR=/tmp/inspect_codegen

./build.sh

# Inspect the generated code
cat /tmp/inspect_codegen/custom_op.cpp
cat /tmp/inspect_codegen/CMakeLists.txt
```

---

## Code Generation Flow

### Parsing Phase

```cpp
// FallbackCodegen::GenerateFromModelJson
std::vector<OpTarget> targets = ParseModelJsonFromString(model_json);

// targets contains only the nodes whose ascend_ir.source is empty
if (targets.empty()) {
    // No nodes need fallback; skip codegen
    return {success=true, skipped=true};
}
```

Filtering logic:

```cpp
bool needs_fallback = false;
if (node.contains("ascend_ir") && node["ascend_ir"].is_object()) {
    const auto& ascend_ir = node["ascend_ir"];
    // Only nodes with a missing or empty source need fallback
    if (!ascend_ir.contains("source") ||
        (ascend_ir.contains("source") && ascend_ir["source"].get<std::string>() == "")) {
        needs_fallback = true;
    }
}
```

### Schema Query

```cpp
// Query the operator signature from the PyTorch Dispatcher
c10::OperatorName op_name(full_op_name, overload);
auto handle = torch::Dispatcher::singleton().findSchema(op_name);
const auto& schema = handle->schema();
```

### Code Generation

Generated files:

- `custom_op.cpp`: GE custom operator implementation
- `CMakeLists.txt`: build script

**Structure of the generated code**:

- One `EagerExecuteOp` subclass per operator
- Helper functions: `MapGeDtypeToTorch`, `ReadScalarInput`, `CreateNpuTensor`
- Registration macros: `REG_OP` + `REG_AUTO_MAPPING_OP`

### Compilation Phase

```bash
cd output_dir
cmake -B build \
  -DTorch_DIR=$TORCH_PKG_DIR \
  -DTORCH_NPU_DIR=$TORCH_NPU_DIR \
  -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
cmake --build build -j4
```

Generated artifact:
- `output_dir/build/libcustom_op.so`

---

## Error Handling

### Error Types

| Error scenario | `success` | `skipped` | error_message |
|---------|-----------|-----------|---------------|
| JSON parsing failed | `false` | `false` | `"cannot parse model.json content: <parse_error>"` |
| No fallback needed (all nodes succeeded) | `true` | `true` | `"Skipped: no nodes need fallback codegen"` |
| Fallback needed but environment variables are missing | `false` | `false` | `"TORCH_PKG_DIR and TORCH_NPU_DIR must be set when fallback codegen is needed"` |
| Schema not found | `false` | `false` | `"schema not found for '<op>' with overload '<overload>'"` |
| Compilation failed | `false` | `false` | `"compilation failed: <error>"` |

### GeGraphBuilder Handling Logic

In `ge_graph_builder.cc`:

```cpp
auto codegen_result = FallbackCodegen::GenerateFromModelJson(graph_json);

if (!codegen_result.success) {
    // Add an Error-level ValidationIssue and abort the flow
    result.issues.push_back({
        severity: Error,
        code: "FALLBACK_CODEGEN_FAILED",
        message: codegen_result.error_message
    });
    result.ok = false;
    return;
} else if (codegen_result.skipped) {
    // Add a Warning-level ValidationIssue and continue the flow
    result.issues.push_back({
        severity: Warning,
        code: "FALLBACK_CODEGEN_SKIPPED",
        message: codegen_result.error_message
    });
}
```

### Common Errors and Solutions

| Error message | Cause | Solution |
|---------|------|---------|
| `ASCEND_HOME_PATH not set` | CANN path not set in online mode | `source ~/Ascend/cann/set_env.sh` |
| `acl/acl.h: No such file` | g++ cannot find the CANN headers | Check that `ASCEND_HOME_PATH` is correct |
| `cannot find -lascendcl` | g++ cannot find the CANN libraries | Check that `ASCEND_HOME_PATH/lib64` exists |
| `libcust_opapi.so: undefined symbol` | ABI incompatibility | Compile with the CMake method, or check that library versions match |
| `basic_string::_M_construct null not valid` | Internal object corruption (cross-SO boundary issue) | Run in a real CANN environment; avoid the stub |
| `TORCH_PKG_DIR and TORCH_NPU_DIR must be set` | Fallback nodes exist but environment variables are not set | Set the required environment variables as described in Section 5 |
| `schema not found for '<op>'` | The operator is not registered with the PyTorch Dispatcher | Check that the `torch.library.Library` registration is loaded correctly |

---

## Debugging Guide

### Debugging Online Mode

If online compilation fails, troubleshoot as follows:

**1. Inspect the generated g++ command** (set `KEEP_ARTIFACTS=1`):

```bash
head -10 $OUTPUT_DIR/custom_op.cpp
```

**2. Run the g++ command manually**:

```bash
# Copy the g++ command from the comments in custom_op.cpp
g++ -shared -fPIC ... -o test.so custom_op.cpp
```

**3. Use the CMake method** (alternative):

```bash
cd $OUTPUT_DIR
cmake -B build -DTorch_DIR=$TORCH_PKG_DIR ...
cmake --build build -j4
```

**4. Check the environment variables**:

```bash
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
ls $ASCEND_HOME_PATH/include/acl/acl.h  # Should exist
ls $ASCEND_HOME_PATH/lib64/libascendcl.so  # Should exist
```

### Inspecting the Generated Code

**Offline mode**:

```bash
cat $OUTPUT_DIR/custom_op.cpp
cat $OUTPUT_DIR/CMakeLists.txt
```

**Online debug mode** (with the g++ command in comments):

```bash
# The header of the generated cpp file contains the full g++ command
head -20 $OUTPUT_DIR/custom_op.cpp
```

### About the Generated Files

**`custom_op.cpp` generated in offline mode** (does not contain the g++ command):

```cpp
// AUTO-GENERATED by FallbackCodegen
#include <torch/torch.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <acl/acl.h>
...
```

**`custom_op.cpp` generated in online mode (debug)** (contains the full g++ command in comments):

```cpp
// AUTO-GENERATED by FallbackCodegen
//
// Online compilation command (g++):
//   g++ -shared -fPIC -std=c++17 -O2 \
//       -I/path/to/torch/include \
//       -I/path/to/cann/include \
//       -I/path/to/torch_npu/include \
//       -L/path/to/torch/lib \
//       -L/path/to/cann/lib64 \
//       -Wl,-rpath,... \
//       -Wl,--no-as-needed \
//       -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -lge_runner -lopapi \
//       -Wl,--as-needed \
//       -o libcust_opapi.so custom_op.cpp
//
// If g++ compilation fails, try CMake method:
//   cmake -B build && cmake --build build -j4
//
#include <torch/torch.h>
...
```

You can copy the g++ command directly to compile and debug manually, or use the CMake method as an alternative.

---

## Performance Considerations

### Compilation Overhead

| Mode | First load | Subsequent loads |
|------|---------|---------|
| Offline mode | No compilation overhead (code generation only) | Must be compiled manually |
| Online mode | Compilation overhead (a few seconds) | Can be cached (`KEEP_ARTIFACTS=1`) |

### Compilation Cache

In online mode, if a `.so` file already exists in `OUTPUT_DIR`, it is not recompiled (it is loaded directly). In production environments, set `KEEP_ARTIFACTS=1` and use a fixed `OUTPUT_DIR` to take advantage of the cache.

### Runtime Overhead

- **Dispatcher call**: each Execute goes through `op_handle.callBoxed`, estimated at <1us
- **BuildStack**: iterates over input_kinds to build the call stack, estimated at <100ns
- **WriteBack**: involves D2D copies; the cost depends on the output tensor size

---

## Notes

### Multi-Developer Collaboration

- **Default behavior**: fallback codegen is not enabled, so other developers are unaffected
- **Before enabling**: make sure torch and torch_npu are installed and the environment variables are set correctly
- **CI/CD configuration**: if you want to enable it in CI, set the environment variables explicitly in the build script

### Safety

- The generated code is based on the PyTorch schema and is ABI-compatible with ATen kernels
- Only operator wrapper code is generated; operator implementation logic is not modified

### Online Mode Dependency Requirements

**Required**:
- `ASCEND_HOME_PATH`: CANN installation path (contains `include/acl/acl.h` and `lib64/libascendcl.so`)
- `g++`: a compiler with C++17 support

**Recommended**:
- Use a real CANN environment (`source ~/Ascend/cann/set_env.sh`)
- Avoid using a CANN stub in test environments (ABI compatibility issues may occur)

---

## Example Projects

### fallback_inplace_optional_case

Tests scenarios with multiple inplace parameters and multiple optional parameters.

- **Tested features**:
  - Multiple inplace parameters: `dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()`
  - Multiple optional parameters: `multi_optional_op(Tensor x, float alpha=1.0, float beta=0.0, ...) -> Tensor`
- **Directory**: `examples/fallback_inplace_optional_case/`
- **Run**: `cd examples/fallback_inplace_optional_case && bash run.sh`

### fallback_e2e_test

End-to-end A/B testing that verifies full aten type support in Fallback Codegen.

- **Model A**: Tensor-only (regression baseline)
- **Model B**: Full Type Spectrum (int/float/bool/int[]/Scalar/Tensor[])
- **Directory**: `examples/fallback_e2e_test/`
- **Run**: `cd examples/fallback_e2e_test && bash run.sh`

### sym_int_case

A simplified end-to-end test that verifies support for non-tensor inputs in the fallback mechanism.

- **Test model**: SimpleMixedModel (dynamic/static tensors + static/dynamic non-tensors + Converter + Custom operators)
- **Directory**: `examples/sym_int_case/`
- **Run**: `cd examples/sym_int_case && bash run.sh`

---

## Related Documents

### Design Documents
- [Fallback Feature Design Document](../features/fallback_codegen_design.md) — initial design (including implementation evolution notes)
- [Overall Design Document](../features/overall_design_20260630.md), Section 5 — Fallback Codegen feature
- [Design of Automatic GE Custom Operator Generation Based on ATen IR Callbacks](../features/aten_ir_callback_codegen_20260715.md) — InputKind encoding, BuildStack/WriteBack implementation, user integration guidelines

### Implementation Documents
- csrc/fallback_codegen/README.md — module implementation document

### Build Documents
- [Build Guide](build.md) — TorchDelegate build guide (including `TORCH_NPU_DIR` configuration)
- [Epair Packaging Guide](build_epair.md) — Python API usage guide
