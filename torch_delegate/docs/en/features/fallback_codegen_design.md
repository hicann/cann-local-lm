# Fallback Codegen Feature Design Document

> [!NOTE] Document Status
> This is the initial design draft of Fallback Codegen, and parts of the implementation have since evolved. For current usage, see [Fallback Codegen User Guide](../guide/fallback_codegen_guide.md).

**Status**: Implemented
**Date**: 2026-05-15
**Last Revised**: 2026-07-20

---

> **Implementation evolution note** (updated 2026-06-30)
>
> This document was originally designed as a standalone CLI tool (`tools/fallback_codegen --epair ... --output-dir ...`).
> In the actual implementation, Fallback Codegen has been integrated into the `GeGraphBuilder::Build()` flow,
> triggered automatically via the environment variable `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1`,
> with no need for separate compilation or CLI invocation.
>
> For current implementation details, see:
> - `docs/en/guide/fallback_codegen_guide.md` (unified user guide)
> - `docs/en/features/overall_design_20260630.md` section 5 (Fallback Codegen feature)

---

## Background and Goals

### Problem Scenario

A PyTorch `ExportedProgram` contains a large number of operators, and the torchair converter can only cover a subset of them. For uncovered operators (including user-defined operators and third-party namespace operators such as `xllm_ascend`), without handling, the entire GE graph cannot be built.

### Design Goals

Fallback Codegen lets operators not covered by the converter automatically fall back to PyTorch Dispatcher calls, without manually writing GE custom operator code:

1. **Identify fallback nodes from `ep_json_with_ascend_ir.json`**: nodes whose `ascend_ir.source` is empty or missing
2. **Automatically generate GE custom operator code based on the PyTorch Schema**: obtain the schema via `findSchemaOrThrow`, and automatically generate an `EagerExecuteOp` subclass
3. **Compile online and load via dlopen**: generate `libcust_opapi.so`, and after `dlopen`, register it into GE
4. **Device consistency**: create NPU `at::Tensor` with zero copy via `at_npu::native::from_blob`, so data always stays on the NPU device

### Trigger Location

Fallback Codegen is not a standalone executable but a step in the `GeGraphBuilder` constructor (`csrc/backend/ge_graph_builder.cc`, search for `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` to locate):

```cpp
const char *enable_codegen_env = std::getenv("TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN");
bool enable_codegen = true;
if (enable_codegen_env != nullptr) {
  enable_codegen = (std::string(enable_codegen_env) == "1");
}

if (enable_codegen) {
  auto codegen_result = torch_delegate::fallback::FallbackCodegen::GenerateFromModelJson(graph_json);
  // ...
}
```

The environment variable `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` controls whether it is enabled; it is enabled by default when unset.

---

## Overall Architecture

```text
┌─────────────────────────────────────────────────────────┐
│ Python side (torch_delegate.api)                         │
│                                                          │
│   ExportedProgram                                        │
│     ↓ torchair converter                                 │
│   ep_json_with_ascend_ir.json (with ascend_ir annotations)│
│     ↓ epair_save                                         │
│   model.epair (ZIP archive)                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ C++ side (GeGraphBuilder constructor)                    │
│                                                          │
│   1. Read ep_json_with_ascend_ir.json from the archive  │
│   2. FallbackCodegen::GenerateFromModelJson(json)        │
│      ├── Parse JSON, identify nodes with empty          │
│      │   ascend_ir.source                                │
│      ├── For each node call Dispatcher findSchema to get │
│      │   the signature                                   │
│      ├── Generate custom_op.cpp (EagerExecuteOp subclass │
│      │   + REG_OP)                                       │
│      ├── Generate CMakeLists.txt (written to disk when  │
│      │   KEEP_ARTIFACTS=1)                               │
│      ├── g++ compiles to libcust_opapi.so                │
│      └── dlopen loads it; schema and REG_OP auto-register│
│          into GE                                         │
│   3. Continue building the GE Graph                      │
└─────────────────────────────────────────────────────────┘
```

---

## epair File Format

### epair File Structure

The epair file reuses the PT2 Archive format (ZIP) and adds two files:

```text
model.epair (ZIP structure)
├── epair_manifest.json               ← epair metadata (added)
├── ep_json_with_ascend_ir.json       ← EP JSON with ascend_ir (added)
├── models/
│   └── model.json                    ← standard EP JSON (without ascend_ir)
└── data/
    ├── weights/
    │   ├── model_weights_config.json
    │   └── weight_0, weight_1, ...
    ├── constants/
    │   ├── model_constants_config.json
    │   └── tensor_0, tensor_1, ...
    └── model_sample_inputs.pt
```

The filename constants are defined in `csrc/common/epair_archive_constants.h:24-25`:
- `kAnnotatedModelJson = "extra/ep_json_with_ascend_ir.json"`
- `kAnnotatedManifestJson = "extra/epair_manifest.json"`

### Node `ascend_ir.source` Field

Each node object in `ep_json_with_ascend_ir.json` may optionally contain an `ascend_ir` field, in which `source` identifies the conversion origin:

| `source` Value | Meaning | Fallback Handling |
|-------------|------|---------------|
| `"converter"` | torchair converter successfully converted to a GE operator | Not needed |
| `""` or field missing | Mock node / not covered by converter | Fallback Codegen required |

The filter logic of `ParseTargetsFromJson` (`fallback_codegen.cc:400-463`):
- A node must have an `ascend_ir` field that is an object (placeholder, output, and other non-compute nodes lack this field and are skipped)
- `ascend_ir.source` missing or an empty string → fallback needed
- Skip `torch.ops.aten._assert_tensor_metadata.default` (used only for graph structure validation, not involved in computation)

### Operator target Parsing

The target format on the FX graph: `torch.ops.<namespace>.<op_name>.<overload>`

- `torch.ops.aten.topk.default` → namespace=aten, op_name=topk, overload=""
- `torch.ops.aten.add.Tensor` → namespace=aten, op_name=add, overload="Tensor"
- `torch.ops.xllm_ascend.rope_in_place.default` → namespace=xllm_ascend, op_name=rope_in_place, overload=""

`default` is converted to an empty string (PyTorch Dispatcher uses an empty string to denote the default overload).

### Higher-Order Operator (HOP) Expansion

PyTorch's `auto_functionalized_v2` is a HOP that handles inplace operators; the real operator name is in the `inputs[0].arg.as_operator` field. `ParseTargetFromJson` (`fallback_codegen.cc:319-398`) detects and expands it:

```json
{
  "target": "torch.ops.higher_order.auto_functionalized_v2",
  "inputs": [
    {"name": "", "arg": {"as_operator": "torch.ops.my_ops.dual_inplace_.default"}},
    {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
    ...
  ]
}
```

It is replaced with `torch.ops.my_ops.dual_inplace_.default`, which then goes through the standard parsing logic.

---

## Schema-Based Code Generation Approach

### Approach Overview

Obtain the operator schema via `torch::Dispatcher::singleton().findSchema(op_name)`, extract parameter types, alias_info, default values, and other information, and automatically generate GE custom operator code.

### SchemaInfo Structure

`SchemaInfo` (`fallback_codegen.h:56-67`) records key information extracted from the schema:

```cpp
struct SchemaInfo {
    std::string full_op_name;   // namespace::op_name
    std::string overload;
    std::string op_name;
    std::string namespace_name;
    std::vector<SchemaArgInfo> arguments;
    std::vector<SchemaArgInfo> returns;
    bool is_inplace = false;    // any argument has a write alias
    bool is_out = false;        // overload == "out"
    bool has_variadic_input = false;
    int num_fixed_inputs = 0;
};
```

### Parameter Type Identification

The `InputKind` enum covers all common PyTorch parameter types (`fallback_codegen.h:14-37`):

- Tensor / Optional[Tensor] / List[Tensor]
- int / Optional[int] / List[int] / Optional[List[int]]
- float / Optional[float] / List[float] / Optional[List[float]]
- bool / Optional[bool] / List[bool] / Optional[List[bool]]
- Scalar / Optional[Scalar] / List[Scalar]
- str / Optional[str]
- ScalarType / Optional[ScalarType]

`GetSchemaInfo` (`fallback_codegen.cc:457-646`) precisely identifies each parameter type through `c10::TypePtr` methods such as `isSubtypeOf`, `castRaw<c10::OptionalType>`, and `cast<c10::ListType>`, and records:
- `has_default`: whether a default value exists
- `default_value`: the default value (int/double/bool/string/int_list/double_list can be extracted directly; other types are recorded as null)
- `alias_info`: alias annotation (e.g. `Tensor(a!)` indicates a write alias, used to identify inplace operations)
- `is_out`: whether it is an out parameter
- `is_variadic`: whether it is a variadic parameter (TensorList)

### GE Operator Identifier Generation

`GenerateGeOpIdentifier` (`fallback_codegen.cc:298-308`) converts `namespace::op_name` into a GE operator identifier:

- `aten::add` + `Tensor` → `aten__add_Tensor`
- `aten::relu_` + `` → `aten__relu_`
- `aten::add` + `out` → `aten__add_out`
- `my_ops::scale_add` + `` → `my_ops__scale_add`

Format: `namespace__op_overload` (`::` replaced with `__`, overload joined with `_`).

### REG_OP Generation

`GenerateGeOpReg` (`fallback_codegen.cc:840-956`) generates `REG_OP` registration code based on the schema:

- **Input declarations**: choose `INPUT` / `OPTIONAL_INPUT` / `DYNAMIC_INPUT` based on `InputKind`
  - `TENSOR` / `OPTIONAL_TENSOR` → `TensorType::ALL()`
  - `INT` / `INT_OPT` / `INT_LIST` → `IntTensorType()`
  - `FLOAT` / `FLOAT_OPT` / `FLOAT_LIST` → `FloatTensorType()`
  - `BOOL` / `BOOL_OPT` / `BOOL_LIST` → `BoolTensorType()`
  - `STRING` / `STRING_OPT` → `StringTensorType()`
  - `SCALAR` / `SCALAR_OPT` / `SCALAR_LIST` → `TensorType::ALL()`
  - `SCALAR_TYPE` / `SCALAR_TYPE_OPT` → `IntTensorType()`
- **Output declarations**:
  - Non-inplace: generate `.OUTPUT()` / `.DYNAMIC_OUTPUT()` from `returns`
  - inplace: skip returns with a write alias, and instead generate `.OUTPUT()` from arguments with a write alias
- **Attributes**: `op_name` (String), `overload` (String), `input_kinds` (ListInt), `num_outputs` (Int), `is_inplace` (Bool)
- **InputsDataDependency**: mark all non-tensor inputs (int/float/bool/scalar, etc.) as data dependencies
- **REG_AUTO_MAPPING_OP**: automatic mapping registration, so that GE takes the `EagerExecuteOp::Execute` path

### Execute Method Generation

`GenerateExecuteMethod` (`fallback_codegen.cc:988-1283`) generates the `EagerExecuteOp::Execute` method; the core flow is:

1. **Get input tensors**: call `ctx->GetInputTensor(i)` for each non-variadic input
2. **Pre-allocate outputs for out versions**: get shape/dtype/format from `ctx->GetOutputTensor(i)`, and call `ctx->MallocOutputTensor(i, ...)`
3. **Build the args vector**: process each parameter according to `InputKind`:
   - `TENSOR` / `OPTIONAL_TENSOR` → `CreateNpuTensor(ge_input)` (fill `c10::IValue()` when input is null)
   - `INT` / `INT_OPT` / `FLOAT` / `FLOAT_OPT` / `BOOL` / `BOOL_OPT` → `ReadScalarInput` (reinterpret_cast after D2H copy)
   - `INT_LIST` / `FLOAT_LIST` / `BOOL_LIST` → `ReadScalarInput` + construct vector in a loop
   - `SCALAR` / `SCALAR_OPT` → distinguish int64/float based on `GetDataType`
   - `SCALAR_LIST` → same as above, construct vector in a loop
   - `STRING` / `STRING_OPT` → `ReadScalarInput` + construct `std::string`
   - `SCALAR_TYPE` / `SCALAR_TYPE_OPT` → `ReadScalarInput` + `static_cast<at::ScalarType>`
   - `TENSOR_LIST` (variadic) → loop `ctx->GetInputTensor(tensor_idx + i)`
   - Missing with a default value → use the `default_value` extracted from the schema
   - Missing without a default value → `c10::IValue()` (None)
4. **Set NPU stream**: `c10_npu::getStreamFromExternal(ge_stream, device_id)` + `NPUStreamGuard`
5. **Call the operator**: `op_handle.callBoxed(args)` (not `call`, because `callBoxed` supports arbitrary parameter combinations)
6. **Write back outputs**:
   - Non-inplace: the tensors at the end of `args` are outputs, `MallocOutputTensor` + `aclrtMemcpyAsync` (D2D)
   - inplace with void return: extract results from saved tensor references, `MakeOutputRefInput` makes the output reference the input
   - inplace with explicit returns: same as the non-inplace path

### Tensor Conversion (GE ↔ PyTorch)

`CreateNpuTensor` (`fallback_codegen.cc:814-822`, the implementation in the generated code) uses `at_npu::native::from_blob` to create an NPU tensor with zero copy:

```cpp
at::Tensor CreateNpuTensor(const gert::Tensor* ge_tensor) {
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < ge_tensor->GetShape().GetShape().GetDimNum(); i++) {
        sizes.push_back(ge_tensor->GetShape().GetShape().GetDim(i));
    }
    at::ScalarType torch_dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());
    auto options = at::TensorOptions().dtype(torch_dtype).device(at::kPrivateUse1);
    return at_npu::native::from_blob(const_cast<void*>(ge_tensor->GetAddr()), sizes, options);
}
```

**Lifetime**:
- GE tensor memory is managed by the GE framework and is valid while the `Execute` method runs
- `at::Tensor` is a stack object that is destroyed automatically after the function returns, but it does not free the GE tensor memory
- During `callBoxed`, `at::Tensor` stays valid and requires no manual management

### Scalar Input Reading (ReadScalarInput)

GE places Const node data for custom ops in device memory (`kOnDeviceHbm`), which cannot be read directly via `GetAddr()` and requires a D2H copy:

```cpp
std::vector<uint8_t> ReadScalarInput(const gert::Tensor* tensor) {
    size_t byte_size = tensor->GetSize();
    std::vector<uint8_t> buf(byte_size);
    aclrtMemcpy(buf.data(), byte_size, tensor->GetAddr(), byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    return buf;
}
```

**Reason** (code comment `fallback_codegen.cc:824-828`): `custom_node_converter.cc` hardcodes the `kOnDeviceHbm` placement, so the `kPlacementHost` setting and `HostInputs` registration have no effect on custom ops. If GE fixes this issue in the future, it can be changed to read directly via `GetAddr()`.

### Stream Coordination

The GE Execute method obtains the GE stream, and PyTorch operators use the same stream, executing synchronously:

```cpp
aclrtStream ge_stream = static_cast<aclrtStream>(ctx->GetStream());
c10_npu::NPUStream npu_stream = c10_npu::getStreamFromExternal(ge_stream, device_id);
c10_npu::NPUStreamGuard guard(npu_stream);
op_handle.callBoxed(args);
```

**Constraint**: PyTorch operators must execute synchronously inside the GE Execute method and cannot be asynchronous. The GE framework synchronizes the stream uniformly after Execute returns.

---

## Inplace Operation Handling

### inplace Identification

Identified via the schema `alias_info`:

```cpp
const auto &first_arg = schema.arguments().at(0);
if (first_arg.alias_info() != nullptr && first_arg.alias_info()->isWrite()) {
    info.is_inplace = true;
}
```

Any parameter with an `isWrite()` alias marks the op as inplace.

### inplace Execute Generation

For inplace operations with empty `returns` (such as `dual_inplace_(Tensor(a!) a, Tensor(b!) b, ...) -> ()`):

1. **Save tensor references**: `callBoxed` clears `args`, so save `args[i].toTensor()` in advance
2. **Call the operator**: `op_handle.callBoxed(args)`; the PyTorch operator directly modifies the input tensor's memory
3. **Write back to GE inputs**: `aclrtMemcpyAsync` copies from the PyTorch tensor back to the GE input tensor (if the addresses differ)
4. **Output references input**: `ctx->MakeOutputRefInput(output_idx, input_idx)`; the GE output directly references the input memory

### inplace REG_OP

```cpp
REG_OP(my_ops__dual_inplace_)
    .INPUT(a, TensorType::ALL())
    .INPUT(b, TensorType::ALL())
    .INPUT(scale_a, FloatTensorType())
    .INPUT(scale_b, FloatTensorType())
    .OUTPUT(a, TensorType::ALL())   // output name is the same as the input
    .OUTPUT(b, TensorType::ALL())
    .ATTR(op_name, String, "my_ops::dual_inplace_")
    ...
```

---

## Optional Parameter Handling

### Missing Parameter Filling

When the `inputs` array length in the EP JSON is smaller than the number of schema parameters, missing parameters are filled in the following order:

1. `as_none` placeholder → use the schema default value or `c10::IValue()` (None)
2. Trailing omission → use the schema default value or `c10::IValue()` (None)

### REG_OP Full Declaration

Regardless of whether a given parameter is provided in the EP JSON, `REG_OP` declares the full set of parameters. The GE Graph Builder creates Const nodes for missing parameters (using default values), ensuring that the `Execute` method can read all parameters from the GE inputs.

---

## Compilation Flow

### Online Compilation (Default)

The online compilation flow of `GenerateFromTargets` (`fallback_codegen.cc:1422-1601`):

1. **Create a temporary directory**: `mkdtemp("/tmp/fbcg_XXXXXX")`
2. **Generate `custom_op.cpp`**: `EmitCustomOpCpp(schemas, gcc_cmd)`, with the full g++ command in a comment at the top of the file
3. **Generate `CMakeLists.txt`**: `EmitCMakeLists(output_dir, torch_pkg_dir, torch_npu_dir)` (written to disk only when `KEEP_ARTIFACTS=1`)
4. **Build the g++ command**: `BuildGccCommandFromPaths` assembles `-I`/`-L`/`-rpath` and other parameters
5. **g++ compilation**: `CompileWithGcc(gcc_cmd)`, executed via `popen` with output captured
6. **dlopen loading**: `dlopen(so_path, RTLD_NOW | RTLD_GLOBAL)`, so that schema and REG_OP are registered globally
7. **Clean up temporary files**: cleaned up by default; when `KEEP_ARTIFACTS=1`, copied to `${OUTPUT_DIR}/${ge_op_identifier}/`

### g++ Command

The command format generated by `BuildGccCommandFromPaths` (`fallback_codegen.cc:1385-1432`):

```bash
g++ -shared -fPIC -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -g \
    -I${TORCH_PKG_DIR}/include \
    -I${TORCH_PKG_DIR}/include/torch/csrc/api/include \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/include/graph \
    -I${ASCEND_HOME_PATH}/include/register \
    -I${TORCH_NPU_DIR}/include \
    -I${TORCH_NPU_DIR}/.. \
    -L${TORCH_PKG_DIR}/lib \
    -L${TORCH_NPU_DIR}/lib \
    -L${ASCEND_HOME_PATH}/lib64 \
    -Wl,-rpath,${TORCH_PKG_DIR}/lib \
    -Wl,-rpath,${TORCH_NPU_DIR}/lib \
    -Wl,-rpath,${ASCEND_HOME_PATH}/lib64 \
    -Wl,--no-as-needed \
    -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -lge_runner -llowering -lopapi \
    -Wl,--as-needed \
    -o libcust_opapi.so custom_op.cpp
```

### Generated CMakeLists.txt

The CMakeLists.txt generated by `EmitCMakeLists` (`fallback_codegen.cc:1340-1462`) supports two usage modes:

- **Environment variables**: `TORCH_PKG_DIR`, `TORCH_NPU_DIR`, `ASCEND_HOME_PATH`
- **CMake variables**: `-DTORCH_PKG_DIR=...`, `-DTORCH_NPU_DIR=...`, `-DASCEND_HOME_PATH=...`

Used when `KEEP_ARTIFACTS=1` for users to manually recompile and debug.

---

## Error Handling

### CodegenResult

```cpp
struct CodegenResult {
    bool success = false;
    bool skipped = false;       // true when no node needs fallback
    std::string error_message;
    std::string generated_cpp_path;     // valid when KEEP_ARTIFACTS=1
    std::string generated_cmake_path;   // valid when KEEP_ARTIFACTS=1
    void* so_handle = nullptr;          // dlopen handle
    std::vector<std::string> generated_op_names;
};
```

### Error Scenarios

| Scenario | `success` | `skipped` | `error_message` |
|------|-----------|-----------|-----------------|
| JSON parsing failure | `false` | `false` | `cannot parse model.json content: ...` |
| No fallback needed | `true` | `true` | `Skipped: no nodes need Fallback Codegen` |
| `TORCH_PKG_DIR` / `TORCH_NPU_DIR` not set | `false` | `false` | `TORCH_PKG_DIR and TORCH_NPU_DIR must be set when Fallback Codegen is needed` |
| Schema not found | `false` | `false` | `schema not found for '...' with overload '...'` |
| g++ compilation failure | `false` | `false` | `g++ compilation failed: ...` |
| dlopen failure | `false` | `false` | `dlopen failed: ...` |

### GeGraphBuilder Integration

In the `GeGraphBuilder` constructor, codegen failure directly `return`s (`init_status = FAILED`), aborting graph construction. When codegen is skipped (`skipped=true`), only a log is printed and the subsequent flow continues.

---

## Limitations

1. **Depends on PyTorch Dispatcher**: user operators must already be registered with the Dispatcher (via `TORCH_LIBRARY` / `torch.library`), otherwise `findSchema` fails
2. **Depends on torch_npu**: the generated code uses `at_npu::native::from_blob` and `c10_npu::NPUStream`, and must link against `libtorch_npu.so`
3. **Depends on CANN**: g++ compilation requires `ASCEND_HOME_PATH` to point to the CANN installation path, and runtime requires `libascendcl.so`, `libge_runner.so`, `libopapi.so`
4. **Performance overhead**: invoking through Dispatcher `callBoxed` has a slight overhead; scalar inputs require a D2H copy
5. **Synchronous execution**: PyTorch operators execute synchronously within GE Execute and cannot leverage asynchronous streams

---

## Reference Resources

- **Environment variables**: [../guide/environment_variables.md](../guide/environment_variables.md) — TorchDelegate environment variable reference
- **epair format**: [../guide/build_epair.md](../guide/build_epair.md) — epair file structure and `ascend_ir` field description
- **Build guide**: [../guide/build.md](../guide/build.md) — TorchDelegate build configuration
- **Source code**:
  - `csrc/fallback_codegen/fallback_codegen.h` — public interfaces and data structures
  - `csrc/fallback_codegen/fallback_codegen.cc` — implementation
  - `csrc/backend/ge_graph_builder.cc` — trigger location (`GeGraphBuilder` constructor)
- **Tests**: `tests/cpp/test_fallback_codegen.cc` — unit tests
- **Example**: `examples/fallback_inplace_optional_case/` — inplace + optional parameter example
