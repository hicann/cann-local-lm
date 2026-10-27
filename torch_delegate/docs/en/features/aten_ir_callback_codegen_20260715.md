# Automatic Generation of GE Custom Operators Based on ATen IR Callbacks

> [!NOTE] Document status
> This is the design draft dated 2026-07-15, describing the custom operator generation scheme based on ATen IR callbacks. For current usage instructions, see [Fallback Codegen User Guide](../guide/fallback_codegen_guide.md).

## Purpose

This document describes the design requirements and implementation details of the automatic GE custom operator generation scheme based on ATen IR callbacks. The target readers are GE graph engine developers, torchair adapter layer developers, and users integrating custom operators.

The scheme aims to solve the problem that users repeatedly write deliverables when integrating custom operators into GE through the PyTorch custom operator mechanism. Through automatic code generation and the Dispatcher callback mechanism, it achieves automatic reuse of computation logic.

## Scope

This document covers:
- Background and goals of the scheme
- Three-stage design flow (prototype mapping, converter graph construction, Execute execution)
- Input/output mapping rules, value dependency registration, schema metadata attributes
- User integration specifications and constraints
- Framework implementation priorities

This document does not cover:
- Support for CPU custom operators (this scheme targets NPU device scenarios only)
- Internal implementation details of the libtorch Dispatcher
- Complete conversion rules from ATen IR to GE IR (only the part reused through callbacks is covered)

---

## Overview

### Software overview

#### Project introduction

This project is a new feature of the GE graph engine that simplifies the process of integrating user custom operators into GE. Previously, users had to:
1. Implement a converter (completing ATen IR → GE IR conversion)
2. Implement the deliverable that enters the GE graph (implementing the `Execute` function via the interfaces in `custom_op.h`)

This scheme eliminates the repetitive work of writing deliverables through automatic code generation and Dispatcher callbacks.

#### Product environment

This product is a component where the GE graph engine and the torchair adapter layer work together, running in the following environment:

**System architecture:**
```text
PyTorch (TORCH_LIBRARY registration) → libtorch Dispatcher → ATen IR
                                           ↓
torchair (FX Graph tracing) → GE Converter → GE ComputeGraph
                                           ↓
GE Runtime (Execute) → Dispatcher callback → PrivateUse1 implementation → NPU operator execution
```

**External interfaces:**
1. PyTorch TORCH_LIBRARY/TORCH_LIBRARY_IMPL: users register custom operators through this interface
2. torchair converter: traverses the FX Graph to build GE nodes
3. libtorch Dispatcher: GE Execute calls back ATen IR through this interface

**Related products:**
- libtorch: PyTorch C++ frontend, providing the Dispatcher mechanism
- torchair: the PyTorch-to-GE adapter layer, responsible for graph tracing and conversion
- GE Runtime: the graph execution engine; this scheme mainly involves the runtime/v2 path

---

### Competitive analysis

#### Comparison dimensions

Comparison of custom operator mechanisms in mainstream deep learning frameworks:

| Dimension | TensorRT | PyTorch | JAX | Inductor | TensorFlow | **GE (this scheme)** |
|------|----------|---------|-----|----------|------------|---------------|
| **Core API** | IPluginV3 | TORCH_LIBRARY | custom_ops | custom_op | REGISTER_OP | **REG_OP + Dispatcher** |
| **Registration mechanism** | Plugin Registry | Dispatcher | JIT mounting | Inductor backend | Kernel Registry | **Operator Factory** |
| **Schema definition** | C++ interface description | Python def string | Python function signature | Python function signature | C++ OP_DEF | **ATen schema parsing** |
| **Shape inference** | getOutputDimensions | Meta key | Abstract evaluation | Symbolic tracing | ShapeFn | **Meta key (reuse PyTorch)** |
| **Execution model** | Device Sink | Host Dispatch | XLA JIT | AOT compilation | TFRT/CPU | **Dispatcher callback** |
| **Serialization** | serialize/deserialize | None | None | None | Bundle packaging | **OM file (existing)** |
| **Version compatibility** | Plugin Version | Unconstrained | Unconstrained | Unconstrained | OpVersion | **To be enhanced** |
| **Primary language** | C++ | Python/C++ | Python | Python | C++ | **C++ + Python (torchair)** |
| **Applicable scenario** | Inference optimization | Training + inference | Scientific computing | PyTorch 2.x | Production deployment | **Online inference (torchair)** |

#### Borrowed design analysis

This scheme borrows the following good designs from competitors:

| Design decision | Reference | Reason |
|----------|----------|----------|
| **Mandatory Meta key** | PyTorch | PyTorch requires a Meta implementation to guarantee compile-time shape inference correctness. The GE scheme follows this design to ensure no out-of-bounds writes in static shape scenarios. |
| **Dispatcher multi-backend routing** | PyTorch | The PyTorch Dispatcher can route to different backends (CPU/CUDA/PrivateUse1). GE borrows the PrivateUse1 key design to implement NPU backend adaptation. |
| **Schema string definition** | PyTorch, JAX | Schema-driven design is concise and easy to maintain. GE CodeGen parses the ATen schema to automatically generate REG_OP, eliminating manual maintenance. |
| **Serialization mechanism** | TensorRT | TensorRT's serialize/deserialize supports engine persistence. GE already has an OM file serialization mechanism, and operator version compatibility should be further enhanced. |
| **OP_DEF + Kernel separation** | TensorFlow | TensorFlow separates schema definition from Kernel implementation. GE's REG_OP + IMPL_OP design is similar, making it easy for operator prototypes and implementations to evolve independently. |

#### Difference analysis

Key differences between this scheme and competitors:

| Dimension | GE scheme | Competitor scheme | Reason for difference |
|------|----------|----------|----------|
| **Applicable scenario** | Online scenarios only (torchair) | TensorRT: offline inference; PyTorch: training + inference | GE dynamic shape scenarios rely on Dispatcher callbacks, and offline compilation (atc) does not support runtime registration |
| **Execution model** | Dispatcher callback (Host) | TensorRT: Task Sink (Device) | Dispatcher callbacks flexibly support dynamic shapes but introduce Host scheduling overhead; static shape scenarios should retain the Sink mode |
| **Serialization support** | OM file serializes operator prototypes | TensorRT: full serialization (with version); PyTorch: no serialization | GE OM files need an enhanced operator version compatibility mechanism, referring to TensorRT Plugin Version |
| **Operator packaging** | so_in_om (partial support) | TensorFlow: full Bundle packaging | GE should enhance operator Bundle packaging to support distributing operator dependencies with the model |
| **Meta validation** | Manual validation | PyTorch: automatic validation in torch.compile | GE should provide a Meta implementation validation tool to reduce user burden |

#### Design trade-offs

Based on the competitive analysis, the design trade-offs of this scheme are:

1. **Performance vs. usability**: choose usability first (Schema-driven + Dispatcher), sacrificing some performance (Host scheduling overhead), suitable for online scenarios
2. **Flexibility vs. constraints**: choose constraints first (mandatory Meta key), ensuring correctness in static shape scenarios
3. **Compile time vs. runtime**: choose runtime registration (depending on the PyTorch Dispatcher), suitable for online scenarios; offline scenarios are left for later support

---

### Software features

This scheme provides the following core features:

1. **Prototype auto-generation**: CodeGen automatically generates GE operator prototypes (REG_OP) from the ATen schema and registers them with the operator factory
2. **Converter auto graph construction**: the torchair converter traverses the FX Graph, automatically builds GE nodes, and encodes schema information as node attributes
3. **Execute auto execution**: GE provides a built-in generic Execute implementation that dynamically builds Dispatcher calls by reading node attributes

Feature relationship diagram:

```text
ATen Schema → [CodeGen] → REG_OP + value dependency registration + metadata attributes
                                    ↓
FX Graph → [Converter] → GE Graph (Const nodes, merge operators)
                                    ↓
GE Execute → [BuildStack] → Dispatcher call → [WriteBack] → GE output Tensor
```

### Design constraints

#### Constraints users must satisfy

1. **Schema definition**: register the complete schema through `TORCH_LIBRARY`; only use data types supported by both GE and PyTorch
2. **Meta key implementation**: must register the Meta key for symbolic shape inference during the torchair compile stage
3. **PrivateUse1 key implementation**: must register the PrivateUse1 key computation implementation for GE Execute to call back through the Dispatcher
4. **so load order**: `libcust_opapi.so` must be loaded before entering GE execution

#### Constraints handled automatically by the framework

1. **Schema-driven**: all code generation is based on schema analysis rather than hard coding
2. **Zero-copy first**: prefer zero-copy solutions in NPU scenarios
3. **Unified carrier model**: all non-Tensor parameters are uniformly converted into Tensor input edges, eliminating path splitting

#### Unsupported scenarios

- Schemas containing `Device` / `Layout` / `Generator` / `MemoryFormat` parameters
- Scalar default values containing the `complex` type (not supported in P0)
- Containing `Tensor[]` but unable to statically determine the upper bound of output count
- Cases where schema parsing fails

### Assumptions and dependencies

#### Dependencies of this iteration

1. **libtorch Dispatcher**: relies on the Dispatcher mechanism to look up and invoke ATen IR
2. **torchair converter**: relies on torchair to provide FX Graph tracing and node construction
3. **GE operator factory**: relies on GE to provide the operator registration mechanism (REG_OP/IMPL_OP)
4. **GE runtime/v2**: relies on GE runtime v2 to provide the Execute execution environment

#### Assumptions

1. The user has correctly registered TORCH_LIBRARY and TORCH_LIBRARY_IMPL
2. The shape inference in the Meta key implementation is correct
3. The computation logic in the PrivateUse1 key implementation is correct
4. `libcust_opapi.so` is loaded at runtime

---

## Feature requirements analysis & design

### Overall introduction

This scheme implements automatic integration of custom operators through a three-stage design:

| Stage | Responsibility | Input | Output |
|------|------|------|------|
| **Step 1: prototype mapping** | CodeGen generates REG_OP from the schema | ATen schema | REG_OP + value dependency registration + metadata attributes |
| **Step 2: converter graph construction** | Traverse the FX Graph to build GE nodes | FX Graph + schema | GE Graph (including Const nodes, merge operators) |
| **Step 3: Execute execution** | Convert from gert::Tensor to torch API calls | GE inputs + metadata attributes | Dispatcher call + WriteBack |

Flow diagram:
```text
[Build/load stage] CodeGen scans ATen schemas registered via TORCH_LIBRARY
                automatically generates REG_OP code and registers it with the GE operator factory
                registers value dependencies (InputsDataDependency)
                saves schema metadata as node attributes
    ↓
[Graph tracing stage] torchair traverses the FX Graph, calling the converter for each call_function node
    ↓
converter handles parameters:
  - Tensor parameters: connect upstream output edges directly
  - Non-Tensor constants: create Const nodes (HOST memory)
  - Non-Tensor dynamic values: connect upstream operator output edges
  - List parameters: generate a merge operator
  - Optional None: optional inputs are not connected
    ↓
converter creates nodes in the GE graph and writes metadata attributes
    ↓
[Runtime] GE Execute is invoked:
  1. Determine each parameter type based on input_kinds
  2. Retrieve values from input edges (Tensor wrapping / scalar extraction)
  3. Invoke ATen IR through the Dispatcher (PrivateUse1 key)
  4. Write results back to GE output Tensors
```

---

### Functional requirements

#### Functional requirement 1 Prototype mapping rules (REG_OP generation)

**Introduction**

CodeGen automatically generates GE operator prototypes from the ATen schema, including input/output mapping, value dependency registration, and schema metadata attributes. This is the core mechanism that frees users from manually writing deliverables.

**Input**

- **Source**: ATen schemas registered by the user via TORCH_LIBRARY
- **Quantity**: Determined by the number of operators registered by the user
- **Unit of measure**: schema entry
- **Valid input range**: ATen schemas supported by both GE and PyTorch (excluding Device/Layout/Generator/MemoryFormat parameters)
- **Timing requirement**: generated once during the build stage

**Processing**

**Operator identifiers and naming:**
- schema name `my_ops::my_op` → REG_OP identifier `my_ops__my_op` (`::` → `__`)
- schema name `my_ops::add.Tensor` → REG_OP identifier `my_ops__add_Tensor` (`.` → `_`)
- The node type string in the GE graph still keeps the schema name to ensure dump readability

**Input parameter mapping rules:**

Core design: all non-Tensor parameters → Tensor input edges

| Schema parameter | REG_OP declaration | Value dependency | dtype constraint |
|------------|------------|--------|-----------|
| `Tensor name` | `.INPUT(name, TensorType::ALL())` | No | Unconstrained |
| `Tensor? name=None` | `.OPTIONAL_INPUT(name, TensorType::ALL())` | No | Unconstrained |
| `Tensor[] name` | `.DYNAMIC_INPUT(name, TensorType::ALL())` | No | Unconstrained |
| `int name` (no default) | `.REQUIRED_INPUT(name, IntTensorType())` | **Yes** | int64 |
| `int name=v` (with default) | `.OPTIONAL_INPUT(name, IntTensorType())` | **Yes** | int64 |
| `float name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, FloatTensorType())` | **Yes** | float32 |
| `bool name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, BoolTensorType())` | **Yes** | bool |
| `Scalar name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, TensorType::ALL())` | **Yes** | Polymorphic (expressed by tensor dtype) |
| `int[] name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, IntTensorType())` | **Yes** | 1-dim int tensor |
| `str name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, StringTensorType())` | **Yes** | string |
| `ScalarType name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, IntTensorType())` | **Yes** | int64 (enum value) |

**Required/optional determination rules:**
- Non-Optional parameter without a default value → `.REQUIRED_INPUT`
- Parameter with a default value or Optional → `.OPTIONAL_INPUT`

**Value dependency registration:**

All non-Tensor parameters (except `Tensor`, `Tensor?`, `Tensor[]`) must be registered as value dependencies:

```cpp
// REG_OP example
REG_OP(my_ops__my_op)
    .INPUT(x, TensorType::ALL())
    .REQUIRED_INPUT(dim, IntTensorType())           // int → value dependency
    .OPTIONAL_INPUT(alpha, TensorType::ALL())       // Scalar → value dependency
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(my_ops__my_op)

// Register value dependencies in IMPL_OP
IMPL_OP(my_ops__my_op)
    .InputsDataDependency({"dim", "alpha"})         // list of non-Tensor parameter names
```

**Schema metadata attributes:**

Preserve schema metadata as node attributes for use by Execute:

```cpp
REG_OP(my_ops__my_op)
    // ... input/output declarations ...

    // Schema metadata attributes
    .ATTR(op_name, String, "my_ops::my_op")         // Dispatcher lookup
    .ATTR(input_kinds, ListInt, {...})              // type code for each parameter
    .ATTR(num_outputs, Int, 1)                      // number of outputs
    .ATTR(is_inplace, Bool, false)                  // inplace flag
    .OP_END_FACTORY_REG(my_ops__my_op)
```

| Attribute | Type | Purpose | Source |
|------|------|------|------|
| `op_name` | String | Dispatcher looks up OperatorHandle | schema name |
| `input_kinds` | ListInt | InputKind code for each parameter | schema parameter types |
| `num_outputs` | Int | WriteBack loop bound | number of schema outputs |
| `is_inplace` | Bool | WriteBack mode selection | schema inplace annotation |

**InputKind code definitions:**

```cpp
enum InputKind {
    // Tensor class (does not use value dependencies)
    TENSOR = 0,
    OPTIONAL_TENSOR = 1,
    TENSOR_LIST = 2,

    // Scalar class (uses value dependencies)
    INT = 10,
    INT_OPT = 11,
    FLOAT = 12,
    FLOAT_OPT = 13,
    BOOL = 14,
    BOOL_OPT = 15,
    SCALAR = 16,
    SCALAR_OPT = 17,

    // List class (uses value dependencies)
    INT_LIST = 20,
    INT_LIST_OPT = 21,
    FLOAT_LIST = 22,
    BOOL_LIST = 23,

    // Special class (uses value dependencies)
    STRING = 30,
    STRING_OPT = 31,
    SCALAR_TYPE = 32,
    SCALAR_TYPE_OPT = 33,
};
```

**Exception handling:**
- Schema parsing failure: skip REG_OP generation, and the converter triggers fallback at runtime
- Unsupported parameter types: skip generation

**Output**

- **Where output goes**: GE operator factory (registered via REG_OP)
- **Quantity**: one REG_OP per operator
- **Unit of measure**: operator prototype
- **Timing**: generated once during the build stage
- **Valid output range**: prototypes conforming to GE operator registration specifications

---

#### Functional requirement 2 Converter graph construction

**Introduction**

The torchair converter traverses the FX Graph, builds the corresponding GE node for each custom operator node, handles parameter mapping, and writes metadata attributes.

**Input**

- **Source**: torchair FX Graph (call_function nodes)
- **Quantity**: determined by the number of custom operators in the FX Graph
- **Unit of measure**: FX Node
- **Valid input range**: custom operator nodes with registered schemas
- **Timing requirement**: built in real time during the graph tracing stage

**Processing**

**Parameter handling rules:**

| FX args form | Converter behavior | GE graph structure |
|-------------|---------------|----------|
| Tensor (FX Node) | Connect upstream output edge directly | Input edge connects to upstream node |
| Constant literal | Create a Const node | Const node (HOST memory) connects edge |
| FX Node (0-dim tensor) | Connect upstream output edge directly | Input edge connects to upstream node |
| FX Node (SymInt output, such as `aten.size`) | Connect upstream output edge | Input edge connects to `aten.size` and similar nodes |
| List parameter (`int[]`) | Generate a merge operator | Merge operator output connects edge |
| None (Optional) | No edge | OPTIONAL_INPUT lacks edge |
| Omitted (use default value) | Create a Const node holding the default value | Const node connects edge |

**Const node creation:**

```python
def create_const_node(value, input_kind):
    # Select dtype based on input_kind
    dtype_map = {
        INT: "int64",
        INT_OPT: "int64",
        FLOAT: "float32",
        FLOAT_OPT: "float32",
        BOOL: "bool",
        BOOL_OPT: "bool",
        SCALAR: infer_dtype_from_value(value),  # Python type inference
        STRING: "string",
        SCALAR_TYPE: "int64",  # enum value
    }

    dtype = dtype_map[input_kind]
    const_node = CreateConstNode(value, dtype, placement=HOST)
    return const_node
```

**Const node characteristics:**
- The value tensor is stored in HOST memory
- dtype is selected based on the schema type or the Python literal type
- Execute reads with zero overhead (no D2H synchronization)

**List parameter handling:**

```python
def process_list_arg(args_list, input_kind):
    tensors = []
    for arg in args_list:
        if is_literal(arg):
            # Constant: create a Const node
            tensors.append(create_const_node(arg, input_kind))
        elif is_fx_node(arg):
            # Upstream node: reference directly
            tensors.append(arg)

    # TODO: generate a merge operator to combine multiple tensors into a single tensor
    merge_node = CreateMergeOp(tensors, dtype=get_dtype(input_kind))
    return merge_node.output
```

**Exception handling:**
- Merge operator not yet implemented (P0 TODO): list parameter handling is currently skipped
- Unsupported parameter types: runtime fallback

**Output**

- **Where output goes**: GE ComputeGraph (nodes)
- **Quantity**: one GE Node per FX Node
- **Unit of measure**: GE Node
- **Timing**: built in real time during the graph tracing stage
- **Valid output range**: nodes conforming to GE graph specifications

---

#### Functional requirement 3 Execute execution conversion

**Introduction**

The GE Execute function dynamically builds the Dispatcher call stack by reading node attributes, calls back ATen IR to implement the computation logic, and writes results back to GE output Tensors.

**Input**

- **Source**: GE runtime v2 execution environment
- **Quantity**: one call per execution
- **Unit of measure**: Execute function call
- **Valid input range**: gert::Tensor inputs (NPU device HBM memory)
- **Timing requirement**: executed in real time at runtime

**Processing**

**Overall flow:**

```cpp
graphStatus Execute(gert::EagerOpExecutionContext* ctx) {
    // 1. Get metadata
    auto op_name = ctx->GetAttr<std::string>("op_name");
    auto input_kinds = ctx->GetAttr<std::vector<int>>("input_kinds");
    auto is_inplace = ctx->GetAttr<bool>("is_inplace");
    auto num_outputs = ctx->GetAttr<int>("num_outputs");

    // 2. Build the call stack
    c10::Stack stack = BuildStack(ctx, input_kinds);

    // 3. Dispatcher call
    static auto& op_handle = GetCachedHandle(op_name);
    op_handle.callBoxed(&stack);

    // 4. WriteBack
    WriteBack(ctx, stack, is_inplace, num_outputs);

    return GRAPH_SUCCESS;
}
```

**BuildStack implementation:**

```cpp
c10::Stack BuildStack(gert::EagerOpExecutionContext* ctx,
                      const std::vector<int>& input_kinds) {
    c10::Stack stack;
    int input_idx = 0;

    for (int kind : input_kinds) {
        auto* tensor = ctx->GetInputTensor(input_idx);

        // Handle missing optional edges
        if (tensor == nullptr) {
            if (IsOptional(kind)) {
                stack.emplace_back(c10::nullopt);
            } else {
                GELOGE("Required input %d is missing", input_idx);
                throw std::runtime_error("Required input missing");
            }
            input_idx++;
            continue;
        }

        // Extract the value based on type
        switch (kind) {
            case TENSOR:
            case OPTIONAL_TENSOR:
                stack.emplace_back(WrapNpuTensor(tensor));
                break;

            case INT:
            case INT_OPT:
                stack.emplace_back(ExtractInt(tensor));
                break;

            case FLOAT:
            case FLOAT_OPT:
                stack.emplace_back(ExtractFloat(tensor));
                break;

            case BOOL:
            case BOOL_OPT:
                stack.emplace_back(ExtractBool(tensor));
                break;

            case SCALAR:
            case SCALAR_OPT:
                stack.emplace_back(ExtractScalar(tensor));
                break;

            case INT_LIST:
            case INT_LIST_OPT:
                stack.emplace_back(ExtractIntList(tensor));
                break;

            // ... handling of other types
        }

        input_idx++;
    }

    return stack;
}
```

**Tensor wrapping:**

```cpp
at::Tensor WrapNpuTensor(const gert::Tensor* ge_tensor) {
    auto sizes = GetShape(ge_tensor->GetStorageShape());
    auto dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());
    int32_t device_id = GetCurrentDeviceId();

    return torch::from_blob(
        const_cast<void*>(ge_tensor->GetAddr()),
        sizes,
        [](void*) {},  // GE manages the memory; torch is not responsible for freeing it
        torch::TensorOptions()
            .dtype(dtype)
            .device(c10::Device(c10::DeviceType::PrivateUse1, device_id))
    );
}
```

**WriteBack implementation:**

```cpp
void WriteBack(gert::EagerOpExecutionContext* ctx, c10::Stack& stack,
               bool is_inplace, int num_outputs) {
    for (int i = 0; i < num_outputs; ++i) {
        at::Tensor result = stack[i].toTensor();
        if (!result.is_contiguous()) {
            result = result.contiguous();
        }

        if (is_inplace && i == 0) {
            // inplace: write the result back to input 0
            const gert::Tensor* input_0 = ctx->GetInputTensor(0);
            aclrtMemcpy(const_cast<void*>(input_0->GetAddr()),
                        input_0->GetSize(),
                        result.data_ptr(), result.nbytes(),
                        ACL_MEMCPY_DEVICE_TO_DEVICE);
            ctx->MakeOutputRefInput(0, i);
        } else {
            // non-inplace: allocate the output and write
            ge::DataType ge_dtype = MapTorchDtypeToGe(result.scalar_type());
            gert::Tensor* ge_out = ctx->MallocOutputTensor(
                i, result.sizes(), ge::FORMAT_ND, ge_dtype);
            aclrtMemcpy(ge_out->GetAddr(), ge_out->GetSize(),
                        result.data_ptr(), result.nbytes(),
                        ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
    }
}
```

**Exception handling:**
- Required input missing: report an error and throw
- Dispatcher call failure: return `GRAPH_FAILED`
- Unsupported dtype: report an error and throw

**Output**

- **Where output goes**: GE output Tensor (gert::Tensor)
- **Quantity**: determined by the number of schema outputs
- **Unit of measure**: gert::Tensor
- **Timing**: executed in real time at runtime
- **Valid output range**: tensors with contiguous memory layout and ND format

---

### Non-functional requirements

#### Maintainability

1. **Schema-driven**: all code generation is based on schema analysis; after changing the schema, simply regenerate without manual maintenance
2. **Generic Execute implementation**: all custom operators share the same Execute function and configure behavior through attributes, reducing code duplication
3. **InputKind coding**: segmented coding makes it easy to quickly determine processing branches and is easy to extend with new types

#### Testability

1. **Unit tests**: functions such as BuildStack, WrapNpuTensor, and WriteBack can be tested independently
2. **Integration tests**: verify the complete flow by registering an example operator
3. **User constraint validation**: the correctness of the Meta key implementation and PrivateUse1 key implementation can be validated through test cases

#### Portability

1. **Depends on libtorch**: requires libtorch Dispatcher support
2. **Depends on torchair**: requires torchair converter support
3. **Depends on GE v2 runtime**: requires the runtime v2 execution environment

#### Platformization requirements

This feature is an online-scenario feature of GE, involving calls to the torchair adapter layer. It does not involve chip differentiation, but the support capability of the underlying Dispatcher interface on each platform must be confirmed.

#### Cross-feature analysis

| Scenario | Applicability | Analysis |
|------|--------|----------|
| Static shape | **Not applicable** | The scheme targets NPU device scenarios and calls back ATen IR through the Dispatcher. Static shape scenarios (TaskSink mode) do not involve Dispatcher calls; the Execute function executes the operator on the Host side rather than in TaskSink mode. |
| Dynamic shape | **Applicable** | The Execute function builds the call stack at runtime and invokes ATen IR through the Dispatcher, which is the typical execution mode for dynamic shape scenarios. The scheme design supports dynamic shape inference (depending on the user-provided Meta key implementation). |
| Dynamic shape with static subgraph (graph splitting) | **Not applicable** | The scheme is mainly used in online scenarios, building GE graphs through the torchair converter. Static subgraphs (DavinciModelKernel) take the v1 TaskSink path and do not involve Dispatcher calls. |
| Offline scenario (atc compilation) | **Not applicable** | The scheme relies on the torchair converter to build GE graphs at runtime, so offline compilation (atc) scenarios do not apply. Users need to register operators through TORCH_LIBRARY, which is specific to online scenarios. |
| Online scenario (framework adaptation) | **Applicable** | The scheme mainly targets torchair framework adaptation, traversing the FX Graph through the converter to build GE nodes and calling back ATen IR through Execute. This is the typical online scenario. |

---

### Performance

#### Model compile time

This feature involves the following compile stages:

1. **CodeGen (prototype mapping)**: scans ATen schemas registered via TORCH_LIBRARY and generates REG_OP. The added code has a small impact, mainly schema parsing and code generation, and the increase in compile time is controllable (estimated <1s).

2. **torchair converter (graph construction)**: traverses the FX Graph to build GE nodes. The added code is parameter handling logic, with a small impact on compile time.

**Impact assessment:** the compile time impact is controllable and does not affect key stages such as optimizeStage1, optimizeStage2, build, and loadmodelonline.

#### OM size and load memory footprint

This feature applies only to online scenarios and does not involve OM file generation. There is no additional memory footprint at load time.

#### Execution performance

**Key execution path:**
1. BuildStack: builds the call stack, involving traversal of input_kinds and value extraction
2. Dispatcher call: invokes ATen IR through `op_handle.callBoxed`
3. WriteBack: result write-back, involving `aclrtMemcpy`

**Performance impact assessment:**
- BuildStack: traverses input_kinds on each execution, estimated <100ns (small graphs)
- Dispatcher call: depends on the libtorch Dispatcher implementation, estimated <1us
- WriteBack: involves D2D copy; time depends on the output tensor size

**Performance optimization suggestions:**
1. OpCache: cache `op_handle` to avoid repeated lookups (P2)
2. Avoid dynamic memory allocation in BuildStack (per RT2 constraints)
3. Use compile-time constant types for scalar parameters (per user integration specifications)

---

### Interface design

#### New/modified interface description

This feature adds no new GE public interfaces and mainly implements through the following internal mechanisms:

1. **REG_OP auto-generation**: automatically generate operator prototypes through CodeGen
2. **Generic Execute implementation**: all custom operators share the same Execute function

The user interface is PyTorch TORCH_LIBRARY/TORCH_LIBRARY_IMPL, not a GE interface.

#### Interface checklist

| **Check item** | **Sub-check item** | **Involved** |
| --- | --- | :--- |
| **Interface description** | Whether the interface needs review, and the review needs to focus on interface compatibility and interface constraints | **Not involved** (no new GE interfaces) |
| **Interface description** | Whether supplementary documentation is needed | **Involved** (user integration specification documentation needs to be added) |
| **Interface description** | Whether the interface documentation clearly explains the interface prototype, functions, return values, etc. | **Not involved** (no new GE interfaces) |
| **Interface compatibility** | Interface behavior compatibility, whether behavior changes before and after the modification | **Not involved** (no modified GE interfaces) |
| **Interface compatibility** | Whether the interface is compatible, and whether the new interface works properly on older versions | **Not involved** (no new GE interfaces) |
| **Interface compatibility** | Whether the interface involves constraints such as usage scenarios and call timing | **Involved** (user so load order constraint) |
| **Interface constraints** | Whether clear errors are reported when call constraints are not satisfied | **Involved** (error reported when a required input is missing) |
| **Interface constraints** | Whether separate test cases need to be designed | **Involved** (test cases for user integration specifications need to be designed) |

---

### Software design

#### Key data structures

**InputKind coding:**
```cpp
enum InputKind {
    // Tensor class (does not use value dependencies)
    TENSOR = 0,
    OPTIONAL_TENSOR = 1,
    TENSOR_LIST = 2,

    // Scalar class (uses value dependencies)
    INT = 10,
    INT_OPT = 11,
    FLOAT = 12,
    FLOAT_OPT = 13,
    BOOL = 14,
    BOOL_OPT = 15,
    SCALAR = 16,
    SCALAR_OPT = 17,

    // List class (uses value dependencies)
    INT_LIST = 20,
    INT_LIST_OPT = 21,
    FLOAT_LIST = 22,
    BOOL_LIST = 23,

    // Special class (uses value dependencies)
    STRING = 30,
    STRING_OPT = 31,
    SCALAR_TYPE = 32,
    SCALAR_TYPE_OPT = 33,
};
```

**Helper functions:**
```cpp
bool IsOptional(InputKind kind);
bool IsValueDependency(InputKind kind);
```

#### Key techniques/algorithms

1. **Dispatcher callback mechanism**: invoke ATen IR through `op_handle.callBoxed(&stack)`
2. **Value dependency registration**: non-Tensor parameters are registered through `InputsDataDependency`
3. **Zero-copy Tensor wrapping**: wrap GE Tensors through `torch::from_blob` to avoid data copies

#### Flow design

**Three-stage flow:**

```text
Stage 1: prototype mapping
  ATen schema → CodeGen → REG_OP + value dependency registration + metadata attributes

Stage 2: converter graph construction
  FX Graph → converter → GE Graph (Const nodes, merge operators)

Stage 3: Execute execution
  GE inputs → BuildStack → Dispatcher call → WriteBack → GE outputs
```

#### Modifications to submodules

This feature involves the following submodules:

| Module | Change | Need to notify MDE |
|------|---------|-----------------|
| `graph_metadef/` | REG_OP auto-generation mechanism | **Yes** (operator registration mechanism extension) |
| `runtime/v2/` | Generic Execute implementation | **Yes** (v2 executor extension) |
| torchair (external component) | Converter graph construction logic | **Yes** (torchair needs to be updated in sync) |

#### Error handling

**System errors**

- **Memory allocation failure**: `MallocOutputTensor` fails in WriteBack, return `GRAPH_FAILED`
- **Dispatcher lookup failure**: `findSchemaOrThrow` throws (must be caught and an error returned)

**Interface errors**

- **Required input missing**: return `GRAPH_FAILED` and log the missing input index
- **Unsupported dtype**: return `GRAPH_FAILED` and log the unsupported dtype
- **Dispatcher call failure**: return `GRAPH_FAILED`

---

### Safety check

#### Coding rules

Refer to the coding rules document.

#### Coding checklist

| **Check item** | **Description** | **Involved** |
| --- | --- | --- |
| **Whether resource lifetime management is involved (such as thread pools, memory pools, etc.)** | An excessively long resource lifetime may cause resource shortage, slow response, system crashes, and other serious system stability and reliability problems | **Involved** (the OpCache lifetime must be clarified; process-level is recommended) |
| **Whether new threads are created** | 1) Newly created threads need to copy context information such as ThreadLocalContext and ErrorContext. 2) Newly created threads need to set a thread name | **Not involved** |

---

### Compatibility check

This feature applies only to online scenarios and does not involve OM file generation; there is no need to execute old OMs on new versions.

---

### DT design

#### Test boundaries

- **Test entry**: Execute function call
- **Test exit**: Dispatcher call succeeds, WriteBack results are correct
- **Stub modules**: libtorch Dispatcher (calls can be simulated), gert::Tensor (input/output can be simulated)

#### Test design

**Case types: UT, ST**

| **Test category** | **Key test item** | **Test method** | **Case type** |
| --- | --- | --- | --- |
| **Function** | Correctness of extracting each type in BuildStack | Construct inputs of different InputKinds and verify the extraction results | UT |
| **Function** | Zero-copy correctness of WrapNpuTensor | Construct a GE Tensor and verify that the wrapped torch Tensor addresses match | UT |
| **Function** | WriteBack inplace/non-inplace modes | Construct different is_inplace attributes and verify the write-back logic | UT |
| **Function** | Complete flow (CodeGen → Converter → Execute) | Register an example operator and verify the end-to-end execution result | ST |
| **Performance** | Execute execution performance | Construct inputs of different sizes and measure execution time | ST |
| **Precision** | Operator computation precision | Verify the output precision after the Dispatcher call | ST |
| **Compatibility** | Support for different schema forms | Test different parameter types (Tensor/Scalar/lists, etc.) | ST |
| **Cross-feature** | Dynamic shape scenarios | Test shape inference and execution for dynamic shape inputs | ST |

#### Test framework design

The current test framework (gtest) meets the UT requirements. The ST tests require an integrated torchair and libtorch environment.

---

### Acceptance criteria

1. After the user registers an operator through TORCH_LIBRARY, no manual writing of GE deliverables is required
2. CodeGen can correctly generate REG_OP, value dependency registration, and metadata attributes
3. Converter can correctly build GE nodes (including Const nodes and merge operators)
4. Execute can correctly build the call stack, call back the Dispatcher, and write back results
5. UT/ST test case pass rate is 100%
6. Execution performance meets expectations (BuildStack <100ns, Dispatcher <1us)

---

## Appendix

### Appendix A Data type mapping

| GE DataType | PyTorch ScalarType |
|-------------|-------------------|
| DT_FLOAT | kFloat |
| DT_FLOAT16 | kHalf |
| DT_DOUBLE | kDouble |
| DT_INT8 | kInt8 |
| DT_INT16 | kInt16 |
| DT_INT32 | kInt32 |
| DT_INT64 | kInt64 |
| DT_UINT8 | kUInt8 |
| DT_BOOL | kBool |
| DT_BF16 | kBFloat16 |

---

### Appendix B User custom operator integration specifications

#### Functional correctness constraints (must be satisfied)

**1. Schema definition**
Register the complete schema through `TORCH_LIBRARY`, using only data types supported by both GE and PyTorch. Inplace semantics are explicitly annotated with `Tensor(a!)`; multiple outputs are explicitly listed after `->`.

**2. Meta key implementation (required)**
The Meta key must be registered. The torchair FX graph tracing stage relies on Meta tensors for symbolic shape inference, and its absence directly causes graph capture to fail. The Meta implementation must **correctly infer the output shape and dtype**; incorrect shape inference causes Execute to write out of bounds in static shape scenarios. The Meta implementation performs pure shape inference only and does not call any NPU/device interfaces.

```cpp
TORCH_LIBRARY_IMPL(my_ops, Meta, m) {
    m.impl("my_op", [](const at::Tensor& x, const at::Tensor& y) {
        // Correctly infer the output shape and dtype; do not perform actual computation
        return at::empty(compute_output_shape(x, y),
                         x.options().dtype(compute_output_dtype(x, y)));
    });
}
```

**3. PrivateUse1 key implementation (required)**
The computation implementation for the PrivateUse1 key must be registered; GE Execute hits this key through the Dispatcher. If only Meta is registered, the dispatch fails at runtime and returns `GRAPH_FAILED`. The operator implementation uses `getCurrentNPUStream()` to obtain the current stream; GE Execute has already bridged the GE stream to the current NPU stream before calling the Dispatcher, so the operator needs no extra handling.

**4. so load order**
`libcust_opapi.so` must be loaded before entering GE execution (`torch.ops.load_library` or `dlopen`); if not loaded, `findSchemaOrThrow` throws directly.

#### Performance best practices

**5. Use compile-time constant types for scalar parameters**
Scalar parameters such as `alpha`, `dim`, `keepdim`, and `scale` are defined as `float/int/bool` in the schema and should not be defined as `Tensor`. `torch.compile` tracing freezes them into constants, taking the Const node HOST memory path, and Execute reads them with zero overhead.

```cpp
// Recommended: define scalar parameters directly as float
m.def("my_op(Tensor x, Tensor y, float alpha=1.0) -> Tensor");

// Not recommended: pass scalars as Tensor
m.def("my_op(Tensor x, Tensor y, Tensor alpha) -> Tensor");
```

**6. Operator implementations should return contiguous tensors directly**
In the PrivateUse1 implementation, the output tensor should have a contiguous memory layout. Non-contiguous tensors trigger an extra `contiguous()` copy during WriteBack.

**7. Use ND format for outputs**
Operator outputs use the standard multi-dimensional array format (ND). Non-ND formats are outside the scope of this scheme.

#### Constraint summary

| Item | Category | Consequence of violation |
|------|------|---------|
| Register a schema with data types within the range supported by both GE/PyTorch | Function required | Unsupported types fall back to the manual implementation path |
| Annotate inplace with `Tensor(a!)` explicitly | Function required | Inplace semantics misidentified, abnormal memory management |
| Implement the Meta key with correct shape inference | Function required | Missing: FX graph tracing fails; wrong shape: Execute writes out of bounds |
| Implement the PrivateUse1 key | Function required | Runtime dispatch fails, returns `GRAPH_FAILED` |
| Load so before GE execution | Function required | `findSchemaOrThrow` throws |
| Use `float/int/bool` for scalar parameters, not `Tensor` | Performance recommended | Each Execute introduces D2H synchronization, interrupting the NPU pipeline |
| Operator implementation returns contiguous tensors | Performance recommended | WriteBack triggers an extra `contiguous()` copy |
| Use ND format for outputs | Performance recommended | Outside the scope of this scheme, requiring format conversion |

---

### Appendix C Framework implementation priorities

| Priority | Feature | Description |
|--------|------|------|
| P0 | REG_OP generation + value dependency registration | Core prototype mapping |
| P0 | Converter basic graph construction (Tensor/scalar) | Core conversion logic |
| P0 | Execute BuildStack + Dispatcher call | Core execution logic |
| P0 | Exception handling + Fallback interface | Ensures robustness |
| P1 | List parameter merge operator | Extends parameter types |
| P1 | Special type handling for String/ScalarType | Extends parameter types |
| P1 | Stream bridging | NPU execution correctness |
| P2 | OpCache performance optimization | Avoids repeated parsing |
| P2 | DFX (dump/profiling integration) | Observability |
| P3 | Complex schema support (Tensor[], multiple outputs) | Expands coverage |

---

### Appendix D Design document review results

#### Competitive analysis review

- [x] Competitive comparison analysis: completed the comparison of custom operator mechanisms in mainstream frameworks
      - Comparison dimensions covered: API design, registration mechanism, shape inference, execution model, serialization, version compatibility
      - Frameworks analyzed: TensorRT, PyTorch, JAX, Inductor, TensorFlow (5 frameworks)
- [x] Borrowed design analysis: clearly marked the borrowed design points and reference competitors
      - Mandatory Meta Key → reference PyTorch
      - Dispatcher multi-backend routing → reference PyTorch PrivateUse1
      - Schema-driven CodeGen → reference PyTorch/JAX
      - Serialization mechanism → reference TensorRT
      - OP_DEF + Kernel separation → reference TensorFlow
- [x] Difference analysis: analyzed the differences between the GE scheme and competitors
      - Reasons for differences explained: GE applies only to online scenarios and relies on Dispatcher callbacks
      - Design trade-offs explained: usability first, constraints first, runtime registration
- [x] Design trade-off explanation: clearly defined trade-off decisions for performance vs. usability, flexibility vs. constraints, and compile time vs. runtime

#### Cross-feature impact

- [x] Cross-feature impact: analyzed scenario by scenario according to cross_feature_check.md;
      the scheme applies only to online scenarios (torchair framework adaptation), applies to dynamic shape scenarios,
      does not apply to static shape or offline scenarios,
      and does not involve graph splitting (dynamic shape with static subgraph) scenarios.

#### Key feature design principles and software constraints

- [x] Key feature design principles:
      1. Loaded `graph_metadef.md`; the scheme follows the module independence principle (REG_OP generation as an operator registration extension does not affect the core AscendIR structure);
      2. Loaded `rt2_runtime.md`; the scheme follows RT2 constraints:
         - Do not modify the computation graph at load time (CodeGen generates at the build stage and does not affect Lowering)
         - Avoid dynamic memory allocation at execution time (BuildStack uses an on-stack `c10::Stack`)
         - Support multi-threaded calls (the Execute function has no shared resources)
      3. The scheme does not break existing design principles.

---

### Appendix E TODO items

| Item | Description | Priority |
|------|------|--------|
| Concrete merge operator implementation | Which GE operator the converter generates to merge list elements | P0 |
| String tensor storage format in Const nodes | The concrete format of GE String tensors | P1 |
| ScalarType enum value mapping table | PyTorch ScalarType ↔ int64 mapping | P1 |
| Complete value dependency registration example | Full API usage of InputsDataDependency | P1 |
| Placement conversion mechanism details | The concrete flow of GE automatic placement conversion | P1 |
| OpCache performance optimization | Avoid repeated parsing of input_kinds | P2 |
