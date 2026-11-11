# TorchDelegate Overall Architecture

## Project Background and Positioning

### Problem Statement

When deploying models based on `torch.export` to an NPU (such as Ascend), there are two typical approaches:

1. **AOTInductor (AOTI)**: All compilation is completed in a Python environment, producing a `.so`. The advantage is no compilation overhead at runtime; the disadvantage is that compilation is bound to Inductor, making the NPU backend hard to replace, and compilation must be done on a machine with a Python/Torch environment.

2. **torch.export + deferred compilation**: Only the computation graph is exported, and compilation happens on the target device. The advantage is flexible deployment; however, the standard PT2 format does not carry backend IR information, so the target machine must redo IR conversion, which is inefficient.

### TorchDelegate's Positioning

TorchDelegate is an **intermediate path**:

- IR conversion (aten op → Ascend IR) is completed on the Python side, but **no graph partitioning and no compilation**
- IR information is inlined into the EP JSON and packed together with weights into an `.epair`
- The C++ side loads the `.epair` and completes graph partitioning → Codegen → compilation → deployment on the target device

Comparison with AOTI:

| Dimension | AOTI | TorchDelegate IR Archive |
|------|------|--------------------------|
| Python-side artifact | `.so` (fully compiled artifact) | `.epair` (IR + weights) |
| Graph partitioning timing | Python side | C++ side (deferred) |
| Compilation timing | Python side | C++ side (deferred) |
| Backend binding | Bound to Inductor | Pluggable (via `NodeIRConverter`) |
| Deployment environment requirements | Must match the compilation environment | Flexible, decided on the C++ side |

---

## Overview of the Four Modules

```text
┌────────────────────────────────────────────────────────┐
│  Module 1 (Python): ep → epair                         │
│  ExportedProgram → (node-level IR conversion) → .epair │
└─────────────────────────┬──────────────────────────────┘
                          │ .epair file
                          ▼
┌────────────────────────────────────────────────────────┐
│  Module 2 (C++): Graph construction (GeGraphBuilder)   │
│  Parse EP JSON, distinguish by the ascend_ir.source    │
│  field: converter nodes (directly build GE ops) and    │
│  fallback nodes                                        │
├────────────────────────────────────────────────────────┤
│  Module 3 (C++): Compile backend (Backend)             │
│  IUnitEmitter: compile along the way during emit,      │
│  write compilation artifacts into the output epair     │
├────────────────────────────────────────────────────────┤
│  Module 4 (C++): Heterogeneous codegen (HeteroCodegen) │
│  Drive Emitter to generate Runner source → runner.so   │
│  Final output: deployable epair (weights + NPU model + │
│  Runner .so)                                           │
└────────────────────────────────────────────────────────┘
```

| Module | Name | Language | Input | Output |
|------|------|------|------|------|
| 1 | EP → EPair serialization | Python | `ExportedProgram` | `.epair` file (IR + weights) |
| 2 | Graph partitioning (EpairPartitioner) | C++ | `.epair` | `PartitionedGraph` (NPU/CPU partitions, topologically ordered) |
| 3 | Compile backend (Backend) | C++ | `ExecutionUnit` (`NpuSubgraph` / `CpuSegment`) | Emitter implementation + compilation artifacts written into archive |
| 4 | Heterogeneous codegen (HeteroCodegen) | C++ | `PartitionedGraph` + `IUnitEmitter` instances | Runner `.so` + deployable epair |

---

## Inter-module Data Flow

```text
ExportedProgram
      │
      │ (1) torch._export.serde.serialize(ep)
      ▼
  EP JSON bytes
      │
      │ (2) Iterate nodes, call NodeIRConverter.convert_node()
      ▼
  EP JSON + ascend_ir extension fields
      │
      │ (3) PT2ArchiveWriter packs weights + manifest
      ▼
  .epair file (input epair)
      │
      │ (4) EpairPartitioner::partition()
      │     EpairArchiveReader reads model.json + weights
      ▼
  PartitionedGraph (list of ExecutionUnit, topologically ordered)
      │
      │ (5) HeteroCodegen::emit(graph, out_archive)
      │     ├─ CpuSegment  → ICpuUnitEmitter::emit()
      │     │    → Generate inlined at::_ops call code
      │     └─ NpuSubgraph → IDeviceUnitEmitter::emit()
      │          → Trigger GE compilation (GeGraphBuilder + aclgrphBuildModel or GeSession)
      │          → ge_aot: write .om into out_archive
      │          → Generate corresponding execution code (aot: load and execute from archive; jit: execute via Session)
      │     SyncEmitter: insert aclrtSynchronizeStream(g_stream) after each Device Emitter
      │
      │ (6) HeteroCodegen::compile_and_package(out_archive)
      │     System compiler compiles runner.cpp → runner.so, writes into out_archive
      ▼
  Output .epair (deployable: original weights + compiled NPU model + Runner .so)
      │
      │ (7) load_runner(epair_path)
      ▼
  RunnerFn → run(inputs) → outputs
```

---

## Epair Format Overview

epair reuses the **PT2 Archive format**. The file structure is identical to PT2, with two files added under the `extra/` directory (see [epair_format_design.md](epair_format_design.md) for complete field descriptions):

```text
model.epair (ZIP structure)
├── extra/                                 ← epair-specific, added
│   ├── epair_manifest.json                ← conversion metadata (version, backend, statistics)
│   └── ep_json_with_ascend_ir.json        ← EP JSON with ascend_ir fields
├── models/
│   └── model.json                         ← standard EP JSON
└── data/
    ├── weights/
    │   ├── weight_0, weight_1 ...
    │   └── model_weights_config.json
    └── constants/
        ├── tensor_0, tensor_1 ...
        └── model_constants_config.json
```

A deployable epair contains the following in addition to the input epair:

```text
model_compiled.epair (ZIP structure)
├── ... (inherits all contents of the input epair)
├── runner.so                              ← HeteroCodegen compilation artifact
└── npu/
    ├── subgraph_0.om                      ← NPU compilation artifact in ge_aot mode
    └── subgraph_1.om
```

### The Only Format Extension

Each node object in `models/model.json` may carry an additional optional field `ascend_ir`:

```json
{
  "target": "aten::mm.default",
  "inputs": [...],
  "outputs": [...],
  "metadata": {},
  "ascend_ir": {
    "source": "converter",
    "ops": [
      {
        "name": "MatMul",
        "type": "MatMul",
        "input": ["x", "weight"],
        "attr": {"transpose_x1": {"b": false}, "transpose_x2": {"b": false}},
        "inputDesc": [{"name": "x1", "dtype": "DT_FLOAT", "shape": {"dim": ["3", "3"]}, "layout": "ND"}],
        "outputDesc": [{"name": "y", "dtype": "DT_FLOAT", "shape": {"dim": ["3", "3"]}, "layout": "ND"}]
      }
    ],
    "mapping": {"inputs": {}, "outputs": {"mm": "MatMul:0"}}
  }
}
```

- The `source` field identifies the conversion origin: `"converter"` means converted by the Python converter, an empty string means a fallback node
- A node without an `ascend_ir` field or with an empty `source`: the C++ side treats it as a fallback node and `FallbackCodegen` generates a GE custom operator
- A node with `ascend_ir` and `source` equal to `"converter"`: the C++ side directly builds the GE operator node

`epair_manifest.json` contains fields such as `version`, `backend_name`, `total_nodes`, and `converted_nodes`.

---

## Key Design Principles

### Strict Error Reporting

For all external inputs (`.epair` file contents, EP JSON structure, field types), a strict parsing policy is adopted: an unrecognized field type, a missing required field, or a type mismatch always produces an Error-level `ValidationIssue` — never a silent skip and never a guess at intent.

### Component Size Principle

When a single component's implementation stays within 50 lines, prefer aggregation over splitting; refactor and split only after the implementation complexity increases.

### Deferred Graph Partitioning

Graph partitioning is done on the C++ side, and the Python side only performs IR annotation. This means:

- The `.epair` file can be reused across different hardware configurations (partitioning strategy can be adjusted on the C++ side)
- The Python-side implementation stays simple and does not need to be aware of graph connectivity or subgraph boundaries

### Codegen Instead of Interpreted Execution

The C++ side adopts a codegen model (aligned with AOTI):

- `HeteroCodegen` drives each `IUnitEmitter` to generate Runner C++ source
- The source is compiled into `runner.so`, and the deployment side loads and executes it via `load_runner`
- The graph structure is not interpreted at runtime, so the execution path is fixed and predictable

### Restrained Format Extension

Only minimal extensions are made to the PT2 format:

- No new top-level directories and no changes to how weights/constants are stored
- The only new file: `extra/epair_manifest.json`
- The only format change: the optional `ascend_ir` field on node JSON objects
- This ensures that `.epair` can be partially read by standard PT2 tools (just ignore the `ascend_ir` field)

### Replaceable Backend

`EmitterFactory` chooses the `IDeviceUnitEmitter` implementation (`ge_aot` or `ge_jit`) based on the `CompileOptions.backend` string; the current default is `ge_aot`. Integrating a new backend only requires implementing a new `IDeviceUnitEmitter` without affecting the upstream or downstream interfaces.

### Node-level IR, Not Subgraph-level

The IR conversion granularity is a single node, not a subgraph. This yields:

- A simpler conversion interface (`convert_node` only looks at a single node)
- Subgraph boundaries are determined by the C++ side based on the contiguity of the `ascend_ir` field
- The Python side does not need to understand the graph topology

---

## Dependency Reuse Principle

This project maximizes reuse of existing PyTorch and CANN capabilities instead of reinventing the wheel. Typical reuse points:

- **Serialization**: ZIP reading reuses `caffe2::serialize::PyTorchStreamReader`, JSON parsing reuses nlohmann/json
- **CPU operator invocation**: generates static dispatch `at::_ops::op_name::call(...)`, aligned with the AOTI `CppWrapperCpu` pattern
- **Device synchronization**: reuses the generic `aclrtSynchronizeStream` interface from the CANN runtime

---

## Phase Dependency Overview

This project divides dependency boundaries by phase, reflecting an architecture where the "front end is heavy and the back end is light":

| Phase | External Dependencies | Description |
|------|---------|------|
| IR conversion (Python side) | Python software stack such as PyTorch and torchair | Heaviest dependency, requires a full Python environment |
| Compilation (C++ side) | Graph compiler (GE), system C++ compiler | Requires a compilation toolchain, no Python needed |
| Execution (deployment side) | ACL runtime | Lightest dependency, only the runtime library is needed |

From IR conversion to execution, the dependency chain narrows at each phase. The final deployment artifact (`runner.so` + `.epair`) only needs the ACL runtime to run, with no Python environment or graph compiler required. See each module's documentation for the specific dependency constraints.

---

## User Interface Overview

This project exposes three entry points to users:

- **Python side**: call `export_to_epair(ep)` + `epair_save(contents, path)` to complete IR conversion and obtain an `.epair` file carrying Ascend IR annotations
- **C++ compile side**: call `EpairPartitioner::partition` to partition the graph, then use `HeteroCodegen` to complete codegen and GE compilation, obtaining a deployable `.epair`
- **C++ deployment side**: call `load_runner(epair_path)` to load the Runner and return a directly callable inference function
