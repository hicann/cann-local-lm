# ExecuTorch-NPU

English | [简体中文](README.md)

ExecuTorch-NPU is the Ascend AOT Backend for ExecuTorch. During the export phase, it converts a PyTorch
`ExportedProgram` into a GE graph and generates the OM and external weights through CANN; during the runtime phase,
the ExecuTorch C++ Runtime loads the PTE/PTD and executes them on Ascend NPUs via AscendCL.

## Workflow

```text
ExecuTorch-NPU source code
    │  Build
    ▼
executorch_npu wheel + C++ Backend SDK
    │  Python export
    ▼
model.pte + weights.ptd
    │  C++ application loading
    ▼
ExecuTorch Runtime → AscendBackend → AscendCL → Ascend NPU
```

Build, export, and runtime are three independent phases that can reside on different machines and in different Python environments.

Model export must use the official PyTorch `+cpu` wheel. ExecuTorch-NPU does not install Torch automatically;
see [Supported Versions](docs/supported_versions_en.md) for the specific combinations.

The export and compilation phases do not require an NPU device to be present: the target chip is passed only via `ACL_SOC_NAME`,
and only the runtime phase requires a matching NPU, driver, and AscendCL runtime. This mode is called Lite export;
see [Model Export](docs/export_en.md) for the mechanism and constraints.

## Choose Documentation by Task

| Goal | Document |
|---|---|
| Complete end-to-end validation for the first time | [Quickstart](docs/quickstart_en.md) |
| Review build dependencies and generate the wheel | [Build from Source](docs/build_en.md) |
| Review export dependencies and generate PTE/PTD | [Model Export](docs/export_en.md) |
| Review runtime dependencies and load PTE/PTD in C++ | [C++ Runtime](docs/runtime_en.md) |
| Troubleshoot build, export, or runtime errors | [Troubleshooting](docs/troubleshooting_en.md) |
| Integrate the headers, static libraries, and CMake targets inside the wheel | [C++ Backend SDK](docs/cpp_backend_sdk_en.md) |

See the [Documentation Index](docs/README_en.md) for the full documentation catalog.

## Artifact Boundaries

| Artifact | Produced in | Purpose |
|---|---|---|
| `executorch_npu-*.whl` | Build from source | Python AOT Backend and C++ Backend SDK |
| `model.pte` | Model export | ExecuTorch Program, Delegate metadata, and OM |
| `ascend.weights.package.ge_aot.ptd` | Model export | External weights named data |
| User Runner/application | C++ application build | Loads PTE/PTD and invokes the ExecuTorch Runtime |

The wheel does not include the CANN Toolkit, drivers, firmware, NNRT, TorchAir, torch-delegate, model files,
or a generic Runner executable.

## Core Capabilities

- Uses ExecuTorch delegation, PTE, PTD, Program, and Method.
- OM and weights are deterministically associated through named-data keys, SHA-256, and logical lengths.
- Supports multi-Method weight reuse and explicit state contracts.
- Export and compilation are fully decoupled from NPU devices; artifacts can be delivered across machines to the runtime environment.
- The C++ Backend integrates with the ExecuTorch Runtime through static registration; users do not call Backend-internal interfaces directly.

## License

CANN Open Software License Agreement Version 2.0
