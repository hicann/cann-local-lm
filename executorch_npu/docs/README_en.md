# ExecuTorch-NPU Documentation Index

English | [简体中文](README.md)

## The Six Questions Answered First

| Question | Single primary document |
|---|---|
| What are the build dependencies? | [Build from Source: Build Dependencies](build_en.md#build-dependencies) |
| How do I build the wheel and Backend libraries? | [Build from Source: Run the Build](build_en.md#running-the-build) |
| What are the export dependencies? | [Model Export: Dependency Boundaries](export_en.md#dependency-boundaries) |
| How do I export PTE/PTD? | [Model Export: Phase B Generates PTE/PTD](export_en.md#phase-b-generate-pteptd-from-the-pt2) |
| What are the C++ compilation and on-device runtime dependencies? | [C++ Runtime: Dependency Boundaries](runtime_en.md#dependency-boundaries) |
| How do I load PTE/PTD? | [C++ Runtime: Load the Model](runtime_en.md#loading-pteptd) |

A given command or dependency is fully described only in its primary document; other documents reference it via links to avoid content drift across multiple places.

## Getting Started Tutorial

| Document | Contents |
|---|---|
| [Quickstart](quickstart_en.md) | MobileNetV2 build, export, Runner compilation, and NPU validation |

## How-To Guides

| Document | Contents |
|---|---|
| [Build from Source](build_en.md) | Build dependencies, `build.sh`, wheel artifacts, and installation verification |
| [Model Export](export_en.md) | CPU Torch, TorchAir, torch-delegate, PTE/PTD generation |
| [C++ Runtime](runtime_en.md) | C++ compilation dependencies, on-device runtime dependencies, Module loading and execution |
| [Troubleshooting](troubleshooting_en.md) | Issues categorized by build, export, C++ compilation, and NPU runtime |

## Reference Documentation

| Document | Contents |
|---|---|
| [Supported Versions](supported_versions_en.md) | ExecuTorch, CPU Torch, Torchvision, CANN, and device matrix |
| [C++ Backend SDK](cpp_backend_sdk_en.md) | SDK layout inside the wheel, CMake targets, state APIs, and linking boundaries |
| [Artifact Protocol](protocol_en.md) | Protocol rules for PTE/PTD, Manifest, named data, and state |

The field-level Manifest reference schema is located at
[`examples/manifest_schema/manifest-v1.schema.json`](../examples/manifest_schema/manifest-v1.schema.json).

## Design Notes

| Document | Contents |
|---|---|
| [Overall Design](design/executorch_npu_design_en.md) | Compilation, artifacts, runtime, state, and compatibility design |
| [Architecture Stack Diagram](design/executorch_npu_stack_en.md) | End-to-end component layering and code mapping |
| [Qwen3.5 static-state](design/qwen3_5_export_design_en.md) | Qwen Prefill/Decode state contract |
| [Gemma4 static-state](design/gemma4_e2b_it_export_design_en.md) | Gemma4 state adapter design |

`docs/passes/` holds development and debugging materials and is not part of the main user path for installation, export, or runtime.
