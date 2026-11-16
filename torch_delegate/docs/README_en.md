# TorchDelegate Documentation

English | [简体中文](README.md)

This directory contains user guides, API references, design documents, and constraints for TorchDelegate.

## Documentation Layers

| Layer | Directory | Content |
|-------|-----------|---------|
| User guides | `en/guide/` | Build and installation, epair packaging, Fallback Codegen usage, environment variables |
| API reference | `en/api/` | Python API and C++ API |
| Design documents | `en/features/` | Architecture, epair format, Fallback Codegen, Lite Context, and other design materials |
| Constraints | `en/constraints/` | CANN Stub, CXX11 ABI compatibility, and other development constraints |

Chinese documents are located in the `zh/` directory. The directory structure and file names are identical to those in `en/`.

## Reading Paths

- First-time users: read [Build and Installation](en/guide/build.md), then follow [epair Packaging Guide](en/guide/build_epair.md) to export a model.
- Python integration: read [Python API Reference](en/api/python_api.md).
- C++ deployment: read [EpairModelLoader API](en/api/epair_model_loader.md).
- Custom operators: read [Fallback Codegen User Guide](en/guide/fallback_codegen_guide.md).
- Implementation details: read [Architecture](en/features/architecture.md) and [Epair Format Design](en/features/epair_format_design.md).

## Maintenance Rules

- Place new user guides in `zh/guide/`, new APIs in `zh/api/`, new design documents in `zh/features/`, and new constraints in `zh/constraints/`.
- When a Chinese document is added or modified, add or modify the file with the same name under `en/`.
- Use lowercase English file names with underscores between words.
- Mark the design date and scope at the beginning of historical design documents.
- Common Python API functions and data classes are maintained in `python_api.md`; new standalone APIs should be split into separate files (for example, `model_loader.md`).
