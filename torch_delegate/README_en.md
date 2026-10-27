# TorchDelegate

English | [简体中文](README.md)

## Overview

**TorchDelegate** is a model deployment framework built on `torch.export`. Using a **delegate mechanism**, it partitions a PyTorch model and delegates the pieces to different execution backends: NPU subgraphs go to the accelerator backend for compilation and execution, CPU parts fall back to the libtorch runtime, and the result is a single deployment artifact (`.epair` format) that supports Python export and C++ loading/runtime execution.

---

## Key Features

- **Built on torch.export**: reuses the PyTorch 2.0+ export path to preserve FX graph integrity and operator semantics
- **Automatic operator delegation**: identifies subgraphs that can be delegated to the NPU and falls back to libtorch for the rest
- **Three-stage API**: `export_to_epair → epair_save → epair_load`, with a clear export/serialize/load path and support for eager fallback execution
- **C++ runtime**: standalone `libtorch_delegate_backend.so` that loads `.epair` and performs partitioning, compilation, and execution on the C++ side

---

## Architecture Overview

```text
ExportedProgram ──▶ [Python Converter] ──▶ .epair artifact ──▶ [C++ Runtime]
                                                        │
                                    NPU subgraph ──────▶ GE Engine
                                    CPU operators ─────▶ libtorch fallback
```

| Module | Language | Responsibility |
|------|------|------|
| **Converter** | Python | Parses `ExportedProgram`, converts NPU-compatible nodes to backend IR, and generates `.epair` |
| **Backend** | C++ | Loads `.epair`, parses EP JSON, builds the GE Graph, and loads weights |
| **Fallback Codegen** | C++ | Generates CPU fallback code for unsupported nodes |
| **Common** | C++ | Shared infrastructure (logging, archive reading, Base64, ABI compatibility) |

---

## Build and Installation

```bash
# One-shot build (third-party libraries → C++ → single whl)
./build.sh

# Build and run tests
./build_test.sh
```

The only release artifact is `output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl`, which contains both the Python API and the C++ artifacts (headers, shared libraries, CMake config package). It is independent of the Python version and installs on 3.10 through 3.13.

For detailed build instructions, see [docs/en/guide/build.md](docs/en/guide/build.md).

---

## Local Verification

The steps below take you from zero to exporting your first `.epair` model and loading it in C++ within 5 minutes.

### Install CANN

```bash
# Install the CANN Toolkit (see docs/en/guide/build.md for details)
# Default path for root
source /usr/local/Ascend/cann/set_env.sh
# Regular user
source ${HOME}/Ascend/cann/set_env.sh
```

### Build and Install

```bash
./build.sh
pip install output/package/torch_delegate-*-py3-none-linux_*.whl
```

C++ prefix and runtime library paths (the whl no longer ships `set_env.sh`):

```bash
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

### Run the Example

```bash
cd examples
# Create and activate the test virtual environment (installs the whl produced above)
bash prepare_env.sh
source ./venv_test/bin/activate

# Run the example
cd export_and_load_example
bash run.sh
```

`run.sh` automatically:
- checks the environment (CANN, TorchDelegate, Python, cmake)
- generates the example epair file
- builds the C++ loader
- runs the end-to-end verification

Example successful output:

```text
 [1/4] Checking environment
  [✓] ASCEND_HOME_PATH=/usr/local/Ascend/cann
  [✓] TORCH_DELEGATE_ROOT=/path/to/venv/lib/python3.12/site-packages/torch_delegate
  ...
  [4/4] Running example
========================================
 Verification passed
========================================
```

More examples are available in the [examples/](examples/README_en.md) directory.

## Documentation Index

| Document | Description |
|------|------|
| [docs/README_en.md](docs/README_en.md) | Documentation overview and reading path |
| [docs/en/guide/build.md](docs/en/guide/build.md) | Build, installation, and environment setup |
| [docs/en/guide/build_epair.md](docs/en/guide/build_epair.md) | epair packaging guide |
| [docs/en/guide/fallback_codegen_guide.md](docs/en/guide/fallback_codegen_guide.md) | Fallback Codegen user guide |
| [docs/en/api/python_api.md](docs/en/api/python_api.md) | Python API reference |
| [docs/en/api/epair_model_loader.md](docs/en/api/epair_model_loader.md) | C++ API reference |
| [docs/en/features/architecture.md](docs/en/features/architecture.md) | Overall architecture |
| [examples/README_en.md](examples/README_en.md) | Examples directory |

## Contributing

Before contributing, read the [Contributing Guide](../CONTRIBUTING_en.md).

## License

This project is licensed under the [CANN Open Software License Agreement Version 2.0](../LICENSE).
