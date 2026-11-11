# epair_runner — C++ epair loading and accuracy validation

English | [简体中文](README.md)

This example demonstrates how to load a `.epair` file with the C++ `td::EpairModelLoader` and run GE graph inference. It provides the Delegate-path accuracy validation used in the `accuracy_compare` flow.

## Directory Structure

```text
epair_runner/
├── CMakeLists.txt       # Build configuration
├── epair_runner.cpp     # Main program
├── tensor_io.h          # Tensor binary I/O
├── tensor_io.cpp
└── README.md            # This document
```

## Prerequisites

| Dependency | Description |
|------|------|
| TorchDelegate | Installed torch_delegate whl (or the install directory from a source build) |
| CANN | Installed, with `ASCEND_HOME_PATH` set |
| libtorch | Standalone libtorch distribution or the Python torch wheel |
| libtorch_npu | (Optional) used for custom op registration |

## Environment Variables

| Variable | Required | Description |
|------|------|------|
| `ASCEND_HOME_PATH` | Yes | CANN installation path |
| `TORCH_DELEGATE_ROOT` | Yes | TorchDelegate install directory |
| `TORCH_PKG_DIR` | Yes | Python torch package directory |
| `TORCH_NPU_DIR` | No | libtorch_npu source path |

## Build and Run

### Set up the environment

```bash
source /usr/local/Ascend/cann/set_env.sh
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
```

### Build

```bash
cmake -S . -B build \
  -DTorchDelegate_ROOT=${TORCH_DELEGATE_ROOT} \
  -DCANN_ROOT=${ASCEND_HOME_PATH} \
  -DTORCH_PKG_DIR=${TORCH_PKG_DIR}
cmake --build build -j$(nproc)
```

### Run

```bash
./build/epair_runner <epair_file> <output_dir> <device_id> <input0.bin> [input1.bin ...]
```

Arguments:
- `epair_file` — Path to the `.epair` file
- `output_dir` — Output directory (stores `output_*.bin`)
- `device_id` — NPU device ID (e.g. `"0"`)
- `input*.bin` — Input binary files (multiple inputs supported)

### Using with accuracy_compare

`accuracy_compare/run.sh compare` calls this binary automatically for Delegate-path inference.

## Key APIs

| API | Description |
|-----|------|
| `td::EpairModelLoader(archive_path)` | Loads the epair and builds the GE Graph |
| `loader.GetGEGraph(&graph)` | Extracts the GE Graph |
| `ge::GeSession(options)` | Creates a GE session |
| `session.AddGraph(id, graph)` | Registers the graph |
| `session.RunGraph(id, inputs, outputs)` | Runs synchronously |

## Custom Ops

You can specify the custom op library path through the `CUSTOM_OP_LIBRARY_PATH` environment variable:

```bash
export CUSTOM_OP_LIBRARY_PATH=/path/to/custom_op/lib
```

Alternatively, specify `libcust_opapi.so` in cmake via `-DCUST_OPAPI_DIR=/path/to/custom_op/build`.
