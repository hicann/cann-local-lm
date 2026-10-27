# Example Obtain a Graph from EPAIR and run it with GeSession

English | [简体中文](README.md)

This example demonstrates the minimal C++ integration path for the TorchDelegate installation package: load an `.epair`, extract a `ge::Graph` through `td::EpairModelLoader`, then execute the model directly with `ge::GeSession` and `gert::Tensor`.

## Directory Structure

```text
export_and_load_example/
├── CMakeLists.txt         # C++ build configuration
├── main.cpp               # Minimal C++ example
├── export_ep_and_epair.py # Export pt2 and epair
├── export_ep_and_epair_lite.py # [Experimental] export pt2 and epair without a CANN runtime
├── load_epair_run_eager.py      # Load and run the epair directly on the Python side
├── compare_single_op_vs_delegate.py  # Single-op vs Delegate accuracy comparison
├── run.sh                         # One-shot validation script
└── README.md                      # This document
```

## Usage

### Prerequisites

1. Install CANN and source its environment
2. Build and install the TorchDelegate whl (run `./build.sh` in the repository root, then `pip install output/package/torch_delegate-*.whl`, or simply `bash examples/prepare_env.sh`)

### One-shot run

```bash
# 1. Set up the CANN environment (replace with your actual installation path)
source /usr/local/Ascend/cann/set_env.sh

# 2. Set up the TorchDelegate environment (derived automatically after whl installation; no set_env.sh needed)
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# 3. Run everything
cd examples/export_and_load_example
bash run.sh
```

`run.sh` performs the following automatically:
1. Environment checks (CANN, TorchDelegate, Python, cmake)
2. Generate the sample epair (Python)
3. Build the C++ example (cmake)
4. Run Delegate whole-graph inference
5. Single-op vs Delegate accuracy comparison (Python)

### Step-by-step

#### Generate the sample epair

```bash
python export_ep_and_epair.py
```

By default this generates `epair_model.epair` in the current directory.

#### Export without a CANN runtime

If the export environment has no CANN, `torch_npu`, or NPU device, use the Lite-mode script:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python export_ep_and_epair_lite.py
```

The script uses a CPU `ExportedProgram`, builds the GE IR inside `lite_export_context`,
and then saves the following outside the context:

- `ep_model_lite.pt2`
- `epair_model_lite.epair`
- `input_lite_0.bin`

Lite mode only generates the EPAIR; it does not load, compile, or execute the GE graph locally. The generated
`epair_model_lite.epair` can be copied to a deployment host with a matching CANN environment and loaded and run
with the C++ steps below.

#### Build the C++ example

```bash
cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT} \
  -DCANN_ROOT=${ASCEND_HOME_PATH} \
  -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cmake --build build -j$(nproc)
```

#### Run the example

```bash
TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${LD_LIBRARY_PATH:-}"
./build/export_and_load_example epair_model.epair
```

On success, the program performs:

- `EpairModelLoader::EpairModelLoader(archive_path)` — load the epair
- `EpairModelLoader::GetGEGraph(&graph)` — extract the GE graph
- `GeSession::AddGraph(graph_id, graph)` — register the graph
- `GeSession::RunGraph(graph_id, inputs, outputs)` — run the graph

#### Single-op vs Delegate accuracy comparison

```bash
python compare_single_op_vs_delegate.py
```

The script performs the following automatically:
1. **Single-op execution**: runs the model op by op in PyTorch eager mode (mm → add → transpose), printing intermediate results at each step
2. **Delegate execution**: invokes the C++ `export_and_load_example` binary to run the whole graph through GE
3. **Accuracy comparison**: compares the outputs element by element and reports the maximum absolute and relative errors

To verify running the epair directly on the Python side, you can additionally run:

```bash
python load_epair_run_eager.py
```

## Key APIs

| API | Header | Description |
|-----|--------|------|
| `td::EpairModelLoader` | `torch_delegate/epair_model_loader.h` | Loads the `.epair` and builds the GE Graph |
| `td::EpairModelLoader::GetGEGraph` | `torch_delegate/epair_model_loader.h` | Extracts the `ge::Graph` |
| `ge::GEInitializeV2` | `ge/ge_api_v2.h` | GE global initialization |
| `ge::GeSession` | `ge/ge_api_v2.h` | Online compilation session |
| `GeSession::AddGraph` | `ge/ge_api_v2.h` | Registers a graph |
| `GeSession::RunGraph` | `ge/ge_api_v2.h` | Runs synchronously with `gert::Tensor` |
| `gert::Tensor` | `ge/ge_api_v2.h` | GeSession v2 input/output tensor |

## Flow

1. **Initialize GE**: `ge::GEInitializeV2(options)`
2. **Load the epair**: `td::EpairModelLoader loader(epair_path)` — the constructor parses the archive and builds the GE Graph
3. **Extract the GE graph**: `loader.GetGEGraph(&graph)` — obtains a copy of the internal GE Graph
4. **Create a session**: `ge::GeSession ge_session(options)`
5. **Add the graph and run**: `ge_session.AddGraph(graph_id, graph)` → `ge_session.RunGraph(graph_id, inputs, outputs)`
6. **Clean up resources**: `ge_session.RemoveGraph(graph_id)` → `ge::GEFinalizeV2()`
