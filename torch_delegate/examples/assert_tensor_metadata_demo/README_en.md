# assert_tensor_metadata_demo

English | [简体中文](README.md)

Demonstrates the `aten._assert_tensor_metadata` nodes that `torch.export` inserts automatically, and how to remove them after export before exporting an epair and loading and running it on the C++ side.

## Background

When exporting, `torch.export` automatically inserts `aten._assert_tensor_metadata` runtime check nodes to verify that a tensor's dtype / device / layout match those at export time.

The schema of these nodes is `-> ()`. They have **no compute semantics**, but they are rejected by downstream GE / torchair converters, causing the epair or `torch.compile` path to fail.

### Trigger conditions

`_assert_tensor_metadata` is inserted during export by `_insert_aten_to_metadata_assert_pass`, immediately before each `aten.to.*` node. Therefore, any model that uses type conversions such as `tensor.to()` / `tensor.float()` / `tensor.bfloat16()` will produce these nodes in its exported result.

> Reference: `torch/_export/utils.py::_insert_aten_to_metadata_assert_pass`

This example uses **no custom ops** (no `torch.library` / `xllm_placeholder`) and does **not rely on fallback codegen** (no torchair converter). Standard `aten.to.dtype` alone is enough to trigger it.

## Model

```python
class SimpleModel(nn.Module):
    def forward(self, x):
        y = torch.mm(x, self.weight)
        y = y.to(torch.float32)       # aten.to.dtype  -> _assert_tensor_metadata
        y = torch.relu(y)
        return y.to(torch.bfloat16)   # aten.to.dtype  -> _assert_tensor_metadata
```

Exported graph (key parts):

```text
%mm = aten.mm.default(%x, %p_weight)
%_assert_tensor_metadata_default = aten._assert_tensor_metadata.default(%mm, dtype=float32, ...)
%to = aten.to.dtype(%mm, torch.float32)
%relu = aten.relu.default(%to)
%_assert_tensor_metadata_default_1 = aten._assert_tensor_metadata.default(%relu, dtype=bfloat16, ...)
%to_1 = aten.to.dtype(%relu, torch.bfloat16)
return (to_1,)
```

## Removal method

```python
target = torch.ops.aten._assert_tensor_metadata.default
for node in list(ep.graph.nodes):
    if node.op == "call_function" and node.target is target:
        ep.graph.erase_node(node)
ep.graph.lint()
ep.graph_module.recompile()
```

This is identical to `accuracy_compare/export_program.py::_remove_assert_tensor_metadata`.

## Directory Structure

```text
assert_tensor_metadata_demo/
├── README.md              # This document
├── export_demo.py         # Python side: export epair + save inputs/reference outputs
├── main.cpp               # C++ side: load and run with EpairModelLoader
├── CMakeLists.txt         # C++ build configuration
└── run.sh                 # One-shot validation script
```

## Requirements

1. **torch 2.9.0+**
2. **torch_npu** + **torchair**
3. **torch-delegate** built
4. **CANN 9.2.0+**

## Quick Start

```bash
# 1. Load the CANN environment
source /path/to/Ascend/cann/set_env.sh

# 2. Activate the Python environment with torch 2.9.0
source /path/to/venv/bin/activate

# 3. Set up the TorchDelegate environment (derived automatically after whl installation; no set_env.sh needed)
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

# 4. Run everything
cd examples/assert_tensor_metadata_demo
bash run.sh
```

`run.sh` performs the following automatically:
1. Exports the epair on the Python side and saves the input binaries and reference outputs
2. Builds epair_loader on the C++ side
3. Loads the epair on the C++ side, runs inference, and compares accuracy

### Step-by-step

#### Export the epair

```bash
python3 export_demo.py
```

Generated files:
- `generate/simple_model.pt2` - ExportedProgram
- `generate/simple_model.epair` - epair file
- `generate/input_0.bin` - Sample input
- `generate/ref_output.bin` - Reference output
- `generate/graph_before.py` - Graph text before removal
- `generate/graph_after.py` - Graph text after removal

#### Build the C++ example

```bash
cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT} \
  -DCANN_ROOT=${ASCEND_HOME_PATH} \
  -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cmake --build build -j$(nproc)
```

#### Run the C++ loader

```bash
TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${LD_LIBRARY_PATH:-}"
./build/epair_loader generate/simple_model.epair
```

## Flow

```python
torch.export.export(strict=False)
    │
    ▼
ExportedProgram  (with _assert_tensor_metadata)
    │
    ├── print graph  (BEFORE)
    │
    ▼
remove_assert_tensor_metadata()
    │
    ├── print graph  (AFTER)
    │
    ▼
torch.export.save()  ──►  .pt2
    │
    ▼
torch_delegate.export_to_epair()  ──►  .epair
    │
    ▼
C++ EpairModelLoader
    │
    ├── CompileAndLoad(options, stream)
    ├── RunModelWithStreamAsync(stream, inputs, outputs)
    ├── aclrtSynchronizeStream(stream)
    └── Accuracy comparison (C++ output vs Python reference output)
```

## References

- `accuracy_compare/export_program.py` — Original `_remove_assert_tensor_metadata` implementation
- `accuracy_compare/ge_converters.py` — Fallback converter (no-op)
- `torch/_export/utils.py::_insert_aten_to_metadata_assert_pass` — Upstream insertion logic
- `export_and_load_example/` — Basic epair loading example
