# SymPack Optimization Verification

English | [简体中文](README.md)

Verifies how the torchair `optimize_sym_pack` pass eliminates redundant Pack operators. Two paths are compared: the torchair torch.compile path and the epair export + C++ load and inference path.

---

## Test Model

### PackReproduceModel

When the int parameters (m, n) do not equal any tensor dimension, `_optimize_sym_input` cannot match m/n to tensor.size(). The SymInts enter the GE graph as standalone placeholders and become CPU Data nodes. Pack takes Data as input, with `_inputs_all_sym=True`:

```text
Group A: view([m, n]) × 4 → 4 Pack[Data_m, Data_n] (3 redundant)
Group B: view([n, m]) × 2 → 2 Pack[Data_n, Data_m] (1 redundant)
```

The optimize_sym_pack pass merges the 6 redundant Packs into 2 unique Packs, eliminating all redundancy in the subgraph of each Data node.

## Directory Layout

```text
sym_pack_optimization_case/
├── torchair_pack_optimization.py   # Pack optimization verification on the torchair torch.compile path
├── export_pack_model.py            # torch.export + epair export + Pack analysis
├── main.cpp                        # C++ EpairModelLoader load + inference + accuracy comparison
├── CMakeLists.txt                  # C++ build configuration
├── run.sh                          # One-shot verification script
└── README.md                       # This document
```

## Requirements

1. **torch 2.9.0+cpu**
2. **torch_npu** + **torchair**
3. **torch-delegate** built
4. **CANN 9.2.0+**
5. **ASCEND_HOME_PATH** and **TORCH_DELEGATE_ROOT** set

## Quick Start

```bash
# 1. Load the CANN environment
source /path/to/cann/set_env.sh

# 2. Load the TorchDelegate environment (derived automatically after whl install, no set_env.sh needed)
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

# 3. Run the test
cd examples/sym_pack_optimization_case
bash run.sh
```

## Detailed Steps

### Pack optimization on the torchair torch.compile path

```bash
python3 torchair_pack_optimization.py
```

Uses `torch.compile(dynamic=True)` with the torchair backend to verify that the `optimize_sym_pack` pass eliminates redundant Packs. The pbtxt output is dumped to `graph_dump_output/`.

### torch.export + epair export

```bash
python3 export_pack_model.py
```

Generated files:
- `generated/model.epair` — Epair file
- `generated/model.pt2` — PT2 archive
- `generated/input_*.bin` / `generated/input_meta.txt` — Test input data
- `generated/ref_output.bin` — Reference output

### Build the C++ example

```bash
mkdir -p build && cd build
cmake .. \
    -DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT} \
    -DCANN_ROOT=${ASCEND_HOME_PATH} \
    -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cmake --build . -j$(nproc)
```

### C++ inference + accuracy comparison

```bash
TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${LD_LIBRARY_PATH:-}"

# Enable GE Dump to export the compiled graph for the step 5 check
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH="${PWD}/ge_graph_dump"

./build/sym_pack_optimization_case generated/model.epair
```

### Check that no Pack operator exists in the GE Dump graph

```bash
grep -rl 'type: "Pack"' ge_graph_dump/ --include='ge_proto*.txt' | wc -l
# Expected: 0
```

## Technical Notes

### How SymPack optimization works

In the original GE graph, every `view([m, n])` produces a `Pack([Data_m, Data_n])`, and repeated views with the same arguments produce duplicate Packs. `optimize_sym_pack` groups by input semantics and deduplicates:

```text
Before optimization: the GE Graph has 3 Data([m, n, x]) + 6 Pack
After optimization:  the GE Graph has 3 Data([x, merged_A([m,n]), merged_B([n,m])]) + 0 Pack
```

Inside EpairModelLoader, InputBindingMap adapts the inputs automatically: `user_inputs[x, m, n] → ge_inputs[x, merged_A([m,n]), merged_B([n,m])]`.
