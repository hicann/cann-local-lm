# Inplace + Return Test Case

English | [简体中文](README.md)

This example tests the **inplace input + Tensor return value** combination in the fallback path.

---

## Test Goal

### Core scenario an inplace operator that also returns a Tensor

```python
# inplace_scale_out: a *= scale (modifies in place), returns a (the modified tensor)
my_ops.define("inplace_scale_out(Tensor(a!) a, float scale) -> Tensor")
```

This combines `dual_inplace_` (pure inplace, returns `()`) and `scale_add` (pure return, no inplace):
- **Inplace input**: `a` is marked `(a!)` and is modified in place
- **Tensor return value**: the operator returns a Tensor rather than `()`

What is verified:
- `auto_functionalized_v2` correctly unpacks the combination of inplace input and return value
- REG_OP declares the complete schema
- The model uses both the returned value and the in-place modified input
- The GE graph correctly handles the mutation output and the return-value output

---

## Directory Layout

```text
inplace_return_case/
├── custom_ops.py           # Operator definitions + CPU/NPU/Meta implementations
├── custom_ops_reg.cpp      # C++ operator registration (CPU + PrivateUse1)
├── export_model.py         # Model export + test data generation
├── fallback_runner.cpp     # C++ runner (load epair + execute)
├── compare_accuracy.py     # Accuracy comparison script
├── CMakeLists.txt          # Build configuration
├── run.sh                  # One-shot run script
├── README.md               # This document
└── output/                 # Generated at runtime
    ├── model_inplace_return.epair
    └── input_*.bin / input_*.txt
```

---

## Requirements

1. **torch 2.9.0+cpu**
2. **torch_npu** + **torchair**
3. **torch-delegate** built
4. **CANN 9.2.0+**
5. **libtorch_npu** sources available

---

## Quick Start

```bash
# 1. Load the CANN environment
source /path/to/cann/set_env.sh

# 2. Activate the Python environment
source /path/to/venv_test/bin/activate

# 3. Run the test
cd examples/inplace_return_case
bash run.sh
```

---

## Detailed Steps

### Export the model

```bash
python3 export_model.py
```

Generated files:
- `output/model_inplace_return.epair` - Epair file
- `output/input_*.bin` / `output/input_*.txt` - Test input data

### Build the C++ runner

```bash
mkdir -p build && cd build
cmake .. -DCANN_ROOT=${ASCEND_HOME_PATH} -DTORCH_NPU_DIR=${TORCH_NPU_DIR}
cmake --build . -j$(nproc)
```

### Run the runner

```bash
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR=./output_dir
./build/fallback_runner --custom-ops build/libcustom_ops.so output/model_inplace_return.epair
```

### Compare accuracy

```bash
python3 compare_accuracy.py
```

---

## Technical Notes

### auto_functionalized_v2 with inplace inputs and return values

PyTorch wraps inplace operators in `auto_functionalized_v2`. When an operator has both an inplace input and a Tensor return value:

```json
{
    "target": "torch.ops.higher_order.auto_functionalized_v2",
    "inputs": [
        {"name": "", "arg": {"as_operator": "torch.ops.my_ops.inplace_scale_out.default"}},
        {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "scale", "arg": {"as_float": 2.0}}
    ]
}
```

The outputs of `auto_functionalized_v2` include:
1. **Mutation output**: the tensor modified in place (the parameter marked `a!`)
2. **Return-value output**: the Tensor returned by the operator

The fallback codegen must distinguish these two output kinds and generate the corresponding REG_OP.
