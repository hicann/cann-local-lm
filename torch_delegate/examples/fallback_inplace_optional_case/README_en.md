# Fallback Inplace/Optional Test Cases

English | [简体中文](README.md)

This example tests two key features in the fallback scenario:

1. **Multiple inplace parameters**: a single op definition with multiple inplace parameters
2. **Multiple optional parameters**: an op definition with multiple optional parameters, where only some are passed at call time

---

## Test Goals

### multiple inplace parameters

```python
# dual_inplace_: a *= scale_a; b *= scale_b
my_ops.define("dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()")
```

Checkpoints:
- Both x and y are modified in place
- REG_OP declares the full schema (a, b, scale_a, scale_b)
- Unpacked correctly by auto_functionalized_v2

### multiple optional parameters

```python
# multi_optional_op: result = x * alpha + beta + gamma + delta
my_ops.define(
    "multi_optional_op(Tensor x, float alpha=1.0, float beta=0.0, float gamma=0.0, float delta=0.0) -> Tensor"
)
```

Checkpoints:
- When calling `multi_optional_op(x, gamma=3.0)`, the EP JSON contains only x and gamma
- REG_OP declares all 5 parameters (x, alpha, beta, gamma, delta)
- Execute reads all parameters from GE inputs; GE supplies default values for the missing alpha/beta/delta

- When calling `multi_optional_op(x, alpha=2.0, gamma=3.0)`, the EP JSON contains x, alpha, and gamma
- REG_OP declares all 5 parameters
- Execute reads all parameters from GE inputs

---

## Directory Structure

```text
fallback_inplace_optional_case/
├── custom_ops.py           # Op definitions + CPU/NPU/Meta implementations
├── custom_ops_reg.cpp      # C++ op registration (CPU + PrivateUse1)
├── export_model.py         # Model export + test data generation
├── fallback_runner.cpp     # C++ runner (load epair + execute)
├── compare_accuracy.py     # Accuracy comparison script
├── CMakeLists.txt          # Build configuration
├── run.sh                  # One-shot run script
├── README.md               # This document
└── output/                 # Generated at runtime
    ├── model_inplace_optional.epair
    └── input_*.bin / input_*.txt
```

---

## Requirements

1. **torch 2.9.0+cpu** (the py311gcc13 venv is recommended)
2. **torch_npu** + **torchair**
3. **torch-delegate** built
4. **CANN 9.2.0+**
5. **libtorch_npu** source available

---

## Quick Start

```bash
# 1. Load the CANN environment
source ~/Ascend/cann/set_env.sh

# 2. Activate the Python environment
conda activate py311gcc13

# 3. Set TORCH_NPU_DIR
export TORCH_NPU_DIR=/path/to/libtorch_npu

# 4. Run the tests
cd /path/to/torch-delegate/examples/fallback_inplace_optional_case
bash run.sh
```

---

## Detailed Steps

### Export the model

```bash
python3 export_model.py
```

Generated files:
- `output/model_inplace_optional.epair` - Epair file
- `output/input_*.bin` / `output/input_*.txt` - Test input data

### Build the C++ Runner

```bash
mkdir -p build && cd build
cmake .. -DCANN_ROOT=${ASCEND_HOME_PATH} -DTORCH_NPU_DIR=${TORCH_NPU_DIR}
cmake --build . -j$(nproc)
```

Generated files:
- `build/fallback_runner` - C++ executor
- `build/libcustom_ops.so` - Custom op library

### Run the Runner

```bash
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR=./output_dir
./build/fallback_runner --custom-ops build/libcustom_ops.so output/model_inplace_optional.epair
```

### Accuracy comparison

```bash
python3 compare_accuracy.py
```

---

## Expected Output

```text
========================================
[1/6] Check environment
========================================
  [✓] ASCEND_HOME_PATH=/path/to/cann
  [✓] python3 3.11.0

========================================
[2/6] Prepare libtorch_npu
========================================
  [✓] TORCH_NPU_DIR=/path/to/libtorch_npu

========================================
[3/6] Export model
========================================
  Export model_inplace_optional
  Fallback ops (3):
    - torch.ops.my_ops.dual_inplace_.default
    - torch.ops.my_ops.multi_optional_op.default (x2)

========================================
[4/6] Build C++ Runner + libcustom_ops
========================================
  [✓] fallback_runner compiled
  [✓] libcustom_ops.so compiled

========================================
[5/6] Run Runner
========================================
  [✓] custom_op.cpp (15000 bytes)
  [✓] libcust_opapi.so (80000 bytes)

========================================
[6/6] Accuracy comparison
========================================
Input group 0 (batch_size=4, alpha=2.0, gamma=3.0):
    PASS (max_abs=1.23e-06, max_rel=2.34e-06)
Input group 1 (batch_size=8, alpha=1.5, gamma=2.5):
    PASS (max_abs=2.34e-06, max_rel=3.45e-06)
Input group 2 (batch_size=2, alpha=3.0, gamma=1.0):
    PASS (max_abs=1.11e-06, max_rel=2.22e-06)

========================================
Test summary
========================================
[PASS] All tests passed!
```

---

## Technical Notes

### Inplace markers

You must use the `(a!)` and `(b!)` markers; otherwise PyTorch will not treat the op as inplace:

```python
# Correct
my_ops.define("dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()")

# Incorrect (will not be recognized as inplace)
my_ops.define("dual_inplace_(Tensor a, Tensor b, float scale_a, float scale_b) -> ()")
```

### auto_functionalized_v2

PyTorch wraps inplace ops in `auto_functionalized_v2`:

```json
{
    "target": "torch.ops.higher_order.auto_functionalized_v2",
    "inputs": [
        {"name": "", "arg": {"as_operator": "torch.ops.my_ops.dual_inplace_.default"}},
        {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
        {"name": "b", "arg": {"as_tensor": {"name": "y"}}},
        {"name": "scale_a", "arg": {"as_float": 0.5}},
        {"name": "scale_b", "arg": {"as_float": 2.0}}
    ]
}
```

Fallback codegen must unpack this correctly and generate the REG_OP.

### Default values for optional parameters

When a parameter is missing from the EP JSON, fallback codegen:
1. Still declares the parameter in REG_OP (full schema)
2. Reads the parameter from GE inputs in Execute
3. Has the GE Graph Builder create a Const node for the parameter (using the default value)

---

## Troubleshooting

### Error "dual_inplace_ is not recognized as inplace"

**Cause**: The op definition is missing the `(a!)` or `(b!)` marker.

**Solution**: Check the op definitions in `custom_ops.py` and make sure the correct alias markers are used.

### Error "multi_optional_op parameter count mismatch"

**Cause**: The number of parameters declared in REG_OP does not match the number of inputs connected in the GE graph.

**Solution**: Check the REG_OP definition in `output_dir/*/custom_op.cpp` and make sure all parameters are declared.

### Error "accuracy comparison failed"

**Cause**: The inplace operation was not executed correctly, or the default values of optional parameters were not applied correctly.

**Solution**: Check the Execute method in `output_dir/*/custom_op.cpp` and make sure all parameters are read correctly.

---

## References

- sym_int_case: `../sym_int_case/`
- accuracy_compare: `../accuracy_compare/`
- TorchDelegate documentation: `../../docs/en/guide/build.md`
