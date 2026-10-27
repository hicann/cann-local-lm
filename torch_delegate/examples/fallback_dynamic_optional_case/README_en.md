# Fallback Dynamic/Optional Tests

English | [简体中文](README.md)

Verifies that fallback codegen correctly supports **dynamic inputs (Tensor[])** and **optional inputs (Tensor?)**.

## Test Scenarios

| Scenario | Op | schema | Description |
|------|------|--------|------|
| Dynamic input | `my_addn` | `(Tensor[] tensors) -> Tensor` | Tensor[] maps to GE DYNAMIC_INPUT; the number of runtime instances is variable |
| Optional input absent | `my_addmm` | `(Tensor, Tensor, Tensor, Tensor?, *, Scalar beta=1) -> Tensor` | bias=None, the OPTIONAL_INPUT is not connected |
| Optional input present | `my_addmm` | Same as above | bias has a valid value, the OPTIONAL_INPUT is connected to a Const |

### Key codegen checkpoints

- **DYNAMIC_INPUT**: `GetDynamicInputTensor(ir_idx, i)` + `GetIrInputInstanceInfo` to get the instance count
- **OPTIONAL_INPUT (Tensor?)**: does not crash when `GetOptionalInputTensor(ir_idx)` returns nullptr
- **OPTIONAL_INPUT (Scalar with default)**: Scalar parameters with defaults are declared as OPTIONAL_INPUT, and the default value is used when they are missing from the EP JSON
- **IR index alignment**: `GetRequiredInputTensor`/`GetOptionalInputTensor` use the index declared in the IR and are unaffected by DYNAMIC_INPUT expansion or unconnected OPTIONAL inputs

## File Structure

```text
fallback_dynamic_optional_case/
├── custom_ops.py              # Python custom op definitions (my_addn, my_addmm)
├── custom_ops_reg.cpp         # C++ custom op registration (CPU + PrivateUse1)
├── export_model.py            # Model export + test data generation
├── fallback_runner.cpp        # C++ Runner: load epair and run inference
├── compare_accuracy.py        # Accuracy comparison
├── CMakeLists.txt             # Build configuration
├── run.sh                     # One-shot validation script
└── README.md                  # This file
```

## Run

```bash
# Prerequisite: source the CANN environment
source <cann_path>/cann/set_env.sh

# Run everything
bash run.sh
```

### Output Files

| File | Description |
|------|------|
| `output/model_addn.epair` | Dynamic-input model |
| `output/model_addmm_no_bias.epair` | Model with the optional input absent |
| `output/model_addmm_with_bias.epair` | Model with the optional input present |
| `output/input_*.bin` | Test input data |
| `output/delegate_output_*.bin` | Delegate inference outputs |
