# E2E Test

English | [简体中文](README.md)

A simplified end-to-end test that verifies fallback support for non-tensor inputs.

## Test Models

### SimpleMixedModel (dynamic shape scenario)

`SimpleMixedModel` has the following characteristics:
- **Dynamic tensor input** (x, dynamic batch)
- **Static tensor input** (static_tensor)
- **Static non-tensor input** (scale, int)
- **Dynamic non-tensor input** (batch_size, from x.size(0))
- **Converter operators** (sigmoid, tanh, add)
- **Custom operators** (scale_add, fallback)

### SimpleStaticModel (static shape scenario)

`SimpleStaticModel` has the following characteristics:
- **Static tensor input** (x, fixed batch=4)
- **Static tensor input** (static_tensor)
- **Static non-tensor input** (scale, int)
- **Static non-tensor input** (multiplier, int, not a SymInt)
- **Converter operators** (sigmoid, tanh, add)
- **Custom operators** (scale_add, fallback)

All input shapes are fixed and `dynamic_shapes` is not used, verifying the scalar input handling path of fallback operators in a purely static scenario.

## Usage Steps

### Check the environment
```bash
bash run_step1.sh
```
Checks whether Python, GCC, CANN, torch, torch_npu, and torch_delegate are available.

### Export the model
```bash
bash run_step2.sh
```
- Exports `model_simple.epair`
- Unpacks it and checks the placeholders in the JSON
- Verifies the correctness of dynamic/static inputs

### Build the C++ runner
```bash
bash run_step3.sh
```
- Builds `fallback_runner`
- Builds `libcustom_ops.so`

### Run the test
```bash
bash run_step4.sh
```
- Runs 3 sets of inputs with different batch sizes
- Writes output files to `output_dir/`

### Compare accuracy
```bash
bash run_step5.sh
```
- Compares the PyTorch reference output with the delegate output
- Verifies whether the accuracy is within tolerance

## Expected Results

All 5 steps should pass:
- Step 1: environment check passes
- Step 2: model export succeeds and the JSON contains the correct placeholders
- Step 3: the C++ runner builds successfully
- Step 4: all 3 input sets execute successfully
- Step 5: accuracy comparison passes

## Notes

- Make sure protobuf is installed: `pip install protobuf`
- Make sure the py311gcc13 environment is activated
- Make sure the CANN environment is sourced
