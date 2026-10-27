# TorchDelegate Examples

English | [简体中文](README.md)

This directory contains end-to-end validation examples for TorchDelegate, covering model export, C++ integration, the Fallback mechanism, and accuracy comparison.

## Example Overview

| Example directory | Description |
|---------|------|
| [`export_and_load_example`](./export_and_load_example/) | Minimal C++ integration: load an `.epair`, build a GE Graph, run inference, and compare accuracy against eager |
| [`inplace_input`](./inplace_input/) | Inplace input behavior consistency check: compares the `torch.export` / `AOTI` / `torch.compile` / `torch_delegate epair` paths |
| [`sym_int_case`](./sym_int_case/) | Fallback mechanism support for dynamic/static non-tensor inputs (`int scale`, dynamic `batch_size`) |
| [`sym_pack_optimization_case`](./sym_pack_optimization_case/) | Sym Pack optimization check: compares torchair `optimize_sym_pack` (removes redundant Packs) with epair export + C++ loading and inference |
| [`assert_tensor_metadata_demo`](./assert_tensor_metadata_demo/) | Demonstrates how to remove the `aten._assert_tensor_metadata` nodes that `torch.export` inserts automatically, then export an epair and load and run it on the C++ side |
| [`accuracy_compare`](./accuracy_compare/) | End-to-end accuracy comparison (Reference vs Delegate) using the BeamSearch model to verify numerical consistency on the NPU |
| [`inplace_return_case`](./inplace_return_case/) | Fallback codegen correctness for the combination of inplace inputs and tensor return values |
| [`fallback_inplace_optional_case`](./fallback_inplace_optional_case/) | Fallback codegen correctness for multiple inplace parameters and multiple optional parameters |
| [`fallback_dynamic_optional_case`](./fallback_dynamic_optional_case/) | Fallback codegen correctness for dynamic inputs (Tensor[]) and optional inputs (Tensor?) |
| [`run_model_multi_instance`](./run_model_multi_instance/) | Multi-process concurrency example: one process per instance (avoiding the process-level lock of GE synchronous `RunGraph`), with producer-consumer threads inside each worker calling `RunModel` in parallel to measure host concurrent submission and device multi-stream parallelism |
| [`common`](./common/) | Shared utility scripts (`build_libtorch_npu.sh`, reused by the examples) |

## Environment Setup

Before running the examples, prepare a Python virtual environment:

```bash
# Create a virtual environment and install all dependencies (torch, torch_npu, triton, triton-ascend, etc.)
cd examples
bash prepare_env.sh
```

The script does the following:
1. Create the `venv_test` virtual environment
2. Install the Python dependencies listed in `requirements.txt`
3. Install `torch==2.9.0` (CPU build), `triton==3.5.0`, and `triton-ascend==3.2.1`
4. Download and install `torch_npu` from the Daily build

## Batch Run

Use `run_all.sh` to run all examples with a single command:

```bash
bash run_all.sh
```

Prerequisites:
- The `ASCEND_HOME_PATH` environment variable is set (`source CANN set_env.sh`)
- The torch_delegate whl is installed (`TORCH_DELEGATE_ROOT` is derived automatically by `run_all.sh` from the installed package, so sourcing set_env.sh is not required)

Execution steps:
1. Create the `venv_test` virtual environment (skipped if it already exists)
2. Check the environment variables
3. Activate the virtual environment
4. Run each enabled example in sequence
5. Aggregate the PASS/FAIL/SKIP results and write them to `case_result.txt`

You can enable or disable individual cases by editing the `CASES` array in `run_all.sh`:

```bash
# enabled  = run
# disabled = skip
# ignored  = ignore
CASES=(
  "export_and_load_example|run.sh|enabled"
  "inplace_input|run.sh|enabled"
  "sym_int_case|run.sh|enabled"
  "sym_pack_optimization_case|run.sh|enabled"
  "assert_tensor_metadata_demo|run.sh|enabled"
  "accuracy_compare|run.sh|ignored"
  "fallback_inplace_optional_case|run.sh|enabled"
  "fallback_dynamic_optional_case|run.sh|enabled"
  "run_model_multi_instance|run.sh|enabled"
)
```
