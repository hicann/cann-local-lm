# host_cpu_mixed_case: CPU/NPU Mixed Scheduling (HostCpu Backend) End-to-End Example

This example verifies TorchDelegate's CPU/NPU mixed scheduling capability:
operators with no NPU execution path that can only run on CPU (custom
namespaces with CPU-only implementations) share one `ge::Graph` with NPU
operators and are scheduled uniformly by GE.

## Model Shapes

| Model | Structure | Scenario covered |
|------|------|---------|
| `model_mixed` | mm(NPU) → `hcpu_ops::feature_scale`(CPU) → relu(NPU) | CPU segment between NPU segments |
| `model_pure_cpu` | `feature_scale`(CPU) → `feature_merge`(CPU) | Pure CPU segment |
| `model_non_adjacent` | `feature_scale`(CPU) → mm(NPU) → relu(NPU) → `feature_merge`(CPU, shared input) | Non-adjacent CPU segments + shared input |
| `model_recommender` | `feature_gather`(CPU embedding lookup) → 3-layer MLP(NPU Task Tower) → `score_rank`(CPU ranking/scoring) | Approximation of the recommendation-network three-stage architecture |

`hcpu_ops::feature_scale` / `hcpu_ops::feature_merge` only register CPU
kernels (no PrivateUse1 implementation); export-time device ownership
resolution marks them `device="cpu"` and persists the decision into the epair
archive. At load time, FallbackCodegen generates custom operators registered
to the GE HostCpu backend (`REG_OP_BACKEND` + `ge::OpBackend::kHostCPU`);
`Execute` calls the PyTorch Dispatcher CPU implementation.

> Known workaround: the index input of `model_recommender` is passed as
> float32 and converted back to int64 inside the CPU kernel for the lookup.
> When the GE runtime stages Data (user runtime inputs) of HostCpu operators
> from device to host, it computes the copy length at 4 bytes per element, so
> DT_INT64 inputs get truncated; a fail-fast check was added on the graph
> build side (`FallbackNodePlanner::BuildInputDefs`). Once GE is fixed, the
> input can revert to int64.

## Run

```bash
bash run.sh
```

Four phases:

1. **Export and ownership validation**: export the models, assert manifest v2
   (`version=2`, the `cpu_nodes` counter) and correct `ascend_ir.device`
   injection;
2. **epair content validation**: unpack the `.epair` and check the manifest
   and device fields;
3. **C++ load and execution**: `host_cpu_runner` loads the epair archives one
   by one; `EpairModelLoader` completes HostCpu codegen during construction
   (g++ compile + dlopen registration), and after `CompileAndLoad` one
   `RunModelWithStreamAsync` executes the mixed graph (NPU device required);
   the runner first `dlopen`s `libcustom_ops.so` (C++-side `TORCH_LIBRARY`
   registration, schema matching `custom_ops.py`) — the C++ deployment process
   and the Python export process do not share the Dispatcher registry;
4. **Accuracy comparison**: delegate outputs are compared element-wise against
   the eager CPU baseline.

Without an NPU device, phases 3/4 are skipped automatically (export and
ownership resolution are still verified).

## Key Observations

- Ownership report: the full ownership inventory is printed at the end of
  export (both NPU and CPU sides, CPU side graded by namespace — aten→CPU is
  flagged as a suspicious WARNING, custom-namespace→CPU is the expected INFO);
- manifest v2: `cpu_nodes` counts the CPU nodes;
- Mixed graph execution: externally still one `RunGraphWithStreamAsync`;
  host↔device copies are inserted by GE at segment boundaries, and data flows
  in host memory within a CPU segment with zero copies.
