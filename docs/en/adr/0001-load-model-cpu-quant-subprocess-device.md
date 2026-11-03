# ADR-0001: converter main process load_model hard-coded to CPU; lightweight tool subprocess supports device selection

English | [简体中文](../../zh/adr/0001-load-model-cpu-quant-subprocess-device.md)

> Status: Accepted
> Date: 2026-08-25
> Related design .md: [cli_design.md](../cann_lm_converter/cli/cli_design.md) "Runtime Environment Support" / "Relationship between converter and torch_npu"

## Background (Context)

The converter has three stages in its full pipeline, each with different compute-domain requirements:

| Stage | What it does | Compute-domain requirement |
|------|--------|-----------|
| Load `load_model` + swap | Load cann_local_lm model, materialize int4, setattr to replace nn.Linear | Only needs tensor operations (CPU is fine) |
| Quantization GPTQ (`--quant-method gptq`) | CANN lightweight tool runs Hessian matrix accumulation `X^T X` + Cholesky inversion + per-column update | Needs real forward + backward, must run on a compute device (NPU/GPU) |
| Export epair (`torch.export` + torchair GE conversion) | Trace model → FX graph → GE IR | Only needs fake-meta (for tracing) + torchair stubs (C symbols) |

Key constraints:

1. **cann_local_lm fused ops are only implemented on CPU**: Ops in `cann_local_lm/models/fused_ops/` (`npu_quant_matmul` / `npu_dynamic_quant` / `fused_infer_attention_score`, etc.) are provided by `_library.py` with schema + fake-meta + CPU PoC impl, **do not support real NPU execution** (only support torch.export tracing). Therefore the cann_local_lm model loaded by `load_model` cannot be forwarded on NPU.
2. **converter venv is CPU-only**: `torch 2.9.0+cpu` + `torchair` (source-compiled `NO_ASCEND_SDK=1`) + lib stubs (`libtorch_npu.so` etc. stubbed, providing C symbols for the torchair GE engine, avoiding real NPU `aclrtSetDevice→107001`). **The torch_npu python package is not installed.**
3. **GPTQ needs real op eager**: dopt_lm uses `transformers.AutoModelForCausalLM` + real torch_npu/torch.cuda ops eager to run GPTQ (Hessian requires batch forward accumulation, Cholesky requires compute power). dopt_lm `SUPPORTED_DEVICE_TYPES = {"cpu","cuda","npu"}` **includes CPU** (CPU uses `dispatch_dtype=float32`), but 7B+ model Hessian is several GB, extremely slow on CPU (only suitable for small models/debugging).
4. **Environment conflict is unsolvable**: If the converter main process installs real torch_npu, it will hit the `_LazyTorchair` proxy overriding the source torchair (`torch_npu/dynamo/__init__.py` unconditionally installs `sys.modules['torchair']=_LazyTorchair()`, hijacking to the bundled wrapper, whose `fx2ge_converter.parse_input` has an AssertionError failing on the `input_ids` placeholder) + torch_npu version dependency (`npu_chunk_gated_delta_rule` schema only in ≥2.7.1.post8) + `_get_deterministic_level` API missing. See cli_design.md "Relationship between converter and torch_npu" for details.

## Decision (Decision)

**The main process (load_model + swap + export) is hard-coded to CPU venv; the lightweight tool subprocess (`--quant-method gptq`) uses the system Python + real compute device (NPU/GPU), isolated via subprocess.**

- `load_model(hf_path, dtype, device="cpu")` is hard-coded to `device="cpu"`; the `--device` CLI parameter is not exposed.
- `--quant-device` only controls the lightweight tool subprocess device (`npu`/`cuda`/`cpu`), passed to the subprocess via `os.environ["DEVICE"]`; if `None`, `_detect_device_type` performs spawn detection or inherits the parent process `DEVICE`, falling back to `cpu` when there is no NPU/GPU (dopt_lm supports CPU float32, 7B+ is extremely slow).
- The subprocess uses a standalone Python (`DOPT_PYTHON`, auto-detected by `_env_common.sh` as the system python with torch_npu installed, see build_guide.md Q7) + real torch_npu/torch.cuda + NPU/GPU env (`LD_LIBRARY_PATH` without stubs).

> **Status change (2026-08-31)**: `--quant-device` has been removed from the converter CLI. Subprocess device is now auto-detected by `auto_quant._detect_device_type` (reads `os.environ["DEVICE"]`, which external envs like `set_env_npu.sh` can set; if unset, performs spawn detection, falling back to `cpu` when no accelerator), and the `_run_dopt_subprocess` `env.setdefault("DEVICE", device)` mechanism is retained. The CLI `os.environ["DEVICE"]` trigger path described here no longer exists, but the `_detect_device_type` + subprocess env mechanism is retained.

Rationale: Export only needs fake-meta tracing (self-provided by `_library.py`, no torch_npu needed); GPTQ needs real op eager (must run on a compute device). The two have different compute domains, and the environment conflict (installing torch_npu breaks GE conversion) is unsolvable, so subprocess isolation is used for the two environments.

## Consequences (Consequences)

- **Positive**:
  - Avoids the `_LazyTorchair` proxy override pitfall + torch_npu version dependency (converter correctness does not depend on any torch_npu version)
  - converter venv and NPU env are independent and do not pollute each other
  - Pure CPU machines are usable (`--quant-path` consumes offline lightweight tool output + export stubs)
- **Negative/Cost**:
  - subprocess has disk-bridge IO (temp dir via `tempfile.mkdtemp`, cleaned up with `shutil.rmtree` after successful swap, retained on failure for debugging)
  - ~~`auto_quant.py` and `dopt_wrapper.py` each have a copy of `_build_layer_strategy` (duplicated, planned to merge into a single file to eliminate, not yet landed)~~ **Landed (2026-08-31)**: `auto_quant._build_layer_strategy` deleted, `dopt_wrapper._build_layer_strategy` renamed to `build_layer_strategy`, single implementation
  - Scenario 2 memory peak is 2× (2 HF models from the lightweight tool inside the subprocess + main process cann_local_lm, in separate processes; not a true one-step integration)
- **Constraints introduced**:
  - `load_model` is always CPU (cann_local_lm fused ops are only implemented on CPU, no plan to port to NPU)
  - `--quant-device` only affects the subprocess, not the main process
  - Pure CPU environment (no NPU/GPU) `--quant-method gptq` falls back to `cpu` (dopt_lm supports this, 7B+ is extremely slow); or use `--quant-path` to consume offline output
  - cann_local_lm model layer names must be consistent with transformers layer names (the HF model layer names quantized by the lightweight tool must be portable to the cann_local_lm model, relying on naming consistency)

## Alternatives (Alternatives)

### Alternative A: Main process also uses a compute device (install real torch_npu)

converter venv installs real torch_npu (e.g., 2.9.0.post6) + renames `torch_npu/dynamo/torchair` → `torchair.disabled` (so that `dynamo/__init__.py`'s `if not exists` check hits → does not install the proxy → source torchair is used directly).

Rejection reasons:
1. Hits torch_npu version dependency (`npu_chunk_gated_delta_rule` schema only in ≥2.7.1.post8, must match torch version)
2. Requires a one-time rename of the wrapper, high complexity
3. converter does not need real torch_npu ops (fake-meta is self-provided by `_library.py`), installing it instead introduces the proxy pitfall
4. Even if the rename workaround works, the main process still needs to simultaneously satisfy CPU stubs (export) + NPU (GPTQ), an environmental configuration contradiction

### Alternative B: Run GPTQ on CPU

The lightweight tool runs GPTQ on CPU (Hessian + Cholesky all on CPU).

> **Status change (2026-08-26)**: dopt_lm now supports CPU (`SUPPORTED_DEVICE_TYPES` includes `cpu`, `train.py`/`modelutils.py` have CPU branches with `dispatch_dtype=float32`). `--quant-device cpu` or auto-fallback when no accelerator is available has **landed** (`auto_quant._detect_device_type` no longer errors on pure CPU). Therefore this option has changed from "rejected" to "**supported, with limited performance**".

Performance-limited (still recommend NPU/GPU or `--quant-path` as fallback):
1. 7B+ model Hessian matrix is several GB, extremely slow on CPU (unacceptable time cost, only suitable for small models/debugging)
2. GPTQ calibration requires batch forward to accumulate activations, CPU throughput is insufficient

> Note: CPU GPTQ still uses **subprocess isolation** (`DEVICE=cpu` passed to the subprocess, system Python + CPU torch), which does not violate this ADR's main decision (load_model hard-coded to CPU + subprocess isolates the two environments).

### Alternative C: Memory bridge (in-process, no subprocess spawn)

converter main process imports dopt_lm to run GPTQ in-process, saving the subprocess.

Rejection reasons:
1. Environment conflict is unsolvable (converter venv CPU stubs contradicts the lightweight tool's need for real NPU, cannot be satisfied simultaneously in the same process)
2. dopt_lm's `args` global singleton + `ptq_loop` reentrancy risk (in-process reentrancy will error)
3. Memory peak 3-4× (cann_local_lm + lightweight tool's 2 HF copies + state_dict)

## Related

- See cli_design.md "Runtime Environment Support (NPU / GPU / Pure CPU)" three-environment matrix + "Relationship between converter and torch_npu" three coupling issues
- quant_design.md "One-click quantization scheme selection" three-option comparison (A modify lightweight tool / B memory / C subprocess, chose C i.e. this ADR)
- If cann_local_lm fused ops are ported to NPU in the future (supporting real execution): the "load_model hard-coded to CPU" constraint of this ADR can be relaxed, but subprocess isolation may still be retained (to avoid environment conflicts)
