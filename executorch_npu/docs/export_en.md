# Exporting the Model to PTE/PTD

"Model export" involves two distinct phases: the model is first captured by `torch.export`, and then ExecuTorch-NPU
completes the ExecuTorch lowering and CANN AOT compilation. The two phases can be executed consecutively in the same environment, or
separated via a PT2 file.

For building from source, see [Build from Source](build_en.md); for C++ loading and execution, see [C++ Runtime](runtime_en.md).

## Two-Phase Export Flow

```text
Phase A: Model capture
Model implementation + Checkpoint + Example inputs
→ torch.export.export
→ ExportedProgram
→ Optional: torch.export.save(..., "model.pt2")

Phase B: CANN AOT export
ExportedProgram or model.pt2
→ ExecuTorch Edge lowering
→ AscendPartitioner
→ torch-delegate
→ TorchAir provider
→ CANN OfflineCompiler
→ model.pte + ascend.weights.package.ge_aot.ptd
```

| Phase | Main inputs | Main outputs | Dependency ownership |
|---|---|---|---|
| A. `torch.export` | PyTorch model, weights, example inputs | `ExportedProgram` / PT2 | Owned by the model implementation |
| B. ExecuTorch-NPU | `ExportedProgram` / PT2, target SoC | PTE/PTD | Owned by ExecuTorch-NPU |

## Dependency Boundaries

### Dependencies Shared by Both Phases

Both phases must use compatible Python and Torch. Phase A itself does not depend on ExecuTorch. Currently supported:

| Dependency | Requirement |
|---|---|
| PyTorch | Official CPU wheel 2.11.0–2.13.0 |

ExecuTorch-NPU does not install Torch automatically. All supported versions must first be installed as `+cpu`-tagged wheels
from the official PyTorch CPU index. For version combinations, see [Supported Versions](supported_versions_en.md).

Default combination:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0+cpu" \
  "torchvision==0.26.0+cpu"
```

If Phase A and Phase B use different environments, it is recommended to keep exactly the same Torch version. PT2 is not a stable
exchange format across arbitrary Torch versions.

### Phase A: Minimal Dependencies for `torch.export`

Generic documentation can only specify the following minimal set:

| Dependency | Pinned | Description |
|---|---:|---|
| CPU Torch | Yes | Basic `torch.export` capability |
| Model implementation package | No | Transformers, Torchvision, custom model code, etc. |
| Checkpoint/Tokenizer dependencies | No | Determined by how the model is loaded |
| torch_npu | No | Install only when required for model construction, fused operators, or NPU tensor capture |
| Custom operator registration | No | May still be required for PT2 loading and subsequent lowering |

### Phase B: Pinned CANN AOT Dependencies

| Dependency | Purpose |
|---|---|
| ExecuTorch-NPU wheel | Partitioner, Backend, and artifact protocol |
| ExecuTorch 1.5.0 | Edge lowering and PTE/PTD serialization |
| CANN Toolkit GE/TBE | Graph optimization and offline compilation |
| torch-delegate | EPAIR, GE Graph, and native bridge |
| TorchAir provider | ATen/FX to GE graph conversion |

Installing the ExecuTorch-NPU wheel installs ExecuTorch 1.5.0 and its regular Python dependencies, but does not
install or replace Torch:

```bash
python -m pip install /path/to/executorch_npu-*.whl
```

### Choosing a TorchAir Provider

Only one source of TorchAir is needed:

| Condition | Choice |
|---|---|
| The model does not depend on torch_npu | Install a standalone TorchAir matching the current Torch |
| The same environment already has torch_npu installed for the model implementation, and its bundled TorchAir is compatible with Torch/CANN | Reuse the TorchAir bundled with torch_npu directly |
| Phase A uses torch_npu, but Phase B is a separate environment | Phase B re-selects a standalone TorchAir, or installs the same compatible torch_npu |

Do not install both a standalone TorchAir and the torch_npu-bundled TorchAir in the same environment. Just because Phase A
once used torch_npu does not mean that a separate Phase B environment automatically has TorchAir.

Source installation entry point for standalone TorchAir:

```bash
git clone https://gitcode.com/ascend/torchair.git
cd torchair
bash build.sh -i
```

TorchAir must be built against the current Torch version. torch-delegate does not need to be downloaded separately: its source code is located
in the `../torch_delegate` directory at the same level as this repository. Choose either of the two installation methods:

```bash
# Method 1: Build the whl from the in-repo source and install it
cd ../torch_delegate
bash build.sh
python -m pip install output/package/torch_delegate-*.whl

# Method 2: Install the released torch_delegate whl package directly
python -m pip install /path/to/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

For build details, see `../torch_delegate/docs/build.md` in the repository.

### Exporting Without an NPU Device (Lite Export)

Neither Phase A nor Phase B **requires an NPU device to be present**. The target chip is passed in only via `ACL_SOC_NAME`;
a device is required only during the runtime phase. This is also required by design constraint 11, see
[Design Document](design/executorch_npu_design_en.md).

```bash
# Phase A: DEVICE=cpu
export DEVICE=cpu
python -m examples.qwen3_5.export.pt2

# Phase B: host-only
export ACL_SOC_NAME=Ascend910_9362
export EXECUTORCH_NPU_HOST_ONLY=1
python -m examples.qwen3_5.export.pte
```

TorchAir's GE engine initialization couples "initializing GE" and "binding the device" into a single C++ call,
while Phase B only needs the former. The device dependency spans three layers, and `EXECUTORCH_NPU_HOST_ONLY=1` handles them separately:

| Layer | Location | Behavior without a device | Handling |
|---|---|---|---|
| Python | `torchair.core._backend._get_device_id()` | Calls `npu.current_device()` because torch_npu is already imported, throwing `aclInit 107001` | Replaced with reading `ASCEND_DEVICE_ID` |
| C++ | `_torchair.InitializeGraphEngine()` | Unconditionally calls `acl.rt.set_device()` internally, returning 107001 | Stub out the C++ leaf function |
| Op converter | `torchair._utils.check_platform.is_arch35()` | Reads the physical device name through `torch_npu.npu.get_device_name()`; without a device the device index resolves to `None`, so `None < 0` raises `TypeError`, and TorchAir only catches `ImportError` | `get_device_name()` is replaced with returning `acl_soc_name`, leaving the architecture decision to TorchAir's own device-name rule |

The third layer covers 25 converter modules in the currently pinned TorchAir revision (including
`native_batch_norm_legit_no_training`, `max_pool2d_with_indices`, `native_group_norm`, `std`, and `npu_rotary_mul`).
The result selects GE op versions, attributes, layouts, and output semantics, so it cannot be hardcoded to `False`;
only the device-name source is replaced. Without torch_npu the function takes its `ImportError` branch and returns `False`,
which matches the currently registered SoCs; adding an arch35 target (a device name containing `Ascend910_95` or `Ascend950`)
requires torch_npu so that `acl_soc_name` takes part in the decision.

The C++ layer cannot be bypassed with parameters: passing `0` for `deviceId` reports `ACL set device id failed`, and passing `-1`
triggers `assert device_index_ >= 0`. Setting `ge_run_with_torch_npu` to `0`, switching to a standalone
TorchAir, or even completely blocking torch_npu are all ineffective.

After stubbing, `EpairModelLoader` still requires GE to be initialized. EPAIR loading and offline compilation
share a single initialization managed by `OfflineCompiler`:

```python
ge.offline_compile.build_initialize({
    "ge.socVersion": soc,
    "ge.graphRunMode": "0",
    "ge.externalWeightDir": session_workspace,
})
```

During initialization, the session holds a dedicated `ge.externalWeightDir` temporary directory; the current CANN does not support
setting this directory in the graph-level options of `build_model`. After each OM is saved, the compiler moves the weights from this build
into the corresponding partition's `weight/`, and residual files from failed builds are also cleaned up.
GE completes finalization first, and then the session directory is deleted; weights from different methods will not be mixed into subsequent artifacts.
The `meta.json` in the session directory is CANN's host-side index, not a FileConstant tensor,
and must be excluded when moving weights — it must not be written into the PTD or registered as an external weight binding.
The compiler does not change the calling process's working directory or `ASCEND_WORK_PATH`.

Within the same process, the compilation entry point, compatibility patches, temporary environment variables, and GE operations share a single reentrant lock.
Reentrancy is used when the compilation entry point invokes inner operations such as loading/initialization; compilations from different threads execute serially.
Temporary environment variables are restored even when exceptions occur. A GE session pins the SoC and
`EXECUTORCH_NPU_HOST_ONLY` mode used the first time; subsequent inconsistent requests report an explicit error, and switching configurations requires a new process.
After GE is shut down, no new requests are accepted; normal shutdown and process exit perform native finalization only once.
TorchAir compatibility adaptations are centralized in `compiler/compatibility.py`.

In device mode, TorchAir initialization is triggered by the torch-delegate converter using the full compilation configuration.
Do not call `initialize_graph_engine()` without arguments before the converter runs: this caches an empty configuration,
causing subsequent initializations with options such as `ge.exec.enableEngineParallel` to report conflicts.
Offline GE session initialization serves a different purpose from this step; both modes require real export and runtime regression.


Memory sampling is disabled by default. Set `ASCEND_EDGE_MEMORY_DIAGNOSTICS=1` to enable it;
GC is actively triggered only when `ASCEND_EDGE_MEMORY_COLLECT_GC=1` is also set, so that log sampling
does not change compilation time by default.

Prerequisites for using Lite export:

| Requirement | Description |
|---|---|
| `EXECUTORCH_NPU_STAGING` points to a large-capacity disk | GE compilation temporary weights can peak at 37 GB (Gemma4 E2B). The default `/tmp` is often on the root partition; when it fills up, `Write data failed, errno: -1` is reported |
| CANN Toolkit from the same source | You must `source` the `set_env.sh` of the same Toolkit. Overriding only `ASCEND_OPP_PATH` causes cross-version mixing of OPP operator implementations and the TBE framework |
| `PYTHONPATH` appended rather than overwritten | Overwriting loses the CANN paths injected by `set_env.sh`, resulting in `No module named 'ge'` |

Verify that the current process indeed cannot see the device:

```bash
ASCEND_RT_VISIBLE_DEVICES="" python -c \
  'import torch, torch_npu; print("device_count =", torch.npu.device_count())'
```

Boundaries of Lite export:

- Does not change runtime-phase requirements: loading and executing PTE/PTD still requires a matching NPU, driver, and AscendCL.
- Without a device, the OM produced by GE may carry a host os/arch suffix (e.g. `model_linux_aarch64.om`),
  so the artifact is bound to that host platform; cross-platform deployment requires recompilation on the target platform. The packaging side locates
  files by the `model*.om` prefix and does not hardcode `.om`.
- The stubbed objects are TorchAir internal symbols and must be re-verified after a TorchAir upgrade.
- Lite export only changes how the GE engine is initialized and does not replace any operators, so it produces
  graphs numerically equivalent to device-side export.

## Environment Organization

### Single-Environment Mode

Model capture and CANN AOT execute consecutively in the same Python environment. The environment dependencies are the union of both phases:

```text
Model dependencies
+ CPU Torch
+ Optional torch_npu
+ ExecuTorch-NPU / ExecuTorch
+ CANN GE/TBE
+ torch-delegate
+ One TorchAir provider
```

This is the most straightforward approach for small models and simple export scripts. If the model already requires torch_npu and its bundled
TorchAir is compatible, there is no need to build a standalone TorchAir.

### PT2 Split-Environment Mode

Phase A saves the PT2, and Phase B loads it later:

```text
Model environment               CANN AOT environment
Model impl + optional torch_npu ExecuTorch-NPU + ExecuTorch
torch.export.export             CANN GE/TBE + torch-delegate
torch.export.save               One TorchAir provider
          │
          └──────── PT2 ───────→ torch.export.load → PTE/PTD
```

PT2 can remove Phase B's dependency on the Checkpoint, Tokenizer, and most model construction code, but it cannot guarantee
removal of the following dependencies:

- Custom operator registrations retained in the graph.
- NPU device or torch_npu type registrations.
- Model patch modules required to load the PT2.
- A Torch version compatible with the saving side.

The Qwen and Gemma examples in the repository already use staged PT2/PTE scripts and can serve as references for complex models.

## Initializing the CANN AOT Environment

```bash
conda activate executorch_npu
source /path/to/cann/set_env.sh
export PYTHONNOUSERSITE=1
```

When using torch-delegate in source form (in-repo `../torch_delegate`), appended paths must preserve
the original values injected by CANN:

```bash
export TORCH_DELEGATE_ROOT="$(cd ../torch_delegate && pwd)"
export PYTHONPATH="$TORCH_DELEGATE_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_DELEGATE_ROOT/build/release/csrc:${LD_LIBRARY_PATH:-}"
```

Do not overwrite `PYTHONPATH`, otherwise `ge` or `tbe` will be lost.

## Phase A: Capture and Save the PT2

The following code shows only the minimal generic steps; `model`, `example_inputs`, dynamic shapes, and model patches are the responsibility
of the model implementation:

```python
from pathlib import Path

import torch

exported = torch.export.export(model.eval(), example_inputs)
torch.export.save(exported, Path("model.pt2"))
```

Multi-method or Prefill/Decode models usually save multiple PT2 files separately. Real examples:

- `examples/qwen3_5/export/pt2.py`
- `examples/gemma4/export/pt2.py`

## Phase B: Generate PTE/PTD from the PT2

```python
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower

from executorch_npu import (
    AscendCompileOptions,
    AscendPartitioner,
    aten_edge_compile_config,
)

exported = torch.export.load("model.pt2")
options = AscendCompileOptions(
    method_name="forward",
    delegate_index=0,
    acl_soc_name="Ascend910B3",
)
program = to_edge_transform_and_lower(
    exported,
    partitioner=[AscendPartitioner(options.to_compile_specs())],
    compile_config=aten_edge_compile_config(),
).to_executorch()

output_dir = Path("artifacts")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "model.pte").write_bytes(program.buffer)
program.write_tensor_data_to_file(output_dir)
```

If split environments are not needed, you can directly replace `torch.export.load("model.pt2")` with the
`ExportedProgram` produced by Phase A, without writing the PT2 to disk.

`acl_soc_name` must exactly match the name returned by `aclrtGetSocName()` on the runtime device.

## Pre-Export Checks

For both TorchAir profiles, first confirm the CPU Torch and CANN Python environment:

```bash
python - <<'PY'
import importlib.util
import torch

assert "+cpu" in torch.__version__, torch.__version__
assert importlib.util.find_spec("ge.offline_compile") is not None
assert importlib.util.find_spec("tbe") is not None
print("torch:", torch.__version__)
print("torch_npu:", importlib.util.find_spec("torch_npu"))
PY
```

Standalone TorchAir profile:

```bash
python -c 'import torchair; print(torchair.__file__)'
```

The torch_npu-bundled TorchAir profile requires importing torch_npu first:

```bash
python -c 'import torch_npu, torchair; print(torch_npu.__file__); print(torchair.__file__)'
```

Confirm that the actual `torchair` path comes from the chosen provider. torch_npu should not be loaded
in the standalone TorchAir profile.

## Inspecting Artifacts

```text
artifacts/
├── model.pte
└── ascend.weights.package.ge_aot.ptd
```

- The PTE stores the ExecuTorch Program, Delegate processed bytes, Manifest, and OM named data.
- The PTD stores external weight named data.
- The PTE and PTD must be produced by the same Phase B export and delivered as a pair.

CANN AOT usually does not require an actual NPU, but does require a complete GE/TBE environment. If Phase A constructs the model
on an NPU or executes fused operators, the corresponding device and torch_npu environment are required.

## Next Steps

For loading artifacts with C++, see [C++ Runtime](runtime_en.md). For export errors, see
[Troubleshooting: Model Export](troubleshooting_en.md#model-export).
