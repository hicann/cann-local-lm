# Python API Reference

## Overview

`torch_delegate` provides three core functions that implement the complete PyTorch `ExportedProgram` → `.epair` workflow:

```text
ExportedProgram → export_to_epair() → EpairContents → epair_save() → .epair file
                                                                  ↓
                                                          epair_load() → EpairContents

.epair file → EpairModelLoader(path) → GE graph → GE Offline_compile (offline compilation path)
```

> The `EpairModelLoader` class (`.epair` → GE graph for offline compilation) is documented in [EpairModelLoader (Python API)](model_loader.md).

**Import:**

```python
from torch_delegate import export_to_epair, epair_save, epair_load
from torch_delegate import EpairContents, EpairManifest, EpairConfig
```

**Package information and C++ prefix:**

```python
import torch_delegate

torch_delegate.__version__      # Version injected at build time (debug builds carry a +debug suffix)
torch_delegate.cmake_prefix()   # C++ artifact prefix, can be added directly to CMAKE_PREFIX_PATH
```

Equivalent CLI: `torch-delegate-config --cmake-prefix` / `torch-delegate-config --version`.

> Full export mode depends on `torchair` provided by `torch_npu`. Since the whl declares no runtime dependencies, pip installs neither `torch` nor `torch_npu` automatically: pre-install `torch>=2.9.0` in the environment where you install the whl, then install a matching `torch_npu`; when `torch_npu` is missing, `export_to_epair()` raises a `RuntimeError` and prompts you to install a matching version. For environments without a CANN runtime, use [Lite export mode](#experimental-feature-lite-export-mode-no-cann-runtime).

**Experimental feature (Lite export mode) imports:**

```python
from torch_delegate import LiteConfig, lite_export_context
```

> Lite export mode and its related interfaces are experimental. The interfaces may change in the future and backward compatibility is not guaranteed. See [Experimental feature: Lite export mode](#experimental-feature-lite-export-mode-no-cann-runtime).

---

## Core functions

### `export_to_epair(ep, config)`

Converts a PyTorch `ExportedProgram` into `EpairContents` annotated with Ascend IR.

**Signature:**

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ep` | `ExportedProgram` | Required | The exported program obtained from `torch.export.export()` |
| `config` | `EpairConfig` | `EpairConfig()` | Conversion configuration, controls operator decomposition, pattern fusion, and similar behavior |

**Returns:** An `EpairContents` object

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `RuntimeError` | torchair conversion failed (for example, an unsupported operator) |
| `ValueError` | Malformed ExportedProgram |

**Example:**

```python
import torch
from torch_delegate import export_to_epair, epair_save

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y) + 1.0

model = MyModel()
ep = torch.export.export(model, (torch.randn(3, 3), torch.randn(3, 3)))
contents = export_to_epair(ep)
print(f"Converted: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")
epair_save(contents, "model.epair")
```

**Notes:**

- The function temporarily patches an internal PyTorch function (`do_auto_functionalized`) and restores it after the call returns
- The `ascend_ir.source` field in the conversion result indicates the conversion source: `"converter"` means successful conversion, an empty string means a fallback node

---

### `epair_save(contents, f)`

Serializes `EpairContents` into a `.epair` file.

**Signature:**

```python
def epair_save(contents: EpairContents, f: str | os.PathLike) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `contents` | `EpairContents` | The object returned by `export_to_epair()` |
| `f` | `str \| os.PathLike` | Output file path; the `.epair` suffix is recommended |

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `OSError` | File write failed |
| `TypeError` | contents is not of type EpairContents |

**File format:**

A `.epair` file is a PT2-compatible ZIP archive that contains:

| Path | Content |
|------|---------|
| `models/model.json` | Standard EP JSON |
| `data/weights/` | Model weights |
| `extra/epair_manifest.json` | Conversion metadata |
| `extra/ep_json_with_ascend_ir.json` | EP JSON with ascend_ir annotations |

**Example:**

```python
from torch_delegate import export_to_epair, epair_save

contents = export_to_epair(ep)
epair_save(contents, "model.epair")
```

---

### `epair_load(archive_path)`

Loads a `.epair` file and returns `EpairContents`.

**Signature:**

```python
def epair_load(archive_path: str | os.PathLike) -> EpairContents
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `archive_path` | `str \| os.PathLike` | Path to the `.epair` file |

**Returns:** An `EpairContents` object

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | File does not exist |
| `zipfile.BadZipFile` | File is not a valid ZIP archive |
| `KeyError` | Archive is missing required files (such as manifest.json) |

**Example:**

```python
from torch_delegate import epair_load

contents = epair_load("model.epair")
print(f"Backend: {contents.manifest.backend_name}")
print(f"Converted: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")

# Execute in Python eager mode
result = contents.ep.module()(input_tensor)
```

---

## Data classes

### `EpairManifest`

Epair conversion metadata that records conversion statistics.

```python
@dataclasses.dataclass
class EpairManifest:
    version: int
    backend_name: str
    cann_version: str
    total_nodes: int
    converted_nodes: int
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | `int` | Format version number, currently 1 |
| `backend_name` | `str` | Backend name, always `"ascend_ge"` |
| `cann_version` | `str` | CANN build environment version, such as `"9.2.0"`. `"unknown"` when unavailable |
| `total_nodes` | `int` | Total number of operator nodes in the EP JSON |
| `converted_nodes` | `int` | Number of nodes successfully converted to Ascend IR |

---

### `EpairContents`

The Epair conversion result, containing the complete conversion artifacts.

```python
@dataclasses.dataclass
class EpairContents:
    manifest: EpairManifest
    ep_json: dict[str, Any]
    ep: ExportedProgram
```

| Field | Type | Description |
|-------|------|-------------|
| `manifest` | `EpairManifest` | Conversion metadata |
| `ep_json` | `dict[str, Any]` | EP JSON dict with `ascend_ir` annotations |
| `ep` | `ExportedProgram` | Executable ExportedProgram |

**Inspecting conversion results:**

```python
contents = export_to_epair(ep)

# Check conversion coverage
rate = contents.manifest.converted_nodes / contents.manifest.total_nodes
print(f"Conversion rate: {rate:.1%}")

# Analyze fallback nodes
for node in contents.ep_json["graph_module"]["graph"]["nodes"]:
    ascend_ir = node.get("ascend_ir", {})
    if ascend_ir.get("source") == "":
        print(f"Fallback: {node['target']}")
    elif "source" not in ascend_ir:
        print(f"Unannotated: {node['target']}")
```

---

### `EpairConfig`

Epair conversion configuration that controls conversion behavior.

```python
@dataclasses.dataclass
class EpairConfig:
    remove_noop_ops: bool = True
    pattern_fusion_pass: bool = True
    whitelist_decompositions: set[Any] = field(default_factory=...)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `remove_noop_ops` | `bool` | `True` | Whether to remove no-op nodes (such as identity, reshape) |
| `pattern_fusion_pass` | `bool` | `True` | Whether to enable pattern fusion optimization |
| `whitelist_decompositions` | `set[Any]` | See the table below | Set of operators allowed to be decomposed |

**Default decomposed operators:**

| Operator | Description |
|----------|-------------|
| `torch.ops.aten.linear.default` | Linear layer, decomposed into matmul + add |
| `torch.ops.aten.matmul.default` | Matrix multiplication |
| `torch.ops.aten.layer_norm.default` | LayerNorm |
| `torch.ops.aten.softmax.int` | Softmax |
| `torch.ops.aten.log_softmax.int` | LogSoftmax |
| `torch.ops.aten.__and__.Tensor` | Bitwise AND |
| `torch.ops.aten.__or__.Tensor` | Bitwise OR |
| `torch.ops.aten.isfinite.default` | Tests whether elements are finite, decomposed into abs + ne + mul |
| `torch.ops.aten.negative.default` | Negation, decomposed into neg (alias normalization) |
| `torch.ops.aten.absolute.default` | Absolute value, decomposed into abs (alias normalization) |
| `torch.ops.aten.relu6.default` | ReLU6, decomposed into hardtanh |
| `torch.ops.aten.arccos.default` | Arc cosine, decomposed into acos (alias normalization) |
| `torch.ops.aten.arcsin.default` | Arc sine, decomposed into asin (alias normalization) |
| `torch.ops.aten.arctan.default` | Arc tangent, decomposed into atan (alias normalization) |
| `torch.ops.aten.arccosh.default` | Inverse hyperbolic cosine, decomposed into acosh (alias normalization) |
| `torch.ops.aten.arcsinh.default` | Inverse hyperbolic sine, decomposed into asinh (alias normalization) |
| `torch.ops.aten.conv2d.default` | 2-D convolution, decomposed into convolution |
| `torch.ops.aten.square.default` | Square, decomposed into pow.Tensor_Scalar |

**Configuration examples:**

```python
from torch_delegate import export_to_epair, EpairConfig

# Disable pattern fusion (for debugging)
config = EpairConfig(pattern_fusion_pass=False)
contents = export_to_epair(ep, config)

# Disable all operator decomposition
config = EpairConfig(whitelist_decompositions=set())
contents = export_to_epair(ep, config)

# Decompose linear only
config = EpairConfig(whitelist_decompositions={torch.ops.aten.linear.default})
contents = export_to_epair(ep, config)

# Extend the default decomposed operators
config = EpairConfig()
config.whitelist_decompositions.update({
    torch.ops.aten.native_batch_norm.default,
    torch.ops.aten.add.Tensor,
})
contents = export_to_epair(ep, config)
```

---

## Experimental feature Lite export mode (no CANN runtime)

> **Experimental feature**: All interfaces described in this section (`LiteConfig`, `lite_export_context`, and the various `Lite*` exceptions) are experimental. The interfaces may change in the future and backward compatibility is not guaranteed.

Lite export mode lets you build the GE IR required by EPAIR in an environment without CANN, without `torch_npu`, and without an NPU device. It uses `lite_export_context` to stub TorchAir and `torch_npu` dependencies within the process, replacing native capabilities only during graph construction; it does not load, compile, or execute GE graphs.

**Constraints:**

- Before entering `lite_export_context`, the process must not have loaded the real `torchair` or `torch_npu`. If `torch_npu` is installed, set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before importing `torch`
- Lite mode and full mode (the default `export_to_epair`) are mutually exclusive and cannot be mixed within the same process
- Lite mode does not allow any unconverted (fallback / unannotated) nodes
- Lite mode does not support multithreading; the context is driven by process-level state, and the state cannot be fully restored after exiting

### `lite_export_context(config)`

Enters the process-level, build-only EPAIR export mode.

**Signature:**

```python
@contextlib.contextmanager
def lite_export_context(config: LiteConfig) -> Iterator[None]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `LiteConfig` | Target platform configuration, including device model, HF32, and target CANN version |

**Returns:** A context manager; within the `with` block, `export_to_epair()` performs Lite conversion.

**Exceptions:**

| Exception | Condition |
|-----------|-----------|
| `LiteConfigError` | `config` has an invalid type or field value |
| `LiteImportError` | The process has already loaded conflicting dependencies such as `torchair` / `torch_npu`, or importing TorchAir fails after stubbing |
| `LiteStateError` | Conflict with the process mode (for example, already in full mode), cross-thread use, or inconsistent configuration when nested |

**Example:**

```python
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from torch_delegate import (
    LiteConfig,
    epair_save,
    export_to_epair,
    lite_export_context,
)

ep = torch.export.export(model, example_inputs)

config = LiteConfig(target_soc="Ascend910_9362", target_cann_version="9.2.0")

with lite_export_context(config):
    contents = export_to_epair(ep)

# epair_save does not call TorchAir, so it can run outside the context.
epair_save(contents, "model.epair")
```

**Notes:**

- After entering the context, the conversion flow reuses `TorchAirGEConverter`; the manifest no longer records the local CANN version (`cann_version` is `"unknown"`).
- Nesting the context is allowed only when the configurations are exactly the same; otherwise a `LiteStateError` is raised.

---

### `LiteConfig`

Target platform configuration for Lite export mode.

```python
@dataclasses.dataclass(frozen=True)
class LiteConfig:
    target_soc: str
    allow_hf32: bool = False
    target_cann_version: str | None = None
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_soc` | `str` | Required | Target deployment device model, such as `"Ascend910_9362"`; the value semantics must match TorchAir platform detection |
| `allow_hf32` | `bool` | `False` | Whether HF32 is enabled for the target graph |
| `target_cann_version` | `str \| None` | `None` | Target deployment CANN version, a reserved field; an error is raised if it is consulted by dependency version queries |

---

### Lite exception types

All Lite mode exceptions derive from `LiteExportError` (base class `RuntimeError`) and can be caught uniformly:


| Exception | Base class | Condition |
|-----------|------------|-----------|
| `LiteExportError` | `RuntimeError` | Base class for all Lite export failures |
| `LiteImportError` | `LiteExportError` | The process has already loaded conflicting native dependencies (`torchair` / `torch_npu`), or TorchAir import fails after stubbing |
| `LiteStateError` | `LiteExportError` | The operation conflicts with the process / context state (mode exclusivity, cross-thread, inconsistent configuration) |
| `LiteDependencyError` | `LiteExportError` | A capability the converter depends on is unavailable in Lite mode |
| `LiteConversionError` | `LiteExportError` | TorchAir conversion fails in Lite mode or produces fallback / unannotated nodes |

---

## Exception handling

### Handling conversion failures

```python
from torch_delegate import export_to_epair, epair_save

try:
    contents = export_to_epair(ep)
    epair_save(contents, "model.epair")
except RuntimeError as e:
    print(f"Conversion failed: {e}")
    # Check which operators were not converted
    for node in ep.graph.nodes:
        if node.op == "call_function":
            print(f"  Operator: {node.target}")
```

### Handling load failures

```python
from torch_delegate import epair_load

try:
    contents = epair_load("model.epair")
except FileNotFoundError:
    print("File not found; run export_to_epair first")
except KeyError as e:
    print(f"Corrupted file, missing: {e}")
```

---

## Logging control

Control log output through environment variables:

```bash
# Enable debug logging
export TORCH_DELEGATE_DEBUG=1
python my_script.py
```

---

## Related documentation

- [epair_format_design.md](../features/epair_format_design.md) — Detailed design of the `.epair` format
- [build_epair.md](../guide/build_epair.md) — User guide (with a complete workflow example)
- [architecture.md](../features/architecture.md) — Overall architecture overview
