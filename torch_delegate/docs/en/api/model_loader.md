# EpairModelLoader (Python API)<a id="section_python_epair_model_loader"></a>

## Overview

`torch_delegate.EpairModelLoader` loads a `.epair` archive and builds a GE graph for GE Offline_compile. This class is the Python wrapper of the C++ `td::EpairModelLoader` (via ctypes + C ABI), covering only the "fetch GE graph" offline compilation path; for execution-stage interfaces (`CompileAndLoad` / `RunModel`, etc.) use the [C++ API](epair_model_loader.md).

**Import:**

```python
from torch_delegate import EpairModelLoader, EpairModelLoaderError
```

## Class `EpairModelLoader`

```python
class torch_delegate.EpairModelLoader(epair_file)
```

On construction, it creates the C++ `td::EpairModelLoader` through the C ABI and completes archive parsing and GE graph construction (no compilation); `get_ge_graph()` returns an independently owned GE graph copy on every call. The instance manages its own lifetime (construction / `close()` / destructor); multi-threaded concurrent use of the same instance is not guaranteed to be safe.

### `__init__(epair_file)`

Loads the backend library, creates the C++ loader, and completes `.epair` parsing and GE graph construction (no compilation); raises on failure.

**Signature:**

```python
def __init__(self, epair_file: str | os.PathLike[str]) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|------|------|------|
| `epair_file` | `str \| os.PathLike[str]` | Path of the `.epair` file; must be non-empty |

**Exceptions:**

| Exception type | Trigger condition |
|---------|---------|
| `TypeError` | `epair_file` is not `str` or `os.PathLike[str]` |
| `ValueError` | `epair_file` is an empty path, or the C++ side returns `INVALID_PARAM` |
| `EpairModelLoaderError` | Backend library failed to load (including old whl versions without the C ABI), GE is not initialized, archive corrupted / graph construction failed (`FAILED`), internal error (`INTERNAL_ERROR`) |

> GE must be initialized (`ge.offline_compile.build_initialize()`) before constructing the loader; without initialization the C ABI usually returns `FAILED`, surfaced as `EpairModelLoaderError`.

Exception messages contain the status code name and the thread-local error string, allowing precise fault localization.

### `get_ge_graph()`

Returns an independent copy of the internal GE graph; can be called repeatedly without affecting other copies.

**Signature:**

```python
def get_ge_graph(self) -> ge.graph.Graph: ...
```

**Returns:** `ge.graph.Graph` — a graph object from the GraphEngine Python API shipped with CANN, ready to be fed into `ge.offline_compile.build_model` for offline compilation.

**Exceptions:**

| Exception type | Trigger condition |
|---------|---------|
| `EpairModelLoaderError` | Fetching the graph after `close`, the GraphEngine Python API being unavailable / empty graph creation failing, the API not exposing the native handle, or the C++ side failing to copy the graph |

### `close()`

Explicitly destroys the C++ loader and releases the `.epair` parsing and GE graph construction artifacts; repeated calls are idempotent (no-op).

**Signature:**

```python
def close(self) -> None: ...
```

### `with` context manager

Supports the `with` statement: `close()` is called automatically on context exit; the destructor calls `close()` as a fallback, guaranteeing no leaks.

**Signature:**

```python
def __enter__(self) -> EpairModelLoader: ...
def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

## Example: initialize GE and compile offline

```python
import torch_delegate
from ge.offline_compile import (
    build_finalize,
    build_initialize,
    build_model,
    save_model,
)

build_initialize({"ge.socVersion": "Ascend910B", "ge.graphRunMode": "0"})
try:
    with torch_delegate.EpairModelLoader("model.epair") as loader:
        ge_graph = loader.get_ge_graph()
        model = build_model(ge_graph)
        save_model("./model", model)
finally:
    build_finalize()
```

**Notes:**

- The example is a runnable minimal flow: call `build_initialize()` to initialize GE before constructing `EpairModelLoader`; `ge.socVersion` must match the target deployment SoC, and `build_finalize()` releases GE resources when the flow finishes
- `get_ge_graph()` returns an object from the GraphEngine Python API shipped with CANN (`ge.graph.Graph`, bridged to the backend C ABI via empty graph creation and native pointer access); this API is installed with CANN under `python/site-packages`. When CANN is not installed or the `ge` module is not importable, calls raise `EpairModelLoaderError` with a clear message
- No guarantee of multi-threaded concurrency safety for the same instance (same as the C++ side); error strings are thread-isolated
- Old whl versions do not export the C ABI symbols; calls fail with a clear `EpairModelLoaderError` at load time — reinstall a matching torch_delegate version
