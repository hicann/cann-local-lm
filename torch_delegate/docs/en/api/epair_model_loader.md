# EpairModelLoader<a id="ZH-CN_TOPIC_TD_EPAIR_MODEL_LOADER"></a>

## Overview<a id="section_overview"></a>

`EpairModelLoader` is the core C++ class of TorchDelegate. It encapsulates the complete workflow of parsing a `.epair` archive, building a GE Graph, compiling, and executing. The class uses the PIMPL pattern to hide implementation details and keeps a stable ABI.

TorchDelegate also provides a [Python API](python_api.md) for model export, saving, and loading, including conversion of PyTorch `ExportedProgram` into `.epair` files.

Header and library files are distributed with the whl. After installation they are located at `<site-packages>/torch_delegate/inc/torch_delegate/` and `<site-packages>/torch_delegate/lib/`, and the CMake configuration package is at `<site-packages>/torch_delegate/lib/cmake/TorchDelegate/`. C++ projects integrate through `find_package(TorchDelegate)`; the prefix is obtained via `torch-delegate-config --cmake-prefix` (or `torch_delegate.cmake_prefix()`). `LD_LIBRARY_PATH` must be configured manually at runtime. See the [Build guide](../guide/build.md).

**Data flow:**

```text
PyTorch Model
    │ torch.export.export()
    ▼
ExportedProgram
    │ [Python] export_to_epair()
    ▼
EpairContents
    │ [Python] epair_save()
    ▼
.epair file (ZIP Archive)
    │ [C++] EpairModelLoader(archive_path)
    ▼
Compile & NPU inference
    │ CompileAndLoad() → RunModel() / RunModelWithStreamAsync()
    ▼
Inference results
```

---

## C++ API<a id="section_cpp_api"></a>

### Class index<a id="section_class_index"></a>

- **[Status codes](#section_status)**
- **[EpairModelLoader constructor](#section_constructor)**
- **[CompileAndLoad](#section_compile_and_load)**
- **[RunModel](#section_run_model_sync)**
- **[RunModelWithStreamAsync](#section_run_model)**
- **[GetGEGraph](#section_get_ge_graph)**

---

### Status codes<a id="section_status"></a>

#### Header/library<a id="section_status_files"></a>

- Header: `#include <torch_delegate/td_types.h>`
- Library: `libtorch_delegate_backend.so`

#### Description<a id="section_status_desc"></a>

`td::Status` is a type alias of `int64_t` used for return values of the TorchDelegate C++ API.

#### Status code definitions<a id="section_status_def"></a>

<a id="table_status"></a>
| Constant | Value | Description |
| --- | --- | --- |
| `td::SUCCESS` | 0 | Operation succeeded |
| `td::FAILED` | 1 | Generic failure |
| `td::INVALID_PARAM` | 2 | Invalid parameter |
| `td::INTERNAL_ERROR` | 3 | Internal error |
| `td::UNIMPLEMENTED` | 4 | Feature not implemented |

---

### EpairModelLoader constructor<a id="section_constructor"></a>

#### Header/library<a id="section_constructor_files"></a>

- Header: `#include <torch_delegate/epair_model_loader.h>`
- Library: `libtorch_delegate_backend.so`

#### Description<a id="section_constructor_desc"></a>

Constructs a model loader from a `.epair` archive. Construction immediately parses the archive contents and builds the GE Graph, but does not compile. If initialization fails, the error state is recorded internally and no exception is thrown.

#### Prototype<a id="section_constructor_proto"></a>

```cpp
explicit EpairModelLoader(const char *archive_path)
```

#### Parameters<a id="section_constructor_params"></a>

<a id="table_constructor"></a>
| Parameter | Input/Output | Description |
| --- | --- | --- |
| archive_path | Input | Path to the `.epair` archive file; must be non-null. Relative and absolute paths are supported; a relative path is resolved against the process's current working directory (CWD). |

#### Constraints<a id="section_constructor_constraints"></a>

- `archive_path` must point to a valid `.epair` file (a ZIP archive produced by `epair_save()`); relative and absolute paths are supported
- The constructor does not throw; if initialization fails, subsequent method calls return non-`SUCCESS` error codes (`INVALID_PARAM` or `FAILED`)

---

### CompileAndLoad<a id="section_compile_and_load"></a>

#### Header/library<a id="section_compile_and_load_files"></a>

- Header: `#include <torch_delegate/epair_model_loader.h>`
- Library: `libtorch_delegate_backend.so`

#### Description<a id="section_compile_and_load_desc"></a>

Compiles and loads the model onto the specified stream, preparing it for graph execution, including allocating and managing the memory and compute stream resources required to run the graph. If called again after the model has been compiled and loaded, the interface returns `SUCCESS` directly.

`GEInitializeV2` must be initialized before this interface is called. Once CompileAndLoad succeeds, inference can be performed with `RunModelWithStreamAsync`.

#### Prototype<a id="section_compile_and_load_proto"></a>

```cpp
Status CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream)
```

#### Parameters<a id="section_compile_and_load_params"></a>

<a id="table_compile_and_load"></a>
| Parameter | Input/Output | Description |
| --- | --- | --- |
| options | Input | Compile options; keys and values are of type `ge::AscendString`. In most cases an empty `{}` can be passed. For supported configuration items, see [options parameter description](https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/options_parameters_description.md). |
| stream | Input | ACL stream pointer, created via `aclrtCreateStream`. If used together with `RunModelWithStreamAsync`, pass a valid value and keep it consistent with the stream used for subsequent execution; if they are not the same stream, call `aclrtSynchronizeStream` after loading to synchronize the stream. May be `nullptr`, in which case an internal default stream is used. |

#### Return value<a id="section_compile_and_load_ret"></a>

<a id="table_compile_and_load_ret"></a>
| Parameter | Type | Description |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`: compilation and loading both succeeded<br>`td::FAILED`: initialization failed (archive parsing or Graph construction failed) / Session creation failed / any of AddGraph / CompileGraph / LoadGraph failed<br>`td::INVALID_PARAM`: a null path was passed to the constructor |

#### Constraints<a id="section_compile_and_load_constraints"></a>

- `GEInitializeV2` must be initialized before calling
- The `stream` should be the same one used by subsequent `RunModelWithStreamAsync` calls; if not, call `aclrtSynchronizeStream` on the stream after `CompileAndLoad` to synchronize
- The method has two stages: compile and load. The compile stage does not allocate memory resources needed for model execution; the load stage allocates and manages memory and compute stream resources
- The lifetime of the compiled model is bound to the `EpairModelLoader` instance and is cleaned up automatically on destruction

---

### RunModel<a id="section_run_model_sync"></a>

#### Header/library<a id="section_run_model_sync_files"></a>

- Header: `#include <torch_delegate/epair_model_loader.h>`
- Library: `libtorch_delegate_backend.so`

#### Description<a id="section_run_model_sync_desc"></a>

Runs the model synchronously and outputs the execution results. The interface blocks until graph execution completes; on return the output data is ready, with no further stream synchronization needed. Input tensor data may reside on the Host or Device side; the caller does not need to allocate memory for output tensors, as GE allocates Host memory automatically after execution and returns the values. This interface must be called after `CompileAndLoad` succeeds and wraps the GE synchronous interface `RunGraph`.

#### Prototype<a id="section_run_model_sync_proto"></a>

```cpp
Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)
```

#### Parameters<a id="section_run_model_sync_params"></a>

<a id="table_run_model_sync"></a>
| Parameter | Input/Output | Description |
| --- | --- | --- |
| inputs | Input | List of input tensors. Each `gert::Tensor` must have `Shape`, `DataType`, and `Format` set correctly. Data may reside on the Host or Device side (Device-side memory is allocated via `aclrtMalloc`). If the data is on the Host side while the graph executes on the Device, each input is allocated and copied once when it enters the data queue, which may become a performance bottleneck for large or numerous inputs. |
| outputs | Output | List of output tensors. The caller does not need to allocate memory; pass an empty vector, and GE allocates Host memory automatically after execution and returns the values. |

#### Return value<a id="section_run_model_sync_ret"></a>

<a id="table_run_model_sync_ret"></a>
| Parameter | Type | Description |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`: run succeeded<br>`td::FAILED`: initialization failed (archive parsing or Graph construction failed) / model not loaded (`CompileAndLoad` not called or failed) / `RunGraph` failed / mixed with `RunModelWithStreamAsync` (the same instance has already called the asynchronous interface, so the call is rejected with an error)<br>`td::INVALID_PARAM`: a null path was passed to the constructor |

#### Constraints<a id="section_run_model_sync_constraints"></a>

- `CompileAndLoad` must have been called successfully first; this interface does not perform loading automatically
- The interface executes synchronously and blocks until graph execution completes; outputs can be read directly after return without calling `aclrtSynchronizeStream`
- This interface is mutually exclusive with `RunModelWithStreamAsync`; synchronous and asynchronous execution must not be mixed for the same model. If `RunModelWithStreamAsync` has already been called on the same instance, this interface returns `FAILED` directly and logs a clear message about the exclusivity (the GE layer `GeSession` also checks the execution mode; this interface intercepts the call earlier in `EpairModelLoader` to produce a friendly error)
- The same instance does not support concurrent calls from multiple threads; do not use the same instance concurrently across threads
- When input tensor data resides on the Device side, its Device-side storage address must be 32-byte aligned, otherwise undefined errors may occur
- Input memory must be allocated by the caller before the call and released by the caller afterwards
- Output memory is allocated by GE on the Host side and returned with values; the caller does not need to preallocate it

---

### RunModelWithStreamAsync<a id="section_run_model"></a>

#### Header/library<a id="section_run_model_files"></a>

- Header: `#include <torch_delegate/epair_model_loader.h>`
- Library: `libtorch_delegate_backend.so`

#### Description<a id="section_run_model_desc"></a>

Runs the model asynchronously and outputs the execution results. Both input and output tensor data reside on the Device side. This interface must be called after `CompileAndLoad` succeeds.

#### Prototype<a id="section_run_model_proto"></a>

```cpp
Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs)
```

#### Parameters<a id="section_run_model_params"></a>

<a id="table_run_model"></a>
| Parameter | Input/Output | Description |
| --- | --- | --- |
| stream | Input | Specifies the Stream on which the graph runs. |
| inputs | Input | List of input tensors. Each `gert::Tensor` must have `Shape`, `DataType`, and `Format` set correctly, and the data resides on the Device side (allocated via `aclrtMalloc`). |
| outputs | Output | List of output tensors. Pass an empty vector; GE allocates Device memory with the built-in allocator, and the lifetime matches the graph. It must be released explicitly via `outputs.clear()` before the graph is unloaded (before `EpairModelLoader` is destroyed); if Tensors with allocated memory are passed in, GE uses that memory to store outputs, and the allocated output memory must be large enough to hold the model output data. |

#### Return value<a id="section_run_model_ret"></a>

<a id="table_run_model_ret"></a>
| Parameter | Type | Description |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`: run succeeded<br>`td::FAILED`: initialization failed (archive parsing or Graph construction failed) / model not loaded (`CompileAndLoad` not called or failed) / `RunGraphWithStreamAsync` failed / mixed with `RunModel` (the same instance has already called the synchronous interface, so the call is rejected with an error)<br>`td::INVALID_PARAM`: a null path was passed to the constructor |

#### Constraints<a id="section_run_model_constraints"></a>

- `CompileAndLoad` must have been called successfully first; this interface does not perform loading automatically
- The Device-side storage addresses of both input and output Tensors must be 32-byte aligned, otherwise undefined errors may occur
- Input memory must be allocated before the call and is released by the caller afterwards
- If `outputs` is empty, output memory is allocated by GE's built-in allocator and its lifetime matches the graph; it must be released explicitly via `outputs.clear()` before the graph is unloaded. External allocators are not supported currently
- The interface returns asynchronously; outputs can only be read after synchronizing the stream via `aclrtSynchronizeStream`. If you want to read outputs immediately after the call returns (synchronous semantics), use the [RunModel](#section_run_model_sync) interface
- This interface is mutually exclusive with [RunModel](#section_run_model_sync); synchronous and asynchronous execution must not be mixed for the same model. If `RunModel` has already been called on the same instance, this interface returns `FAILED` directly and logs a clear message about the exclusivity

---

### GetGEGraph<a id="section_get_ge_graph"></a>

#### Header/library<a id="section_get_ge_graph_files"></a>

- Header: `#include <torch_delegate/epair_model_loader.h>`
- Library: `libtorch_delegate_backend.so`

#### Description<a id="section_get_ge_graph_desc"></a>

Gets a copy of the internal GE Graph. The method copies the internally built GE Graph via `CopyFrom`; the caller can add the Graph to a `ge::GeSession` and execute it, enabling finer-grained control (for example, managing the Session lifetime itself or adding multiple Graphs).

#### Prototype<a id="section_get_ge_graph_proto"></a>

```cpp
Status GetGEGraph(ge::Graph *graph) const
```

#### Parameters<a id="section_get_ge_graph_params"></a>

<a id="table_get_ge_graph"></a>
| Parameter | Input/Output | Description |
| --- | --- | --- |
| graph | Output | Output parameter; must be non-null. This interface returns a copy of the internal Graph, and the caller takes ownership of the returned Graph. |

#### Return value<a id="section_get_ge_graph_ret"></a>

<a id="table_get_ge_graph_ret"></a>
| Parameter | Type | Description |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`: retrieval succeeded<br>`td::FAILED`: initialization failed (archive parsing or Graph construction failed) / internal Graph is empty / copy failed<br>`td::INVALID_PARAM`: a null path was passed to the constructor |

#### Constraints<a id="section_get_ge_graph_constraints"></a>

- The constructor must have completed successfully (model initialization succeeded)
- The `graph` pointer must be non-null and must be a valid Graph object
- The returned Graph is a copy; the caller owns it
- **Input shapes may differ from the EP JSON**: after the internal GE Graph is built, the SymPack optimization pass runs (merging redundant Pack nodes, removing dead Data, and reordering Data indices). At this point the number, index, and shape of Data nodes in the graph may all differ from the original input definitions in the EP JSON. To obtain the actual input specifications after optimization (Data node count, index, shape, dtype, placement), iterate over nodes with `type == "Data"` via `ge::Graph::GetAllNodes()` and read their `index` attribute and `TensorDesc`, or export the graph structure with the GE Dump tool. **Do not construct `RunGraph` inputs directly from the input signature in the EP JSON.** For details on how the SymPack optimization pass affects input structure and on the runtime input adaptation mechanism, see [SymPack merge optimization and AssertTensorMetadata filtering](../features/sympack_merge_20260710.md)

---
## Related documentation<a id="section_references"></a>

- [Python API reference](python_api.md) — Complete Python-side API documentation
- [Build guide](../guide/build.md) — Environment setup and build instructions
- [epair format description](../guide/build_epair.md) — `.epair` file structure and user guide
