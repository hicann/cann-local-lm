# EpairModelLoader Synchronous Execution Interface (RunModel)

> [!NOTE] Document Status
> This is a design draft dated 2026-09-24 that describes the EpairModelLoader synchronous execution interface. For the current interface documentation, see [EpairModelLoader API](../api/epair_model_loader.md).

## Introduction

### Purpose

Add a synchronous execution interface `RunModel` to `td::EpairModelLoader`, wrapping GE's `ge::GeSession::RunGraph`. The interface blocks until graph execution completes, and output data is ready upon return, so users no longer need to call `aclrtSynchronizeStream` manually to synchronize the stream, lowering the barrier to entry on the C++ deployment side.

### Scope

| File | Change |
|------|----------|
| `csrc/inc/external/torch_delegate/epair_model_loader.h` | Add `RunModel` interface declaration (ABI boundary, additive only) |
| `csrc/backend/epair_model_loader.cc` | Add `Impl::RunModel` implementation and outer forwarding |
| `tests/cpp/ge_stub/ge_stub.cc` | Add `ge::GeSession::RunGraph` stub implementation |
| `tests/cpp/test_epair_model_loader.cc` | Add UT cases for the synchronous interface |
| `docs/en/api/epair_model_loader.md` | Add a RunModel interface description section |
| `examples/run_model_multi_instance` | Multi-threaded multi-instance end-to-end example (producer-consumer, 4 instances on 4 threads calling `RunModel`) |

---

## Overall Summary

### Software Overview

Currently `EpairModelLoader` only exposes the asynchronous execution interface `RunModelWithStreamAsync` (corresponding to `ge::GeSession::RunGraphWithStreamAsync`). After the call returns, the output is not yet ready; the user must call `aclrtSynchronizeStream` on the stream themselves before reading the output. The GE layer already provides the synchronous interface `RunGraph` (blocks until graph execution completes), but the td layer had not wrapped it. This feature fills in the synchronous execution semantics, forming a one-to-one correspondence with the GE API:

| GE Interface | Semantics | td Wrapper Interface |
|---------|------|-------------|
| `RunGraph` | Synchronous (blocks until graph execution completes) | `RunModel` (added by this feature) |
| `RunGraphWithStreamAsync` | Asynchronous (specified stream) | `RunModelWithStreamAsync` (existing) |

### Software Functions

```cpp
Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);
```

- Runs the model synchronously, blocking until graph execution completes before returning; `outputs` can be read directly after return
- Input tensor data may reside on the Host side or the Device side (Host inputs incur one memory allocation and copy when entering the data queue)
- The caller does not need to allocate memory for output tensors; after execution, GE automatically allocates Host memory and assigns and returns it (aligned with the GE `RunGraph` capability)

### Design Constraints

- **ABI boundary**: `epair_model_loader.h` is an external ABI boundary. `RunModel` uses only already-exposed types (`std::vector<gert::Tensor>`, `Status`) and is a new non-virtual member function (a PIMPL class), so it does not break the existing ABI
- **Precondition state checks**: Consistent with the asynchronous interface — when `init_status_ != SUCCESS`, the initialization error code is passed through; when the model is not loaded (`CompileAndLoad` not called or failed), `FAILED` is returned and an ERROR log is printed
- **SymPack input adaptation reuse**: When `input_binding_map_` is non-empty, `BuildGeInputs` is reused to bind/pack user inputs into GE inputs, with behavior identical to the asynchronous interface
- **No stream parameter**: Aligned with the GE `RunGraph` semantics, no additional stream parameter or synchronization combination logic is introduced, avoiding over-design
- **Mutual exclusion constraint**: Aligned with GE `RunGraph`, `RunModel` and `RunModelWithStreamAsync` are mutually exclusive; the same model cannot mix synchronous and asynchronous interfaces. GE `GeSession` has an internal execution mode check (`CheckRunMode`), but its error is hard to diagnose; therefore a defensive check is added at the `EpairModelLoader` layer — `Impl` records the interface mode (synchronous/asynchronous) of the first successful execution, after which a call to the other interface is intercepted before reaching GE, returning `FAILED` and printing a clear log explaining that the two interfaces are mutually exclusive, with UT cases added
- **Concurrency constraint**: This interface wraps GE's synchronous `RunGraph`; multi-instance concurrency behavior is governed by that GE interface; the same instance does not support multi-threaded concurrent calls

---

## Feature 1 RunModel Synchronous Execution Interface

### Overview

`RunModel` is implemented in `EpairModelLoader::Impl`. The execution path is consistent with `RunModelWithStreamAsync`, except that the underlying GE call is replaced with the synchronous `RunGraph`:

```text
RunModel(inputs, outputs)
    │ Check init_status_ / loaded_flag_ / run_mode_ (mutually exclusive with the async interface; mixing them is rejected with an error)
    ▼
input_binding_map_ empty?
    ├─ Yes → session_->RunGraph(graph_id_, inputs, outputs)
    └─ No  → BuildGeInputs() constructs ge_inputs
            → session_->RunGraph(graph_id_, ge_inputs, outputs)
    ▼
ge::Status → td::Status mapping (print TDLOGE on failure)
```

### Functional Requirement 1 Synchronous Execution

- After a successful `CompileAndLoad`, call `RunModel`; the interface blocks until graph execution completes, returns `SUCCESS`, and `outputs` has been populated
- The caller does not need to pre-allocate memory for `outputs`; after execution, GE automatically allocates Host memory and assigns and returns it
- Input data may reside on the Host side or the Device side; when on the Host side while the graph executes on the Device, each input entering the data queue incurs one memory allocation and copy, which may be a performance bottleneck when inputs are large or numerous; Device-side input addresses must be 32-byte aligned

### Functional Requirement 2 Error Handling

| Scenario | Return Value | Behavior |
|------|--------|------|
| Construction failure (archive parsing / Graph construction failure) | Pass through `init_status_` (`INVALID_PARAM` / `FAILED`) | Consistent with existing interfaces |
| Model not loaded | `FAILED` | `TDLOGE` prints "model is not loaded" |
| `BuildGeInputs` input index out of range | `INVALID_PARAM` | `TDLOGE` prints the out-of-range index |
| `ge::GeSession::RunGraph` failure | `ToTdStatus(ge_status)` | `TDLOGE` prints ge_status and graph_id |

---

## Non-Functional Requirements

- **Maintainability**: `RunModel` and `RunModelWithStreamAsync` share the state checks, input adaptation (`BuildGeInputs`), and error mapping (`ToTdStatus`) logic, with no new abstraction layer
- **Testability**: The GE stub adds a `RunGraph` stub implementation, reusing the existing `SetRunGraphStatus` fault injection entry
- **Compatibility**: Only adds an interface; it does not modify the `.epair` format, existing interface signatures, or behavior; existing user code requires no changes

---

## DT Design

### Test Boundary

- **Test entry**: C++ side `td::EpairModelLoader::RunModel`
- **Test exit**: returned status code, `outputs` population result, ERROR logs
- **Stubbed module**: GE stub (`libge_runner_v2` stub, `tests/cpp/ge_stub/ge_stub.cc`), injecting execution failure via `GeApiStub::SetRunGraphStatus`

### Test Design

| Category | Test Item | Method | Case Type |
|------|--------|------|----------|
| Functional | Execution rejected when not loaded | Construct loader then call `RunModel` directly → returns `FAILED` | UT |
| Functional | Synchronous execution succeeds | `CompileAndLoad` → `RunModel` → returns `SUCCESS` and `outputs` non-empty | UT |
| Functional | Execution failure propagation | Call `SetRunGraphStatus(ge::FAILED)` then execute → returns `FAILED` | UT |
| Functional | Mutual exclusion with async interface (defensive) | On the same instance, call `RunModelWithStreamAsync` successfully first, then call `RunModel` → returns `FAILED` and prints the mutual-exclusion log | UT |
| Functional | Reverse mutual exclusion of async interface (defensive) | On the same instance, call `RunModel` successfully first, then call `RunModelWithStreamAsync` → returns `FAILED` and prints the mutual-exclusion log | UT |
| Functional | SymPack input adaptation | An epair containing `InputBindingMap` goes through the `BuildGeInputs` path, with output consistent with the async interface | ST (`examples/sym_pack_optimization_case`) |
| Functional | Host/Device-side input | Execute with Host-side and Device-side data as input respectively → both succeed with consistent results | ST (real device) |
| Functional | Automatic output memory allocation | Pass an empty vector for `outputs` → GE allocates Host memory and assigns and returns it, directly readable | ST (real device) |
| End-to-end | Sync/async output consistency | On a real device, load the same model into two instances, one executing `RunModel` and the other `RunModelWithStreamAsync` + `aclrtSynchronizeStream` (the two interfaces are mutually exclusive and cannot be mixed on the same instance); max‖diff‖=0 | ST |
| End-to-end | Multi-threaded multi-instance task distribution | Under a producer-consumer model, 4 `EpairModelLoader` instances take tasks in parallel and call `RunModel`; all tasks succeed and output checksums are consistent | ST/example (`examples/run_model_multi_instance`) |
| Compatibility | Existing interface behavior unchanged | Full regression of existing UTs passes | UT |

### Test Framework

UTs use GoogleTest (`tests/cpp/test_epair_model_loader.cc` + GE stub) and run under an ASAN build (`./build_test.sh` enables it by default); sync/async output consistency can be verified on a real device by reusing the `examples/export_and_load_example` flow, and the multi-threaded multi-instance end-to-end case is described in the example project `examples/run_model_multi_instance`.

---

## Acceptance Criteria

1. **Functional acceptance**: After a successful `CompileAndLoad`, call `RunModel`; it returns `SUCCESS` synchronously, the output can be read directly, and no additional stream synchronization is needed
2. **UT acceptance**: The 5 new UT cases (not loaded rejection / synchronous execution success / failure propagation / mutual exclusion rejection with async interface / reverse mutual exclusion rejection of async interface) all pass under the ASAN build, with no regressions in existing cases
3. **Interface documentation acceptance**: `docs/en/api/epair_model_loader.md` contains a RunModel section (function description / function prototype / parameter description / return value description / constraint description), with cross-references to the async interface
4. **ABI acceptance**: Only a new non-virtual member function is added; the existing binary interface is unaffected
5. **Example acceptance**: `examples/run_model_multi_instance` can complete model export, C++ build, and a 4-instance 4-thread producer-consumer run; all tasks succeed and output checksums are consistent
