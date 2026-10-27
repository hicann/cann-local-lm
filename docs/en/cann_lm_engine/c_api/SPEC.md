---
upstream: 基础需求和约束.md
upstream_version: v0.9
spec_version: 0.1.0
status: draft
module: c_api
depends_on: pipeline+engine_executor/SPEC.md (v0.4.0)
---

# C API Specification (SPEC)

English | [简体中文](../../../zh/cann_lm_engine/c_api/SPEC.md)

> **Module positioning**: C API is the stable ABI boundary layer of the Engine, wrapping the C++ API (`EngineExecutor` / `Context` / `RequestHandle`) and exposing it externally via `extern "C"` functions + opaque handles. It does not replace the C++ API; it provides a **compiler-agnostic, ABI-stable** entry point for cross-language bindings (Python / Rust / Go / Java) and service wrappers.

---

## Chapter 1 Finalization Purpose

Define the What / Why of the C API: what capabilities it provides, why this layer is needed, and the boundary relationship with the C++ API. The How (opaque handle implementation, type mapping, memory lifecycle) is in DESIGN.

## Chapter 2 Scope

### 2.1 In Scope

- All `extern "C"` function signatures and type declarations in the C header file `c_api.h`
- 1:1 mapping of status codes, enum values to the C++ API
- Memory management conventions (library allocates / caller releases)
- Naming conventions and ABI version strategy

### 2.2 Out of Scope

- C++ API internal implementation details (PIMPL / PipelineExecutorBase / Scheduler, etc.) — see `docs/pipeline+engine_executor/`
- Low-level runtime (IModelManager) — see `docs/model_executor/`
- Tokenizer internals (TokenizerBase / BPE / SP) — see `docs/tokenizer/`
- C adaptation of multimodal Prompt (ModalityEmbedding / Tensor handle) — outside first phase scope, future extension
- Specific implementations of Python / Rust / Go bindings — done independently after C API is ready

## Chapter 3 Expectations

Add a layer of C ABI stable boundary above the C++ API:

- **ABI stability**: All exported symbols declared via `extern "C"`, symbol names do not vary with C++ compiler version (no name mangling), compatible across GCC / Clang / MSVC.
- **Cross-language binding**: Python (ctypes / cffi), Rust (bindgen), Go (cgo), Java (JNI) can directly call the C API without a C++ compiler.
- **Zero intrusion**: C API is a thin wrapper of the C++ API, does not change any C++ API signatures or behavior; C++ API remains independently usable.
- **Exception safety**: C boundary forbids exception propagation, all C++ → C crossings are guarded by `try/catch(...)`, exceptions mapped to `CANN_LM_ENGINE_FAILURE`.
- **Memory clarity**: `Result.token_ids` and `Result.text` are allocated by library `malloc`, released by caller via `CANN_LMEngine_ResultDestroy`; opaque handles managed by paired `Create` / `Destroy`.

## Chapter 4 Functional Requirements

### 4.1 Behavior Specification

C API behavior semantics are **completely consistent with the C++ API** (see `pipeline+engine_executor/SPEC.md` Chapter 4), the only difference is the interface expression form (C functions + opaque handles vs C++ class members).

#### 4.1.1 Executor Lifecycle

| Behavior | C Function | C++ Equivalent | Semantics |
|------|--------|----------|------|
| Path-load construction | `CANN_LMEngine_ExecutorCreate` | `EngineExecutor(path, path, opts)` | Success returns `SUCCESS` + `*out` points to valid handle; failure returns `FAILURE`, `*out = NULL` |
| Buffer-load construction | `CANN_LMEngine_ExecutorCreateFromBuffer` | `EngineExecutor(artifact, artifact, opts)` | Same as above |
| Destruction | `CANN_LMEngine_ExecutorDestroy` | `~EngineExecutor()` | Releases Executor and its held Context table; passing NULL is safe no-op |

#### 4.1.2 Context Management

| Behavior | C Function | C++ Equivalent | Semantics |
|------|--------|----------|------|
| Create | `CANN_LMEngine_ContextCreate` | `CreateContext(Context*)` | On success `*out` receives `ContextId` (non-zero), `SUCCESS`; on failure returns C++ corresponding status code, `*out = 0` |
| Destroy | `CANN_LMEngine_ContextDestroy` | `DestroyContext(ContextId)` | Idempotent (already destroyed returns `SUCCESS`) |
| Cancel | `CANN_LMEngine_ContextCancel` | `CancelContext(ContextId)` | Cancels all in-flight requests on that Context; thread-safe |

> **Context representation**: C API directly uses `CANN_LMEngine_ContextId` (`uint64_t`) to identify Context, does not introduce an additional opaque Context handle. The Context's internal state (KV handle / history, etc.) is carried by the `Context` C++ object held by the Executor wrapper.

#### 4.1.3 Inference

| Behavior | C Function | C++ Equivalent | Semantics |
|------|--------|----------|------|
| Synchronous generation | `CANN_LMEngine_ExecutorGenerate` | `Generate(Prompt, Context, SamplingParams, Result*)` | Blocks until completion; `Result` internal pointers allocated by library |
| Asynchronous generation | `CANN_LMEngine_ExecutorGenerateAsync` | `GenerateAsync(...)` | Returns `Request` handle immediately; non-streaming when callback is NULL |

> **Prompt**: First phase C API only accepts `const char* prompt_text` (pure text). C adaptation of multimodal embedding (requires Tensor handle design) is deferred to future versions.

> **SamplingParams**: C struct uses pointer + count to express array fields (`stop_token_ids` / `stop_strings`), avoiding C++ `std::vector`.

#### 4.1.4 Request Query and Control

| Behavior | C Function | C++ Equivalent | Semantics |
|------|--------|----------|------|
| Query status | `CANN_LMEngine_RequestGetStatus` | `RequestHandle::GetStatus()` | Returns `RequestStatus` enum; NULL handle returns `QUEUED` (safe default) |
| Is terminal state | `CANN_LMEngine_RequestIsDone` | `RequestHandle::IsDone()` | Returns `int` (0 = non-terminal, 1 = terminal) |
| Single request cancel | `CANN_LMEngine_RequestCancel` | `RequestHandle::Cancel()` | Idempotent; NULL handle returns `SUCCESS` |
| Blocking wait | `CANN_LMEngine_RequestWait` | `RequestHandle::Wait(Result*)` | Terminal semantics same as C++: `SUCCESS`/`FAILURE` fills `Result`, `CANCELED` does not fill |
| Non-blocking get | `CANN_LMEngine_RequestTryGetResult` | `RequestHandle::TryGetResult(Result*)` | Non-terminal returns `INVALID_STATE` |
| Destruction | `CANN_LMEngine_RequestDestroy` | `~RequestHandle()` | Releases handle; passing NULL is safe no-op |

#### 4.1.5 Result Memory Management

| Behavior | C Function | Semantics |
|------|--------|------|
| Release | `CANN_LMEngine_ResultDestroy` | `free(result->token_ids)` + `free(result->text)`, set NULL; safe no-op on already-zeroed Result |

> **Ownership convention**: `CANN_LMEngine_Result` is a stack-allocated POD struct (caller declares), but its internal `token_ids` and `text` pointers are allocated by library `malloc`. The caller **must** call `ResultDestroy` to release after use, otherwise memory leak. `finish_reason` is a value type, no release needed.

### 4.2 Streaming Callback

C API uses C function pointer + `user_data` to replace C++ `std::function<void(TokenId, std::string)>`:

```c
typedef void (*CANN_LMEngine_StreamCallback)(
    CANN_LMEngine_TokenId  token,
    const char*            text,
    size_t                text_len,
    void*                 user_data
);
```

- The `text` pointer is only valid during the callback invocation (points to C++ `std::string` internal buffer), may be invalid after callback returns — caller should not save this pointer.
- Callback is triggered on pipeline internal thread, caller **should not** block or call Engine synchronous API within callback (same as C++ API constraint).
- `text_len` is guaranteed to equal `strlen(text)` (text is null-terminated).

## Chapter 5 Non-Functional Requirements

| Metric | Requirement | Evidence Source |
|------|------|----------|
| ABI stability | **MUST**: all exported symbols are `extern "C"`, symbol names do not vary with compiler version | C API design constraint |
| Header purity | **MUST**: `c_api.h` only includes `<stddef.h>` / `<stdint.h>`, does not depend on any C++ standard library header | C API design constraint |
| Exception safety | **MUST**: C boundary forbids exception propagation, all crossings via `try/catch(...)` → `FAILURE` | C API design constraint |
| Memory safety | **MUST**: `Result` internal pointers paired with `ResultDestroy`; opaque handles paired with `Destroy`; NULL parameters handled safely | C API design constraint |
| Naming consistency | **MUST**: all function names start with `CANN_LMEngine_` prefix, classified by `Executor` / `Context` / `Request` / `Result` | C API design constraint |
| Status code consistency | **MUST**: C `int` status code values 1:1 correspond to C++ `Status` constant values (0–5) | C API design constraint |
| Thread safety | Same as C++ API: `GenerateAsync` can be concurrent across threads; same ContextId concurrent returns `INVALID_STATE` | pipeline+engine_executor SPEC §5 |
| Latency | C API wrapper overhead **should** be negligible (one layer of function pointer forwarding + type conversion) | — |

## Chapter 6 Constraints and Dependencies

### 6.1 Behavior Constraints

| Constraint | Requirement | Evidence Source |
|------|------|----------|
| ABI stability | Exported symbols via `extern "C"`; symbol set only adds, never modifies or deletes (backward compatible) | C API core constraint |
| Exception boundary | C functions use `try/catch(...)` to catch all C++ exceptions, map to `FAILURE`; C side has no exception concept | C API core constraint |
| Memory convention | `Result` internal pointers library-allocated + caller-released; handles paired Create/Destroy | C API core constraint |
| Header isolation | `c_api.h` does not include C++ headers; does not expose C++ types (`std::string` / `std::vector`, etc.) | C API core constraint |
| Behavior equivalence | C API behavior semantics completely consistent with C++ API; difference only in expression form | pipeline+engine_executor SPEC |

### 6.2 Error Classification Table

| Failure Scenario | Return Status Code | Description |
|----------|-----------|------|
| NULL parameter | `INVALID_ARGUMENT` (2) | `executor` / `prompt_text` / `out` and other required parameters are NULL |
| Context does not exist | `NOT_FOUND` (4) | `context_id` not found in Executor's Context table |
| Same Context concurrent | `INVALID_STATE` (3) | Same as C++ API |
| Construction failure | `FAILURE` (1) | Model loading failure and other internal errors |
| C++ exception escape | `FAILURE` (1) | `try/catch(...)` fallback |
| Cancel | `CANCELED` (5) | Same as C++ API |
| Normal completion | `SUCCESS` (0) | Same as C++ API |

### 6.3 Software Dependencies

- C11 compiler (caller)
- C++17 compiler (library build side, `c_api.cpp` compiled with C++)
- Link `libcann_lm_engine.a` (same static library, contains both C++ API and C API symbols)

## Chapter 7 Backward Compatibility

### 7.1 ABI Version Strategy

- Current version is C API v1 (first version)
- **Backward compatibility principle**: existing function signatures and type definitions are not modified or deleted; new functionality extends via new functions / new types / new enum values
- If breaking changes are needed, create a new v2 header (`c_api_v2.h`) or version prefix (`CANN_LMEngine2_`), not affecting v1 consumers

### 7.2 Relationship with C++ API Version

- C API follows C++ API version evolution, but C API ABI is not broken by C++ API implementation changes
- New C++ API methods do not automatically appear in C API — C API needs to explicitly add corresponding C functions

## Chapter 8 Verification Cases

| Case ID | Given / When / Then |
|---------|---------------------|
| TC_C_001 | **Given** pure C compiler (`cc`) **When** compile `c_api_smoke.c` and link `libcann_lm_engine.a` **Then** compile and link succeed, run exit 0 |
| TC_C_002 | **Given** `NULL` parameter **When** call `ExecutorCreate(NULL, NULL, NULL, NULL)` **Then** returns `INVALID_ARGUMENT` |
| TC_C_003 | **Given** valid path **When** call `ExecutorCreate` **Then** returns `SUCCESS`, `*out` non-null |
| TC_C_004 | **Given** stub implementation **When** call `ContextCreate` **Then** returns `FAILURE` (stub expected) |
| TC_C_005 | **Given** non-existent `context_id` **When** call `ExecutorGenerate` **Then** returns `NOT_FOUND` |
| TC_C_006 | **Given** zero-initialized `Result` **When** call `ResultDestroy` **Then** safe no-op, internal pointers remain NULL |
| TC_C_007 | **Given** `NULL` handle **When** call `RequestGetStatus(NULL)` **Then** returns `QUEUED` (safe default) |
| TC_C_008 | **Given** `NULL` handle **When** call `RequestCancel(NULL)` **Then** returns `SUCCESS` (idempotent) |
| TC_C_009 | **Given** `NULL` handle **When** call `ExecutorDestroy(NULL)` **Then** safe no-op |
| TC_C_010 | **Given** C API produces `Result` **When** read `result.text` and `result.token_ids` **Then** data valid; after calling `ResultDestroy` pointers become NULL |

### 8.1 Verification Environment

- Compiler: GCC / Clang (C + C++ mixed compilation)
- Build system: CMake (`LANGUAGES C CXX`)
- Verification method: `c_api_smoke.c` (pure C compiled smoke test, `-Wall -Wextra -Wpedantic`)

---

## Change Log

| Version | Date | Changes |
|------|------|----------|
| 0.1.0 | 2026-08-14 | Initial version: C API specification finalized. Defines 15 `extern "C"` functions (Executor 3 + Context 3 + Inference 2 + Request 6 + Result 1), opaque handle types, POD struct types, status code 1:1 mapping, memory management conventions (library allocates + caller ResultDestroy), naming conventions (`CANN_LMEngine_` prefix classified by Executor/Context/Request/Result), ABI version strategy, 10 verification cases. Links pipeline+engine_executor SPEC v0.4.0, base v0.9. |
| 0.1.1 | 2026-08-14 | Directory restructure: code path `c_api/` → `api/c/`, C++ API `api/` → `api/cpp/`. §3 file paths, §4 code example include paths synchronized. Functionality unchanged, ABI unchanged. |
