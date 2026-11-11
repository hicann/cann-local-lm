# CANN Stub Design Document

English | [简体中文](../../zh/constraints/cann_stub.md)

## Background

Modules in this project need to call CANN/ACL runtime APIs (`aclInit`, `aclrtMalloc`, etc.). In CI environments without NPU hardware, a stub `libascendcl.so` is required so that tests are decoupled from real CANN hardware.

Core constraint: the stub's `.so` output path matches the relative path used by real CANN, so whichever library is linked, the compile and packaging artifacts are identical. There is no need for separate "stub mode" and "production mode" compile switches; only `LD_LIBRARY_PATH` at runtime decides which library is loaded.

---

## Directory Structure

The stub is test infrastructure, so it lives in `tests/cpp/cann_stub/` rather than `csrc/`.

```text
tests/cpp/
├── cann_stub/
│   ├── ascendcl.cc            # Stub implementation backing libascendcl.so
│   └── cann_stub_control.h    # Test-only control interface (Reset/query/error injection)
└── test_cann_stub.cc          # GTest unit tests
```

**File naming rule**: one `.cc` file per `.so`; the file name equals the `.so` name with the `lib` prefix removed. Currently only `libascendcl.so` exists, hence only `ascendcl.cc`. If a stub for `libacl_op_compiler.so` is needed later, add `acl_op_compiler.cc`.

---

## Shared Library Naming and Paths

Build artifact path (`LIBRARY_OUTPUT_DIRECTORY`):

```text
${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64/libascendcl.so
```

This maps to real CANN's relative path:

```text
real: <CANN_ROOT>/lib64/libascendcl.so
stub: <build>/tests/cpp/cann_stub/lib64/libascendcl.so
```

The SONAME is set to `libascendcl.so` (identical to the real library) via `OUTPUT_NAME "ascendcl"`.

**Only one `.so` is needed**: all 9 target APIs are currently exported by real CANN's `libascendcl.so`.

---

## CMake Integration

### CANN_ROOT Cache Variable

```cmake
set(CANN_ROOT "" CACHE PATH "CANN installation root (contains include/ and lib64/ symlinks)")
```

**CANN_ROOT validation** (must be non-empty and the header must exist, otherwise `FATAL_ERROR`):

- If `CANN_ROOT` is empty: fail with a hint to pass `-DCANN_ROOT=<path>`
- If `CANN_ROOT` is non-empty but `${CANN_ROOT}/include/acl/acl_rt.h` does not exist: fail, indicating the header path is wrong

### stub target

```cmake
add_library(cann_stub SHARED tests/cpp/cann_stub/ascendcl.cc)
set_target_properties(cann_stub PROPERTIES
    OUTPUT_NAME "ascendcl"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64"
)
target_include_directories(cann_stub PRIVATE ${CANN_ROOT}/include)
target_compile_options(cann_stub PRIVATE ${DELEGATE_COMMON_FLAGS})
target_link_options(cann_stub PRIVATE ${DELEGATE_COMMON_FLAGS})
```

- `target_include_directories` points to `${CANN_ROOT}/include` (real headers are used; no type definitions are copied)
- Complies with `DELEGATE_COMMON_FLAGS` (`-Wall -Wextra -Werror` plus optional ASAN)

### Test executable linking

```cmake
target_link_libraries(torch_delegate_tests PRIVATE cann_stub ...)
```

### Runtime LD_LIBRARY_PATH injection

The environment variable is injected into ctest through `set_tests_properties`, so no manual setup is needed:

```cmake
set_tests_properties(torch_delegate_tests PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64:$ENV{LD_LIBRARY_PATH}"
)
```

**Why no `USE_CANN_STUB` switch is needed**: the stub has the same SONAME and ABI interface as real CANN's `libascendcl.so`, so modules built against the stub are fully equivalent to those built against the real library. Tests load the stub at runtime while production loads the real library; no distinction is needed at compile time.

---

## Header Strategy

`ascendcl.cc` directly `#include <acl/acl_rt.h>` and uses the type definitions from the real CANN headers. **No ACL type is copied or redefined**. `cann_stub_control.h` includes the same header to use types such as `aclError`.

---

## Stub Behavior Contract

| Function | Signature Source | Return Value | Side Effects |
|------|---------|--------|--------|
| `aclInit` | `acl_rt.h:858` | `ACL_SUCCESS` | Sets `initialized_ = true` |
| `aclFinalize` | `acl_rt.h:870` | `ACL_SUCCESS` | Sets `initialized_ = false` |
| `aclrtSetDevice` | `acl_rt.h:1284` | `ACL_SUCCESS` | Records `current_device_` |
| `aclrtResetDevice` | `acl_rt.h:1311` | `ACL_SUCCESS` | Clears `current_device_` (sets it to -1) |
| `aclrtCreateStream` | `acl_rt.h:2280` | `ACL_SUCCESS` | `new`s a sentinel object, writes it to `*stream`, and adds it to `live_streams_` |
| `aclrtDestroyStream` | `acl_rt.h:2328` | `ACL_SUCCESS` | `delete`s the sentinel and removes it from `live_streams_` |
| `aclrtSynchronizeStream` | `acl_rt.h:2356` | `ACL_SUCCESS` | No-op |
| `aclrtMalloc` | `acl_rt.h:1626` | `ACL_SUCCESS` | `malloc(size)` writes to `*devPtr` and is recorded in `live_allocs_` |
| `aclrtFree` | `acl_rt.h:1869` | `ACL_SUCCESS` | `free(devPtr)` and removes it from `live_allocs_` |

**`aclrtMalloc` uses real `malloc`**: some modules may actually read and write tensor data, so real allocation plus ASAN can detect out-of-bounds accesses and leaks.

**`aclrtCreateStream` allocates a sentinel object**: gives the stream handle a stable non-null pointer so ASAN can detect use-after-free.

---

## Global State Structure (`CannStubState`)

Singleton, containing:

| Field | Type | Description |
|------|------|------|
| `initialized_` | `bool` | `aclInit`/`aclFinalize` state |
| `current_device_` | `int32_t` | -1 means unset |
| `live_streams_` | `std::unordered_set<void*>` | Currently live stream handles |
| `live_allocs_` | `std::unordered_map<void*, size_t>` | pointer → size |
| `state_mutex_` | `std::mutex` | Guards all fields |
| `next_error_` | `aclError` | One-shot error injection; `ACL_SUCCESS` means none |

---

## Test Control Interface (`cann_stub_control.h`)

Intended for `#include` by test files only; declared in the `cann_stub` namespace:

```cpp
namespace cann_stub {
    void Reset();                          // Resets all state; called from each test's SetUp()
    bool IsInitialized();
    int32_t CurrentDevice();
    size_t LiveStreamCount();
    size_t LiveAllocCount();
    void SetNextError(aclError err);       // One-shot error injection (the next API call returns this error)
    std::vector<size_t> LiveAllocSizes();  // Snapshot of the sizes of all currently live allocations
}
```

Standard GTest fixture pattern:

```cpp
class CannStubTest : public ::testing::Test {
protected:
    void SetUp()    override { cann_stub::Reset(); }
    void TearDown() override {
        EXPECT_EQ(cann_stub::LiveStreamCount(), 0u);
        EXPECT_EQ(cann_stub::LiveAllocCount(),  0u);
    }
};
```

---

## Verification

```bash
# Configure (CANN_ROOT must be specified, otherwise FATAL_ERROR)
TORCH_CMAKE_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -B build \
      -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
      -DCANN_ROOT=$ASCEND_HOME_PATH \
      -DENABLE_ASAN=ON

cmake --build build -j$(nproc)

# Run tests (ctest injects LD_LIBRARY_PATH automatically via set_tests_properties)
cd build && ctest --output-on-failure
```

---

## Extension Adding a New Stub API

To add a new stub API, follow these steps:

1. Locating the function signature in the real CANN headers
2. Determining the return value and side effects according to the behavior contract table
3. Adding the function body to the corresponding `.cc` file
4. Adding a test in `test_cann_stub.cc`
