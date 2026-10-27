# harmony_acl Build System Analysis (Reference Document)

English | [简体中文](../../zh/reference/harmony_acl_build_analysis.md)

> Source project: `/mnt/workspace/cann-recipes-harmony-infer/harmony_acl`
> Analysis date: 2026-08-24
> Purpose: serves as a reference template for porting CANN-LM `engine/` to a HarmonyOS PC (compiling to .so / executable UT).
> This document is self-contained — a future session can load it cold-start and understand it without revisiting the source project.

---

## 1. What is harmony_acl

A subproject under Huawei's `cann-recipes-harmony-infer` repository. It provides a unified **ACL (AscendCL) interface** aligned with the Ascend platform, enabling developers to deploy model inference on the **HarmonyOS (OHOS) device side** using the same set of interfaces. Core artifacts:

- `libhiai_ascendcl.so` — dynamic library wrapping the ACL interface
- `ascendcl_ut` — GTest unit test executable
- Also supports building on **OHOS / Linux / Windows** three platforms (the same `CMakeLists.txt` switches platforms via branches)

This is exactly the form CANN-LM engine wants: compile the engine into .so + executable UT, and be able to run it on a HarmonyOS PC.

---

## 2. Directory Structure

```
harmony_acl/
├── build.sh                 # Main build entry (OHOS cross-compile + dependency header download)
├── build_ut.sh / build_ut.bat # Windows/MinGW UT build (independent of build.sh)
├── ohos.toolchain.cmake     # OHOS cross-compile toolchain file (core)
├── README.md                # Prerequisites + build/run instructions
├── scripts/
│   ├── CMakeLists.txt       # The real build logic (three-platform branches)
│   └── CMakeLists.txt.back  # Historical version (compare to see the evolution)
├── src/                     # Library source
│   ├── acl_*.cpp/.h         # ACL interface implementation
│   ├── ascend_string.cpp
│   ├── ndk/                 # HarmonyOS NDK layer: dynamic loading, hiai_* proxies
│   └── utils/               # log.h / assert.h / securestl.h and other utilities
├── tests/cpp/engine/                # GTest tests
│   ├── main.cpp             # test main
│   ├── aippParam_ut.cpp
│   ├── compiler_ut.cpp
│   └── ndk/                 # NDK test stubs + headers
├── include/                 # (generated at build time) where ACL/graph headers land
├── depends/                 # (filled at build time) downloaded dependency header packages
└── build_ohos/              # (artifacts) libhiai_ascendcl.so / ascendcl_ut / libc++_shared.so
```

Key point: `include/` and `depends/` are **generated/filled when build.sh runs**; they do not statically exist in the repo.

---

## 3. OHOS Cross-Compile Toolchain (`ohos.toolchain.cmake`)

This is the **most critical file** for porting to HarmonyOS. 33 lines, annotated line by line:

```cmake
set(CMAKE_SYSTEM_NAME OHOS)            # Target system = OpenHarmony (not Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)    # Target CPU = ARM64

# Force the target triple, cutting off dependency on host (Windows/Linux) libraries
set(CMAKE_C_COMPILER_TARGET "aarch64-linux-ohos")
set(CMAKE_CXX_COMPILER_TARGET "aarch64-linux-ohos")

set(OHOS_SDK "<SDK path>" CACHE PATH "OpenHarmony SDK path")
set(CMAKE_C_COMPILER   "${OHOS_SDK}/native/llvm/bin/clang")     # HarmonyOS SDK bundled clang
set(CMAKE_CXX_COMPILER "${OHOS_SDK}/native/llvm/bin/clang++")
set(CMAKE_SYSROOT "${OHOS_SDK}/native/sysroot")                 # sysroot (headers + base libs)

set(CMAKE_LIBRARY_ARCHITECTURE "aarch64-linux-ohos")
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "/usr/lib/aarch64-linux-ohos")

# find mode: only look for libs/headers inside the sysroot, never use the host's
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
list(APPEND CMAKE_FIND_ROOT_PATH "${OHOS_SDK}/native")

# Force the HarmonyOS SDK-bundled lld linker
set(CMAKE_EXE_LINKER_FLAGS   "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)

# Compile macros required by HarmonyOS
add_compile_definitions(__OHOS__ __MUSL__=1)
```

**Key takeaways**:
1. Use the **SDK-bundled clang/clang++** (not system gcc), with target triple `aarch64-linux-ohos`.
2. **sysroot** points to the SDK's `native/sysroot`; all headers/libs are found only from the sysroot (`ONLY` mode), avoiding host library pollution.
3. HarmonyOS uses **musl libc** (hence `__MUSL__=1`), different from Linux glibc.
4. The linker is forced to **lld**.
5. The `__OHOS__` macro is the switch for `#ifdef __OHOS__` platform branches in the source — the source switches HarmonyOS paths based on this.

> Note: what `build.sh` actually uses is the **official SDK toolchain** `${OHOS_SDK}/native/build/cmake/ohos.toolchain.cmake` (via `-D CMAKE_TOOLCHAIN_FILE=...`); the repo-root `ohos.toolchain.cmake` is a hand-written simplified/comparison version. Pick one of the two; the official one is more complete.

---

## 4. `build.sh` Main Build Flow (155 lines)

### 4.1 Automatic Dependency Header Download (`auto_download_header_with_git` + `deal_after_download`)

Uses git sparse-checkout to pull headers from three CANN repos on gitcode:

| Repo | sparse-checkout path | Lands at |
|------|----------------------|----------|
| `cann/runtime` | `include/external/acl` | `include/acl/` |
| `cann/ge` | `inc` | `include/acl/` + `include/graph/` (selected specific .h) |
| `cann/metadef` | `inc` | `include/graph/` (selected specific .h) |
| `js/json` (nlohmann) | full | `depends/nlohmann/` |

If git fails, the README gives a fallback path of manually downloading a zip into `depends/`. The script also detects a zip and prefers extracting it; otherwise it uses the clone result.

**Takeaway**: if CANN-LM depends on CANN headers, it can reuse this sparse-checkout auto-pull mechanism rather than committing the headers into the repo.

### 4.2 cmake Configure and Compile

```bash
CMD_LINE_TOOLS_DIR=M:\\harmony-pc\\command-line-tools    # <- the only env var you need to change
OHOS_SDK=${CMD_LINE_TOOLS_DIR}/sdk/default/openharmony/
OHOS_INC_DIR=${OHOS_SDK}/native/sysroot/usr/include/
HMS_INC_DIR=${CMD_LINE_TOOLS_DIR}/sdk/default/hms/native/sysroot/usr/include/

cmake -G "Ninja" \
  -D OHOS_INC_DIR=${OHOS_INC_DIR} \
  -D HMS_INC_DIR=${HMS_INC_DIR} \
  -D OHOS_STL=c++_shared \
  -D OHOS_ARCH=arm64-v8a \
  -D OHOS_PLATFORM=OHOS \
  -D CMAKE_TOOLCHAIN_FILE=${OHOS_SDK}/native/build/cmake/ohos.toolchain.cmake \
  ../scripts
cmake --build . -j
```

**Takeaways**:
- `CMD_LINE_TOOLS_DIR` is the extraction directory of Huawei's **command-line-tools** (the native dev environment); this line must be changed before building.
- Standard variables passed to the official toolchain: `OHOS_STL` (c++_shared/static), `OHOS_ARCH` (arm64-v8a), `OHOS_PLATFORM=OHOS`.
- Uses the **Ninja** generator.
- Artifacts land under `build/` (the script does `mkdir build && cd build`). Note the README says UT is in `build/ut`, while build.sh writes `build/` — a minor doc-vs-script inconsistency bug.

---

## 5. `scripts/CMakeLists.txt` Three-Platform Build Logic (246 lines, core)

Structure: `add_library(hiai_ascendcl SHARED ...)` produces the main .so, then handles UT and stub libraries per platform branch.

### 5.1 Main Library (shared by all platforms)

```cmake
add_library(hiai_ascendcl SHARED ${ACL_SRC})   # source = src/*.cpp + src/ndk/*.cpp
if(GNU OR Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3")   # Note: this line was originally before the branch; Windows errors (see §7)
```

### 5.2 OHOS Branch (`OHOS_PLATFORM STREQUAL "OHOS"`)

```cmake
# RPATH set to $ORIGIN so the .so finds dependencies in its own directory
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

find_library(HILOG_LIB NAMES hilog_ndk.z)            # HarmonyOS logging library (required)
target_link_libraries(hiai_ascendcl PUBLIC ${HILOG_LIB})

# HarmonyOS system library neural_network_core (IMPORTED)
find_library(NNCORE_LIB NAMES neural_network_core)
add_library(neural_network_core SHARED IMPORTED)
set_target_properties(neural_network_core PROPERTIES IMPORTED_LOCATION "${NNCORE_LIB}")

# HMS-side libhiai_foundation.so (IMPORTED, path passed in via -DHMS_LIB_DIR)
add_library(hiai_foundation SHARED IMPORTED)
set_target_properties(hiai_foundation PROPERTIES IMPORTED_LOCATION "${HMS_LIB_DIR}/libhiai_foundation.so")

# GTest: local source FetchContent (scripts/googletest-1.14.0/)
FetchContent_Declare(googletest SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest-1.14.0)
FetchContent_MakeAvailable(googletest)

# UT executable
add_executable(ascendcl_ut ${UT_TEST_SRC})
target_link_libraries(ascendcl_ut PUBLIC hiai_ascendcl neural_network_core hiai_foundation gtest gtest_main ${HILOG_LIB})

# ★ HarmonyOS auto-signing (POST_BUILD): binary-sign-tool.jar self-signs .so / ut / libc++_shared.so
```

**HarmonyOS-specific key points**:
- Links `hilog_ndk.z` (HarmonyOS logging).
- The system library `neural_network_core` is found via `find_library` in the sysroot.
- Third-party `.so` (`libhiai_foundation.so`) uses an **IMPORTED target** + a path variable passed in.
- GTest uses **local source** FetchContent (HarmonyOS has restricted network access, cannot use GIT_REPOSITORY).
- **Signing is mandatory**: `binary-sign-tool.jar` + java, selfSigns `ascendcl_ut`, `libhiai_ascendcl.so`, and `libc++_shared.so`. The artifact name gets a `_signed` suffix. Without signing it cannot run on a HarmonyOS device.
- Copies `libc++_shared.so` to the artifact directory (the C++ runtime at runtime).

### 5.3 Linux Branch (`UNIX AND NOT APPLE`)

No HarmonyOS system libraries; instead **self-built stub libraries** simulate `hiai_foundation` / `neural_network_core` (compiled into .so using the stub sources under `tests/cpp/`), then links the UT. GTest also uses local source FetchContent. Used for running UT on the host.

### 5.4 Windows Branch (`WIN32`)

- `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON` (DLL does not export symbols by default; must enable).
- `CMAKE_SHARED_LIBRARY_PREFIX ""` (drop the MinGW lib prefix).
- MSVC: `_CRT_SECURE_NO_WARNINGS NOMINAX UNICODE _UNICODE` + `/utf-8`.
- GTest uses `GIT_REPOSITORY` (Windows network is OK), tag `release-1.12.1`.
- Also self-built stub libraries.

### 5.5 UT Build (`build_ut.sh` / `build_ut.bat`)

**Independent of `build.sh`**, specifically for running UT on a Windows host:
- `build_ut.bat` invokes the MSYS2 UCRT64 environment and runs `build_ut.sh`.
- `build_ut.sh` uses **MinGW** (`-G "MinGW Makefiles"`, gcc/g++), outputs to `build_ut/`, and only goes through the WIN32 branch of CMakeLists.
- Relationship to OHOS: OHOS UT is produced by `build.sh` going through the OHOS branch (signed, runs on HarmonyOS); Windows UT is produced by `build_ut.*` going through the WIN32 branch (runs on host).

---

## 6. HarmonyOS PC Execution Environment Requirements (runtime)

Distilled from the README + CMakeLists, running artifacts on a HarmonyOS PC requires:

| Item | Description |
|------|-------------|
| **Signing** | `.so` and executables must be signed with `binary-sign-tool.jar -selfSign`, otherwise the system refuses to load them |
| **libc++_shared.so** | HarmonyOS C++ runtime library; must be copied to the artifact directory (already uses `$ORIGIN` RPATH to find it nearby) |
| **LD_LIBRARY_PATH** | The README has `export LD_LIBRARY_PATH=$(pwd)` — lets the executable find the .so in the same directory |
| **System library paths** | `hilog_ndk.z`, `neural_network_core`, etc. are under the HarmonyOS system `/system/lib64/ndk/`, bundled with the device |
| **Dynamic loading** | `src/ndk/ndk_proxy.cpp` dlopens `libhiai_foundation.so` etc. at runtime; the path switches to `/system/lib64/ndk/` based on the `__OHOS__` macro |

---

## 7. Cross-Platform Adaptation Pitfalls (distilled from `windows_ut_adaptation_spec.md`)

This 117-line spec records the pitfalls hit when adapting UT to Windows; it applies equally to CANN-LM's future cross-HarmonyOS/Linux/Windows work. Categorized:

### P0 Compile Blockers
- **POSIX-specific headers**: `<dirent.h>` does not exist on non-POSIX platforms; remove if unused.
- **Compiler-specific flags**: `-O0 -g3` is GCC/Clang-specific; MSVC errors. Wrap with `if(GNU OR Clang)` and place it **after** the platform branches.
- **Hardcoded platform suffixes**: `.so` vs `.dll` — at dynamic-load sites use a three-way `#ifdef __OHOS__ / _WIN32 / else` branch.
- **Path separator logic**: `find("/") == npos || find("\\") == npos` should be `&&` (a logic bug).

### P1 Link/Runtime Failures
- **Windows DLL symbol export**: `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON` must be at the start of the WIN32 branch.
- **MSVC definitions covering all targets**: not just UT; the library target needs them too (`strncpy` etc. trigger C4996).

### P2 Warnings/Potential
- Duplicate macro definitions (MSVC C4005).
- `float32_t` (C++23 `<stdfloat>`) has poor compatibility; change to `float`.

**General rule**: all **platform-related** behavior in the source is isolated with three-way `__OHOS__` / `_WIN32` / `else (Linux)` macros; in CMake, isolated with `OHOS_PLATFORM`/`UNIX`/`WIN32` three-way branches. This is the core pattern for cross-HarmonyOS + host compilation.

---

## 8. Borrowable Points for the CANN-LM Engine (PLAN preview)

Compared to this project's `cann_lm/engine/CMakeLists.txt` (current state: single Linux/GCC, STATIC library, `find_package(GTest REQUIRED)` host-installed, no platform branches):

| harmony_acl practice | CANN-LM current state | Borrowing direction |
|----------------------|-----------------------|---------------------|
| `ohos.toolchain.cmake` + `-DCMAKE_TOOLCHAIN_FILE` cross-compile | No toolchain, pure native build | Introduce an OHOS toolchain file |
| `add_library(... SHARED ...)` produces .so | `add_library(cann_lm_engine STATIC ...)` | Add an OHOS branch to produce SHARED .so |
| Three-platform branches (OHOS/Linux/Windows) | Single branch | Add `if(OHOS_PLATFORM)/elseif(UNIX)/elseif(WIN32)` |
| GTest local source FetchContent (HarmonyOS restricted network) | `find_package(GTest REQUIRED)` host-installed | OHOS branch switches to local FetchContent |
| HarmonyOS auto-signing + copy libc++_shared.so | None | OHOS artifacts need a signing script |
| `$ORIGIN` RPATH | None | Set RPATH for finding .so nearby |
| `hilog_ndk.z` / `__OHOS__` / `__MUSL__` | No platform macros | Source needs POSIX dependency audit + platform macros |
| Dependency headers pulled via sparse-checkout | No CANN header dependency (currently) | If integrating CANN runtime later, reuse the mechanism |

**Biggest gap**: the CANN-LM engine is currently a **STATIC library + host GCC + host GTest**, with no cross-compilation / platform branches / signing chain at all. To get onto a HarmonyOS PC, it needs a new OHOS toolchain config + CMake platform branches + a signing script, and a POSIX-proprietary dependency audit of the source. See the compatibility matrix in the follow-up PLAN document for details.

---

## 9. One-Page Quick Reference

```
Toolchain:  OHOS SDK native/llvm/clang, aarch64-linux-ohos, sysroot, lld
Compile macros: __OHOS__ __MUSL__=1
STL:        c++_shared (need to copy libc++_shared.so to the artifact directory)
Main lib:   add_library(hiai_ascendcl SHARED ...)
UT:         add_executable(ascendcl_ut ...), links gtest + main lib + IMPORTED system .so
GTest:      local source FetchContent (ohos/linux); GIT_REPOSITORY (windows)
Signing:    binary-sign-tool.jar -selfSign 1, artifacts *_signed
Run:        export LD_LIBRARY_PATH=$(pwd); ./ascendcl_ut
Platform isolation: source __OHOS__/_WIN32/else; CMake OHOS_PLATFORM/UNIX/WIN32
Pitfalls:   POSIX headers (dirent), conditionalize GCC flags, .so/.dll suffix, Windows symbol export
```
