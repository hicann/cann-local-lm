English | [简体中文](../../../zh/cann_lm_engine/build_reference/cann_lm_harmony_port_plan.md)

# CANN-LM Engine HarmonyOS PC Porting Path and Gap List (PLAN)

> Companion references: `docs/build_reference/harmony_acl_build_analysis.md` (harmony_acl build analysis), `docs/build_reference/build_guide.md` (actual post-implementation overall guide)
> Date: 2026-08-24
> Goal: Compile `cann_local_lm/engine/` on HarmonyOS PC into **.so** or **executable UT** and be able to run it.
> This document is a Phase 3 PLAN output (base.DESIGN + chip-adaptation.COMPATIBILITY-MAP).

> **Actual implementation (2026-10-15, completed)**: This PLAN has been executed, with a few **intentional deviations** from the original plan to adapt to the host (aarch64-linux-glibc) minimal environment. **When reading this document, this section takes precedence; the main text §2–§5 retains the original planning record for backtracking**:
> - **Compiler**: Not "OHOS SDK-bundled clang", but **host `clang-18`** (`find_program(clang-18)` inside `ohos.toolchain.cmake`); the minimal SDK **has no `llvm/bin`**.
> - **Builder**: Not `cmake -G Ninja`, but **`cmake -G "Unix Makefiles"` + `make -j$(nproc)`**; the minimal SDK **has no `build-tools/ninja`**, and ninja is not installed on the host.
> - **SDK**: Not the full "Huawei command-line-tools", but **minimal extraction** from the OpenHarmony open-source mirror (only sysroot + `llvm/lib` + `llvm/include` + `build/cmake/ohos.toolchain.cmake`, about 82M).
> - **CMake organization**: Instead of adding an `OHOS_PLATFORM` branch to the existing `cann_local_lm/engine/CMakeLists.txt` per §Step B, an **independent bypass** `cann_local_lm/engine/ohos/CMakeLists.txt` is used (source list manually mirrored), with zero changes and zero regression to Linux build files.
> - **Script paths**: Migrated from the top-level `scripts/` to **`scripts/cann_lm_engine/`** (one extra level, so `REPO_ROOT` inside scripts uses `../..`).
> - **Artifacts**: Actually produces **11 artifacts** (2 `.so` + 9 executables: 6 UT including `ut_text_embedding` + `bbit` + 2 smoke), each with `*_signed`.
> - **Packaging**: `build_package_ohos.sh` produces signed wheel `cann_lm_engine-*+signed*.whl` (`.so` embedded in `cann_local_lm/lib/`).
>
> For the actual flow, see `build_guide.md` §3 and `build_linux_guide.md`; for individual scripts, see the file table in `build_guide.md` §1.
>
> **Update (2026-10-20)**: Now supports **dual-SDK routes**, with the compiler in `ohos.toolchain.cmake` auto-selecting one of two:
> - **Route A (recommended)**: Full command-line-tools native subtree, `llvm/bin/clang` is ARM-native and runs directly on host (back to the "SDK-bundled clang" idea, but ARM version instead of x86_64, no qemu needed), **with built-in builtins resource dir** (pitfall #1 auto-solved, no need to补 `/usr/lib/clang/18`, no host clang-18 needed). 11 artifacts all green.
> - **Route B**: Minimal OpenHarmony mirror SDK (no `llvm/bin`), falls back to host `clang-18` + manual builtins (i.e., the 2026-10-15 implemented route).
> Both routes produce identical artifacts (OHOS musl/AArch64, 11 artifacts). See `build_linux_guide.md` §2.

---

## 1. Current State in One Sentence

The engine is **pure standard C++17**, **zero POSIX proprietary dependencies**, **zero external library dependencies** (only internal source + GTest), currently built with **a single Linux/GCC + STATIC library + host `find_package(GTest)`**. **Portability is extremely high** — the porting work is ~90% on the build system side, with almost no changes needed on the source side.

---

## 2. Compatibility Matrix (COMPATIBILITY-MAP: source platform Linux/glibc → target HarmonyOS/musl)

Item-by-item comparison from scanning the harmony_acl reference:

| Dimension | Source Platform (current Linux host) | Target Platform (HarmonyOS PC / OHOS) | Compatible? | Handling |
|------|---------------------|------------------------|------|------|
| Compiler | System gcc/g++ | OHOS SDK-bundled clang (`aarch64-linux-ohos`) | No (different compilers) | Introduce toolchain file |
| Target triple | Host x86_64-linux-gnu | `aarch64-linux-ohos` | No | Toolchain enforces |
| libc | glibc | **musl** | Partial differences | `__MUSL__=1`, audit musl-missing APIs |
| ABI/architecture | x86_64 | arm64-v8a | No | Cross-compile |
| STL | libstdc++ | **libc++_shared** | Different | `OHOS_STL=c++_shared`, copy `libc++_shared.so` |
| `<vector>/<memory>/<algorithm>/<atomic>/<mutex>/<thread>/<chrono>/<random>/<cmath>` | ✓ | ✓ (standard library) | ✓ | No changes needed |
| `<filesystem>` (dfx_collector persistence) | ✓ | ✓ (clang/musl supported) | ✓ (mind linking) | C++17 already enabled, HarmonyOS clang usually header-only passes; if link errors occur, add `-lc++filesystem` or similar (verify in practice) |
| POSIX headers (`dirent`/`dlfcn`/`pthread`/`unistd`/`sys/*`) | — | **No POSIX proprietary headers** | ✓ | Engine source has zero such includes, **no pitfalls** |
| POSIX APIs (realpath/dlopen/opendir/...) | — | None | ✓ | Zero usage |
| Platform macros | None | Need `__OHOS__`/`__MUSL__=1` | Need to add | Source currently has no platform branches — add later if CANN runtime/logging is integrated |
| Linker | ld | **lld** (toolchain enforced) | No | Toolchain sets `-fuse-ld=lld` |
| GTest | `find_package(GTest REQUIRED)` host-installed | Host-side unavailable, HarmonyOS network restricted | No | OHOS branch switches to **local source FetchContent** |
| Library type | `STATIC` | Need `SHARED` to produce .so | Change | OHOS branch `SHARED` |
| Signing | None | **Must selfSign** | No | Add `binary-sign-tool.jar` POST_BUILD |
| RPATH | None | `$ORIGIN` to find .so nearby | Need to add | Set `CMAKE_INSTALL_RPATH $ORIGIN` |
| External system libraries | None | hilog (if using HarmonyOS logging) | As needed | Engine currently has no logging library dependency, can skip linking for now |
| Dynamic loading of third-party .so | None | (harmony_acl has it, this project does not yet) | N/A | This project has no dlopen, not involved for now |

**Matrix conclusion**: **No P0 compilation blockers**. Zero POSIX dependencies in source is the biggest advantage — unlike harmony_acl which had to handle `dirent`/`realpath`/`.so vs .dll`. The only thing to audit is `<filesystem>` linking under HarmonyOS clang/musl (most likely fine, a single test run will tell).

---

## 3. .so vs Executable UT — Decision

Your goal: "compile into a so, or compile executable ut files". The two are not mutually exclusive; harmony_acl produces **both simultaneously**. Recommendation:

| Artifact | Value | Difficulty | Recommendation |
|------|------|------|------|
| **libcann_lm_engine.so** (SHARED library) | For HarmonyOS apps/other modules to dynamically load engine capabilities | Low (change STATIC→SHARED + sign) | ✅ Produce |
| **Executable UT** (each *_ut + bbit) | Directly verify each engine component's logic on HarmonyOS PC | Medium (UT needs to link GTest; HarmonyOS GTest requires local source) | ✅ Produce (running it first validates correctness better than .so alone) |

**Recommended**: **Produce both in parallel**, but **priority = UT executable first**. Rationale: UT can run GREEN directly on HarmonyOS PC, which is the strongest evidence of "truly compilable and executable on HarmonyOS"; .so is just an artifact and does not itself prove logic correctness. UT first, then .so.

> Consistent with harmony_acl: its `build.sh` OHOS branch simultaneously generates `libhiai_ascendcl.so` + `ascendcl_ut`.

---

## 4. Action Path (phased, corresponding to Locked Step's PORT/COMPILE)

> **Implementation summary (2026-10-15)**: This section is the **initial planning conception** (command-line-tools full SDK + `cmake -G Ninja` + adding OHOS branch inside `cann_local_lm/engine/CMakeLists.txt`). **The actual implementation adopted a better route**: minimal SDK (no `llvm/bin`, no `build-tools`) + host `clang-18` + `cmake -G "Unix Makefiles"` + `make -j$(nproc)` + **independent `cann_local_lm/engine/ohos/CMakeLists.txt` bypass** (no change to `engine/CMakeLists.txt`, zero regression risk) + scripts migrated to `scripts/cann_lm_engine/`. The final implementation is governed by `build_guide.md` / `build_linux_guide.md` / `build_win_guide.md`; Steps A–D in this section are retained as planning history reference (where `scripts/build_ohos.sh` path and `cmake -G Ninja` have both been superseded by the current state).

### Prerequisites (environment preparation, you need to provide)
1. **Huawei command-line-tools** (linux x64 version): download and extract, record its directory path (corresponds to harmony_acl's `CMD_LINE_TOOLS_DIR`). Download page: https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos
2. **OHOS SDK**: bundled with command-line-tools, includes `native/llvm/clang`, `native/sysroot`, `native/build/cmake/ohos.toolchain.cmake`.
3. **Java** (for signing): `binary-sign-tool.jar` requires java to run.
4. **Ninja** (`cmake -G Ninja`).
5. **GTest source**: download `googletest-1.14.0` source zip locally (HarmonyOS network restricted, cannot use GIT_REPOSITORY).
6. Host machine ubuntu 24.04+ + build-essential (for running host UT for comparison).

### Step A — Introduce OHOS toolchain (no source changes)
- Add `scripts/ohos.toolchain.cmake` next to `cann_local_lm/engine/` (refer to harmony_acl, or directly use the SDK's official one).
- Add `scripts/build_ohos.sh`: set `CMD_LINE_TOOLS_DIR`/`OHOS_SDK` etc. variables → `cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=... -DOHOS_PLATFORM=OHOS -DOHOS_ARCH=arm64-v8a -DOHOS_STL=c++_shared ..` → `cmake --build`.

### Step B — Add OHOS branch to CMake (modify `cann_local_lm/engine/CMakeLists.txt`)
Currently single-branch. Change to:
```cmake
if(OHOS_PLATFORM STREQUAL "OHOS")
  # SHARED library + $ORIGIN RPATH + GTest local FetchContent + signing POST_BUILD + copy libc++_shared.so
  add_library(cann_lm_engine SHARED ${ENGINE_SRC})
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
  FetchContent_Declare(googletest SOURCE_DIR ${CMAKE_SOURCE_DIR}/../third_party/googletest-1.14.0)
  FetchContent_MakeAvailable(googletest)
  # Each *_ut target re-linked to gtest target (remove find_package(GTest REQUIRED) in this branch)
  # signing custom command
elseif(UNIX AND NOT APPLE)
  # existing Linux branch (STATIC + find_package GTest) retained as-is
elseif(WIN32)
  # optional, later
endif()
```
- Wrap the existing `find_package(GTest REQUIRED)` with `if(NOT OHOS_PLATFORM...)` (OHOS branch uses FetchContent).
- `gtest_discover_tests` may have issues under cross-compilation (cross-execution); the OHOS branch should consider using `ctest` registration commands instead of discover (verify in practice).

### Step C — Audit `<filesystem>` (the only source-side risk point)
- `dfx_collector.h` uses `std::filesystem`. OHOS clang/musl usually passes header-only; if link reports undefined reference, add target link options. **Do not modify source**, only adjust build.

### Step D — Signing + artifact deployment
- Run `binary-sign-tool.jar -selfSign` on `cann_lm_engine.so`, each UT executable, and `libc++_shared.so`.
- Artifacts are集中 in `build_ohos/`.

### Step E — HarmonyOS PC runtime verification
- Push to HarmonyOS PC, `export LD_LIBRARY_PATH=$(pwd)`, run `*_ut` → expect the same GREEN as on host (compare against the 8/20 ctest all-green baseline in PROGRESS.md).
- Run bbit (black-box integration).

### Step F — REVIEW (thorough multiple rounds)
- +1 self-review + +2 external review, until zero new findings or 5 rounds.

---

## 5. Risks and Items to Be Tested

| Risk | Description | Mitigation |
|------|------|------|
| `gtest_discover_tests` fails under cross-compilation | It needs to run the compiled test binary on host for discovery; cross-target binaries cannot run | OHOS branch switches to `add_test(NAME ... COMMAND <target>)` static registration |
| `<filesystem>` linking | Occasionally requires extra linking under HarmonyOS musl/clang | Test in practice; if errors occur, add link per clang's提示 |
| Signing tool path / Java version | harmony_acl hardcodes Windows paths | Make configurable variables + Linux adaptation |
| OHOS SDK version differences | `ohos.toolchain.cmake` variable names vary by version | Prefer the SDK-bundled one, do not write your own |
| Whether UT can exec on HarmonyOS device | HarmonyOS restricts unsigned/unauthorized binaries | Must selfSign + place in an executable directory |

---

## 6. Exit Criteria (aligned with active_workflow)

- [x] Compatibility matrix recorded (this section §2)
- [x] .so vs UT scheme decided (§3: parallel, UT first)
- [x] OHOS toolchain + build script in place (Step A — actually independent `cann_local_lm/engine/ohos/ohos.toolchain.cmake` + `scripts/cann_lm_engine/build_ohos.sh`, see top "Actual implementation")
- [x] CMake OHOS branch connected, host-side can still build as-is (Step B — actually uses bypass `ohos/CMakeLists.txt`, not modifying existing CMake, Linux zero regression)
- [x] `cann_lm_engine.so` produced and signed, loadable (Step D — 11 artifacts all signed, signed wheel produced)
- [ ] UT runs GREEN on HarmonyOS PC (Step E — awaiting device runtime; signed wheel ready for deployment)
- [ ] +1 +2 review passed (Step F)
