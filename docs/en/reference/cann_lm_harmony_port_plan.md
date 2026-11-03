# CANN-LM Engine HarmonyOS PC Porting Path and Gap List (PLAN)

English | [简体中文](../../zh/reference/cann_lm_harmony_port_plan.md)

> Companion reference: `docs/reference/harmony_acl_build_analysis.md` (harmony_acl build analysis)
> Date compiled: 2026-08-24
> Goal: compile `cann_local_lm/engine/` on a HarmonyOS PC into **.so** or **executable UT** and be able to run it.
> This document is a Phase 3 PLAN deliverable (base.DESIGN + chip-adaptation.COMPATIBILITY-MAP).

---

## 1. Current State in One Sentence

The engine is **pure standard C++17**, **zero POSIX proprietary dependencies**, **zero external library dependencies** (only internal source + GTest), currently built with a **single Linux/GCC + STATIC library + host `find_package(GTest)`** setup. **Portability is extremely high** — about 90% of the porting work is on the build-system side; the source side barely changes.

---

## 2. Compatibility Matrix (COMPATIBILITY-MAP: source platform Linux/glibc → target HarmonyOS/musl)

Item-by-item comparison based on the harmony_acl reference template:

| Dimension | Source platform (current Linux host) | Target platform (HarmonyOS PC / OHOS) | Compatible? | Handling |
|-----------|--------------------------------------|---------------------------------------|-------------|----------|
| Compiler | System gcc/g++ | OHOS SDK bundled clang (`aarch64-linux-ohos`) | No (different compiler) | Introduce a toolchain file |
| Target triple | Host x86_64-linux-gnu | `aarch64-linux-ohos` | No | Enforced by the toolchain |
| libc | glibc | **musl** | Partial differences | `__MUSL__=1`; audit missing musl APIs |
| ABI/architecture | x86_64 | arm64-v8a | No | Cross-compile |
| STL | libstdc++ | **libc++_shared** | Different | `OHOS_STL=c++_shared`; copy `libc++_shared.so` |
| `<vector>/<memory>/<algorithm>/<atomic>/<mutex>/<thread>/<chrono>/<random>/<cmath>` | ✓ | ✓ (standard library) | ✓ | No change needed |
| `<filesystem>` (dfx_collector persistence) | ✓ | ✓ (clang/musl supports it) | ✓ (mind linking) | C++17 already on; HarmonyOS clang usually passes header-only; if linking errors appear, add `-lc++filesystem` or similar (verify in practice) |
| POSIX headers (`dirent`/`dlfcn`/`pthread`/`unistd`/`sys/*`) | — | **No POSIX-specific headers** | ✓ | Engine source has zero such includes, **no pitfall** |
| POSIX APIs (realpath/dlopen/opendir/...) | — | None | ✓ | Zero usage |
| Platform macros | None | Needs `__OHOS__`/`__MUSL__=1` | Needs adding | Source currently has no platform branches — add later when integrating CANN runtime/logging |
| Linker | ld | **lld** (enforced by toolchain) | No | Toolchain sets `-fuse-ld=lld` |
| GTest | `find_package(GTest REQUIRED)` host-installed | Not available on host; HarmonyOS has restricted network access | No | OHOS branch switches to **local source FetchContent** |
| Library type | `STATIC` | Needs `SHARED` to produce .so | Change | OHOS branch `SHARED` |
| Signing | None | **selfSign required** | No | Add `binary-sign-tool.jar` POST_BUILD |
| RPATH | None | `$ORIGIN` to find .so nearby | Needs adding | Set `CMAKE_INSTALL_RPATH $ORIGIN` |
| External system libraries | None | hilog (if using HarmonyOS logging) | As needed | Engine currently has no logging library dependency; can skip linking for now |
| Dynamic loading of third-party .so | None | (harmony_acl has it; this project does not yet) | N/A | This project has no dlopen; not applicable for now |

**Matrix conclusion**: **No P0 compile blockers.** The source having zero POSIX dependencies is the biggest advantage — unlike harmony_acl, which also has to handle `dirent`/`realpath`/`.so vs .dll`. The only thing to audit is `<filesystem>` linking under HarmonyOS clang/musl (most likely fine; one run will tell).

---

## 3. .so vs Executable UT — Decision

Your goal: "compile into a so, or compile executable ut files". The two are not mutually exclusive; harmony_acl produces **both at the same time**. Recommendation:

| Artifact | Value | Difficulty | Recommendation |
|----------|-------|------------|----------------|
| **libcanngin_engine.so** (SHARED library) | Lets HarmonyOS apps / other modules dynamically load engine capabilities | Low (change STATIC→SHARED + signing) | ✅ Produce |
| **Executable UT** (each *_ut + bbit) | Directly verify engine component logic on a HarmonyOS PC | Medium (UT needs GTest linked; HarmonyOS GTest requires local source) | ✅ Produce (passing UT first is stronger proof of correctness than the .so) |

**Recommended**: **Produce both in parallel**, but **priority = UT executable first**. Rationale: UT running GREEN on a HarmonyOS PC is the strongest evidence of "really can compile and execute on HarmonyOS"; the .so is just an artifact and by itself does not prove the logic is correct. UT first, then .so.

> Consistent with harmony_acl: its `build.sh` OHOS branch generates both `libhiai_ascendcl.so` + `ascendcl_ut`.

---

## 4. Action Path (phased, corresponding to the Locked Step's PORT/COMPILE)

### Prerequisites (environment prep, you need to provide)
1. **Huawei command-line-tools** (linux x64 version): download and extract, record its directory path (corresponds to harmony_acl's `CMD_LINE_TOOLS_DIR`). Download page: https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos
2. **OHOS SDK**: bundled with command-line-tools; contains `native/llvm/clang`, `native/sysroot`, `native/build/cmake/ohos.toolchain.cmake`.
3. **Java** (for signing): `binary-sign-tool.jar` needs java to run.
4. **Ninja** (`cmake -G Ninja`).
5. **GTest source**: download the `googletest-1.14.0` source zip and place it locally (HarmonyOS has restricted network access, cannot use GIT_REPOSITORY).
6. Host machine ubuntu 24.04+ + build-essential (for running host UT as a control).

### Step A — Introduce the OHOS toolchain (no source changes)
- Add `scripts/ohos.toolchain.cmake` next to `cann_local_lm/engine/` (following harmony_acl, or directly use the SDK's official one).
- Add `scripts/build_ohos.sh`: set `CMD_LINE_TOOLS_DIR`/`OHOS_SDK` and other variables → `cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=... -DOHOS_PLATFORM=OHOS -DOHOS_ARCH=arm64-v8a -DOHOS_STL=c++_shared ..` → `cmake --build`.

### Step B — Add an OHOS branch to CMake (modify `cann_local_lm/engine/CMakeLists.txt`)
Current state is a single branch. Change to:
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
  # Existing Linux branch (STATIC + find_package GTest) kept as-is
elseif(WIN32)
  # Optional, later
endif()
```
- Wrap the existing `find_package(GTest REQUIRED)` with `if(NOT OHOS_PLATFORM...)` (OHOS branch uses FetchContent).
- `gtest_discover_tests` may have issues under cross-compilation (cross-execution); the OHOS branch should consider switching to `ctest` command registration instead of discover (verify in practice).

### Step C — Audit `<filesystem>` (the only source-side risk point)
- `dfx_collector.h` uses `std::filesystem`. OHOS clang/musl usually passes header-only; if linking reports undefined reference, add a target link option. **Do not touch the source**, only adjust the build.

### Step D — Signing + artifact deployment
- Run `binary-sign-tool.jar -selfSign` on `cann_lm_engine.so`, each UT executable, and `libc++_shared.so`.
- Artifacts are collected under `build_ohos/`.

### Step E — HarmonyOS PC runtime verification
- Push to the HarmonyOS PC, `export LD_LIBRARY_PATH=$(pwd)`, run `*_ut` → expect the same GREEN as on the host (compare against the all-green ctest baseline of 8/20 in PROGRESS.md).
- Run bbit (black-box integration).

### Step F — REVIEW (thorough multiple rounds)
- +1 self-review + +2 external review, until zero new findings or 5 rounds.

---

## 5. Risks and Items Pending Practical Verification

| Risk | Description | Mitigation |
|------|-------------|------------|
| `gtest_discover_tests` fails under cross-compilation | It runs the compiled test binary on the host to do discovery; the cross-target binary cannot run | OHOS branch switches to `add_test(NAME ... COMMAND <target>)` static registration |
| `<filesystem>` linking | Occasionally needs extra linking under HarmonyOS musl/clang | Verify in practice; if it errors, add link per clang's hint |
| Signing tool path / Java version | harmony_acl hardcodes a Windows path | Make it a configurable variable + linux adaptation |
| OHOS SDK version differences | `ohos.toolchain.cmake` variable names change across versions | Prefer the SDK-bundled one, do not write your own |
| Whether UT can exec on a HarmonyOS device | HarmonyOS restricts unsigned/unauthorized binaries | Must selfSign + land in an executable directory |

---

## 6. Exit Criteria (aligned with active_workflow)

- [x] Compatibility matrix recorded (this section §2)
- [x] .so vs UT plan decided (§3: parallel, UT first)
- [ ] OHOS toolchain + build script in place (Step A)
- [ ] CMake OHOS branch wired up; host still builds as-is (Step B, no regression)
- [ ] UT runs GREEN on HarmonyOS PC (Step E)
- [ ] `cann_lm_engine.so` produced, signed, and loadable (Step D/E)
- [ ] +1 +2 review passed (Step F)
