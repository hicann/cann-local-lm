# CANN-LM Build and Verification Guide

English | [简体中文](../../zh/reference/build_guide.md)

> Covers two environments: **native Linux build** (dev/CI/host UT) and **HarmonyOS (OHOS) cross-compilation** (produces `libcann_lm_engine.so` for running on HarmonyOS PCs).
> Companion documents: `harmony_acl_build_analysis.md` (reference project analysis), `cann_lm_harmony_port_plan.md` (porting path and gaps).
> Last updated: 2026-08-24.

---

## 0. In One Sentence

The engine is pure standard C++17 with zero POSIX dependencies. It can be built natively on Linux to run UT, and also cross-compiled into a HarmonyOS musl target `.so` + 8 executables (UT/bbit/smoke) with self-signing. **The two build systems are independent of each other** — the HarmonyOS build is a standalone bypass (`cann_lm/engine/ohos/` + `scripts/build_ohos.sh` + `scripts/sign_ohos.sh` + `scripts/build_package_ohos.sh`) that does not modify any existing Linux build files, so the Linux build carries zero risk. HarmonyOS build is a single command: `./scripts/build_ohos.sh` (compile → auto-sign), or one-click `./scripts/build_package_ohos.sh` (compile → sign → produce signed wheel).

---

## 1. Files Involved and Their Roles

| File | Role | Environment |
|------|------|-------------|
| `CMakeLists.txt` (repo root) | Top-level CMake; sets C++17, `add_subdirectory(cann_lm/engine)` | Both |
| `cann_lm/engine/CMakeLists.txt` | **The real Linux build**: defines `cann_lm_engine` (STATIC), 6 UT/smoke/bbit targets, GTest (`find_package`), `gtest_discover_tests` | Linux |
| `cann_lm/engine/**/*.cpp/.h` | Engine source (18 .cpp files); shared by both builds | Both |
| `cann_lm/engine/ohos/ohos.toolchain.cmake` | **HarmonyOS cross-compile toolchain** (self-contained): host clang-18 + OHOS sysroot; flags adopted from the official toolchain; ar/ranlib use host llvm-18 | HarmonyOS |
| `cann_lm/engine/ohos/CMakeLists.txt` | **HarmonyOS build**: compiles the same set of sources into SHARED `libcann_lm_engine.so` + **8 executables** (UT/bbit/smoke); GTest FetchContent from local source; static `add_test` | HarmonyOS |
| `cann_lm/engine/ohos/binary-sign-tool.jar` | **HarmonyOS binary signing tool** (jar, architecture-independent); self-signs with `-selfSign 1`, injects the `.codesign` segment | HarmonyOS |
| `scripts/build_ohos.sh` | **HarmonyOS build + sign entry**: validates `OHOS_SDK_NATIVE` → cmake build → auto-invokes `sign_ohos.sh` | HarmonyOS |
| `scripts/sign_ohos.sh` | **Signing script**: for each of the 2 .so + 8 executables under build_ohos/, produces a `*_signed` copy | HarmonyOS |
| `scripts/build_package_ohos.sh` | **One-click script**: pre-checks java+jar → `build_ohos.sh` (compile+sign) → copy signed .so into wheel → `pip wheel` (produces `cann_lm_engine-*+signed*.whl`) | HarmonyOS |
| `tools/` (contains `binary-sign-tool.jar`) | Vendored signing tool (12MB, jar, **gitignored, not committed**; see tools/README.md) | HarmonyOS |
| `build_ohos/` | **Build artifacts** (gitignored): cross-compiled signed .so + UT/bbit/smoke executables | HarmonyOS |
| `/mnt/workspace/gtest-1.14.0.tar.gz` | GTest source (github v1.14.0, 867K); referenced locally via OHOS CMake FetchContent | HarmonyOS |
| `tests/` (cann_lm/engine/../../tests) | GTest UT sources + bbit black-box | Both |
| `.project/memory/gotchas.md` | Root causes and fixes for the six unavoidable pitfalls of OHOS cross-compilation | HarmonyOS (reference) |
| `docs/reference/harmony_acl_build_analysis.md` | Full analysis of the reference project harmony_acl build system | HarmonyOS (reference) |
| `docs/reference/cann_lm_harmony_port_plan.md` | Compatibility matrix + porting path + gap list | HarmonyOS (reference) |

### Why does HarmonyOS use a standalone CMake instead of modifying the existing one?

The existing `cann_lm/engine/CMakeLists.txt` is a single Linux branch + STATIC library + host `find_package(GTest)`. Adding HarmonyOS support would require introducing an `OHOS_PLATFORM` branch, SHARED library, signing, local GTest source, etc. — a large surface area where the HarmonyOS branch cannot be compiled/verified on the host. **The first version uses a standalone `ohos/CMakeLists.txt` bypass.** The source list is currently duplicated during the transition (later, once stabilized, an `engine_src.cmake` can be extracted for both to share, eliminating the duplication). This way the Linux build files remain untouched — zero regression risk.

---

## 2. Linux Environment Build and Verification

### 2.1 Prerequisites
```bash
sudo apt-get install build-essential cmake libgtest-dev
# cmake >= 3.10; host already has cmake 3.20
```

### 2.2 Build
```bash
cd /mnt/workspace/CANN-LM
BUILD=/tmp/cannlm_build
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
cmake /mnt/workspace/CANN-LM -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j"$(nproc)"
```
All artifacts end up under `$BUILD/cann_lm/engine/` (because the engine's CMakeLists defines targets in that subdirectory).

### 2.3 Artifacts
| File | Role |
|------|------|
| `libcann_lm_engine.a` | STATIC library (linked by all targets) |
| `model_executor_ut` | Core component UT (kv_cache/input_assembler/model_executor/dfx) |
| `ut_c_api` | C API UT |
| `ut_engine_executor` | engine_executor UT |
| `ut_tokenizer` | tokenizer UT |
| `ut_scheduler` | scheduler UT |
| `bbit` | Black-box integration (`--check` + full flow) |
| `api_smoke` / `c_api_smoke` | C++/C API smoke programs |

### 2.4 Verification
```bash
cd "$BUILD/cann_lm/engine"

# All unit tests
ctest --output-on-failure
# Expected: 100% tests passed, 0 failed (1 disabled: TC003 disabled due to stub era)

# Run a single UT directly to see output
./ut_tokenizer
./ut_c_api

# Black-box integration
./bbit --check
./bbit
```
> **Note**: `ctest` must be run inside the `cann_lm/engine/` subdirectory. The top-level `CMakeLists.txt` does not call `enable_testing()`, so running `ctest` at the top level reports "No tests found" — this is the current project state, not a bug.

### 2.5 Current Baseline (measured 2026-08-24, after pull)
- 67 tests passed / 0 failed / 1 disabled (originally 63; new `test_tensor_manager` added 4)
- Compiler: host gcc/g++ (aarch64-linux-gnu)
- GTest: host `libgtest-dev` package
> Sync note: the Linux `CMakeLists.txt` is the source of truth for the source list; the OHOS `cann_lm/engine/ohos/CMakeLists.txt` is a hand-maintained mirror. After a pull, if Linux adds new `.cpp`/UT files, they must also be added to the OHOS list (already done for `stub_built_model.cpp`/`tensor_manager.cpp`/`test_tensor_manager.cpp`).

---

## 3. HarmonyOS (OHOS) Environment Build and Verification

Produces **`libcann_lm_engine.so`** (aarch64-linux-ohos / musl).

### 3.1 Route Selection (why not use the official command-line-tools)
The official Huawei command-line-tools only ships **x86_64-linux** binaries; if your host is x86_64 you can use it directly. **This host is aarch64-linux-glibc**, which shares the same CPU architecture as the target aarch64-linux-ohos and differs only in libc, so we adopt a better route:

> **Host `clang-18` + OHOS SDK sysroot extracted from the open-source mirror** (clang-18 runs natively, no qemu needed, and no need to run the x86_64 NDK clang).

The SDK is taken from the OpenHarmony open-source mirror (direct curl, not the JS-rendered Huawei consumer page):
```
https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz
```

### 3.2 Prerequisites (one-time setup)

```bash
# 1) Compiler
sudo apt-get install clang-18

# 2) Download and extract the SDK (only the needed 82M, not the full 2.5G)
mkdir -p /mnt/workspace/ohos_sdk && cd /mnt/workspace/ohos_sdk
curl -L -o sdk.tar.gz \
  'https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz'

# The outer tar contains nested zips; extract the linux native one
tar -xzf sdk.tar.gz linux/native-linux-x64-5.0.0.71-Release.zip
NATIVE_ZIP=linux/native-linux-x64-5.0.0.71-Release.zip

# Extract what we need: sysroot + libc++ headers/libs + builtins + official toolchain reference
unzip -oq $NATIVE_ZIP \
  'native/sysroot/*' \
  'native/build/cmake/*' \
  'native/llvm/lib/aarch64-linux-ohos/*' \
  'native/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/*' \
  'native/llvm/include/libcxx-ohos/include/c++/v1/*' \
  -d ohos_native

export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native

# 3) Install builtins/crt into clang-18's resource dir (fixes pitfall #1, see §5)
SRC=$OHOS_SDK_NATIVE/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos
DST=/usr/lib/clang/18/lib/aarch64-linux-ohos
sudo mkdir -p "$DST"
sudo cp "$SRC"/{libclang_rt.builtins.a,clang_rt.crtbegin.o,clang_rt.crtend.o} "$DST"/
```

### 3.3 Build

```bash
cd /mnt/workspace/CANN-LM
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
./scripts/build_ohos.sh
```
What `scripts/build_ohos.sh` does: validate `OHOS_SDK_NATIVE` → `cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=... -DOHOS_SDK_NATIVE=... cann_lm/engine/ohos` → `cmake --build` → auto-invoke `scripts/sign_ohos.sh` to sign. Artifacts: 2 .so + 8 executables + a `*_signed` copy of each.

### 3.4 Artifacts (under `build_ohos/`)

| File | Role |
|------|------|
| `libcann_lm_engine.so` | HarmonyOS SHARED library (ELF AArch64, NEEDED=`libc.so` (musl) + `libc++_shared.so`) |
| `libc++_shared.so` | HarmonyOS C++ runtime library, distributed alongside the .so (auto-copied from sysroot during build) |
| `model_executor_ut` | Core component UT (kv_cache/input_assembler/model_executor/dfx) |
| `ut_c_api` / `ut_engine_executor` / `ut_tokenizer` / `ut_scheduler` | Component UTs |
| `bbit` | Black-box integration (`--check` + full flow) |
| `api_smoke` / `c_api_smoke` | C++/C API smoke programs |
| `*_signed` / `*_signed.so` for each of the above | Signed versions (auto-produced by `build_ohos.sh`, see §3.6) |

### 3.5 Verify Artifacts (confirm it is an OHOS musl target, not glibc)

```bash
cd /mnt/workspace/CANN-LM/build_ohos
readelf -d libcann_lm_engine.so | grep NEEDED
# Expected: NEEDED libc++_shared.so, libc.so   <- libc.so rather than libc.so.6 = musl/OHOS
readelf -h libcann_lm_engine.so | grep Machine
# Expected: Machine: AArch64
```
> This .so **cannot be run directly** on the native Linux host (it links musl's `libc.so`; the host is glibc). Runtime verification must be done on a HarmonyOS device (see §3.6).

### 3.6 Signing (auto-completed on host)

After a successful build, `scripts/build_ohos.sh` automatically invokes `scripts/sign_ohos.sh`, which uses `tools/binary-sign-tool.jar` to self-sign each artifact (following harmony_acl: `-selfSign 1`). The jar is architecture-independent and runs directly on host java 21 — **no cert/keystore needed** (selfSign mode uses the descriptor digest as the signature, injecting it into the ELF's `.codesign` segment).

```bash
# Re-run signing alone (without recompiling):
./scripts/sign_ohos.sh                    # Default: sign artifacts under build_ohos/
./scripts/sign_ohos.sh <dir> <jar_path>   # Specify explicitly
```
Signing produces a same-named `*_signed` (for executables) and `*_signed.so` (for .so).

### 3.7 Device Runtime Verification (requires a HarmonyOS PC / device)

**Build the signed wheel** (`build_ohos/` signed .so → embedded in wheel):
```bash
./scripts/build_package_ohos.sh     # compile+sign+produce build_out/cann_lm_engine-*+signed*.whl
```
The signed wheel embeds the `.so` (`cann_lm/lib/`). On the HarmonyOS device, `pip install` then `cllm generate --config testcase.json` to run inference (see [Quick Start](../QUICKSTART.md)).

Convention when no device is connected on the host: **produce signed wheel → push to device → pip install → `cllm generate`**.

```bash
export LD_LIBRARY_PATH=$(pwd)
./model_executor_ut          # Run UT
./bbit --check               # Black-box
```
> The device requires developer mode / allowing self-signed binaries (selfSign is for dev/debug only, not an official release signature).

### 3.8 Current Status (2026-08-24)
- ✅ `.so` + `libc++_shared.so` cross-compiled successfully (ELF AArch64, musl target)
- ✅ **8 executables cross-compiled successfully** (UT/bbit/smoke; GTest local source FetchContent, static add_test)
- ✅ **All artifacts signed** (`*_signed`, host java 21 + binary-sign-tool.jar -selfSign 1)
- ⏳ HarmonyOS device runtime verification (pending device; device pulls after git commit)

---

## 4. Comparison of the Two Builds

| Dimension | Native Linux | HarmonyOS OHOS cross-compile |
|-----------|--------------|------------------------------|
| Entry point | `cmake <repo> && cmake --build` | `./scripts/build_ohos.sh` |
| CMake source of truth | `cann_lm/engine/CMakeLists.txt` | `cann_lm/engine/ohos/CMakeLists.txt` |
| Toolchain | System gcc/g++ | `cann_lm/engine/ohos/ohos.toolchain.cmake` (clang-18 + sysroot) |
| Target triple | aarch64-linux-gnu (host) | aarch64-linux-ohos (musl) |
| Library type | STATIC (`.a`) | SHARED (`.so`) |
| GTest | Host `find_package(GTest)` | Local source FetchContent (github v1.14.0); static `add_test` (`gtest_discover_tests` disabled for cross-compilation) |
| Artifact directory | `$BUILD/cann_lm/engine/` | `build_ohos/` |
| Runnable environment | Host | HarmonyOS device only (after signing) |
| Modifies existing CMake | — | **No** (bypass, zero risk) |

---

## 5. Three Unavoidable Pitfalls of HarmonyOS Cross-Compilation (see `.project/memory/gotchas.md` for details)

| # | Symptom | Root cause | Fix |
|---|---------|------------|-----|
| 1 | `libclang_rt.builtins.a` / `crtbeginS.o` not found | clang-18 only looks in its own resource dir (`/usr/lib/clang/18/lib/aarch64-linux-ohos/`) | Copy the NDK's `libclang_rt.builtins.a` + `clang_rt.crtbegin.o`/`crtend.o` into it |
| 2 | `-lunwind` / `-lc++` not found | These static libraries live in `native/llvm/lib/aarch64-linux-ohos/`, not in the sysroot standard paths | Add `-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos` to the link flags |
| 3 | `<__config_site>` not found | Wrong libc++ header set used: the generic `llvm/include/c++/v1/` is missing generated headers | Switch to `llvm/include/libcxx-ohos/include/c++/v1/` (the OHOS-configured version that includes `__config_site`) |

---

## 6. Environment Variable Quick Reference

| Variable | Meaning | Default / Example |
|----------|---------|-------------------|
| `OHOS_SDK_NATIVE` | Absolute path to the extracted OHOS SDK `native/` (contains sysroot/llvm) | `/mnt/workspace/ohos_sdk/ohos_native/native` |
| `BUILD_DIR` | Output directory of `scripts/build_ohos.sh` (optional) | `<repo>/build_ohos` |
| `CMAKE_BUILD_TYPE` | Build type | Linux: Debug; OHOS: Release |
