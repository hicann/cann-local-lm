English | [简体中文](../../../zh/cann_lm_engine/build_reference/build_guide.md)

# CANN-LM Build and Verification Guide

> Covers two environments: **Linux native build** (development/CI/host UT) and **HarmonyOS (OHOS) cross-compilation** (produces `libcann_lm_engine.so` for HarmonyOS PC runtime).
> Companion documents: `harmony_acl_build_analysis.md` (reference project analysis), `cann_lm_harmony_port_plan.md` (porting path and gaps).
> Maintenance date: 2026-10-15.

---

## 0. In One Sentence

The engine is pure standard C++17 with zero POSIX dependencies. It can be built natively on Linux to run UT, or cross-compiled into a HarmonyOS musl-targeted `.so` + 9 executables (UT/bbit/smoke) and self-signed. **The two build systems are independent** — the HarmonyOS build is a separate bypass (`cann_lm/engine/ohos/` + `build_ohos.sh`/`sign_ohos.sh`/`build_package_ohos.sh` under `scripts/cann_lm_engine/`), which does not modify existing Linux build files and poses zero risk to Linux builds. A single HarmonyOS command does it all: `./scripts/cann_lm_engine/build_ohos.sh` (compile → auto-sign), or one-click `./scripts/cann_lm_engine/build_package_ohos.sh` (compile → sign → produce signed wheel).

---

## 1. Involved Files and Their Roles

| File | Role | Environment |
|------|------|---------|
| `CMakeLists.txt` (repo root) | Top-level CMake, sets C++17, `add_subdirectory(cann_lm/engine)` | Both |
| `cann_lm/engine/CMakeLists.txt` | **Linux build core**: defines `cann_lm_engine` (STATIC) + `cann_backend` (subproject), **13 UT + bbit + 2 smoke (+1 temporary golden_verify)** targets, GTest (`find_package`), `gtest_discover_tests` | Linux |
| `cann_lm/engine/**/*.cpp/.h` | Engine source (18 .cpp files), shared by both builds | Both |
| `cann_lm/engine/ohos/ohos.toolchain.cmake` | **HarmonyOS cross-compilation toolchain** (self-contained): host clang-18 + OHOS sysroot, flags copied from official toolchain; ar/ranlib use host llvm-18 | HarmonyOS |
| `cann_lm/engine/ohos/CMakeLists.txt` | **HarmonyOS build**: compiles the same sources into SHARED `libcann_lm_engine.so` + **9 executables** (UT/bbit/smoke); GTest FetchContent from local source; static `add_test` | HarmonyOS |
| `tools/binary-sign-tool.jar` | **HarmonyOS binary signing tool** (jar, architecture-independent); self-signing via `-selfSign 1`, injects `.codesign` segment; placed in `tools/`, **gitignored, not in repo** (see tools/README.md) | HarmonyOS |
| `scripts/cann_lm_engine/build_ohos.sh` | **HarmonyOS build + sign entry**: validates `OHOS_SDK_NATIVE` (minimal SDK) + host `clang-18` → `cmake -G Unix Makefiles` build → 11-artifact validation + libc++ fallback → auto-calls `sign_ohos.sh` | HarmonyOS |
| `scripts/cann_lm_engine/sign_ohos.sh` | **Signing script**: for each .so ×2 + **9 executables** under build_ohos/, produces `*_signed` | HarmonyOS |
| `scripts/cann_lm_engine/build_package_ohos.sh` | **One-click script**: pre-checks java+jar → `build_ohos.sh` (compile+sign) → copy signed .so into wheel → `pip wheel` (produces `cann_lm_engine-*+signed*.whl`); supports `--clean` for full rebuild | HarmonyOS |
| `build_ohos/` | **Build artifacts** (gitignored): 11 signed artifacts (.so + 9 executables) | HarmonyOS |
| `/mnt/workspace/gtest-1.14.0.tar.gz` | GTest source (github v1.14.0, 867K), referenced locally by OHOS CMake FetchContent (CMake variable `GTEST_TAR` defaults to this path) | HarmonyOS |
| `tests/` | GTest UT source + bbit black-box | Both |
| `docs/build_reference/harmony_acl_build_analysis.md` | Complete analysis of reference project harmony_acl build system | HarmonyOS (reference) |
| `docs/build_reference/cann_lm_harmony_port_plan.md` | Compatibility matrix + porting path + gap list | HarmonyOS (reference) |

### Why Does HarmonyOS Use an Independent CMake Instead of Modifying the Existing One?

The existing `cann_lm/engine/CMakeLists.txt` is a single Linux branch + STATIC library + host `find_package(GTest)`. Adding HarmonyOS would require introducing an `OHOS_PLATFORM` branch, SHARED library, signing, GTest local source, etc. — a large change surface, and the HarmonyOS branch cannot be compile-verified on the host. **The first version uses an independent `ohos/CMakeLists.txt` bypass**. The source list is currently a transitional copy (later, once stable, an `engine_src.cmake` can be extracted for both to share, eliminating the copy). This way, the Linux build files remain untouched — zero regression risk.

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
All artifacts are under `$BUILD/cann_lm/engine/` (because the engine's CMakeLists defines targets in that subdirectory).

### 2.3 Artifacts
All located under `$BUILD/cann_lm/engine/`. Target list (aligned with `cann_lm/engine/CMakeLists.txt`):

| File | Role |
|------|------|
| `libcann_lm_engine.a` | STATIC library (linked by each target); pulls in backend subproject via `target_link_libraries PUBLIC cann_backend` |
| **UT (guarded by GTest_FOUND, registered via `gtest_discover_tests`)** | |
| `model_executor_ut` | Core component UT, aggregates 4 test files: `test_kv_cache_pool`/`test_model_executor`/`test_dfx_collector`/`test_tensor_manager` |
| `ut_input_assembler` | InputAssembler UT (includes gemma4 dual-bool-mask cases, see `docs/mask_fill.md`) |
| `ut_linear_state_pool` | Qwen3.5 linear state pool UT (`support_qwen3.5.md` §4.4) |
| `ut_engine_executor` | engine_executor UT |
| `ut_tokenizer` | tokenizer UT (reads `tests/fixtures/tokenizer/`) |
| `ut_c_api` | C API UT |
| `ut_scheduler` | scheduler UT |
| `ut_sampler` | sampler UT |
| `ut_request_handle` | request_handle UT |
| `ut_backend_shareweights` | weight-sharing UT (shim fast-fail + default 0 no-bind contract) |
| `ut_share_weights_executor` | weight-sharing executor-layer UT (Load share_weights_model_id passthrough) |
| `ut_data_type_utils` | data_type_utils UT |
| `ut_text_embedding` | TextEmbedding UT (SR20260828001) |
| `tokenizer_realdata` | tokenizer real-data UT (**conditional build**: only registered when `tmp/qwen`+`tmp/gemma` `tokenizer.json` are present) |
| **Black-box / Smoke** | |
| `bbit` | Black-box integration (`--check` NULL defense + `--config` full flow; includes `bbit_host_flow` cases) |
| `api_smoke` / `c_api_smoke` | C++ / C API smoke programs |
| `golden_verify` | [Temporary debug] Model accuracy verification tool (delete after verification is complete) |

> Additionally, `stub_ndk` (an external_ndk stub .so for host regression, under `tests/cpp/`) and the `cann_backend` subproject's own demo/test are both `EXCLUDE_FROM_ALL` — not in default build, require explicit `--target` to trigger.
> OHOS cross-compilation (`ohos/CMakeLists.txt`) only takes a **subset** of the above to build 9 executables (6 UT + bbit + 2 smoke); see §3.4.

### 2.4 Verification
```bash
cd "$BUILD/cann_lm/engine"

# All unit tests
ctest --output-on-failure
# Expected: 100% tests passed, 0 failed (1 disabled: TC003 disabled due to stub era)

# Run a specific UT directly to see output
./ut_tokenizer
./ut_c_api

# Black-box integration
./bbit --check
./bbit
```
> **Note**: `ctest` must be run in the `cann_lm/engine/` subdirectory. The top-level `CMakeLists.txt` does not call `enable_testing()`, so running `ctest` at the top-level directory reports "No tests found" — this is the project's current state, not a bug.

### 2.5 Current Baseline (measured 2026-08-24, after pull)
- 67 tests passed / 0 failed / 1 disabled (originally 63; new `test_tensor_manager` added 4)
- Compiler: host gcc/g++ (aarch64-linux-gnu)
- GTest: host `libgtest-dev` package
> ⚠ This 67-item baseline was measured on 2026-08-24. Since then, several UT targets in §2.3 (`ut_text_embedding`/`ut_data_type_utils`/`ut_sampler`/`ut_request_handle`/`ut_backend_shareweights`/`ut_share_weights_executor`/`ut_linear_state_pool`) have been added, so the actual test count is higher than 67 — refer to the actual `ctest` run on your host.
> Sync note: Linux `CMakeLists.txt` is the source-of-truth for the source list; OHOS `cann_lm/engine/ohos/CMakeLists.txt` is a manually-copied mirror. After pulling, if Linux adds new `.cpp`/UT files, they must be synced into the OHOS list (already done for `stub_built_model.cpp`/`tensor_manager.cpp`/`test_tensor_manager.cpp`).

---

## 3. HarmonyOS (OHOS) Cross-Compilation (see platform-specific manuals for details)

Produces **`libcann_lm_engine.so`** (aarch64-linux-ohos / musl) + `libc++_shared.so` + 9 executables, all self-signed. Route: host `clang-18` + minimal OHOS SDK (no `llvm/bin`/`build-tools`, SDK only contributes sysroot/libc++/builtins/toolchain files) + `cmake -G "Unix Makefiles"` + `make` (not Ninja). Host aarch64-glibc and target aarch64-ohos share the same CPU architecture, only libc differs; clang-18 runs natively without qemu.

> **Detailed steps (dependency list / SDK extraction / getting the code / compile commands / FAQ / submission) are in the platform-specific manuals**:
> - Linux host → [`build_linux_guide.md`](build_linux_guide.md)
> - Windows (WSL) → [`build_win_guide.md`](build_win_guide.md)
>
> One command (common to both platforms): `bash scripts/cann_lm_engine/build_ohos.sh` (compile+sign), or one-click `bash scripts/cann_lm_engine/build_package_ohos.sh [--clean]` (compile+sign+produce signed wheel). Three pitfalls in §5, environment variables in §6.

### 3.1 Artifact Quick Reference (flat in `build_ohos/`, 11 total + each `*_signed`)

| File | Role |
|------|------|
| `libcann_lm_engine.so` | HarmonyOS SHARED library (ELF AArch64, NEEDED=`libc.so`(musl)+`libc++_shared.so`) |
| `libc++_shared.so` | C++ runtime library (auto-copied from sysroot during build; fallback if missing) |
| `model_executor_ut` / `ut_engine_executor` / `ut_tokenizer` / `ut_c_api` / `ut_scheduler` / `ut_text_embedding` | Component UTs |
| `bbit` | Black-box integration (`--check` + full flow) |
| `api_smoke` / `c_api_smoke` | C++/C API smoke |
| `*_signed` / `*_signed.so` for each of the above | Signed version (auto-produced by `build_ohos.sh`) |

> Artifact verification (readelf to confirm musl target), standalone re-signing, producing the signed wheel (`.so` embedded in `cann_lm/lib/`), and device runtime flow are all in the platform-specific manuals. Host glibc **cannot directly run** these artifacts (they link against musl's `libc.so`); runtime verification must be on a HarmonyOS device (developer mode, allowing self-signed binaries).

### 3.2 Current Status (measured 2026-10-15)
- ✅ `.so` + `libc++_shared.so` cross-compile passed (ELF AArch64, musl)
- ✅ **9 executables cross-compile passed** (6 UT + bbit + 2 smoke; GTest FetchContent local source, static add_test)
- ✅ **All 11 artifacts signed** (host java + `tools/binary-sign-tool.jar -selfSign 1`)
- ✅ **Signed wheel produced** (`cann_lm_engine-*+signed*.whl` with embedded .so); one-click `build_package_ohos.sh` verified
- ✅ `model_executor_ut` / `ut_text_embedding` restored (backend hiai migration complete)
- ⏳ HarmonyOS device runtime verification (awaiting device; push signed wheel to device, pip install)

---

## 4. Comparison of the Two Builds

| Dimension | Linux Native | HarmonyOS OHOS Cross |
|------|-----------|---------------|
| Entry | `cmake <repo> && cmake --build` | `./scripts/cann_lm_engine/build_ohos.sh` |
| CMake core | `cann_lm/engine/CMakeLists.txt` | `cann_lm/engine/ohos/CMakeLists.txt` |
| Toolchain | System gcc/g++ | `cann_lm/engine/ohos/ohos.toolchain.cmake` (clang-18 + sysroot) |
| Target triple | aarch64-linux-gnu (host) | aarch64-linux-ohos (musl) |
| Library type | STATIC (`.a`) | SHARED (`.so`) |
| GTest | Host `find_package(GTest)` | Local source FetchContent (github v1.14.0); static `add_test` (`gtest_discover_tests` disabled for cross-compilation) |
| Artifact directory | `$BUILD/cann_lm/engine/` | `build_ohos/` |
| Runnable environment | Host | HarmonyOS device only (after signing) |
| Modifies existing CMake | — | **No** (bypass, zero risk) |

---

## 5. Three Inevitable Pitfalls of HarmonyOS Cross-Compilation

> Historical pitfall records were originally in `.project/memory/gotchas.md` (internal memory, gitignored along with `.project/`, not in repo); root causes and solutions are summarized in the table below. build_ohos.sh / ohos.toolchain.cmake have already固化 the corresponding handling.

| # | Symptom | Root Cause | Solution |
|---|------|------|------|
| 1 | `libclang_rt.builtins.a` / `crtbeginS.o` not found | clang-18 only searches its own resource dir (`/usr/lib/clang/18/lib/aarch64-linux-ohos/`) | Copy NDK's `libclang_rt.builtins.a` + `clang_rt.crtbegin.o`/`crtend.o` into it |
| 2 | `-lunwind` / `-lc++` not found | These static libraries are in `native/llvm/lib/aarch64-linux-ohos/`, not in sysroot standard paths | Add `-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos` to link flags |
| 3 | `<__config_site>` not found | Wrong libc++ header set used: the generic `llvm/include/c++/v1/` lacks generated headers | Use `llvm/include/libcxx-ohos/include/c++/v1/` instead (the OHOS-configured version with `__config_site`) |

---

## 6. Environment Variable Quick Reference

| Variable | Meaning | Default/Example |
|------|------|----------|
| `OHOS_SDK_NATIVE` | Absolute path to the extracted OHOS SDK `native/` (contains sysroot + `llvm/lib`/`include` + toolchain; **no** `llvm/bin`, **no** `build-tools`) | `/mnt/workspace/ohos_sdk/ohos_native/native` |
| `BUILD_DIR` | Output directory of `scripts/cann_lm_engine/build_ohos.sh` (optional) | `<repo>/build_ohos` |
| `CMAKE_BUILD_TYPE` | Build type | Linux: Debug; OHOS: Release |
| `GTEST_TAR` | GTest source tarball path (OHOS CMake FetchContent local reference) | `/mnt/workspace/gtest-1.14.0.tar.gz` |
