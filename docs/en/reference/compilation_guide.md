# CANN-LM Compilation Guide

English | [简体中文](../../zh/reference/compilation_guide.md)

## Overview: What build.sh Does

The repository-root `build.sh` is the **single entry point for full build / packaging / UT**. Main entries are `--pkg`/`-u`/`-h`; it also supports subcommands like `check`/`deps`/`cpp`/`wheel`/`third_party`/`package`/`all`/`smoke` (for daily development, see [build_guide.md](../build_guide.md) Part 3 Quick Reference). The `--pkg` full flow:

| Command | Purpose | Artifact |
|----------|---------|----------|
| `bash build.sh --pkg` | Compile all three parts and package | `compile_converter-*.run` under `build_out/` (required) + signed engine wheel under `dist/` (requires complete OHOS environment) |
| `bash build.sh -u` | Compile the parts containing UTs and run them | Test results (no persistent artifact) |
| `bash build.sh -h` | Print help | — |

### The 5 Stages of `--pkg`

```
[1/5] compile      build_cpp.sh --no-tests --target cann_lm_compile
                   → build/cann_lm_compile (epair→OM compilation tool, ~308K)
                   → build/libtorch_delegate_backend.so (C++ backend, ~845K)
[2/5] converter    pure-source pip wheel (--no-deps)
                   → build_out/cann_lm-<ver>-py3-none-any.whl (converter export whl)
[3/5] engine whl   engine python ctypes binding whl (pure python, no OHOS dependency)
                   → build_out/cann_local_lm_engine-<ver>-py3-none-any.whl
[4/5] engine OHOS  OHOS cross-compile + sign (build_package_ohos.sh, requires OHOS SDK)
                   → deploy_ohos/ (signed libcann_lm_engine.so + UT/bbit/smoke)
[5/5] package      self-extracting .run distribution
                   → build_out/cann_local_lm_converter-<ver>-<arch>.run (compile+converter, required)
```

> ⚠️ **If any stage fails, `--pkg` aborts immediately** (chained `|| return 1` exit). It will not continue to subsequent stages, nor will it package incomplete existing artifacts.

### The 3 Segments of `-u`

```
[UT/1] engine host C++ UT     GTest, build engine + UT targets, run with stub NDK
[UT/2] converter C++ UT       tests/cpp/epair_archive, ctest runs 7 cases (requires CANN toolchain, no NPU device)
[UT/3] Python UT              pytest: tests/python/converter + tests/python/llm_models + tests/python
```

---

## Environment Preparation Checklist

Listed under three tiers: "Required / engine cross-compile / Optional". **Every item is actually checked by `build.sh`** (`need_cann` / `need_third_party` / `build_engine` — 5 hard checks).

### Required (used by both `--pkg` and `-u`)

| # | Item | Check logic | How to prepare |
|---|------|-------------|----------------|
| 1 | **CANN toolkit** | `ASCEND_HOME_PATH` is set | After install, `source <cann>/set_env.sh`. Required by the compile stage and converter C++ UT |
| 2 | **C++ third-party libs** | `output/third_party/{json,gtest,makeself}` all exist | Run `bash scripts/build.sh third_party` before compiling — downloads third-party libs to the specified path then runs the install script |
| 3 | **cmake / make** | `command -v cmake && command -v make` | `apt install cmake build-essential` |
| 4 | **pip** | `command -v pip` (converter stage of `--pkg`) | `--no-deps` pure-source packaging; system pip is sufficient |
| 5 | **python torch** | `pip list` | torch version must be >= 2.4 |

### engine cross-compile items (hard-checked in `[3/4]` of `--pkg`; any missing aborts immediately)

`build_engine()` **prints ✓/✗ item by item; any ✗ triggers `return 1` and aborts the whole `--pkg`**. 5 items in total:

| # | Item | Check logic | How to prepare |
|---|------|-------------|----------------|
| 1 | **Compiler** | `$OHOS_SDK_NATIVE/llvm/bin/clang` exists and is executable | Use the full SDK (the native subtree of Huawei's official command-line-tools component ships it; you need to use the matching component in your environment) |
| 2 | **OHOS SDK** | `OHOS_SDK_NATIVE` is set and contains a `sysroot/` subdirectory | `export OHOS_SDK_NATIVE=<sdk>/linux/<ver>/native` |
| 3 | **java + signing jar** | `java` on PATH; `tools/binary-sign-tool.jar` or `cann_lm/engine/ohos/binary-sign-tool.jar` exists | `apt install default-jre`; copy from SDK `.../26.0.0/toolchains/lib/binary-sign-tool.jar` into `tools/` |
| 4 | **GTest source tarball** | `GTEST_TAR` (the third-party gtest `gtest-1.14.0.tar.gz`) file exists | Set with `export GTEST_TAR=<path>/googletest-1.14.0.tar.gz` |
| 5 | **cmake / make** | Same as Required-3 | — |

### Optional / Notes

| Item | Notes |
|------|-------|
| `VERSION` / `ARCH` | Defaults `0.1.0` / `$(uname -m)`, overridable via `export` (affects .run filename) |
| `THIRD_PARTY` | Defaults to `$REPO/output/third_party`, overridable |
| `BUILD_DIR` / `UT_BUILD_DIR` | Defaults `build/` / `/tmp/llm_ut_host`, overridable |
| `PIP` | Defaults to system pip; if `deps/torch-delegate/.venv/bin/pip` exists it is preferred |
| patchelf | If missing, rpath setup is skipped (when packaging the .run at `[4/5]`: `(patchelf not installed, skipping; relies on set_env.sh LD_LIBRARY_PATH at runtime)`), **non-blocking** |


## Packaging Flow (`bash build.sh --pkg`)

### Run

```bash
cd <repo>   # /srv/workspace/.../cann/CANN-LM-main
# First set environment variables per §2.4
bash build.sh --pkg
```

### What a Successful Log Looks Like (read for verification)

```
▶ === CANN-LM full build + package (--pkg) ===

▶ [1/4] compile: build_cpp.sh --no-tests --target cann_lm_compile
  ✓ CANN env path
...
[100%] Built target cann_lm_compile
=== Build complete ===
  308K .../build/cann_lm_compile
  845K .../build/libtorch_delegate_backend.so
  ✓ compile artifacts ready: build/cann_lm_compile + build/libtorch_delegate_backend.so   ← stage-1 success marker

▶ [2/4] converter: build cann_lm wheel (--no-deps pure-source packaging)
  Created wheel for cann-lm: filename=cann_lm-0.1.0-py3-none-any.whl size=163313   ← stage-2 key line
  ✓ converter whl: .../build_out/cann_lm-0.1.0-py3-none-any.whl (160K)                  ← stage-2 success marker

▶ [3/4] engine: OHOS cross-compile (build_package_ohos.sh, hard env check)
  ✓ Compiler: SDK clang (.../23/native/llvm/bin/clang)                               ← 5 prerequisite checks all ✓
  ✓ OHOS_SDK_NATIVE: .../23/native
  ✓ java: /usr/bin/java
  ✓ binary-sign-tool.jar: ready (tools/)
  ✓ GTEST_TAR: /tmp/gtest-1.14.0.tar.gz
  ✓ cmake/make: ready
  ✓ engine cross-compile prerequisites ready (...)
=== [1/2] OHOS cross-compile + sign (build_ohos.sh) ===
=== cmake configure (OHOS cross) ===
=== build (make -j32) ===
[100%] Built target ...
=== Artifact verification ===
  [ok]   libcann_lm_engine.so
  [ok]   libc++_shared.so
  [ok]   bbit / api_smoke / c_api_smoke / ut_engine_executor / ut_tokenizer / ut_c_api / ut_scheduler / model_executor_ut / ut_text_embedding   ← all 11 [ok]
=== sign binaries ===
  ✓ libcann_lm_engine.so -> libcann_lm_engine_signed.so
  ✓ ... (remaining 10)                                                            ← all signed ✓
=== [2/2] package signed wheel (build_package_ohos.sh → cann_local_lm_engine-*+signed*.whl) ===
  ✓ copy signed .so → cann_lm/lib/libcann_lm_engine.so
  ✓ copy libc++_shared.so → cann_lm/lib/
  ✓ pip wheel → build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl   ← stage-3 success marker

▶ [4/4] package .run (compile_converter)
  ... makeself compressing ...
  Self-extractable archive ".../cann_local_lm_converter-0.1.0-x86_64.run" successfully created.
▶ Package results
  533K .../build_out/cann_local_lm_converter-0.1.0-x86_64.run
  845K .../build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl
▶ === --pkg complete ===                                                          ← overall success marker
```

**Key points for reading success**: every stage's `✓` success marker appears + the final `▶ === --pkg complete ===`. If a `✗` or `FATAL_ERROR` appears midway, that stage `return 1`s and execution never reaches `--pkg complete`.

### Common Failure Scenarios and Log Signatures

| Stage | Failure symptom (on screen) | Root cause | Remedy |
|-------|------------------------------|------------|--------|
| **Prerequisite** | `✗ CANN ... ASCEND_HOME_PATH not set` / `✗ missing third_party/json ...` | Didn't source CANN / didn't build third-party libs | Fix §2.1-1/2 |
| **[1/4] compile** | `CMake Error ... Could NOT find GTest / nlohmann_json` | In `output/third_party` json/gtest only have tar, no extracted artifacts | Fix §2.1-2 |
| **[3/4] engine prereq** | Multiple `✗` (missing compiler / invalid OHOS_SDK_NATIVE / missing jar / missing GTest tar) + `✗ engine cross-compile prerequisites not met; aborted --pkg` | Any §2.2 item unmet | Fulfill per §2.2; `export` env vars |
| **[3/4] engine configure** | `CMake Error at .../ohos/CMakeLists.txt:97: FATAL_ERROR "GTest source tarball not found: /mnt/workspace/gtest-1.14.0.tar.gz"` | Didn't `export GTEST_TAR`; used the hardcoded default path | `export GTEST_TAR=<actually-existing tar>` and rerun |
| **[4/4] package** | `✗ engine signed wheel not generated, packaging failed` | Stage 3 produced no artifact (prerequisite aborted, or artifact verification found `[missing]`) | Go back and fix stage 3 |
| **[4/4] package** | `⚠ compile/converter artifacts incomplete, skipping compile_converter.run` | Stage 1/2 artifacts missing but script continued | Run a complete `--pkg` to completion once |


### Artifact Verification (after packaging)

```bash
# ① .run self-extraction integrity ("All good" = pass)
bash build_out/cann_local_lm_converter-0.1.0-x86_64.run --extract=/tmp/x1 --noexec
# Expected output: Verifying archive integrity... All good.

# ② compile tool smoke test (requires CANN env)
source <cann>/set_env.sh && ./build/cann_lm_compile   # printing usage means OK
# ③ engine signed wheel: push to OHOS device, pip install, then cllm generate --config testcase.json
```

---

## UT Flow (`bash build.sh -u`)

### Run

```bash
cd <repo>
source CANN toolkit path   # converter C++ UT requires CANN
bash build.sh -u
```

> If you only run Python UT / engine host UT you can skip CANN; but `-u` runs all three segments in order, and converter C++ UT depends on CANN — if missing, that segment shows `✗`.

### Successful Log Signatures (one success marker per segment)

```
▶ === CANN-LM UT build + run (-u) ===

▶ [UT/1] engine host C++ UT (GTest)
  • cmake configure (BUILD_ENGINE=ON, BUILD_CONVERTER=OFF)
  • cmake build (engine + UT targets)
  • run UT binaries (with stub NDK)
  == ut_request_handle
  ✓   ut_request_handle: PASS
  == ut_scheduler
  ✓   ut_scheduler: PASS
  ...(one PASS/FAIL line per ut_*)
  == model_executor_ut
  ✓   model_executor_ut: PASS
  • bbit --check (black-box self-check)
  ✓ bbit --check: PASS

▶ [UT/2] converter C++ UT (tests/cpp/epair_archive, requires CANN toolchain, no NPU device)
  ✓ CANN CANN toolkit path
  • ctest
100% tests passed, 0 tests failed out of 7        ← 7/7 all passed

▶ [UT/3] Python UT (pytest)
  ✓ tests/python/converter: PASS
  ✓ tests/python/llm_models: PASS
  ✓ tests/python: PASS

▶ === -u all passed ===
```

**Key points for reading success**:
- engine host UT: all `ut_*` lines PASS + `bbit --check: PASS` → `✓ engine host UT all passed` (if this line appears there is **no overall failure**)
- converter C++ UT: `100% tests passed, 0 tests failed out of 7`
- Python UT: three `✓ ...: PASS`
- **Final line**: `▶ === -u all passed ===` = overall success; `▶ === -u has failures (see above) ===` = at least one segment failed

### Common Failure Scenarios

| Segment | Failure symptom | Root cause / Remedy |
|---------|-----------------|---------------------|
| Prerequisite | `✗ missing third_party/gtest ...` | Third-party libs not built; fix §2.1-2 |
| [UT/1] | A single `ut_*: FAIL (exit=...)` | Inspect the specific assertion |
| [UT/2] | `✗ CANN ... ASCEND_HOME_PATH not set` | Didn't source CANN |
| [UT/3] | `✗ tests/...: FAIL` | pytest assertion failure; inspect detailed output above, focusing on whether the torch version is the issue |
---

## Log Reading Quick Reference (judge success/failure at a glance)

| Printed text | Meaning |
|--------------|---------|
| `▶ === --pkg complete ===` | **--pkg overall success** (all 5 stages green) |
| `▶ === -u all passed ===` | **-u overall success** (all 3 segments green) |
| `▶ === -u has failures (see above) ===` | -u has failures; scroll up for `✗`; most often the known model_executor_ut failure |
| `✓ <item> ...` | This check / artifact succeeded |
| `✗ <item> ...` | This item failed (prerequisite check / missing artifact / UT failure) |
| `⚠ <item> ...` | Warning (non-aborting, e.g. patchelf missing, known UT failure) |
| `[ok] <file>` | engine cross-compile artifact exists (verified by build_ohos.sh) |
| `[missing] <file>` | engine cross-compile artifact missing → stage 3 will definitely fail |
| `FATAL_ERROR` | cmake configure fatal error (e.g. GTEST_TAR default path doesn't exist) |
| `Verifying archive integrity... All good` | .run self-extraction integrity passed |
| `100% tests passed, 0 tests failed out of 7` | converter C++ UT all passed |

---

## FAQ

**Q1: I only want to compile compile+converter, and don't want to be blocked by engine prerequisites?**
`--pkg` forces engine cross-compile (any missing item aborts). For building only the first two parts:
```bash
source <cann>/set_env.sh
bash scripts/cann_lm_compile/build_cpp.sh --no-tests --target cann_lm_compile   # compile C++
bash scripts/cann_lm_converter/build_wheel.sh                                    # converter whl
```
(The engine prerequisite check lives in `build_engine()`; not invoking `--pkg` means it never triggers.)

**Q2: I don't have the OHOS SDK / don't want to cross-compile — can `-u` still run?**
Yes. `-u` does not depend on the OHOS SDK (engine host UT uses host gcc + stub NDK). Only `[3/4]` of `--pkg` needs the SDK.

**Q3: engine wheel won't run after being installed on the device?**

Common `.so` load failure causes: developer mode not enabled (self-signed `.so` rejected), `libc++_shared.so` not in the same directory, `.so` name changed. Use `cllm generate --config testcase.json --show-paths` to diagnose paths, or `--lib <path>` to specify the `.so` explicitly.
