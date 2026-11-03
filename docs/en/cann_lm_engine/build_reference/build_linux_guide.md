English | [简体中文](../../../zh/cann_lm_engine/build_reference/build_linux_guide.md)

# Linux Build Guide — cann_llm_open (OHOS Cross-Compilation)

> Goal: Cross-compile the engine + bbit + all UTs on Linux (aarch64), producing signed deployable artifacts.
> Dual-SDK route (compiler in `ohos.toolchain.cmake` auto-selects one of two; `build_ohos.sh` auto-detects SDK):
> - **Route A (recommended)**: The **native subtree** of Huawei command-line-tools — `llvm/bin/clang` is ARM-native and runs directly on the host, **with built-in builtins/crt resource dir** (no need to补充 `/usr/lib/clang/18`), no host clang-18 required.
> - **Route B (minimal)**: SDK extracted from OpenHarmony mirror (no `llvm/bin`) — falls back to **host clang-18** + manually补 builtins.
> Both routes produce identical artifacts (OHOS musl/AArch64); the builder uses host make (the full SDK's build-tools only contains cmake, no ninja).

## 1. Dependency List

| # | Dependency | How to Obtain | Purpose |
|---|------|---------|------|
| 1 | clang-18 / clang++-18 (**Route B only**) | `apt install clang-18 llvm-18` | Host cross-compiler (fallback when minimal SDK has no `llvm/bin/clang`). **Route A uses SDK-bundled clang-15, no host clang-18 needed** |
| 2 | cmake >= 3.16 | `apt install cmake` (host already has 3.20) | Build configuration |
| 3 | make | Bundled with `apt install build-essential` | Build execution (`-G "Unix Makefiles"`) |
| 4 | JDK | `apt install default-jdk` (tested java-21 works) | Signing tool `binary-sign-tool.jar` requires java |
| 5 | OHOS SDK (native subtree, minimal version) | See §2 | sysroot + libc++ headers/libs + builtins + `ohos.toolchain.cmake` |
| 6 | googletest v1.14.0 tarball | See §3 | OHOS CMake FetchContent local reference (cross-compilation cannot use host `find_package(GTest)`) |
| 7 | binary-sign-tool.jar | Place in `tools/` (12MB, `.gitignore` excludes from repo; see `tools/README.md`) | Self-signing |

> **Route B (minimal SDK)**: No `llvm/bin` (no clang), `build-tools` only has cmake, no ninja — compiler uses host clang-18, SDK contributes sysroot/libc++/builtins/toolchain files. Host aarch64-linux-glibc and target share the same CPU architecture, only libc differs; clang-18 runs natively without qemu.
> **Route A (full SDK, recommended)**: `llvm/bin/clang` is ARM-native and runs directly on host, with built-in builtins resource dir (no need to补 `/usr/lib/clang/18`), `build-tools` only has cmake (still uses host make). Both routes produce identical artifacts; the compiler in `ohos.toolchain.cmake` auto-selects one of two.

## 2. Obtain the OHOS SDK

### 2.1 Route A: Full SDK (Recommended)

Download **Command Line Tools for Linux(ARM)** from the Huawei developer official site:
https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos

Download the zip (e.g., `commandline-tools-linux-arm64-26.0.0.851.zip`), extract only the native subtree:

```bash
mkdir -p /mnt/workspace/ohos_sdk/cli_root
unzip -oq commandline-tools-linux-arm64-*.zip \
  'command-line-tools/sdk/default/openharmony/native/*' \
  -d /mnt/workspace/ohos_sdk/cli_root
# Move to a fixed path (build_ohos.sh auto-detects this path)
mv /mnt/workspace/ohos_sdk/cli_root/command-line-tools/sdk/default/openharmony/native \
   /mnt/workspace/ohos_sdk/cli_native
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/cli_native
```

Full SDK native subtree (**includes `llvm/bin/clang`, ARM-native**):
```
$OHOS_SDK_NATIVE/  (= .../cli_native)
  ├── llvm/bin/clang(+clang-15,llvm-ar,ld.lld,...)   ← ARM-native, runs directly on host
  ├── llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/  ← builtins/crt bundled (pitfall #1 auto-solved)
  ├── llvm/lib/aarch64-linux-ohos/                   ← libc++_shared.so/libunwind.a/libc++abi.a
  ├── llvm/include/libcxx-ohos/include/c++/v1/       ← OHOS libc++ headers (with __config_site)
  ├── sysroot/usr/lib/aarch64-linux-ohos/            ← full musl sysroot
  └── build/cmake/ohos.toolchain.cmake               ← official toolchain (reference, actual uses repo version)
```

Verify:
```bash
$OHOS_SDK_NATIVE/llvm/bin/clang --version   # clang 15.0.4, Target: aarch64-unknown-linux-gnu
```
> Route A needs no host clang-18, no need to补 builtins (pitfalls #1/2/3 all auto-solved). Go directly to §4 to compile.

### 2.2 Route B: Minimal SDK (OpenHarmony mirror, no llvm/bin)

SDK is taken from the OpenHarmony open-source mirror (direct curl, not the JS-rendered Huawei consumer page):

```bash
mkdir -p /mnt/workspace/ohos_sdk && cd /mnt/workspace/ohos_sdk
curl -L -o sdk.tar.gz \
  'https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz'

# The outer tar contains nested zips; extract the linux native one
tar -xzf sdk.tar.gz linux/native-linux-x64-5.0.0.71-Release.zip
NATIVE_ZIP=linux/native-linux-x64-5.0.0.71-Release.zip

# Extract only the needed 82M (not the full 2.5G): sysroot + libc++ headers/libs + builtins + official toolchain reference
unzip -oq $NATIVE_ZIP \
  'native/sysroot/*' \
  'native/build/cmake/*' \
  'native/llvm/lib/aarch64-linux-ohos/*' \
  'native/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/*' \
  'native/llvm/include/libcxx-ohos/include/c++/v1/*' \
  -d ohos_native

export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
```

Directory structure after extraction (note: **no `llvm/bin`, no `build-tools`**):
```
$OHOS_SDK_NATIVE/
  ├── sysroot/                       ← target system headers/libs (musl)
  ├── llvm/lib/aarch64-linux-ohos/   ← libc++_shared.so / libunwind.a / libc++.a …
  ├── llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/  ← libclang_rt.builtins.a / crt
  ├── llvm/include/libcxx-ohos/include/c++/v1/       ← OHOS-specific libc++ headers (with __config_site)
  └── build/cmake/ohos.toolchain.cmake              ← official toolchain file (reference, actual uses repo version)
```

### Verify the SDK

```bash
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
[ -d "$OHOS_SDK_NATIVE/sysroot" ] && echo "sysroot ok" || echo "sysroot missing"
ls "$OHOS_SDK_NATIVE/llvm/lib/aarch64-linux-ohos/" | head    # should have libc++_shared.so etc.
clang-18 --version    # host compiler, not the SDK's
```

### First-time setup (Route B only): Install builtins/crt into clang-18 resource dir (solves pitfall #1, see §6)

> Route A uses SDK-bundled clang-15, whose resource dir `llvm/lib/clang/15.0.4/` already includes builtins/crt, **skip this step**.

```bash
SRC=$OHOS_SDK_NATIVE/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos
DST=/usr/lib/clang/18/lib/aarch64-linux-ohos
sudo mkdir -p "$DST"
sudo cp "$SRC"/{libclang_rt.builtins.a,clang_rt.crtbegin.o,clang_rt.crtend.o} "$DST"/
```

## 3. Get the Code and GTest tarball

```bash
git clone ssh://szv-y.codehub.huawei.com:2222/h00651112/cann_llm_open.git
cd cann_llm_open
```

- Write access requires the repo owner to grant permission (otherwise reports CH.00905401).
- Clone to a **pure-English path** (Chinese paths will definitely break on Windows; English is also recommended on Linux).

GTest tarball (for OHOS cross-compilation):
```bash
# Download googletest v1.14.0 (867K), place at the default path expected by OHOS CMakeLists
curl -L -o /mnt/workspace/gtest-1.14.0.tar.gz \
  https://github.com/google/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz
```
> `cann_local_lm/engine/ohos/CMakeLists.txt` references this tarball via FetchContent (CMake variable `GTEST_TAR`, default `/mnt/workspace/gtest-1.14.0.tar.gz`). If missing, it will `FATAL_ERROR` with a download URL prompt. No `find_package(GTest)` needed (host GTest is unavailable under cross-compilation).

## 4. Compile

```bash
cd /mnt/workspace/CANN-LM
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native   # or persist in ~/.bashrc

# Option 1: Compile + sign only
bash scripts/cann_lm_engine/build_ohos.sh

# Option 2: One-click compile + sign + package deploy directory
bash scripts/cann_lm_engine/build_package_ohos.sh
# Add --clean for auto-clean full rebuild: bash scripts/cann_lm_engine/build_package_ohos.sh --clean
```

### Build Flow Description (inside `build_ohos.sh`)

1. **SDK location**: `OHOS_SDK_NATIVE` environment variable; if empty, auto-detect (`cli_native` full SDK > `ohos_native/native` minimal SDK), validates `sysroot` exists.
2. **Compiler check**: SDK-bundled `${OHOS_SDK_NATIVE}/llvm/bin/clang` takes priority (Route A); if absent, check host `clang-18` (Route B). Error only if both are missing.
3. **CMake configure**: `cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=cann_local_lm/engine/ohos/ohos.toolchain.cmake -DOHOS_SDK_NATIVE=... -DOHOS_PLATFORM=OHOS -DCMAKE_BUILD_TYPE=Release -B build_ohos -S cann_local_lm/engine/ohos`.
4. **Make build**: `cmake --build build_ohos -j$(nproc)` (host make, not ninja).
5. **Artifact validation**: 11 artifacts checked one by one `[ok]`/`[missing]` (see §5); if `libc++_shared.so` is missing, fallback-copy from SDK.
6. **Signing**: Auto-calls `scripts/cann_lm_engine/sign_ohos.sh`.

### Green Criteria

Output ends with `=== build_ohos.sh 完成 ===` (Option 1) or `=== build_package_ohos.sh 完成 ===` (Option 2), and the make stage has no errors.

### One-click Script (`build_package_ohos.sh`)

A minimal orchestration layer that only does: `--clean` (delete `build_ohos` for full rebuild) → pre-check java + jar → call `build_ohos.sh` (compile+sign) → copy signed .so into wheel → `pip wheel` (produces `cann_lm_engine-*+signed*.whl`) → final `ls build_out/cann_lm_engine-*+signed*.whl`. SDK/clang-18/GTest/signing are all handled inside `build_ohos.sh`.

## 5. Artifacts

Artifacts are flat in `build_ohos/` (default; `BUILD_DIR` environment variable can override):

| File | Description |
|------|------|
| `libcann_lm_engine.so` | Engine SHARED library (ELF AArch64, NEEDED=`libc.so`(musl)+`libc++_shared.so`) |
| `libc++_shared.so` | HarmonyOS C++ runtime library, auto-copied from sysroot during build |
| `model_executor_ut` | Core component UT (kv_cache/input_assembler/model_executor/dfx/tensor_manager) |
| `ut_engine_executor` / `ut_tokenizer` / `ut_c_api` / `ut_scheduler` / `ut_text_embedding` | Component UTs |
| `bbit` | Black-box integration (`--check` + full flow) |
| `api_smoke` / `c_api_smoke` | C++/C API smoke programs |
| `*_signed` / `*_signed.so` for each of the above | Signed version (auto-produced by `build_ohos.sh`) |

A total of **2 .so + 9 executables**, each with a `*_signed` copy after signing.

### Confirm OHOS musl Target (not glibc)

```bash
cd build_ohos
readelf -d libcann_lm_engine.so | grep NEEDED
# Expected: NEEDED libc++_shared.so, libc.so   ← libc.so instead of libc.so.6 = musl/OHOS
readelf -h libcann_lm_engine.so | grep Machine
# Expected: Machine: AArch64
```
> These artifacts **cannot be run directly** on a Linux host (they link against musl's `libc.so`; the host uses glibc). Runtime verification must be on a HarmonyOS device.

### Produce Signed Wheel (`build_package_ohos.sh`)

`build_package_ohos.sh` automatically copies the signed `.so` into `cann_local_lm/lib/` and runs `pip wheel` to produce `build_out/cann_lm_engine-*+signed*.whl` (embeds `.so` + `libc++_shared.so`) after compile+sign.

On device: `pip install cann_lm_engine-*+signed*.whl` then `cllm generate --config testcase.json` (see [Quick Start](../../QUICKSTART.md)). C++ tests (bbit/smoke) require separate `bash build.sh --smoke`, then push to device and run `./bbit --config testcase.json`.

## 6. Three Inevitable Pitfalls of HarmonyOS Cross-Compilation (first-time environment setup)

> Route A (full SDK + bundled clang-15): all three pitfalls auto-solved (clang-15 finds builtins in its own resource dir; libc++/libcxx-ohos header/library paths are identical for both routes, toolchain already configured). The table below mainly applies to Route B (host clang-18) first-time environment setup.

| # | Symptom | Root Cause | Solution | Route A |
|---|------|------|------|------|
| 1 | `libclang_rt.builtins.a` / `crtbegin.o` not found | clang-18 only searches its own resource dir (`/usr/lib/clang/18/...`) | Copy NDK's builtins/crt into `/usr/lib/clang/18/lib/aarch64-linux-ohos/` (see end of §2.2) | **Auto-solved** (clang-15 resource dir has it built-in) |
| 2 | `-lunwind` / `-lc++` not found | Static libraries are in `llvm/lib/aarch64-linux-ohos/`, not in sysroot standard paths | Add `-L${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos` to link flags (`ohos.toolchain.cmake` already adds it) | Same (paths identical) |
| 3 | `<__config_site>` not found | Wrong libc++ header set used: the generic `llvm/include/c++/v1/` lacks generated headers | Use `llvm/include/libcxx-ohos/include/c++/v1/` instead (the OHOS-configured version with `__config_site`, toolchain already configured) | Same (paths identical) |

> These three pitfalls are handled in `ohos.toolchain.cmake` and the SDK extraction stage; daily compilation will not encounter them again, only referenced for Route B first-time environment setup.

## 7. FAQ

| # | Symptom | Cause / Solution |
|---|------|------------|
| 1 | `ERROR: OHOS_SDK_NATIVE invalid (no sysroot subdirectory)` | SDK path is wrong; confirm `${OHOS_SDK_NATIVE}/sysroot` exists; `export OHOS_SDK_NATIVE=<absolute path to native directory>` |
| 2 | `ERROR: SDK has no llvm/bin/clang and host has no clang-18` | Neither route is satisfied: Route A places the full SDK at `cli_native` (with bundled clang); Route B runs `apt install clang-18 llvm-18` |
| 3 | CMake `FATAL_ERROR: GTest source tarball not found: /mnt/workspace/gtest-1.14.0.tar.gz` | OHOS uses FetchContent local source, not `find_package(GTest)`. Download googletest v1.14.0 to that default path, or override with `-DGTEST_TAR=<your tarball path>` |
| 4 | Signing step reports `java not in PATH` | `apt install default-jdk`; confirm `java -version` produces output |
| 5 | Changed code but artifact md5 unchanged | Incremental state used stale artifacts. `rm -rf build_ohos` or `build_package_ohos.sh --clean` then rerun |
| 6 | Push rejected (master protected branch) | master forbids direct push. Use a feature branch + MR: `git push origin HEAD:refs/heads/oxt/<feature-name>` |
| 7 | Lots of `--gcc-toolchain=... argument unused` warnings | Known behavior of OHOS toolchain on host, **harmless**, does not affect artifacts |

## 8. Environment Variable Quick Reference

| Variable | Meaning | Default/Example |
|------|------|----------|
| `OHOS_SDK_NATIVE` | Absolute path to OHOS SDK `native/` (includes sysroot/llvm) | `/mnt/workspace/ohos_sdk/ohos_native/native` |
| `BUILD_DIR` | `build_ohos.sh` output directory | `<repo>/build_ohos` |
| `CMAKE_BUILD_TYPE` | Build type | OHOS:Release |
| `GTEST_TAR` | GTest tarball path (CMake variable) | `/mnt/workspace/gtest-1.14.0.tar.gz` |

## 9. Current Status (measured 2026-10-20)

- ✅ **Both SDK routes verified passing**:
  - Route A (full command-line-tools SDK + bundled clang-15 ARM-native): 11 artifacts all green, **pitfall #1 auto-solved** (no need to补 `/usr/lib/clang/18`, no host clang-18 needed)
  - Route B (minimal OpenHarmony mirror SDK + host clang-18 + 补 builtins): 11 artifacts all green
- ✅ `libcann_lm_engine.so` + `libc++_shared.so` cross-compile passed (ELF AArch64, musl target; `readelf -d` NEEDED `libc.so` not `libc.so.6`)
- ✅ **9 executables cross-compile passed** (6 UT + bbit + 2 smoke; GTest local source FetchContent, static `add_test`)
- ✅ **All artifacts signed** (`*_signed`, host java-21 + binary-sign-tool.jar `-selfSign 1`)
- ✅ **Signed wheel produced** (`cann_lm_engine-*+signed*.whl` with embedded .so + libc++_shared.so)
- ✅ `model_executor_ut` / `ut_text_embedding` restored (backend hiai migration complete)
- ⏳ ut_sampler/ut_request_handle/ut_data_type_utils and other old backend API UTs were not migrated with hiai, not registered yet (source rewrite is a separate work item)
- ⏳ HarmonyOS device runtime verification (awaiting device; push signed wheel to device, pip install)

## 10. Submit Code

```bash
# Commit message format
git commit -m "DTS:8888888888888 Description:<one-line English>
Details: <supplementary details>
Change-Id: I<40-hex>"

# Generate Change-Id (when uuidgen is unavailable)
python3 -c "import uuid; print('I' + uuid.uuid4().hex)"
```

Push to a feature branch (master is a protected branch):
```bash
git push origin HEAD:refs/heads/oxt/<your-feature-name>
```
