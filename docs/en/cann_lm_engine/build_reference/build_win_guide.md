English | [简体中文](../../../zh/cann_lm_engine/build_reference/build_win_guide.md)

# Windows Build Guide (OHOS Cross-Compilation inside WSL)

> Goal: On a Windows PC, use WSL (Windows Subsystem for Linux) with host clang-18 + minimal OHOS SDK to cross-compile the engine + bbit, producing signed deployable artifacts.
> Maintenance date: 2026-10-15. Companion: `build_linux_guide.md` (the Linux native version of the same flow), `build_guide.md` (overview).

> **Key prerequisite**: The current build scripts (`scripts/cann_lm_engine/*.sh`) are **pure Linux logic** (host `clang-18` + `cmake -G "Unix Makefiles"` + `make`), **with no Windows-native clang.exe/MinGW/`.exe` branch** (the old clang.exe/shim logic has been deleted). Therefore, on Windows you must run via **WSL**, with a flow identical to the Linux guide. **Do not** run natively in Git Bash (Git Bash has no clang-18/make).

---

## 1. Dependency List (inside WSL)

| # | Dependency | How to Obtain | Purpose |
|---|------|---------|------|
| 1 | WSL (Ubuntu distribution) | Built into Windows 10/11 (`wsl --install -d Ubuntu`) or Microsoft Store | Linux runtime host (scripts depend on clang-18/make/bash) |
| 2 | clang-18 / clang++-18 | Inside WSL: `apt install clang-18 llvm-18` | Cross-compiler (`find_program(clang-18)` inside `ohos.toolchain.cmake`) |
| 3 | cmake >= 3.16 + make | Inside WSL: `apt install cmake build-essential` | Builder (`cmake -G "Unix Makefiles"` + `make -j$(nproc)`) |
| 4 | JDK (1.8+, 17/21 tested working) | Inside WSL: `apt install default-jdk` | Signing tool `binary-sign-tool.jar` requires java |
| 5 | OHOS SDK (minimal native subtree) | See §2 | sysroot + llvm/lib (builtins/libc++) + official ohos.toolchain.cmake |
| 6 | GTest source tarball | Download github googletest v1.14.0 | OHOS CMakeLists FetchContent local reference (default `/mnt/workspace/gtest-1.14.0.tar.gz`) |
| 7 | binary-sign-tool.jar | `tools/binary-sign-tool.jar` (not in repo, see tools/README.md) | Artifact self-signing |
| 8 | Code repo | CodeHub clone (see §3) | Engine + bbit source |

> The minimal SDK **has no `llvm/bin` (no bundled clang), no `build-tools` (no bundled cmake/ninja)** — compiler uses host clang-18, builder uses system cmake + make, **does not depend on SDK-bundled toolchain binaries**.

---

## 2. Obtain the OHOS SDK (extract minimal version inside WSL)

This flow does not use the full Windows SDK installed by DevEco Studio (which only contains `.exe`, **cannot** be used for the WSL+clang-18 flow). Instead, it uses the **minimal Linux SDK** extracted from the OpenHarmony open-source mirror: only sysroot + llvm/lib (builtins/libc++ static libs and headers) + official `ohos.toolchain.cmake`.

```bash
# Inside WSL
mkdir -p /mnt/workspace/ohos_sdk && cd /mnt/workspace/ohos_sdk
curl -L -o sdk.tar.gz \
  'https://repo.huaweicloud.com/openharmony/os/5.0.0-Release/ohos-sdk-windows_linux-public.tar.gz'

# The outer tar contains nested zips; extract the linux native one
tar -xzf sdk.tar.gz linux/native-linux-x64-5.0.0.71-Release.zip
NATIVE_ZIP=linux/native-linux-x64-5.0.0.71-Release.zip

# Extract what's needed: sysroot + libc++ headers/libs + builtins + official toolchain reference
unzip -oq $NATIVE_ZIP \
  'native/sysroot/*' \
  'native/build/cmake/*' \
  'native/llvm/lib/aarch64-linux-ohos/*' \
  'native/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/*' \
  'native/llvm/include/libcxx-ohos/include/c++/v1/*' \
  -d ohos_native

export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native
```

> Directory structure after extraction (note **no `llvm/bin`, no `build-tools`**):
> ```
> $OHOS_SDK_NATIVE/
>   ├── sysroot/                                   ← target system headers/libs (musl libc)
>   ├── build/cmake/ohos.toolchain.cmake           ← official toolchain file (reference)
>   ├── llvm/lib/aarch64-linux-ohos/               ← libc++/unwind static libs
>   ├── llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos/  ← builtins/crt
>   └── llvm/include/libcxx-ohos/include/c++/v1/    ← libc++ headers (with __config_site)
> ```

### One-time Patch: Install builtins/crt into clang-18's resource dir

clang-18 only searches its own resource dir (`/usr/lib/clang/18/lib/aarch64-linux-ohos/`) for `libclang_rt.builtins.a` / crtbegin/end; the SDK's copy must be placed there (see §5 pitfall ⑥ for details):
```bash
SRC=$OHOS_SDK_NATIVE/llvm/lib/clang/15.0.4/lib/aarch64-linux-ohos
DST=/usr/lib/clang/18/lib/aarch64-linux-ohos
sudo mkdir -p "$DST"
sudo cp "$SRC"/{libclang_rt.builtins.a,clang_rt.crtbegin.o,clang_rt.crtend.o} "$DST"/
```

### Verify the SDK
```bash
[ -d "$OHOS_SDK_NATIVE/sysroot" ] && echo "SDK ready: $OHOS_SDK_NATIVE"
clang-18 --version     # host clang-18 (not SDK-bundled)
cmake --version        # host cmake
```

---

## 3. Get the Code

```bash
# Inside WSL, clone to the WSL filesystem (do not place under /mnt/c/, see §5 pitfall ②)
git clone ssh://szv-y.codehub.huawei.com:2222/h00651112/cann_llm_open.git
cd cann_llm_open
```
- Write access requires the repo owner to grant permission (otherwise reports CH.00905401).
- Clone to a **pure-English path** (Chinese paths will definitely break the build, see §5 pitfall ③).

---

## 4. Compile

Scripts have been migrated from the top-level `scripts/` to `scripts/cann_lm_engine/`.

```bash
# Inside WSL, set the SDK path for each new shell (or persist in ~/.bashrc)
export OHOS_SDK_NATIVE=/mnt/workspace/ohos_sdk/ohos_native/native

cd <repo>

# Option 1: Compile + sign only (artifacts flat in build_ohos/)
bash scripts/cann_lm_engine/build_ohos.sh

# Option 2: One-click compile + sign + package deploy directory
bash scripts/cann_lm_engine/build_package_ohos.sh
# Add --clean for auto-clean full rebuild of build_ohos:
bash scripts/cann_lm_engine/build_package_ohos.sh --clean
```

### Green Criteria
Output ends with `=== build_ohos.sh 完成 ===` (Option 1) or `=== build_package_ohos.sh 完成 ===` (Option 2), and **the make stage has no errors**.

### Build Flow Description (5 steps inside build_ohos.sh)
1. **SDK validation**: Takes `OHOS_SDK_NATIVE` (default `/mnt/workspace/ohos_sdk/ohos_native/native`), validates `sysroot` subdirectory exists; checks host `clang-18`/`clang++-18` (`command -v`, prompts `apt install clang-18 llvm-18` if missing).
2. **CMake configure**: `cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=<repo>/cann_lm/engine/ohos/ohos.toolchain.cmake -DOHOS_SDK_NATIVE=... -DOHOS_PLATFORM=OHOS -DCMAKE_BUILD_TYPE=Release -B build_ohos -S cann_lm/engine/ohos` (toolchain uses `find_program(clang-18)`, **does not** use SDK-bundled clang).
3. **make build**: `cmake --build build_ohos -j$(nproc)` (Unix Makefiles, **not ninja**).
4. **Artifact validation**: Validates 11 artifacts (see table below) are present; if `libc++_shared.so` is missing, fallback-copy from SDK.
5. **Signing**: Auto-calls `sign_ohos.sh`, for each of 2 .so + 9 executables runs `java -jar binary-sign-tool.jar sign ... -selfSign 1`.

### Artifacts (flat in `build_ohos/`, 11 total + each `*_signed`)

| File | Description |
|------|------|
| `libcann_lm_engine.so` (+`_signed.so`) | Engine SHARED library (aarch64-linux-ohos / musl) |
| `libc++_shared.so` (+`_signed.so`) | C++ runtime library (auto-copied from sysroot during build) |
| `model_executor_ut` (+`_signed`) | Core component UT (kv_cache/input_assembler/model_executor/dfx/tensor_manager) |
| `ut_engine_executor` / `ut_tokenizer` / `ut_c_api` / `ut_scheduler` / `ut_text_embedding` (each +`_signed`) | Component UTs |
| `bbit` (+`_signed`) | Black-box integration (`--check` + full flow) |
| `api_smoke` / `c_api_smoke` (each +`_signed`) | C++/C API smoke programs |

### Packaging (auto-run by Option 2, or run standalone)
### Packaging (auto-executed by Option 2)
`build_package_ohos.sh` copies the signed `.so` into `cann_lm/lib/` and runs `pip wheel` to produce `build_out/cann_lm_engine-*+signed*.whl` (embeds `.so` + `libc++_shared.so`) after compile+sign. On the HarmonyOS device, `pip install` then `cllm generate --config testcase.json` to run inference.

---

## 5. FAQ (Windows / WSL-specific)

| # | Symptom | Cause / Solution |
|---|------|------------|
| 1 | Running natively in Git Bash reports `clang-18: command not found` / `make: not found` | Scripts are Linux logic, **must use WSL**, do not run in Git Bash. Enter Ubuntu via `wsl` then execute |
| 2 | Repo placed under `/mnt/c/...` builds slowly / executable bits lost | Cross-filesystem (Windows ↔ WSL) I/O is slow and permission bits are easily lost. Clone to the WSL filesystem (`~/CANN-LM`, corresponding to `\\wsl$\<distro>\home\<user>\CANN-LM`) |
| 3 | Path contains Chinese → compile/link failure | Avoid Chinese paths even inside WSL, use pure English |
| 4 | Changed code but artifacts unchanged | Incremental state used stale artifacts. `rm -rf build_ohos` or `bash scripts/cann_lm_engine/build_package_ohos.sh --clean` then rerun |
| 5 | `ERROR: clang-18/clang++-18 not found` | Inside WSL: `apt install clang-18 llvm-18` (ohos.toolchain.cmake depends on host clang-18) |
| 6 | `libclang_rt.builtins.a` / `crtbeginS.o` not found | The §2 builtins patch was not done. Copy the SDK's builtins/crt into `/usr/lib/clang/18/lib/aarch64-linux-ohos/` |
| 7 | `ERROR: OHOS_SDK_NATIVE invalid` | SDK path is wrong; confirm `sysroot/` subdirectory exists; `export OHOS_SDK_NATIVE=<native absolute path>` |
| 8 | Signing reports `java not in PATH` | Inside WSL: `apt install default-jdk`, confirm `java -version` produces output |
| 9 | `GTest source tarball not found` | Missing `/mnt/workspace/gtest-1.14.0.tar.gz`. Download github googletest v1.14.0 to that path, or `cmake -DGTEST_TAR=<path>` |
| 10 | Lots of `argument unused: '--gcc-toolchain=...'` warnings | Known behavior of OHOS toolchain on Linux, **harmless**, does not affect artifacts |
| 11 | DevEco Studio-installed SDK unusable | DevEco Windows SDK only contains `.exe`; this flow requires the minimal Linux SDK (with sysroot + llvm/lib, see §2) |
| 12 | Push rejected | master is a protected branch, use a feature branch + MR (see §6) |

---

## 6. Submit Code

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
