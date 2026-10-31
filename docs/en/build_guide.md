# Build Guide

English | [简体中文](../zh/build_guide.md)

> This document focuses on **cann-device-lm build**: build prerequisites (Part 1) → one-click build + install + usage (Part 2) → quick reference/FAQ (Part 3). For environment dependencies, see [env_dependencies.md](env_dependencies.md); for end-to-end examples, see [Quick Start](QUICKSTART.md).
>
> Unless otherwise specified, all commands below are run from the **cann-device-lm repo root** (`cd /path/to/cann-device-lm`). The CANN install path uses `$HOME/Ascend` as an example (adjust as needed; all subsequent `source` command paths must be replaced accordingly).

---

# Part 1: Build Prerequisites

> Aligns with [env_dependencies.md](env_dependencies.md) Part 1 "Build Dependencies". Runtime dependencies (P1 venv / P2 torch_npu) are in env_dependencies Part 2.

## System-level Base Tools

| Tool | Version Requirement | Notes |
|------|---------------------|-------|
| CMake | >= 3.18 | C++ build |
| Python | >= 3.10 | whl packaging (pip) |
| gcc / g++ | >= 8.0 and < 14.0 | `-Wall -Wextra -Werror` zero warnings |
| make | GNU Make | comes with `build-essential` |
| patchelf | | `.run` package sets .so rpath (skip if missing) |
| git-lfs | | pull HF model weights |
| pip | | whl packaging |
| **java (JRE)** | any | only `--pkg` [3/4] OHOS signing needs it |

One-click install (Ubuntu/Debian):

```bash
apt install -y cmake build-essential python3 python3-venv python3-pip patchelf git-lfs default-jre
```

## CANN

CANN is **externally preinstalled** (the user sources its `set_env.sh` first); this repo does not build it.

| Component | Version | Source |
|-----------|---------|--------|
| CANN toolkit | >= 9.3.0 | [master download](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) |
| CANN ops | by SOC | Ascend910 → `Ascend910-ops`; kirinx90 → `ops-nn`/`ops-math`/`ops-transformer` |

Install (values below are examples; adjust by actual version/SOC/architecture):

```bash
# Example: CANN 9.3.0 / aarch64 / Ascend910 / install to ~/Ascend
install_path=$HOME/Ascend
cann_version=9.3.0
arch=aarch64
soc_name=Ascend910

# 1. toolkit
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}

# 2. ops operator package
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}

# 3. Create symlink (GE finds platform_config via this path)
ln -s ${install_path}/cann-${cann_version} ${install_path}/cann
```

Verify after install:

```bash
source $HOME/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH   # Expected: .../cann-9.3.0/cann-9.3.0
```

> All subsequent `source $HOME/Ascend/cann/set_env.sh` commands point to this path. If installed elsewhere, replace `$HOME/Ascend` with the actual path.

## OHOS Cross-Compilation Suite (only needed by `build.sh --pkg` [3/4])

> Only required by the engine cross-compilation stage of `build.sh --pkg`. Not needed for everyday `scripts/build.sh cpp/wheel`.

### Download

command-line-tools must be downloaded manually via browser (the API returns 403 and requires login):

1. Visit [Huawei Developer - Command Line Tools](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos)
2. Log in with a Huawei developer account
3. Download **Command Line Tools for Linux(ARM)** (e.g. `commandline-tools-linux-arm64-26.0.0.851.zip`)

### Contents

After unzipping, the archive contains:

| Directory | Contents |
|-----------|----------|
| `command-line-tools/sdk/default/openharmony/native/llvm/bin/clang` | OHOS-specific compiler (clang 15.0.4, ARM-native, ships with builtins/crt) |
| `command-line-tools/sdk/default/openharmony/native/sysroot/` | OHOS musl sysroot (HarmonyOS libc headers/libs) |
| `command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar` | Binary signing tool (HarmonyOS requires .so signing) |
| `command-line-tools/sdk/default/openharmony/native/build/cmake/ohos.toolchain.cmake` | Official OHOS CMake toolchain |

### Install and Configure

```bash
cd /path/to/cann-device-lm

# 1. Unzip
unzip -oq commandline-tools-linux-arm64-*.zip -d /your/ohos-sdk-path

# 2. Set env var (point to the native subdirectory)
export OHOS_SDK_NATIVE=/your/ohos-sdk-path/command-line-tools/sdk/default/openharmony/native

# 3. Copy signing tool into repo tools/ (build.sh hard-checks tools/binary-sign-tool.jar)
mkdir -p tools/
cp /your/ohos-sdk-path/command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar tools/

# 4. Install java (the signing step requires it)
apt install -y default-jre
```

### Verify

```bash
$OHOS_SDK_NATIVE/llvm/bin/clang --version   # Expected: clang 15.0.4
ls $OHOS_SDK_NATIVE/sysroot                 # Expected: directory exists
ls tools/binary-sign-tool.jar                # Expected: file exists
```

> GTest is auto-downloaded by `--pkg` from gitcode into `output/third_party/gtest/`; no manual setup needed.

---

## Runtime Prerequisite: P1 converter venv

> ⚠️ **Only needed at converter runtime (exporting epair); `--pkg` build does not need it.** Skip to Part 2 if you only need `.run` artifacts.

| Component | Source |
|-----------|--------|
| torch-delegate | In-repo `torch_delegate/` (includes `build.sh`, auto-creates `.venv` + torch 2.9.0+cpu) |
| torchair | Choose one of the two methods below |

```bash
source $HOME/Ascend/cann/set_env.sh
cd /path/to/cann-device-lm

# 1. torch-delegate: build from in-repo source (build.sh auto-creates .venv [torch 2.9.0+cpu] + compiles whl)
cp -a torch_delegate deps/torch-delegate
cd deps/torch-delegate && ./build.sh && .venv/bin/pip install --no-deps output/package/torch_delegate-*.whl

# 2. torchair (choose one):
```

**Option A: Build from source** (`.so` linked against CANN stubs, no torch_npu needed for converter)

```bash
# .venv/bin/pip is the P1 venv created by torch-delegate build.sh above
cd /path/to/cann-device-lm
git clone https://gitcode.com/Ascend/torchair.git deps/torchair && \
  cd deps/torchair && TARGET_PYTHON_PATH="$(realpath ../torch-delegate/.venv/bin/python)" bash build.sh -c && \
  ../torch-delegate/.venv/bin/pip install --no-deps output/torchair-*.whl
```

**Option B: Extract from torch_npu whl** (`.so` from official torch_npu build, must match torch version)

```bash
VENV=deps/torch-delegate/.venv
PY_VER=$($VENV/bin/python -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')

# Download torch_npu whl (no install), extract torchair only
pip download torch_npu==2.9.0.post9 \
  --index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/ \
  --no-deps -d /tmp/torch_npu_whl
unzip -o /tmp/torch_npu_whl/torch_npu-*.whl 'torch_npu/dynamo/torchair/*' -d /tmp/torch_npu_extracted

# Copy to P1 venv site-packages, rename dir to torchair
cp -r /tmp/torch_npu_extracted/torch_npu/dynamo/torchair \
  $VENV/lib/$PY_VER/site-packages/torchair
```

> Both methods produce torchair with the same `.so` files (`_torchair.so`/`_npu_graph_executor.so` etc.), differing only in build environment: Option A uses CANN stubs, Option B uses official torch_npu build. The converter venv does not install torch_npu; Option A is the recommended path.

### Verify

```bash
bash scripts/build.sh check
```

---

# Part 2: Build and Deploy

## One-click Build

```bash
cd /path/to/cann-device-lm
source $HOME/Ascend/cann/set_env.sh
export OHOS_SDK_NATIVE=<extract-dir>/command-line-tools/sdk/default/openharmony/native

bash build.sh --pkg
```

> `--pkg` auto-downloads and builds third-party libs (nlohmann_json / GTest / makeself, from the gitcode mirror); downloads on first run, idempotently skipped afterward.

Artifacts under `dist/` are two `.run` files: `cann_lm_compile_converter-*.run` (Ascend910 build host) and `cann_lm_engine_ohos-*.run` (HarmonyOS device).

## Install

```bash
# Ascend910 build host
bash dist/cann_lm_compile_converter-0.1.0-aarch64.run --extract=/opt/cann-device-lm

# HarmonyOS device
bash dist/cann_lm_engine_ohos-0.1.0-aarch64.run --extract=/data/local/tmp/cann_lm_engine
```

## Usage

```bash
# Compile omc on Ascend910 (the build host must have CANN + driver installed first)
source $HOME/Ascend/cann/set_env.sh    # CANN (provides lib64/driver/opp)
source /opt/cann-device-lm/set_env.sh         # cann-device-lm
/opt/cann-device-lm/cann_lm_compile <ep_json> <weight_info> [output_name]

# HarmonyOS inference
cd /data/local/tmp/cann_lm_engine && ./run.sh
```

> For end-to-end examples, see [Quick Start](QUICKSTART.md).

---

# Part 3: Quick Reference

## Version Configuration (`scripts/build.conf`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CANN_VERSION` | `9.3.0` | CANN version (>= 9.3.0) |
| `TORCH_VERSION` | `2.9.0` | P1 venv torch version |
| `SOC_VERSION` | `Ascend910_9362` | omc build target SOC |
| `CONVERTER_PYTHON` | `${TORCH_DELEGATE_HOME}/.venv/bin/python` | P1 venv python |
| `DOPT_PYTHON` | _(auto-detected)_ | P2 system python (used for NPU quantization) |

> Any variable already set in the environment is not overridden. To switch versions/paths, edit `scripts/build.conf` or export overrides.

## scripts/build.sh subcommand flags

> The root `build.sh` only has `--pkg` (full build + package) and `-u` (UT compile + run). The following are `scripts/build.sh` subcommands (for daily development).

| subcommand | flags | Description |
|-------------|-------|-------------|
| `cpp` | `--debug`/`--release`(default) / `--asan` / `--no-tests` / `--no-compile-tool` / `--target <name>` | Build `libtorch_delegate_backend.so` + `cann_lm_compile` |
| `tests` | `--no-asan` / `--skip-run` | ASAN build + ctest |
| `third_party` | `--third_party_path=PATH` / `--jobs=N` | Build nlohmann_json + GTest + makeself |
| `deps` | (none) | Install cann_lm wheels into the P1 venv |
| `wheel` | (none) | cann_lm + cann_lm_engine whl |
| `package` | (passthrough) | Package `cann_lm_compile-*.run` |
| `all` | (passthrough) | deps → third_party → cpp → wheel |

## FAQ

### Q1: ASCEND_HOME_PATH not set

```bash
source $HOME/Ascend/cann/set_env.sh
```

### Q2: torchair / torch_delegate not installed

See the "Runtime Prerequisite: P1 converter venv" section; install torch-delegate (produces `.venv`) → torchair in order, then `bash scripts/build.sh deps && bash scripts/build.sh check`.

### Q3: omc compilation exit code

`cann_lm_compile` uses `std::quick_exit(0)` to skip GE library static destruction; normal exit code is 0.

### Q4: CANN version not found

Check whether a CANN run package is available locally; if not, download from [master](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master). After installing, run `ln -s cann-9.3.0 cann`.

### Q5: Do I need to change DOPT_PYTHON on a different machine?

No. `_env_common.sh`'s `_resolve_dopt_python` auto-detects a python that can `import torch_npu`.

### Q6: `$DOPT_PYTHON` reports `No module named 'torch_npu'` / `_ctypes` undefined symbol

`LD_LIBRARY_PATH` does not include the python `<prefix>/lib` → loads the system's old libpython. `_env_common.sh` handles this automatically; when running manually:

```bash
export LD_LIBRARY_PATH=$(dirname $(dirname $DOPT_PYTHON))/lib:${LD_LIBRARY_PATH:-}
```

### Q7: torchaudio version mismatch `undefined symbol: torch_library_impl`

torchaudio must match the torch version. Fix:

```bash
pip install torchaudio==2.9.0 --force-reinstall --no-deps --user   # Match torch 2.9.0
```

> See [env_dependencies.md](env_dependencies.md) §2.2 for details.

### Q8: OHOS cross-compilation shows many `clang++: warning: argument unused`

Redundant flags passed by the OHOS toolchain cmake; **harmless** and does not affect build results.

## Related Files

- `build.sh` — root build entry (`--pkg` full build + `-u` UT)
- `scripts/build.sh` — daily development build entry (subcommand dispatch)
- `scripts/build.conf` — version numbers + path defaults
- `scripts/set_env.sh` — unified converter + compile env
- `scripts/cann_lm_compile/build_cpp.sh` — C++ build
- `scripts/cann_lm_compile/package_run.sh` — package the `.run`
- [env_dependencies.md](env_dependencies.md) — environment dependency description
- [Quick Start](QUICKSTART.md) — end-to-end examples
