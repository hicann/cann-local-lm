# Build Guide

English | [简体中文](../zh/build_guide.md)

> This document focuses on **cann-local-lm build**: build prerequisites (Part 1) → one-click build + install + usage (Part 2) → quick reference/FAQ (Part 3). For environment dependencies, see [env_dependencies.md](env_dependencies.md); for end-to-end examples, see [Quick Start](QUICKSTART.md).
>
> Unless otherwise specified, all commands below are run from the **cann-local-lm repo root** (`cd /path/to/cann-local-lm`). The CANN install path uses `$HOME/Ascend` as an example (adjust as needed; all subsequent `source` command paths must be replaced accordingly).

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

CANN is **externally preinstalled** (the user sources its `set_env.sh` first); this repo does not build it. Currently only x86 is supported.

| Component | Version | Source |
|-----------|---------|--------|
| CANN toolkit mobile station | >= 9.3.0 | [cann-mobile-station](https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/Ascend-cann-toolkit_9.3.0_linux-x86_64-mobile-station.run) |
| CANN ops | KirinX90 | [KirinX90](https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/Ascend-cann-kirinx90-ops_9.3.0_linux-x86_64.run) |

Install (values below are examples; adjust by actual version/SOC/architecture):

```bash
# Example: CANN 9.3.0 / x86_64 / KirinX90 / install to ~/Ascend
install_path=$HOME/Ascend
cann_version=9.3.0
arch=x86_64
soc_name=kirinx90

# 1. toolkit (package name carries the -mobile-station suffix)
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run
./Ascend-cann-toolkit_${cann_version}_linux-${arch}-mobile-station.run --install --install-path=${install_path}

# 2. ops operator package
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}

# 3. Create symlink (GE finds platform_config via this path)
ln -s ${install_path}/cann-${cann_version} ${install_path}/cann
```

Verify after install:

```bash
source $HOME/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH   # Expected: .../cann-9.3.0 (single level, the toolkit install root)
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
cd /path/to/cann-local-lm

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
cd /path/to/cann-local-lm

# 1. torch-delegate: build from in-repo source (build.sh auto-creates .venv [torch 2.9.0+cpu] + compiles whl)
mkdir -p deps && cp -a torch_delegate deps/torch-delegate
cd deps/torch-delegate && ./build.sh && .venv/bin/pip install --no-deps output/package/torch_delegate-*.whl

# 2. torchair (choose one):
```

**Option A: Build from source** (no torch_npu needed for converter)

```bash
# .venv/bin/pip is the P1 venv created by torch-delegate build.sh above
cd /path/to/cann-local-lm
git clone https://gitcode.com/Ascend/torchair.git deps/torchair
cd deps/torchair

# ⚠️ Both variables must be set, and do NOT use realpath:
#   1. realpath resolves the .venv/bin/python symlink to the system python path, losing venv activation
#      → finds system torch instead of venv torch. Use $(cd ../torch-delegate && pwd)/.venv/bin/python
#      to get an absolute path while preserving the symlink.
#   2. torchair build.sh main() overwrites PYTHON_BIN_PATH=$(which python3) when TARGET_PYTHON_PATH is empty
#      (build.sh:136-139), so TARGET_PYTHON_PATH must also be set to prevent the overwrite.
# Also: clear stale tools/TORCH_VERSION cache (its presence makes configure skip re-detection, configure:13).
rm -f tools/TORCH_VERSION tools/TORCH_INSTALLED_PATH tools/PYTHON_BIN_PATH tools/COMPILE_FLAGS
rm -rf build/CMakeCache.txt build/CMakeFiles
VENV_PY="$(cd ../torch-delegate && pwd)/.venv/bin/python"
PYTHON_BIN_PATH="$VENV_PY" TARGET_PYTHON_PATH="$VENV_PY" bash build.sh -c

# Install whl (--no-deps avoids pulling in torch_npu; torchair runtime deps need manual install)
../torch-delegate/.venv/bin/pip install --no-deps output/torchair-*.whl
../torch-delegate/.venv/bin/pip install packaging cloudpickle
```

> **Verify configure picked the right torch**: after build, `cat tools/TORCH_VERSION` should show `20900` (torch 2.9.0). If it shows `20701` (torch 2.7.1), the build used the system python's torch — `_torchair.so` will have an ABI mismatch and `import` will fail with `undefined symbol: _ZN2at14RecordFunction6beforeEPKcl`. Clear the `tools/TORCH_VERSION` cache and rebuild with the correct `PYTHON_BIN_PATH`.

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

> Both methods produce torchair with the same `.so` files (`_torchair.so`/`_npu_graph_executor.so` etc.), differing only in build environment: Option A builds from source (linked against local CANN), Option B uses official torch_npu build. The converter venv does not install torch_npu; Option A is the recommended path.
>
> **Option A dependencies**: `--no-deps` skips torchair's declared runtime deps (`protobuf`/`decorator`/`psutil`/`scipy` are usually already provided by CANN site-packages or torch, but `packaging`/`cloudpickle` must be installed manually — `torchair/__init__.py` begins with `from packaging import version`, and a missing `packaging` causes `ModuleNotFoundError`). Alternatively, `pip install -r scripts/requirements-p1.txt` installs them all at once.

### Verify

```bash
# Requires source $HOME/Ascend/cann/set_env.sh first
TORCH_DEVICE_BACKEND_AUTOLOAD=0 deps/torch-delegate/.venv/bin/python -c "import torchair; print('OK', torchair.__file__)"
# Expected: OK /mnt/workspace/.../deps/torch-delegate/.venv/lib/python3.12/site-packages/torchair/__init__.py
```

> This only verifies that `import torchair` succeeds (catches ABI mismatch / missing deps early). The full baseline check (including cann_local_lm / third_party) is at the end of Part 2: "[Environment Check](#environment-check)".

---

# Part 2: Build and Deploy

## One-click Build

```bash
cd /path/to/cann-local-lm
source $HOME/Ascend/cann/set_env.sh
export OHOS_SDK_NATIVE=<extract-dir>/command-line-tools/sdk/default/openharmony/native

bash build.sh --pkg
```

> `--pkg` auto-downloads and builds third-party libs (nlohmann_json / GTest / makeself, from the gitcode mirror); downloads on first run, idempotently skipped afterward.

Artifacts go under `build_out/` (controlled by `build.sh`'s `DIST` variable, default `${REPO}/build_out`, overridable via environment):
- `cann_local_lm_converter-*.run` (Ascend910 build host, **required**) — contains `cann_lm_compile` + `libtorch_delegate_backend.so` + `libcustom_pass_plugin.so` + `op_cpu/` + `op_pass/` + cann_lm whl + set_env.sh
- `cann_local_lm_engine-*.whl` (engine python ctypes binding whl, pure python without .so)
- `deploy_ohos/` (engine OHOS cross-compile artifacts: signed `libcann_lm_engine.so` + UT/bbit/smoke, **requires OHOS SDK**; skipped when SDK is absent, does not affect the 910 OM compilation flow)

> The converter whl is packed inside the `.run` under `wheels/` and removed from `build_out/` after packaging.

## Install

```bash
# Ascend910 build host
bash build_out/cann_local_lm_converter-0.1.0-aarch64.run --extract=/opt/cann-local-lm

# HarmonyOS device (engine signed wheel, from --pkg or build_package_ohos.sh)
pip3 install build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl
```

## Usage

```bash
# Compile omc on Ascend910 (the build host must have CANN + driver installed first)
source $HOME/Ascend/cann/set_env.sh    # CANN (provides lib64/driver/opp)
source /opt/cann-local-lm/set_env.sh         # cann-local-lm
/opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> [output_name]

# HarmonyOS inference (see QUICKSTART.md for full deployment flow)
# cllm generate --config testcase.json --prompt "Hello"
# cllm chat --config testcase.json
```

> **Weight separation**: For non-Kirin SOCs, `/opt/cann-local-lm/set_env.sh` does not set `GE_EXTERNAL_WEIGHT`, so the compiled OM **embeds weights** (e.g. ~4.5G/segment for gemma4 5B). For a smaller OM plus external weight files, `export GE_EXTERNAL_WEIGHT=2` before compilation; the output is then `<out>.om` (tens of MB) + a sibling `weight/` directory (external weights, loaded at runtime). Kirin SOCs set `=2` automatically — see the next section.

> ⚠️ **Kirin platform requirement**: When model weights exceed **2 GB**, external weights (`GE_EXTERNAL_WEIGHT=2`) are **mandatory** — the output is `<out>.omc` plus a sibling `weight/` directory, and both are loaded together at runtime. Embedded mode (`GE_EXTERNAL_WEIGHT=0`) is only valid on Kirin for models up to 2 GB; do not set `=0` beyond that size. On Kirin SOCs `set_env.sh` already sets `=2` automatically, so no action is normally needed — only verify model size if you manually `export GE_EXTERNAL_WEIGHT=0`.

## Compile Environment Variables (GE / mobile)

`set_env.sh` sets the variables below based on `SOC_VERSION`: **Kirin family (`KirinX90` / `Kirin9030` / …) automatically enters the mobile compile chain**; other SOCs leave the mobile-related items unset. Manual configuration is normally unnecessary — only `export` a variable to override its default.

| Variable | Default | Purpose | When to change |
|----------|---------|---------|----------------|
| `SOC_VERSION` | `KirinX90` in the combined package; `Ascend910_9362` in the standalone `cann_lm_compile` package | Target SoC, passed to GE as `ge.socVersion`. **Kirin family → mobile compile chain** (`.omc` + `MobileModelHelper` path); others → generic `.om` path | When targeting a different chip. This is the only mode switch — any non-Kirin value disables the mobile branch |
| `GE_FUSION_SWITCH_FILE` | `fusion_autoub_off.cfg` in the package | GE graph-fusion switch config. Auto-UB fusion is off by default | Point to a custom cfg for fusion tuning; temporarily `unset` to fall back to GE defaults when debugging fusion |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `0` | Prevents torch from auto-loading the npu backend | Do not change. The P1 converter venv has no torch_npu; setting `1` makes the import fail |
| `GE_EXTERNAL_WEIGHT` | Package: `2` under mobile, unset otherwise. Dev-mode `set_env.sh`: always `2` | `2` = external weights (`<out>.omc` + sibling `weight/` directory, loaded at runtime); `0` = weights embedded in the omc | **On Kirin, must be `2` when model weights exceed 2 GB** (embedding is only valid up to 2 GB). On non-Kirin, use `0` for single-file distribution when size is acceptable |
| `GE_MOBILE_WEIGHT_SHARE_OPS` | `MatMulV3,QuantBatchMatmulV4,RmsNorm` | Shared-weight op whitelist on the mobile path (read by GE's `[Mobile]` path). Listed ops whose inputs come from the same `FileConstant` store only one weight copy | Add op names here if duplicated weights bloat the artifact; clear it to fall back to GE's built-in default when debugging shared weights |
| `ASCEND_CUSTOM_OPP_PATH` | `<install-dir>/op_cpu` under mobile | GE Python custom-op plugin dir; registers the `Bitcast` / `TransData` HostCpu kernels (int4 layout constant folding) | Do not change. Missing it breaks int4 quantized graph compilation (`TransData` has no kernel) |
| `ASCEND_GE_PY_PASS_PATH` | `<install-dir>/op_pass` under mobile | GE Python pass plugin dir; registers `QbmX2TransDataPass` (`QuantBatchMatmulV4` x2 int4 `ND→FRACTAL_Z`) | Do not change. Missing it reports `FRACTAL_Z infershape requires x2 4D shape` |

Typical usage:

```bash
# Quantized mobile compile (Kirin): set SOC only, the rest is automatic
source $HOME/Ascend/cann/set_env.sh
export SOC_VERSION=KirinX90
source /opt/cann-local-lm/set_env.sh
/opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> <out>   # → <out>.omc + weight/

# Generic Ascend910 compile: a non-Kirin SOC disables the mobile branch
export SOC_VERSION=Ascend910_9362
source /opt/cann-local-lm/set_env.sh
/opt/cann-local-lm/cann_lm_compile <ep_json> <weight_info> <out>   # → <out>.om

# Embedded weights (single-file distribution; on Kirin only for models ≤ 2 GB)
export GE_EXTERNAL_WEIGHT=0        # may be set before or after source set_env.sh (the default applies only when unset)
```

> Development mode (source tree, no `.run` package) works identically — just replace `source /opt/cann-local-lm/set_env.sh` with `source scripts/cann_lm_compile/set_env.sh`; the plugin dirs then point to the source tree's `cann_local_lm/converter/export/{op_cpu,op_pass}`.

> For end-to-end examples, see [Quick Start](QUICKSTART.md).

## Environment Check

After completing the build above (`--pkg` or `deps` + `cpp`), run the baseline check to confirm all components are ready:

```bash
bash scripts/build.sh check
```

Expected output (all ✓):

```
  ✓ CANN ...
  ✓ P1 venv ...
  ✓ torch (2.9.0+cpu)
  ✓ torchair
  ✓ torch_delegate
  ✓ cann_local_lm (converter)
  ✓ third_party json
```

> If `cann_local_lm (converter)` shows ✗: run `bash scripts/build.sh deps` to install into P1 venv (converter only, not engine).
> If `third_party` shows ✗: run `bash build.sh third_party` (or `--pkg` which builds it automatically).
> If `torchair` shows ✗: see [Q2](#q2-torchair--torch_delegate-not-installed).

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
| `deps` | (none) | Install cann_local_lm wheels into the P1 venv |
| `wheel` | (none) | cann_local_lm + cann_lm_engine whl |
| `package` | (passthrough) | Package `cann_lm_compile-*.run` |
| `all` | (passthrough) | deps → third_party → cpp → wheel |

## FAQ

### Q1: ASCEND_HOME_PATH not set

```bash
source $HOME/Ascend/cann/set_env.sh
```

### Q2: torchair / torch_delegate not installed

See the "Runtime Prerequisite: P1 converter venv" section; install torch-delegate (produces `.venv`) → torchair in order, then `bash scripts/build.sh deps && bash scripts/build.sh check`.

If torchair is installed but `check` still shows `✗ torchair`, the `import torchair` is likely failing (check silences errors with `2>/dev/null`). Reproduce to diagnose:

```bash
source $HOME/Ascend/cann/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 deps/torch-delegate/.venv/bin/python -c "import torchair"
```

Common causes:
- **`ModuleNotFoundError: No module named 'packaging'`**: Option A installed torchair with `--no-deps`, skipping runtime deps. Run `pip install packaging cloudpickle` to fix.
- **`ImportError: undefined symbol: _ZN2at14RecordFunction6beforeEPKcl`**: torchair was compiled against the wrong torch version (e.g. system python's torch 2.7.1 instead of the venv's 2.9.0). Check `cat deps/torchair/tools/TORCH_VERSION` — it should be `20900`; if `20701`, clear the `tools/TORCH_VERSION` cache and rebuild with `PYTHON_BIN_PATH` (not `TARGET_PYTHON_PATH`). See Option A notes for details.

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

### Q9: Kirin platform model debugging tool

[model_run_tool](https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.rc003/CANN-Kit-tools-7.0.1.0.run) — see `./model_run_tool -h` for detailed usage.

### Q10: prefill/decode weight sharing

`GE_EXTERNAL_WEIGHT` and `GE_MOBILE_WEIGHT_SHARE_OPS` must be used together. For the Kirin platform, `GE_EXTERNAL_WEIGHT` only supports values 0 and 2; `GE_MOBILE_WEIGHT_SHARE_OPS` is Kirin-only. This feature requires PC 7.0 B115SPX or later; other versions only do compatibility adaptation.

### Q11: Custom fusion operators required for quantization

`ASCEND_CUSTOM_OPP_PATH` and `ASCEND_GE_PY_PASS_PATH` — the current companion version's quantization operators only support `FRACTAL_Z` format; weight conversion and reordering must be implemented on the CPU side.

## Related Files

- `build.sh` — root build entry (`--pkg` full build + `-u` UT)
- `scripts/build.sh` — daily development build entry (subcommand dispatch)
- `scripts/build.conf` — version numbers + path defaults
- `scripts/set_env.sh` — unified converter + compile env
- `scripts/cann_lm_compile/build_cpp.sh` — C++ build
- `scripts/cann_lm_compile/package_run.sh` — package the `.run`
- [env_dependencies.md](env_dependencies.md) — environment dependency description
- [Quick Start](QUICKSTART.md) — end-to-end examples
