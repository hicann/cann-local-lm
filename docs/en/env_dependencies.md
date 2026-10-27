# cann-device-lm Environment Dependencies Overview

English | [简体中文](../zh/env_dependencies.md)

> **Build dependencies** = what's needed to produce `.whl` / `.run` distribution artifacts (compiling `cann_lm_compile` / `.so` + packaging).
> **Runtime dependencies** = what's needed to produce epair / omc / text generation (running the distribution artifacts to do actual work).
>
> Building distributions only (`bash build.sh --pkg` / `scripts/build.sh wheel|package|cpp`) requires only Part 1; exporting epair / compiling omc / quantizing / generating text requires Part 2.
> This follows the "compile vs runtime boundary" repeatedly emphasized by GE — GE's `quick_install.md` explicitly states "driver and firmware are runtime dependencies; if you only compile source code, they are not required".
>
> Versions come from `scripts/build.conf` + `build_guide.md` + 2026-10-21 local verification; torch_npu compatibility comes from the [official COMPATIBILITY.md](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md). Same source as `scripts/build.sh check` (§3.2).

---

## Overview: Task → Build? / Runtime?

| Task | Part 1 Build? | Part 2 Runtime? | Notes |
|------|:----------:|:----------:|------|
| Compile `cann_lm_compile` + backend `.so` (`scripts/build.sh cpp`) | ✓ | — | C++ compilation, produces `build/` intermediate artifacts |
| Build `cann_lm-*.whl` / `cann_lm_engine-*.whl` (`wheel`) | ✓ | — | pip pure source packaging |
| Build `cann_lm_compile-*.run` / `engine_ohos-*.run` (`package` / `--pkg`) | ✓ | — | makeself self-extracting |
| ③a engine `.so` HarmonyOS cross-compilation (→ `engine_ohos.run`) | ✓ | — | OHOS SDK (incl. SDK clang) + signing |
| UT build (`bash build.sh -u` / `tests`) | ✓ | — | GTest + ASAN (optional) |
| ① Export epair (converter) | — | ✓ | P1 venv + CANN |
| ② kirinx90 omc compilation (run `cann_lm_compile`) | — | ✓ | P1 venv (TBE/tvm) + CANN |
| Quantization Step A (dopt GPTQ) | — | ✓ | P2 torch_npu/NPU (or GPU) |
| Quantization Step B (converter `--quant-path`) | — | ✓ | P1 venv (same as ①) |
| ③ NPU text generation (numerical verification) | — | ✓ | CANN + P2 torch_npu + NPU |
| ③b/c Kirin device text generation | — | ✓ (device side) | Device ships with musl + libneural_network_core.so |

> ✓ = required; — = not needed. **Building distributions only** requires only Part 1; without an NPU/Kirin device you can still build + run Part 2's ①② (CPU, no NPU needed).

---

# Part 1: Build Dependencies (producing `.whl` / `.run` distribution artifacts)

## System-level base tools (required for all builds)

| Tool | Version requirement | Local verified | How to get | Purpose |
|------|---------|---------|---------|------|
| **gcc / g++** | >= 8.0 and < 14.0 | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) | `apt install build-essential` | C++ compilation (`-Wall -Wextra -Werror` zero warnings) |
| **cmake** | >= 3.18 | 3.20.5 | `apt install cmake` | C++ unified build (top-level CMakeLists) |
| **make** | GNU Make | 4.3 | `apt install build-essential` (included) | cmake build execution |
| **Python** | >= 3.10 | 3.12.9 | `apt install python3 python3-venv python3-pip` | whl packaging (pip) |
| **patchelf** | any (skips rpath if missing) | 0.18.0 | `apt install patchelf` | Sets .so rpath when building `.run` packages (missing is non-blocking, falls back to LD_LIBRARY_PATH from `set_env.sh`) |
| **makeself** | >= 2.5.0 | `/usr/bin/makeself` (installed) | Produced by third-party library build, or `apt install makeself` | Builds `.run` self-extracting packages |
| **git-lfs** | any | 3.4.1 | `apt install git-lfs` | `pull_models.sh` pulls HF model weights |
| **pip** | any | comes with python3-pip | `apt install python3-pip` | converter whl packaging (`--no-deps` pure source) |
| **java (JRE)** | any | openjdk 21.0.10 ✓ | `apt install default-jre` | Only needed for `--pkg` [3/4] OHOS signing; not needed for daily `cpp`/`wheel` |

One-line install (Ubuntu/Debian):
```bash
apt install -y cmake build-essential python3 python3-venv python3-pip patchelf git-lfs default-jre
```

> **C++ third-party libraries (nlohmann_json / GTest / makeself)**: Not pre-installed dependencies; `bash scripts/build.sh third_party` automatically downloads + builds them into `output/third_party`. This is a build action, not an environment dependency.

---

## CANN (compiling `cann_lm_compile` C++ requires CANN headers/libs)

CANN is **externally pre-installed** (the user first sources its `set_env.sh` to set `ASCEND_HOME_PATH`); this repo does not build it.

| Component | Version | How to get |
|------|------|------|
| CANN toolkit | >= 9.3.0 | [master download](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) or official release; see `build_guide.md` "Installing CANN packages" |
| CANN ops | per SOC | Ascend910 → `Ascend910-ops`; kirinx90 → `ops-nn`/`ops-math`/`ops-transformer`; 950DT → `950-ops` |

**Version compatibility** (`build.conf` informational):
```
CANN_VERSION=9.3.0
SOC_VERSION=KirinX90      # kirinx90; 910 uses Ascend910_9362; 950DT uses Ascend950DT_9575
```

**Three-step install**:
1. toolkit: `./Ascend-cann-toolkit_9.3.0_*.run --install --install-path=<path>`
2. ops (per SOC): `./Ascend-cann-<soc>-ops_*.run --install --install-path=<same as above>`
3. Symlink: `ln -s cann-9.3.0/cann cann` (so GE finds platform_config; decouples version upgrades from the version-less path)

> CANN is also a runtime dependency (Part 2 §2.2 + ② omc compilation): running `cann_lm_compile` to compile omc + NPU text generation both require CANN lib64. Managing multiple coexisting CANN installations (unset to avoid pollution, etc.) is an ops detail; see `build_guide.md` and `AGENTS.md`.

---

## OHOS cross-compilation toolkit (③a engine `.so`, only for the root `build.sh --pkg` [3/4] stage)

> ⚠️ **OHOS SDK requires manual browser download**: [command-line-tools](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos) must be downloaded through a browser after logging into a Huawei developer account — **it cannot be downloaded by script** (the API returns 403 requiring login). Download **Command Line Tools for Linux(ARM)**, yielding a zip (e.g. `commandline-tools-linux-arm64-26.0.0.851.zip`). Without the OHOS SDK, the root `build.sh --pkg` aborts at the [3/4] engine stage — daily `scripts/build.sh cpp/wheel` is unaffected. Installation steps are in [build_guide.md §1.3](build_guide.md).

`build_engine()` checks each item hard; missing any one aborts `--pkg`. **Not needed for daily development (`scripts/build.sh`)**.

> **Why the OHOS cross-compilation toolkit is needed**: `libcann_lm_engine.so` must be deployed onto Kirin phones (HarmonyOS), which uses musl libc (incompatible with the host's glibc). A .so compiled by the host gcc/clang cannot be loaded by HarmonyOS. command-line-tools provides HarmonyOS-specific clang + sysroot (musl headers/libs), the only legal toolchain for cross-compiling HarmonyOS .so files.

| # | Item | Check logic | Local verified | How to get |
|---|----|---------|---------|---------|
| 1 | **Compiler** | SDK clang (`${OHOS_SDK_NATIVE}/llvm/bin/clang`) | ✓ clang 15.0.4 | Bundled with SDK (inside command-line-tools zip at `llvm/bin/clang`); runs directly on host, no extra install |
| 2 | **OHOS SDK** | `OHOS_SDK_NATIVE` is set and contains `sysroot/` | ✓ | `export OHOS_SDK_NATIVE=<extracted-dir>/command-line-tools/sdk/default/openharmony/native` |
| 3 | **java + signing jar** | `java` in PATH; `tools/binary-sign-tool.jar` or `cann_lm/engine/ohos/binary-sign-tool.jar` exists | java 21.0.10 ✓; jar ✓ | `apt install default-jre`; jar is inside the SDK zip at `command-line-tools/sdk/default/openharmony/toolchains/lib/binary-sign-tool.jar`, copy it into the repo `tools/` (`mkdir -p tools/ && cp ... tools/`) |
| 4 | **cmake / make** | same as §1.1 | 3.20.5 / 4.3 ✓ | same as §1.1 |

> **Why java + signing jar is needed**: HarmonyOS requires all binaries (`.so` + executables) to be **signed** before they can run on the device (security mechanism). `binary-sign-tool.jar` is the signing tool (run with `java`); the cross-compiled `libcann_lm_engine.so` + UT executables must be signed by it (selfSign dev signing) before deployment to the device.

> **GTest (UT, optional)**: OHOS deployment packages containing UT/bbit/smoke executables need GTest to build (`build_ohos.sh` hard-checks `GTEST_TAR`, default `/mnt/workspace/gtest-1.14.0.tar.gz`); the repo already has `output/third_party/gtest/googletest-1.14.0.tar.gz` (produced by `scripts/build.sh third_party`), set `export GTEST_TAR=$REPO/output/third_party/gtest/googletest-1.14.0.tar.gz` to point to it. Producing only `.so` does not need GTest. GTest classification is in §1.1.

---

## ASAN (UT build, optional during development)

| Item | Requirement | Local verified | Purpose |
|----|------|---------|------|
| **libasan** | matches the main gcc version | gcc 13 → `libasan8` (apt installs with gcc) | `build_cpp.sh --asan` / `scripts/build.sh tests` |

ASAN flags include `-static-libasan`, so ctest can run bare without `LD_PRELOAD`. **Before submission, C++ changes must pass ASAN build** (AGENTS.md "ASAN mandatory principle").

```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build
```

---

# Part 2: Runtime Dependencies (producing epair / omc / text generation)

## P1 converter venv (① export epair + quantization Step B consumption)

**CPU-only venv, torch_npu not installed (stub path)**. The converter is device-independent (no NPU needed); it runs `cann_lm_converter.py` (torch.export executes the model + torchair GE conversion) to produce epair artifacts.

> **User constraint (root cause of P1/P2 environment isolation)**: The converter is designed to run without an NPU (even CPU-only machines can export), so it **decouples** torchair from torch_npu — torchair GE conversion does not depend on torch_npu. P1 deliberately does not install torch_npu (stub path, a design decision); P2 installs torch_npu to run on NPU; although both use torch 2.9.0, **P1 venv and P2 system python are isolated** (venv vs system python, different dependencies). Quantization (dopt GPTQ) also runs on NPU/GPU and likewise cannot share P1 with the converter. Therefore P1 uses a dedicated venv (`CONVERTER_PYTHON`), isolated from P2 system python (`DOPT_PYTHON`).

| Component | Version | Local verified | Source |
|------|------|---------|------|
| **torch** | 2.9.0+cpu | 2.9.0+cpu ✓ | torch-delegate `.venv` (`pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu`) |
| **torchair** | 0.1 | OK ✓ | Source build (`gitcode.com/Ascend/torchair.git`, public repo) |
| **torch_delegate** | 0.1.0 | OK ✓ | Merged into cann-device-lm repo `torch_delegate/` directory (`gitcode.com/cann/cann-device-lm`) |
| **torch_npu** | **not installed** | absent ✓ (stub path OK) | If installed, requires **2.9.0.post8** (TorchNPU 26.1.1, paired with CANN 9.3.0, see §2.2) — but the converter takes the stub path and **deliberately does not install it** (design decision, see `cli_design.md` "Relationship between converter and torch_npu") |

> Full package list in [`scripts/requirements-p1.txt`](../../scripts/requirements-p1.txt) (torchair/torch-delegate are not pip packages; require source install).
| numpy / protobuf | any | installed with torchair | torchair runtime dependencies (`ge._ge_graph` imports numpy, `ge_ir_pb2` checks protobuf version) |
| wheel | any | comes with venv | torchair `setup.py bdist_wheel` needs it |

**venv location** (derived from `build.conf`): `${PROJECT_ROOT}/deps/torch-delegate/.venv` (`CONVERTER_PYTHON` = that venv's python)

**Two install methods** (see `build_guide.md` "Installing upstream components"):
- **Method 1, source build**: clone torch-delegate → `./build.sh` (produces .venv) → install torch_delegate whl → clone torchair → `TARGET_PYTHON_PATH=venv bash build.sh -c` → install torchair whl
- **Method 2, direct whl install**: create venv + `pip install torch==2.9.0+cpu` + install prebuilt torch_delegate/torchair whls + numpy/protobuf; set `export CONVERTER_PYTHON=/path/to/venv/bin/python`

**Key checks** (`scripts/build.sh check` check items):
- `CONVERTER_PYTHON` is executable
- `import torch` / `import torchair` / `import torch_delegate` / `import cann_lm` all pass
- `import torch_npu` **must fail** (stub path OK)

**TBE/tvm dependency packages (for ② omc compilation)**: When `cann_lm_compile` (C++ binary) runs `aclgrphBuildModel` to compile operators, it calls back into Python TBE/tvm, which needs the following packages. These packages are **not in the CANN .run package**, but most are auto-dependencies of `te` (pip package) or `torch`, pulled in automatically by `pip install te` / `pip install torch`:

| Package | Source | Notes |
|----|------|------|
| attrs / decorator / numpy / psutil / scipy | pip dependency of `te` | auto-pulled by `pip install te` |
| sympy / networkx | pip dependency of `torch` | auto-pulled by `pip install torch` |
| **requests / pyyaml / cffi** | **no one declares the dependency** | **manual install required** (`pip install requests pyyaml cffi`) |
| tqdm | no dependency | optional, missing is non-blocking |

> TBE/tvm is a **runtime** dependency (called when running `cann_lm_compile` to compile omc in ②), not a build dependency (compiling the `cann_lm_compile` C++ binary does not need it).

---

## CANN runtime + P2 system python + torch_npu/NPU (or GPU) — ③ NPU text generation + quantization Step A

> **PyTorch and TorchNPU install order and constraints**:
> 1. **Install CANN toolkit first, then TorchNPU** (TorchNPU .so depends on CANN lib64 symbols; without CANN you get undefined symbol errors).
> 2. **Install separately per actual Python version** (PyTorch + TorchNPU must be installed in the same Python environment; version compatibility is in the [TorchNPU Software Installation guide](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)).
> 3. **Python versions must match**: PyTorch, TorchNPU, and the running Python must share the same Python version (torch C extensions depend on libpython; mismatch causes ABI incompatibility).
> 4. **Kirin processor series products depend only on PyTorch; TorchNPU is not required** (Kirin uses hiai ACL, bypassing the torch_npu NPU backend). TorchNPU is only used for Ascend NPU (③ NPU text generation / quantization acceleration).

This section is the aggregate group for **runtime tasks that use torch_npu/NPU**: ③ NPU text generation (numerical verification) + quantization Step A (dopt GPTQ) share P2 system python + torch_npu + NPU. **Independent of P1 venv** (P1 venv deliberately does not install torch_npu and takes the stub path; P2 system python installs real torch_npu to run on NPU; the two are isolated, see the binding table below). Only these runtime tasks need it.

> **Quantization Step A can optionally use GPU**: dopt GPTQ can use NPU (torch_npu) or **GPU (CUDA torch) for acceleration** — in that case use a GPU env (CUDA torch, no torch_npu, no NPU needed), which does not hit the triple binding below. **③ NPU text generation requires NPU** (runs omc via `acl.mdl.execute`), cannot use GPU. The table below defaults to the NPU path; for the GPU path, replace the `torch_npu` row with `torch (CUDA)` + a GPU device.

> Full package list in [`scripts/requirements-p2.txt`](../../scripts/requirements-p2.txt) (GPU/NPU common) + [`scripts/requirements-p2-npu.txt`](../../scripts/requirements-p2-npu.txt) (NPU adds torch_npu).

**torch_npu triple version binding** ([authoritative compatibility table COMPATIBILITY.md](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)):

| Binding | Rule | Why mismatch is fatal |
|------|------|-------------|
| **① torch_npu ↔ torch (1:1)** | torch_npu package version number = torch version number (`{torch version}` or `{torch version}.postN`) | C++ extension .so links libtorch.so; mismatch → undefined symbol (same as torchaudio version mismatch, below) |
| **② torch_npu ↔ CANN** | each torch_npu version nominally pairs with a fixed CANN version | torch_npu .so depends on CANN lib64 symbols |
| **③ torch ↔ Python (indirect)** | torch version determines the supported python range | torch C extension depends on libpython |

**Compatibility table excerpt** (rows affecting this machine):

| TorchNPU product version | torch_npu package | torch | nominal CANN | Notes |
|------------------|----------------|-------|-----------|------|
| **26.1.1** | **2.9.0.post8** | **2.9.0** | **CANN 9.3.0** | **← actually used by P2 on this machine**; latest; P1 §2.1 uses this row if installing torch_npu |
| 26.1.1 | 2.7.1.post10 | 2.7.1 | CANN 9.1.X | includes `npu_chunk_gated_delta_rule` |
| 26.1.0 | 2.9.0.post6 | 2.9.0 | CANN 9.1.0 | |
| 26.1.0 | 2.7.1.post8 | 2.7.1 | CANN 9.1.0 | `npu_chunk_gated_delta_rule` available from this version |
| 26.0.0 | 2.9.0.post2 | 2.9.0 | CANN 9.0.0 | |

> **Key rule**: The same TorchNPU product version (e.g. 26.1.1) can pair with multiple torch versions, but each torch version needs its corresponding package version — `2.9.0.post8` only pairs with torch 2.9.0, **not** torch 2.7.1. `.postN` is the release sequence number within the same torch version and **cannot be converted to a product version** (2.7.1.post4 ≠ 2.9.0.post4).
>
> **Root cause of P1/P2 isolation**: P1 §2.1 uses torch **2.9.0+cpu** (converter needs torch.export, venv isolated); this section P2 uses torch **2.9.0** + torch_npu **2.9.0.post8** (TorchNPU 26.1.1); P1/P2 are both torch 2.9.0, but **P1 venv deliberately does not install torch_npu** (stub path, design decision), while P2 system python installs torch_npu to run on NPU — venv and system python are isolated (different dependencies, different environments) and cannot be shared.

**CANN for runtime**: same as Part 1 §1.2 (same install; NPU text generation / quantization Step A / ② omc compilation reuse it, no extra install). torch_npu .so depends on CANN lib64 (see the binding table above).

**P2 system python + torch_npu + NPU device**:

| Component | Version | Local verified | Source |
|------|------|---------|------|
| **python** | 3.12.9 | 3.12.9 ✓ | `/opt/buildtools/python-3.12.9/bin/python3` |
| **torch** | 2.9.0+cpu | pending verification | pre-installed (currently 2.7.1+cpu, needs upgrade) |
| **torch_npu** | 2.9.0.post8 | pending verification | currently 2.7.1.post4 (TorchNPU 26.0.0), needs upgrade to 2.9.0.post8 (TorchNPU 26.1.1, paired with CANN 9.3.0, see compatibility table above) |
| torchvision | 0.22.1 | — | pre-installed (optional) |
| torchaudio | 2.9.0 | pending verification | `pip install torchaudio==2.9.0 --force-reinstall --no-deps --user` (matches torch 2.9.0, installs to `~/.local/` shadowing the system old version) |

**NPU device**: this machine has 2 Ascend910 cards (64GB HBM), `npu-smi` 25.5.5

**Driver and firmware**: NPU runtime dependencies (③ NPU text generation / device text generation needs NPU device + driver firmware); build-only (①②) does not need them. Installation is in the "Installing NPU driver and firmware" chapter of the CANN Software Installation Guide.

**`DOPT_PYTHON` auto-detection** (`_resolve_dopt_python` in `_env_common.sh`):
- Priority: already-exported DOPT_PYTHON → candidate paths (`/opt/buildtools/python-3.12.9/bin/python3` / `/opt/buildtools/Python-3.11.4/bin/python3` / `/usr/local/bin/python3` / `/usr/bin/python3`) + whichever in PATH can `import torch_npu` → if none found, warn (only ③ needs it; Part 1 is non-blocking)
- During detection, LD prepends the candidate python's `<prefix>/lib` + CANN lib64 (prevents loading an old libpython that mis-computes site-packages)
- No change needed when switching machines: a new machine with a torch_npu-capable python in the candidate paths or PATH is auto-detected

**⚠️ Three P2 pitfalls** (from the triple binding above + verification):

1. **CANN version compatibility**: torch_npu 2.9.0.post8 (TorchNPU 26.1.1) nominally pairs with **CANN 9.3.0**, matching this machine's runtime CANN 9.3.0 — no version mismatch risk.
2. **torch_npu version**: 2.9.0.post8 (TorchNPU 26.1.1) includes `npu_chunk_gated_delta_rule` (available from ≥2.7.1.post8 / TorchNPU 26.1.0+), no missing operators.
3. **torchaudio version match**: torchaudio must match the torch version (torchaudio 2.9.0 ↔ torch 2.9.0). If versions mismatch (e.g. system has torchaudio 2.11.0 + torch 2.9.0), `import torchaudio` reports `undefined symbol: torch_library_impl`. Fix:

```bash
pip install torchaudio==2.9.0 --force-reinstall --no-deps --user   # install to ~/.local/, shadowing the system old version
```

**Another P2 environment pitfall** (`build_guide.md` Q8):
- **`_ctypes` undefined symbol / `No module named 'torch_npu'` (actually installed)**: A clean shell sourcing CANN leaves LD without the python `<prefix>/lib` → loads the system's old libpython. Fix: prepend `/opt/buildtools/python-3.12.9/lib` to `LD_LIBRARY_PATH` (`_env_common.sh` already auto-derives and persistently prepends it)

---

## Kirin device side (③b/c device text generation, device ships with it, host does not install)

③b/c runs `cann_lm.generate` on Kirin/HarmonyOS devices (via `libcann_lm_engine.so` + hiai ACL); **the host does not need to install any dependencies** — the device ships with:

| Component | Source | Purpose |
|------|------|------|
| musl libc | provided by HarmonyOS system | C runtime (musl, not glibc) |
| `libneural_network_core.so` | HarmonyOS system library (Hiai/NNRT) | NPU inference (hiai ACL underlying) |
| `libc++_shared.so` | provided with the `engine_ohos.run` package | C++ runtime (same directory as the engine .so) |
| `libcann_lm_engine.so` | cross-compiled on host (Part 1 §1.3) → deployed to device | engine SHARED library (signed version) |

> The host side only needs to complete the Part 1 build (producing `engine_ohos.run`); after the artifact is deployed to the device, the device-side runtime carries it.

---

# Part 3: Quick reference

## env → dependency mapping quick reference table

Look up "which Part 1 section + which Part 2 section" from the task:

| I want to | which env to source | Part 1 build deps | Part 2 runtime deps | NPU/GPU needed? |
|--------|----------------|---------------|---------------|------------|
| Compile cann_lm_compile + build whl/.run | `source scripts/set_env.sh` | §1.1 + §1.2 CANN | — | No |
| ③a engine .so (HarmonyOS cross) | root `bash build.sh --pkg` | §1.1 + §1.3 OHOS toolkit | — | No |
| UT build + run | `bash build.sh -u` | §1.1 + §1.2 + §1.4 ASAN | — | No |
| ① Export epair (converter) | `source scripts/set_env.sh` | — | §1.2 CANN + §2.1 P1 venv | No |
| ② kirinx90 omc compilation | source CANN set_env.sh + source scripts/set_env.sh (SOC_VERSION=KirinX90) | — | §1.2 CANN + §2.1 P1 venv (TBE/tvm) | No |
| Quantization Step A (dopt GPTQ) | `source scripts/set_env_cann.sh --npu` (or GPU env) | — | §2.2 P2 (NPU or GPU) | NPU or GPU |
| Quantization Step B (converter consumption) | `source scripts/set_env.sh` | — | §1.2 CANN + §2.1 P1 venv | No |
| ③ NPU text generation (numerical verification) | `source scripts/set_env_cann.sh --npu` | — | §1.2 CANN + §2.2 P2 | NPU |
| ③b/c Kirin device text generation | `cann_lm.generate` on device | — | §2.3 device side | Kirin device |
| Release full packaging | root `bash build.sh --pkg` | Part 1 all (§1.1-1.4) | — | No |

> **Compile vs runtime boundary**: Building distributions only (`.whl`/`.run`) requires only Part 1; producing epair/omc/text generation requires Part 2. Without an NPU/Kirin device you can do Part 1 build + Part 2's ①② (CPU, no NPU needed).

---

## Verification commands

```bash
# One-line baseline verification (missing required items reports "missing N required items"; P2/torch_npu marked optional)
source /home/developer/Ascend/cann/set_env.sh
bash scripts/build.sh check

# Expected output:
#   ✓ CANN /home/developer/Ascend/cann-9.3.0
#   ✓ P1 venv .../deps/torch-delegate/.venv/bin/python
#   ✓ torch (2.9.0+cpu)
#   ✓ torchair / torch_delegate / cann_lm
#   ✓ third_party json
#   optional (③ NPU text generation / quantization acceleration, Part 2; quantization can be NPU or GPU, see §2.2):
#   ✓ P2 .../python-3.12.9 (for ③ NPU text generation / NPU quantization acceleration)
#   ✓ torch_npu (P2, NPU path; GPU path does not go through this check)
#   ✓ baseline ready (build ①② required items ready; ③ NPU text generation / quantization acceleration optional)
```

> This table shares the same source as the `scripts/build.sh check` check items; env/version changes must update this table and `build.conf` in sync (doc-script same source).
