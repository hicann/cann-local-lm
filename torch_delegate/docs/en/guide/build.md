# TorchDelegate Build Guide

This document describes the one-stop build process for TorchDelegate, including environment preparation, build commands, release package usage, and user integration.

## Prerequisites

| Tool | Version Requirement | Notes |
|------|---------|------|
| CMake | >= 3.18 | - |
| Python | 3.10 ~ 3.13 | The `python3-venv` package must be installed |
| gcc / g++ | >= 8.0 and < 14.0 | - |
| make | GNU Make | Explicitly checked by `build_third_party.sh`; installed with `build-essential` |
| pip | | Required in `--system_python` mode; in the default mode it comes with `python3-venv` |
| python3-venv | | Needed to automatically create the virtual environment during the build |
| torch (CPU) | 2.9.0 | Automatically installed into `.venv` by `build.sh` |
| CANN | See the installation instructions below | The path is specified through the `ASCEND_HOME_PATH` environment variable |
| libasan | Matches the gcc version | AddressSanitizer runtime library. `build_test.sh` enables ASAN by default, derives the path via `gcc -print-file-name=libasan.so` and loads it via `LD_PRELOAD`; ASAN will not work without it. The package name changes with the gcc major version: gcc 7.5.0 corresponds to `libasan4`, gcc 13 to `libasan8`; for other versions, install the matching asan version |

One-command dependency installation (Ubuntu/Debian):

```bash
# Basic build dependencies + ASAN runtime (the libasan package name changes with the gcc major version; gcc 13 corresponds to libasan8, install the matching asan version for other versions)
apt install -y cmake build-essential python3 python3-venv python3-pip libasan8
```

## Installing the CANN Package

### Try master capabilities or develop based on master

Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master) to get the latest version, and download the corresponding package for your product model and environment architecture. The installation commands are as follows; for more guidance, see the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

1. Install the CANN Toolkit development kit package.

   ```bash
   # Make sure the installation package is executable
   chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
   # Installation command
   ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
   ```

   - `${cann_version}`: the CANN package version number.
   - `${arch}`: the CPU architecture, such as `aarch64` or `x86_64`.
   - `${install_path}`: the specified installation path. root users install to `/usr/local/Ascend` by default, and regular users to `${HOME}/Ascend` by default.

2. Install the CANN ops operator package.

   ```bash
   # Make sure the installation package is executable
   chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
   # Installation command
   ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
   ```

   Here `${soc_name}` denotes the NPU model name.

### Try released capabilities or develop based on a released version

If you want to try the capabilities of the **officially released CANN package**, go to the [CANN Download Center](https://www.hiascend.com/cann/download) and select the CANN package of the corresponding version (only CANN 9.2.0 and later are supported) to install.

### Driver and Firmware (optional, required only for running samples)

For downloading and installing the driver and firmware, see the "Preparing Software Packages" and "Installing the NPU Driver and Firmware" sections of the *CANN Software Installation Guide*. The driver and firmware are required for running samples; if you only need a build environment, you can skip them.

## Quick Start

### Setting Up the CANN Environment

Run the corresponding setup script according to the CANN installation path. The default path is `/usr/local/Ascend` for root users and `${HOME}/Ascend` for regular users:

```bash
# Default path for root users
source /usr/local/Ascend/cann/set_env.sh

# Default path for regular users
source ${HOME}/Ascend/cann/set_env.sh

# Custom installation path (replace with the actual installation path)
source ${install_path}/cann/set_env.sh
```

After setup, the `ASCEND_HOME_PATH` environment variable automatically points to the CANN installation directory.

### Local Build

```bash
./build.sh
```

When the build completes, the only release artifact is a platform-tagged whl:

- `output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl` (debug builds use `<version>+debug`)

This whl contains both the Python API and the C++ artifacts (headers, dynamic libraries, CMake configuration package). One package per platform, independent of the Python version.

`build.sh` automatically performs the following steps:
1. Check the currently active Python environment (cmake, gcc, Python, CANN)
2. Prepare the Python environment (by default, automatically create `.venv`; add `--system_python` to use the current environment)
3. Build third-party libraries (nlohmann_json, gtest, libtorch_npu)
4. Build and install the C++ delegate library
5. Build the whl (embed the CMake install tree into the Python package without invoking CMake again)

## Build Commands

### build.sh Parameters

| Parameter | Description |
|------|------|
| `--debug` | Build a debug version |
| `--release` | Build a release version (default) |
| `--cann_3rd_lib_path=PATH` | Specify the third-party library installation path |
| `--output_dir=PATH` | Artifact output path (default: output/package) |
| `--pre_check` | Only validate the environment and exit without building |
| `--skip_third_party` | Skip third-party library builds |
| `--with_libtorch_npu` | Build libtorch_npu (disabled by default) |
| `--system_python` | Use the current Python environment and do not create `.venv` automatically |
| `--help` | Help information |

Python environment strategy:

**Default mode** (without `--system_python`):
1. `.venv` already exists: activate it and reinstall dependencies (pip automatically skips installed packages)
2. `.venv` does not exist: create it automatically and install torch + CANN runtime dependencies

**`--system_python` mode**:
- Use the currently active system Python interpreter directly (`command -v python3`, compatible with venv / conda / system Python)
- Do not create `.venv` and do not automatically install any packages
- Requires the correct torch version to already be installed in the current environment (an error is raised if it is missing or mismatched)
- Suitable for: existing conda environments, CI environments, or scenarios where you want to control dependency installation manually

`Torch_DIR` is always derived from the `torch` package in the selected Python environment and does not accept an externally specified path. Use `--pre_check` to check the environment in advance.

Examples:

```bash
# Default release build (automatically creates .venv)
./build.sh

# Debug build
./build.sh --debug

# Use the current conda/venv environment
./build.sh --system_python

# Enable libtorch_npu compilation
./build.sh --with_libtorch_npu

# Specify the output directory
./build.sh --output_dir=/custom/output
```

### build_test.sh - Test Build

| Parameter | Description |
|------|------|
| `--debug` | Build debug tests (default) |
| `--release` | Build release tests |
| `--asan` | Enable AddressSanitizer (default) |
| `--no-asan` | Disable ASAN |
| `--cann_3rd_lib_path=PATH` | Specify the third-party library path |
| `--skip_third_party` | Skip third-party library builds |
| `--skip_tests` | Only build, do not run tests |
| `--system_python` | Use the current Python environment and do not create `.venv` automatically |

`--debug` and `--release` are mutually exclusive; specifying both raises an error.

Examples:

```bash
# Build and run tests (ASAN is enabled by default)
./build_test.sh

# Use the current Python environment to run tests
./build_test.sh --system_python

# Build release tests
./build_test.sh --release

# Disable ASAN
./build_test.sh --no-asan
```

## Version Configuration

All version numbers are managed centrally in `build.conf`:

```bash
# TorchDelegate
TORCH_DELEGATE_VERSION=0.1.0

# PyTorch (CPU)
TORCH_VERSION=2.9.0

# CANN
CANN_VERSION=9.2.0

# libtorch_npu
LIBTORCH_NPU_RELEASE_TAG=v26.1.0-beta.1-pytorch2.9.0

# Third-party
NLOHMANN_JSON_VERSION=3.11.3
GTEST_VERSION=1.14.0
```

To switch to a new version, only this file needs to be modified. `TORCH_DELEGATE_VERSION` is the single source of the version number: at build time it is injected through the `TORCH_DELEGATE_PACKAGE_VERSION` environment variable into the whl metadata and the runtime `torch_delegate.__version__`; the CMake side uses the same variable, and `pyproject.toml` no longer hardcodes the version.

## whl Installation and Usage

The whl is the only release artifact and carries both the Python API and the C++ artifacts.

### Installation

```bash
pip install output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

- Platform tags are `linux_x86_64` / `linux_aarch64`; Python/ABI tags are fixed to `py3-none`, so the same whl can be installed into Python 3.10 ~ 3.13 without building separately per version.
- `torch_npu` is not installed automatically as a dependency: the full export mode requires it (it provides `torchair`), so install the version matching `torch` yourself (`torch_npu>=2.9.0`); when missing, export fails immediately instead of degrading silently.
- Debug builds have a `+debug` suffix in the version number (e.g. `0.1.0+debug`).

### Package Layout After Installation

```text
<site-packages>/torch_delegate/
├── *.py, converter/, epair/, core/     ← Python API
├── _version.py                         ← __version__ injected at build time
├── inc/torch_delegate/                 ← Public C++ headers
│   ├── epair_model_loader.h
│   └── td_types.h
└── lib/
    ├── libtorch_delegate_backend.so    ← C++ dynamic library
    └── cmake/TorchDelegate/            ← CMake configuration package (Config / Targets / modules)
```

The CMake install prefix is entirely inside the package directory, so `<site-packages>/torch_delegate` is the C++ prefix.

### Obtaining the C++ Prefix (TORCH_DELEGATE_ROOT)

`set_env.sh` is no longer provided. Use either of the following methods instead (they are equivalent):

```bash
# Method 1: CLI (generated at pip install time; the current environment's bin must be in PATH)
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

# Method 2: Python function
export TORCH_DELEGATE_ROOT=$(python3 -c "import torch_delegate; print(torch_delegate.cmake_prefix())")
```

`torch-delegate-config --version` queries the version number.

> **NOTE**: `torch-delegate-config` is a console script that pip installs into the current environment's `bin/` (venv / conda / system Python are all supported). Before use, activate the target environment so that its `bin/` is in `PATH`; in the `pip install --target` scenario the script is not added to PATH, so use method 2 instead.

### Runtime Dependencies (Manual Configuration Required)

`libtorch_delegate_backend.so` depends on libtorch and CANN at runtime. Neither is bundled into the whl, and the dynamic library does not inject RUNPATH pointing to them, so `LD_LIBRARY_PATH` must be configured manually before execution:

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

Notes:

- `${TORCH_PKG_DIR}.libs` exists only on aarch64 (on x86_64 all libtorch libraries are in `torch/lib`); it is harmless when the directory does not exist.
- When using fallback codegen / custom operators, `${TORCH_NPU_DIR}/lib` must also be appended.

### C++-only Usage (Without Installing the Python Package)

The whl is essentially a zip archive, so you can extract the C++ artifacts directly:

```bash
unzip torch_delegate-<version>-py3-none-linux_<arch>.whl -d /tmp/td
# The prefix is /tmp/td/torch_delegate, containing inc/, lib/, lib/cmake/TorchDelegate/
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/td/torch_delegate
```

Alternatively, use `pip install --target=<dir> <whl>`; the prefix is `<dir>/torch_delegate`.

### Release Verification

```bash
# whl content and metadata validation (pytest; requires a whl produced by ./build.sh first)
pytest tests/python/test_wheel_package.py

# End-to-end smoke test: temporary venv install → import → CLI → readelf → find_package compile and run
bash tests/packaging/wheel_smoke_test.sh
```

The smoke script only verifies the packaging and integration path and does not require an NPU device. Use `TORCH_DELEGATE_WHEEL=<path>` to specify a whl and `--python=<interpreter>` to specify the base interpreter.

## User Integration

### C++ Projects

After installing the whl, obtain the C++ prefix with `torch-delegate-config --cmake-prefix` (or `torch_delegate.cmake_prefix()`) and export it as `TORCH_DELEGATE_ROOT`, then integrate it in CMakeLists.txt through `$ENV{TORCH_DELEGATE_ROOT}`. Below is a CMAKE template for an external application:

```cmake
cmake_minimum_required(VERSION 3.18)
project(your_app CXX)

set(TorchDelegate_ROOT_DIR "" CACHE PATH "Path to TorchDelegate install directory")

if(TorchDelegate_ROOT_DIR STREQUAL "" AND DEFINED ENV{TORCH_DELEGATE_ROOT})
    set(TorchDelegate_ROOT_DIR $ENV{TORCH_DELEGATE_ROOT})
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()
set(CANN_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

if(NOT DEFINED ENV{TORCH_PKG_DIR} OR "$ENV{TORCH_PKG_DIR}" STREQUAL "")
    message(FATAL_ERROR "TORCH_PKG_DIR is required and must point to Python torch package directory")
endif()
set(TORCH_PKG_DIR "$ENV{TORCH_PKG_DIR}")
set(TORCH_LIB_DIR "${TORCH_PKG_DIR}/lib")
get_filename_component(_TORCH_PARENT "${TORCH_PKG_DIR}" DIRECTORY)
set(TORCH_BUNDLED_LIB_DIR "${_TORCH_PARENT}/torch.libs")

if(TorchDelegate_ROOT_DIR STREQUAL "")
    message(FATAL_ERROR "TorchDelegate_ROOT_DIR not specified")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TorchDelegate_ROOT_DIR}")
find_package(TorchDelegate REQUIRED)

set(TORCH_LINK_DIRS "")
if(EXISTS "${TORCH_LIB_DIR}")
    list(APPEND TORCH_LINK_DIRS "${TORCH_LIB_DIR}")
endif()
if(EXISTS "${TORCH_BUNDLED_LIB_DIR}")
    list(APPEND TORCH_LINK_DIRS "${TORCH_BUNDLED_LIB_DIR}")
endif()
if(NOT TORCH_LINK_DIRS)
    message(FATAL_ERROR "No torch link directories found from TORCH_PKG_DIR=${TORCH_PKG_DIR}")
endif()

add_executable(your_app main.cpp)

target_link_directories(your_app PRIVATE
    ${CANN_LIB_DIR}
    ${TORCH_LINK_DIRS}
)

target_link_options(your_app PRIVATE
    "LINKER:-rpath-link,${CANN_LIB_DIR}"
)
foreach(_torch_link_dir IN LISTS TORCH_LINK_DIRS)
    target_link_options(your_app PRIVATE
        "LINKER:-rpath-link,${_torch_link_dir}"
    )
endforeach()
unset(_torch_link_dir)

target_link_libraries(your_app PRIVATE
    TorchDelegate::torch_delegate_backend
    ge_runner_v2
    acl_rt
)
```

Note: the application only links `TorchDelegate::torch_delegate_backend` and does not link `torch`, `torch_cpu`, or `c10` directly. However, the final application link still needs the CANN lib directory, `torch/lib`, and `torch.libs` when present, used in both `target_link_directories` and `-rpath-link`.

Configuration example:

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
  -DTORCH_PKG_DIR="${TORCH_PKG_DIR}"
```

`TorchDelegateConfig.cmake` does not manage torch: the torch library path is passed in by the consumer via `-DTORCH_PKG_DIR` (or resolved from `ENV{TORCH_PKG_DIR}` inside CMakeLists), as shown in the template above.

To run fallback codegen, custom operators, or samples that include `libtorch_npu`, `TORCH_NPU_DIR` also needs to be set:

```bash
export TORCH_PKG_DIR=/path/to/site-packages/torch
export TORCH_NPU_DIR=/path/to/libtorch_npu
```

Here `TORCH_PKG_DIR` uniformly points to the Python `torch` package directory. CMake configurations that need `TorchConfig.cmake` will internally append `${TORCH_PKG_DIR}/share/cmake/Torch`; the `share/cmake/Torch` directory is no longer passed in externally.

The runtime library path must be set manually before running (the whl does not inject RUNPATH and does not provide set_env.sh):

```bash
source ${ASCEND_HOME_PATH}/set_env.sh

TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

Using the delegate API:

```cpp
#include "torch_delegate/epair_model_loader.h"
#include "torch_delegate/td_types.h"

// Load the .epair model
td::EpairModelLoader loader("/path/to/model.epair");

// Compile and load onto the device
std::map<ge::AscendString, ge::AscendString> options;
void *stream = nullptr;  // ACL stream pointer
td::Status status = loader.CompileAndLoad(options, stream);

// Run inference asynchronously
std::vector<gert::Tensor> inputs, results;
status = loader.RunModelWithStreamAsync(stream, inputs, results);

// Check the execution result
if (status == td::SUCCESS) {
    // Process outputs
}
```

### Python Projects

```bash
pip install output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

```python
import torch_delegate
from torch_delegate import export_to_epair, epair_save, epair_load

print(torch_delegate.__version__)
print(torch_delegate.cmake_prefix())   # C++ prefix, can be passed to CMAKE_PREFIX_PATH
```

The full export mode requires `torch_npu` (which provides `torchair`); install the version matching `torch` yourself; the whl does not pull it automatically.

## IDE Configuration

### CLion

In **Settings → Build → CMake**, add:

```text
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
-DBUILD_DELEGATE=ON
-DBUILD_TESTS=ON
-DASCEND_INSTALL_PATH=/path/to/cann        # Replace with the actual CANN installation path
-DTorch_DIR=/path/to/venv/lib/python3.10/site-packages/torch/share/cmake/Torch
-DCANN_3RD_LIB_PATH=./output/third_party
-DENABLE_ASAN=ON
```

If you specify torch through environment variables, use the Python torch package directory: `export TORCH_PKG_DIR=/path/to/venv/lib/python3.10/site-packages/torch`.

`ASCEND_INSTALL_PATH` can also be obtained automatically from the `ASCEND_HOME_PATH` environment variable; after sourcing the CANN environment, CLion will inherit it.

### VSCode

Use the CMake Tools extension and configure the same parameters as CLion in `.vscode/settings.json` or CMake Kits. In settings.json:

```json
{
    "cmake.configureSettings": {
        "ASCEND_INSTALL_PATH": "/path/to/cann",
        "Torch_DIR": "/path/to/venv/lib/python3.10/site-packages/torch/share/cmake/Torch",
        "CANN_3RD_LIB_PATH": "${workspaceFolder}/output/third_party"
    }
}
```

## ABI Compatibility

- Public headers do not expose `std::string`, `std::unordered_map`, or `std::filesystem::path`
- String parameters uniformly use `const char*` or `ge::AscendString`
- The `torch_delegate/lib/` directory in the whl contains `libtorch_delegate_backend.so`, and `torch_delegate/inc/` contains the public headers

## FAQ

### Q1 ASCEND_HOME_PATH is not set

**Solution**:
Source the CANN environment script: `source /path/to/Ascend/cann/set_env.sh`

### Q2 torch installation fails due to network issues

**Solution**:
`build.sh` uses the Huawei Cloud PyPI mirror (`https://repo.huaweicloud.com/repository/pypi/simple`) to speed up dependency downloads. If problems persist, check the network connection or configure a proxy.

### Q3 Running a C++ program reports `libtorch_cpu.so` / `libgraph.so` cannot open shared object file

**Solution**:
The whl does not inject RUNPATH, so `LD_LIBRARY_PATH` must be configured manually before running (see "Runtime Dependencies (Manual Configuration Required)"), ensuring it includes `torch/lib`, `torch.libs` (aarch64), `$TORCH_DELEGATE_ROOT/lib`, and `$ASCEND_HOME_PATH/lib64`.

### Q4 `torch-delegate-config command not found`

**Solution**:
This command is installed by pip into the current environment's `bin/`, so you must first activate the environment where the whl was installed (venv / conda). If installed with `pip install --target`, the script is not added to PATH; use
`python3 -c "import torch_delegate; print(torch_delegate.cmake_prefix())"` instead.

### Q5 Export reports missing TorchAir / torch_npu

**Solution**:
The full export mode depends on `torchair` provided by `torch_npu`, which the whl does not install automatically. Install `torch_npu>=2.9.0` matching your `torch` version yourself; for scenarios without a CANN runtime, use `lite_export_context()` instead.


## Output Directory Structure

```text
output/third_party/                                    ← Third-party build artifacts
├── json/
├── gtest/
└── libtorch_npu/

output/package/                                        ← Release artifacts (the only one)
└── torch_delegate-<ver>-py3-none-linux_<arch>.whl     ← Python API + C++ artifacts
```

Debug builds produce `torch_delegate-<ver>+debug-py3-none-linux_<arch>.whl`.

## Related Files

- `build.sh` - One-stop build script (produces a single whl)
- `build.conf` - Centralized version configuration (the single source of version numbers)
- `build_third_party.sh` - Third-party library build script
- `build_test.sh` - Test build and run script
- `setup.py` - whl assembly (embeds the CMake install tree, platform tag, version injection)
- `pyproject.toml` - Python package metadata and entry point
- `torch_delegate/cli.py` - Implementation of `torch-delegate-config`
- `tests/python/test_wheel_package.py` - whl content and metadata validation
- `tests/packaging/wheel_smoke_test.sh` - End-to-end whl smoke test
- `requirements.txt` - Python dependency description
