# TorchDelegate 构建指南

本文档描述 TorchDelegate 的一站式构建流程，包括环境准备、构建命令、发布包使用和用户接入指南。

## 前置条件

| 工具 | 版本要求 | 备注 |
|------|---------|------|
| CMake | >= 3.18 | - |
| Python | 3.10 ~ 3.13 | 需安装 `python3-venv` 包 |
| gcc / g++ | >= 8.0 且 < 14.0 | - |
| make | GNU Make | `build_third_party.sh` 会显式检查；随 `build-essential` 安装 |
| pip | | `--system_python` 模式必需；默认模式下由 `python3-venv` 自带 |
| python3-venv | | 构建时自动创建虚拟环境所需 |
| torch (CPU) | 2.9.0 | 由 `build.sh` 自动安装到 `.venv`|
| CANN | 见下方安装说明 | 路径通过 `ASCEND_HOME_PATH` 环境变量指定 |
| libasan | 与 gcc 版本匹配 | AddressSanitizer 运行时库，`build_test.sh` 默认启用 ASAN，通过 `gcc -print-file-name=libasan.so` 推导路径并 `LD_PRELOAD` 加载；缺失则 ASAN 无法正常工作。包名随 gcc 主版本变化：以 gcc 7.5.0 为例对应 `libasan4`，gcc 13 对应 `libasan8`，其他版本请安装对应版本 asan |

一键安装依赖（Ubuntu/Debian）：

```bash
# 基础构建依赖 + ASAN 运行时（libasan 包名随 gcc 主版本变化，gcc 13 对应 libasan8，其他版本请安装对应版本 asan）
apt install -y cmake build-essential python3 python3-venv python3-pip libasan8
```

## 安装 CANN 包

### 场景1：体验master版本能力或基于master版本进行开发

请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)获取最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

1. 安装CANN Toolkit开发套件包。

   ```bash
   # 确保安装包具有可执行权限
   chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
   # 安装命令
   ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
   ```

   - `${cann_version}`：表示CANN包版本号。
   - `${arch}`：表示CPU架构，如`aarch64`、`x86_64`。
   - `${install_path}`：表示指定安装路径。root用户默认安装在`/usr/local/Ascend`目录，普通用户默认安装在`${HOME}/Ascend`目录。

2. 安装CANN ops算子包。

   ```bash
   # 确保安装包具有可执行权限
   chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
   # 安装命令
   ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
   ```

   其中`${soc_name}`表示NPU型号名称。

### 场景2：体验已发布版本能力或基于已发布版本进行开发

如果您想体验**官网正式发布的CANN包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择对应版本CANN软件包（仅支持CANN 9.1.0及后续版本）进行安装。

### 驱动与固件（可选，仅运行样例依赖）

驱动与固件的下载和安装操作请参考《CANN软件安装指南》中"准备软件包"和"安装NPU驱动和固件"章节。驱动与固件是运行样例依赖，若仅编译环境，可以不安装。

## 快速开始

### 1. 设置 CANN 环境

根据 CANN 安装路径执行对应的 setup 脚本。root 用户默认路径为 `/usr/local/Ascend`，普通用户默认路径为 `${HOME}/Ascend`：

```bash
# root 用户默认路径
source /usr/local/Ascend/cann/set_env.sh

# 普通用户默认路径
source ${HOME}/Ascend/cann/set_env.sh

# 自定义安装路径（根据实际安装路径替换）
source ${install_path}/cann/set_env.sh
```

设置完成后，`ASCEND_HOME_PATH` 环境变量会自动指向 CANN 安装目录。

### 2. 本地构建

```bash
./build.sh
```

构建完成后，唯一发布产物是一个平台标签化 whl：

- `output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl`（debug 构建为 `<version>+debug`）

该 whl 同时包含 Python API 与 C++ 制品（头文件、动态库、CMake 配置包），一个平台一个包、与 Python 版本无关。

`build.sh` 自动完成以下步骤：
1. 检查当前系统激活的 Python 环境（cmake、gcc、Python、CANN）
2. 准备 Python 环境（默认自动创建 `.venv`；加 `--system_python` 使用当前环境）
3. 编译三方库（nlohmann_json、gtest、libtorch_npu）
4. 编译并安装 C++ delegate 库
5. 构建 whl（把 CMake 安装树内嵌进 Python 包，不重复调用 CMake）

## 构建命令

### build.sh 参数

| 参数 | 说明 |
|------|------|
| `--debug` | 构建 debug 版本 |
| `--release` | 构建 release 版本（默认） |
| `--cann_3rd_lib_path=PATH` | 指定三方库安装路径 |
| `--output_dir=PATH` | 产物输出路径（默认：output/package） |
| `--pre_check` | 仅校验环境并退出，不执行构建 |
| `--skip_third_party` | 跳过三方库编译 |
| `--with_libtorch_npu` | 编译 libtorch_npu（默认关闭） |
| `--system_python` | 使用当前 Python 环境，不自动创建 .venv |
| `--help` | 帮助信息 |

Python 环境策略：

**默认模式**（不加 `--system_python`）：
1. `.venv` 已存在：激活并重新安装依赖（pip 自动跳过已安装包）
2. `.venv` 不存在：自动创建并安装 torch + CANN 运行时依赖

**`--system_python` 模式**：
- 直接使用当前系统激活的 Python 解释器（`command -v python3`，兼容 venv / conda / 系统 python）
- 不创建 `.venv`，不自动安装任何包
- 要求当前环境已安装正确版本的 torch（未安装或版本不匹配时报错）
- 适用于：已有 conda 环境、CI 环境、或希望手动控制依赖安装的场景

`Torch_DIR` 始终从所选 Python 环境中的 `torch` 包推导，不接收外部指定路径。可通过 `--pre_check` 提前检查环境。

示例：

```bash
# 默认 release 构建（自动创建 .venv）
./build.sh

# debug 构建
./build.sh --debug

# 使用当前 conda/venv 环境
./build.sh --system_python

# 启用 libtorch_npu 编译
./build.sh --with_libtorch_npu

# 指定输出目录
./build.sh --output_dir=/custom/output
```

### build_test.sh - 测试构建

| 参数 | 说明 |
|------|------|
| `--debug` | 构建 debug 测试（默认） |
| `--release` | 构建 release 测试 |
| `--asan` | 启用 AddressSanitizer（默认） |
| `--no-asan` | 关闭 ASAN |
| `--cann_3rd_lib_path=PATH` | 指定三方库路径 |
| `--skip_third_party` | 跳过三方库编译 |
| `--skip_tests` | 仅编译不运行测试 |
| `--system_python` | 使用当前 Python 环境，不自动创建 .venv |

`--debug` 和 `--release` 互斥，同时指定会报错。

示例：

```bash
# 构建并运行测试（ASAN默认开启）
./build_test.sh

# 使用当前 Python 环境 running tests
./build_test.sh --system_python

# 构建release测试
./build_test.sh --release

# 关闭ASAN
./build_test.sh --no-asan
```

## 版本配置

所有版本号集中在 `build.conf` 中管理：

```bash
# TorchDelegate
TORCH_DELEGATE_VERSION=0.1.0

# PyTorch (CPU)
TORCH_VERSION=2.9.0

# CANN
CANN_VERSION=9.1.0-beta1

# libtorch_npu
LIBTORCH_NPU_RELEASE_TAG=v26.1.0-beta.1-pytorch2.9.0

# Third-party
NLOHMANN_JSON_VERSION=3.11.3
GTEST_VERSION=1.14.0
```

切换到新版本只需修改此文件。`TORCH_DELEGATE_VERSION` 是版本号唯一来源：构建时经环境变量 `TORCH_DELEGATE_PACKAGE_VERSION` 注入 whl 元数据与运行时 `torch_delegate.__version__`，CMake 侧使用同一变量，`pyproject.toml` 不再硬编码版本。

## whl 安装与使用

whl 是唯一发布产物，同时承载 Python API 与 C++ 制品。

### 安装

```bash
pip install output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

- 平台标签为 `linux_x86_64` / `linux_aarch64`；Python/ABI 标签固定 `py3-none`，同一个 whl 可安装进 Python 3.10 ~ 3.13，无需按版本分别构建。
- `torch_npu` 不在依赖中自动安装：完整导出模式需要它（提供 `torchair`），请自行安装与 `torch` 配套的版本（`torch_npu>=2.9.0`）；缺失时导出直接报错，不静默降级。
- debug 构建的版本号带 `+debug` 后缀（如 `0.1.0+debug`）。

### 安装后的包内布局

```
<site-packages>/torch_delegate/
├── *.py, converter/, epair/, core/     ← Python API
├── _version.py                         ← 构建注入的 __version__
├── inc/torch_delegate/                 ← C++ 对外头文件
│   ├── epair_model_loader.h
│   └── td_types.h
└── lib/
    ├── libtorch_delegate_backend.so    ← C++ 动态库
    └── cmake/TorchDelegate/            ← CMake 配置包（Config / Targets / modules）
```

CMake 安装前缀整体位于包目录内，因此 `<site-packages>/torch_delegate` 就是 C++ 前缀。

### 获取 C++ 前缀（TORCH_DELEGATE_ROOT）

不再提供 `set_env.sh`，改用以下任一方式（两者等价）：

```bash
# 方式 1：CLI（pip 安装时生成，需当前环境的 bin 在 PATH 中）
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

# 方式 2：Python 函数
export TORCH_DELEGATE_ROOT=$(python3 -c "import torch_delegate; print(torch_delegate.cmake_prefix())")
```

`torch-delegate-config --version` 查询版本号。

> **NOTE**：`torch-delegate-config` 是 console script，pip 会把它装进当前环境的 `bin/`（venv / conda / 系统 Python 均支持）。使用前需激活目标环境使其 `bin/` 位于 `PATH`；`pip install --target` 场景下脚本不进 PATH，请改用方式 2。

### 运行期依赖（需手动配置）

`libtorch_delegate_backend.so` 运行期依赖 libtorch 与 CANN。二者不打入 whl，动态库也不注入指向它们的 RUNPATH，因此执行前需手动配置 `LD_LIBRARY_PATH`：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

说明：

- `${TORCH_PKG_DIR}.libs` 仅 aarch64 存在（x86_64 的 libtorch 全在 `torch/lib`），目录不存在时无害。
- 使用 fallback codegen / 自定义算子时，还需追加 `${TORCH_NPU_DIR}/lib`。

### C++-only 使用（不安装 Python 包）

whl 本质是 zip 归档，可直接解压取用 C++ 制品：

```bash
unzip torch_delegate-<version>-py3-none-linux_<arch>.whl -d /tmp/td
# 前缀为 /tmp/td/torch_delegate，含 inc/、lib/、lib/cmake/TorchDelegate/
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/td/torch_delegate
```

或 `pip install --target=<dir> <whl>`，前缀为 `<dir>/torch_delegate`。

### 发布验证

```bash
# whl 内容与元数据校验（pytest，需先 ./build.sh 产出 whl）
pytest tests/python/test_wheel_package.py

# 端到端冒烟：临时 venv 安装 → import → CLI → readelf → find_package 编译运行
bash tests/packaging/wheel_smoke_test.sh
```

冒烟脚本只验证打包与接入链路，不需要 NPU 设备。可用 `TORCH_DELEGATE_WHEEL=<path>` 指定 whl、`--python=<interpreter>` 指定基础解释器。

## 用户接入

### C++ 项目

安装 whl 后，用 `torch-delegate-config --cmake-prefix`（或 `torch_delegate.cmake_prefix()`）取得 C++ 前缀并导出为 `TORCH_DELEGATE_ROOT`，再在 CMakeLists.txt 中通过 `$ENV{TORCH_DELEGATE_ROOT}` 接入。下面是外部应用的 CMAKE 模板：

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

说明：应用只链接 `TorchDelegate::torch_delegate_backend`，不直接链接 `torch`、`torch_cpu`、`c10`。但最终应用链接时仍需将 CANN lib 目录、`torch/lib`，以及存在时的 `torch.libs` 同时用于 `target_link_directories` 和 `-rpath-link`。

配置示例：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
  -DTORCH_PKG_DIR="${TORCH_PKG_DIR}"
```

`TorchDelegateConfig.cmake` 不代管 torch：torch 的库路径由消费方通过 `-DTORCH_PKG_DIR`（或 CMakeLists 内解析 `ENV{TORCH_PKG_DIR}`）传入，如上模板所示。

需要运行 fallback codegen、自定义算子或包含 `libtorch_npu` 的示例时，还需要设置 `TORCH_NPU_DIR`：

```bash
export TORCH_PKG_DIR=/path/to/site-packages/torch
export TORCH_NPU_DIR=/path/to/libtorch_npu
```

其中 `TORCH_PKG_DIR` 统一指向 Python `torch` 包目录。需要 `TorchConfig.cmake` 的 CMake 会在内部拼接 `${TORCH_PKG_DIR}/share/cmake/Torch`，外部不再传入 `share/cmake/Torch` 目录。

运行前需手动设置运行时库路径（whl 不注入 RUNPATH，也不提供 set_env.sh）：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh

TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
```

使用 delegate API：

```cpp
#include "torch_delegate/epair_model_loader.h"
#include "torch_delegate/td_types.h"

// 加载 .epair 模型
td::EpairModelLoader loader("/path/to/model.epair");

// 编译并加载到设备
std::map<ge::AscendString, ge::AscendString> options;
void *stream = nullptr;  // ACL 流指针
td::Status status = loader.CompileAndLoad(options, stream);

// 异步运行推理
std::vector<gert::Tensor> inputs, results;
status = loader.RunModelWithStreamAsync(stream, inputs, results);

// 检查执行结果
if (status == td::SUCCESS) {
    // 处理输出
}
```

### Python 项目

```bash
pip install output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

```python
import torch_delegate
from torch_delegate import export_to_epair, epair_save, epair_load

print(torch_delegate.__version__)
print(torch_delegate.cmake_prefix())   # C++ 前缀，可交给 CMAKE_PREFIX_PATH
```

完整导出模式需要 `torch_npu`（提供 `torchair`），请自行安装与 `torch` 配套的版本；whl 不会自动拉取。

## IDE配置

### CLion

在 **Settings → Build → CMake** 中添加：

```
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
-DBUILD_DELEGATE=ON
-DBUILD_TESTS=ON
-DASCEND_INSTALL_PATH=/path/to/cann        # 替换为实际 CANN 安装路径
-DTorch_DIR=/path/to/venv/lib/python3.10/site-packages/torch/share/cmake/Torch
-DCANN_3RD_LIB_PATH=./output/third_party
-DENABLE_ASAN=ON
```

如果通过环境变量指定 torch，使用 Python torch 包目录：`export TORCH_PKG_DIR=/path/to/venv/lib/python3.10/site-packages/torch`。

`ASCEND_INSTALL_PATH` 也可通过环境变量 `ASCEND_HOME_PATH` 自动获取，source CANN 环境后 CLion 会继承。

### VSCode

使用 CMake Tools 扩展，在 `.vscode/settings.json` 或 CMake Kits 中配置同 CLion 的参数。在 settings.json 中：

```json
{
    "cmake.configureSettings": {
        "ASCEND_INSTALL_PATH": "/path/to/cann",
        "Torch_DIR": "/path/to/venv/lib/python3.10/site-packages/torch/share/cmake/Torch",
        "CANN_3RD_LIB_PATH": "${workspaceFolder}/output/third_party"
    }
}
```

## ABI 兼容性

- 对外头文件不暴露 `std::string`、`std::unordered_map`、`std::filesystem::path`
- 字符串参数统一使用 `const char*` 或 `ge::AscendString`
- whl 内 `torch_delegate/lib/` 目录包含 `libtorch_delegate_backend.so`，`torch_delegate/inc/` 包含对外头文件

## 常见问题

### Q1: ASCEND_HOME_PATH 未设置

**解决方案**：
source CANN 环境脚本：`source /path/to/Ascend/cann/set_env.sh`

### Q2: torch 安装网络失败

**解决方案**：
`build.sh` 使用华为云 PyPI 镜像（`https://repo.huaweicloud.com/repository/pypi/simple`）加速依赖下载。如仍有问题，检查网络连接或配置代理。

### Q3: 运行 C++ 程序报 `libtorch_cpu.so` / `libgraph.so` cannot open shared object file

**解决方案**：
whl 不注入 RUNPATH，运行前需手动配置 `LD_LIBRARY_PATH`（见「运行期依赖（需手动配置）」），确保包含 `torch/lib`、`torch.libs`（aarch64）、`$TORCH_DELEGATE_ROOT/lib`、`$ASCEND_HOME_PATH/lib64`。

### Q4: `torch-delegate-config: command not found`

**解决方案**：
该命令由 pip 装入当前环境的 `bin/`，需先激活安装了 whl 的环境（venv / conda）。若用 `pip install --target` 安装，脚本不进 PATH，改用
`python3 -c "import torch_delegate; print(torch_delegate.cmake_prefix())"`。

### Q5: 导出时报缺少 TorchAir / torch_npu

**解决方案**：
完整导出模式依赖 `torch_npu` 提供的 `torchair`，whl 不会自动安装。请自行安装与 `torch` 版本配套的 `torch_npu>=2.9.0`；无 CANN 运行时的场景改用 `lite_export_context()`。


## 输出目录结构

```
output/third_party/                                    ← 三方库产物
├── json/
├── gtest/
└── libtorch_npu/

output/package/                                        ← 发布产物（唯一）
└── torch_delegate-<ver>-py3-none-linux_<arch>.whl     ← Python API + C++ 制品
```

debug 构建产出 `torch_delegate-<ver>+debug-py3-none-linux_<arch>.whl`。

## 相关文件

- `build.sh` - 一站式构建脚本（产出单一 whl）
- `build.conf` - 集中版本配置（版本号唯一来源）
- `build_third_party.sh` - 三方库构建脚本
- `build_test.sh` - 测试用例构建和运行脚本
- `setup.py` - whl 装配（内嵌 CMake 安装树、平台 tag、版本注入）
- `pyproject.toml` - Python 包元数据与 entry point
- `torch_delegate/cli.py` - `torch-delegate-config` 实现
- `tests/python/test_wheel_package.py` - whl 内容与元数据校验
- `tests/packaging/wheel_smoke_test.sh` - whl 端到端冒烟测试
- `requirements.txt` - Python依赖说明
