# 使用C++ 加载和运行PTE/PTD

本文只回答两件事：C++ 应用编译和设备执行分别需要什么，以及如何通过ExecuTorch
`Module` 加载ExecuTorch-NPU生成的PTE/PTD。

## 输入和输出

C++ 应用至少需要：

```text
model.pte
ascend.weights.package.ge_aot.ptd
应用输入数据
```

PTE/PTD的生成方式见 [模型导出](export.md)。协议细节见 [产物协议](protocol.md)。

## 依赖边界

### C++ 应用编译依赖

| 依赖 | 用途 |
|---|---|
| CMake >= 3.24 | 配置应用工程 |
| C++17编译器 | 编译应用 |
| ExecuTorch 1.5.0 full wheel | C++ 头文件、`libexecutorch.so` 和CMake package |
| ExecuTorch-NPU wheel SDK | Backend静态库、状态头文件和CMake接口 |
| CANN Toolkit开发文件 | ACL头文件和 `libascendcl.so` |
| OpenSSL Crypto | Backend Manifest校验 |

C++ 应用编译不要求Torch、TorchAir、torch_npu或torch-delegate。

### 设备执行依赖

| 依赖 | 用途 |
|---|---|
| 已编译应用 | ExecuTorch调用入口 |
| `libexecutorch.so` | ExecuTorch Runtime和Module实现 |
| CANN Runtime、NNRT | OM加载和执行 |
| Ascend Driver、Firmware | 设备运行 |
| 与产物匹配的目标NPU | 执行离线编译结果 |

应用编译完成后，执行路径不需要Python或Torch。使用wheel路径部署
`libexecutorch.so` 时，应保留对应Runtime库或设置正确的动态库搜索路径。

## 准备C++ SDK

在独立C++ 开发环境中只安装开发文件：

```bash
python -m pip install --no-deps "executorch==1.5.0"
python -m pip install --no-deps /path/to/executorch_npu-*.whl
```

这两个命令不会安装Torch。获取两个SDK根目录：

```bash
export EXECUTORCH_ROOT="$(python -c 'import executorch; print(next(iter(executorch.__path__)))')"
export EXECUTORCH_NPU_SDK="$(python -c 'from executorch_npu import sdk_root; print(sdk_root())')"
source /path/to/cann/set_env.sh
```

## 配置CMake工程

最小 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("$ENV{EXECUTORCH_NPU_SDK}/cmake/executorch_npu_backend.cmake")

add_executable(my_runner runner.cpp)
target_link_libraries(my_runner PRIVATE executorch_npu::backend)
```

`executorch_npu::backend` 会传递ExecuTorch Runtime、OpenSSL、AscendCL、头文件和
Backend whole-archive注册要求。应用不应直接定位或逐个链接SDK内的 `.a` 文件。

配置和编译：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$EXECUTORCH_ROOT" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build build -j
```

## 加载PTE/PTD

ExecuTorch 1.5高层Module API直接接受PTE和PTD两个路径：

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::module::Module;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::LoadBackendOptionsMap;

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }

  executorch::runtime::runtime_init();

  BackendOptions<2> ascend_options;
  if (ascend_options.set_option("device_id", 0) != Error::Ok ||
      ascend_options.set_option("state_namespace", "my-request") != Error::Ok) {
    return 1;
  }

  LoadBackendOptionsMap backend_options;
  if (backend_options.set_options("AscendBackend", ascend_options.view()) !=
      Error::Ok) {
    return 1;
  }

  Module module(argv[1], argv[2]);
  return module.load(backend_options) == Error::Ok ? 0 : 1;
}
```

加载成功后，通过 `method_meta("forward")` 获取输入输出元数据，构造ExecuTorch
Tensor，再调用 `module.forward(...)`。仓库内 `tools/ascend_runner.cpp` 是带输入文件和
golden输出比较的验证工具，不是通用应用接口。

## 执行应用

执行前加载目标机器的CANN Runtime环境：

```bash
source /path/to/cann/set_env.sh
./build/my_runner model.pte ascend.weights.package.ge_aot.ptd
```

注意：

- `device_id` 是ACL Runtime的逻辑设备编号，不一定等于 `npu-smi` 展示的物理卡号。
- PTE中的 `acl_soc_name` 必须与 `aclrtGetSocName()` 返回值精确匹配。
- PTE、PTD和ExecuTorch-NPU Backend SDK必须来自兼容版本组合。
- PTE/PTD校验失败时，Backend会在加载阶段拒绝执行。

## Stateful模型

Prefill/Decode state管理接口位于：

```cpp
#include <executorch_npu/state.h>
```

API、隔离键和状态提交语义见 [C++ Backend SDK：Stateful API](cpp_backend_sdk.md#stateful-api)。

## 运行检查

检查动态依赖：

```bash
ldd build/my_runner
```

应能够找到 `libexecutorch.so`、`libascendcl.so`、OpenSSL和系统C++ Runtime。常见编译
和运行错误见 [故障排查](troubleshooting.md)。
