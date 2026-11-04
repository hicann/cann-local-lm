# ExecuTorch-NPU C++ Backend SDK参考

本文件说明wheel内C++ SDK的稳定内容和接口边界。完整的应用编译、PTE/PTD加载和
设备执行步骤见 [C++ 运行](runtime.md)。

## SDK定位

安装wheel后，通过轻量API获取SDK根目录：

```bash
python -c "from executorch_npu import sdk_root; print(sdk_root())"
```

该操作不需要安装Torch，也不会导入Python AOT Backend。

## Wheel内SDK内容

```text
executorch_npu/_sdk/
├── cmake/
│   ├── executorch_npu_backend.cmake
│   └── executorch_npu_executorch.cmake
├── include/executorch_npu/state.h
├── lib/
│   ├── libexecutorch_npu_manifest.a
│   └── libexecutorch_npu_backend.a
└── compatibility.json
```

SDK不包含：

- Runner可执行文件。
- ExecuTorch、CANN Toolkit、驱动、固件或NNRT。
- PTE、PTD、模型权重或输入数据。
- Backend内部类声明、实现源码和测试源码。
- `AscendBackend::init/execute/destroy` 的直接调用接口。

## CMake targets

应用包含：

```cmake
include("$ENV{EXECUTORCH_NPU_SDK}/cmake/executorch_npu_backend.cmake")
```

对外target：

```text
executorch_npu::backend
```

该target传递：

- Backend和Manifest静态库。
- Backend whole-archive注册要求。
- ExecuTorch-NPU公开头文件。
- ExecuTorch 1.5 `executorch::runtime` 或CMake 3.24–3.27兼容链接路径。
- OpenSSL Crypto和 `libascendcl.so`。

以下target属于SDK内部实现，不应由应用直接链接：

```text
executorch_npu_backend_archive
executorch_npu_manifest_archive
executorch_npu::executorch_runtime
```

## Backend注册和调用边界

`libexecutorch_npu_backend.a` 通过静态初始化注册 `AscendBackend`。链接时必须保留注册
对象，因此SDK CMake接口使用whole-archive。

调用链：

```text
用户应用
→ ExecuTorch Module API
→ ExecuTorch Runtime
→ PTE BackendDelegate.id = AscendBackend
→ AscendBackend init / execute / destroy
→ AscendCL
→ Ascend NPU
```

用户应用只调用ExecuTorch Runtime或Module API，不直接调用Backend生命周期方法。

## Stateful API

```cpp
#include <executorch_npu/state.h>
```

公开接口：

```cpp
StateStatus SetStateInitializer(
    int device_id,
    const char* state_namespace,
    const char* state_id,
    const StateTensorData& initializer);

StateStatus ResetState(
    int device_id,
    const char* state_namespace,
    const char* state_id,
    const StateTensorData& source);

StateStatus ReleaseStateNamespace(
    int device_id,
    const char* state_namespace);
```

状态按 `(device_id, state_namespace, state_id)` 隔离。Backend使用current/next
双缓冲，只有执行成功后才提交状态交换。状态协议见 [产物协议](protocol.md)。

## Compatibility metadata

`compatibility.json` 由构建过程生成，记录：

```text
ExecuTorch-NPU package version
Build type
Python version
ExecuTorch version 和 git commit
CANN version
C++ 编译器版本
```

Backend静态库和ExecuTorch Runtime需要保持兼容。不要手工修改该文件，也不要把一个
wheel中的 `.a` 与另一个ExecuTorch版本随意组合。

模型与wheel独立交付：

```text
model.pte
ascend.weights.package.ge_aot.ptd
```

执行机还必须提供CANN Runtime、驱动、固件、NNRT、系统C++ Runtime和OpenSSL。

## Backend二次开发

wheel SDK用于应用集成，不替代ExecuTorch-NPU源码开发环境。修改Backend、Manifest、
ACL执行或状态实现时，应使用本仓库源码重新构建wheel和静态库。普通应用不再需要
编译ExecuTorch 1.5源码。
