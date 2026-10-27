# stub_kernel.so - 自定义算子Schema Stub

简体中文 | [English](README_en.md)

## 用途

将自定义算子schema（如 `xllm_ascend::rope_in_place`）注册到libtorch dispatcher，
使 `FallbackCodegen` 能够查询非原生注册算子的schema信息。

**已验证**： 编译成功（485KB），schema注册已通过rope_in_place.json测试。

---

## 已注册算子

### rope_in_place（inplace版本）

**命名空间**： `xllm_ascend`

**Schema**：
```text
rope_in_place(Tensor(a!) input, Tensor sin_cache, Tensor cos_cache) -> ()
```

**完整Target**： `torch.ops.xllm_ascend.rope_in_place.default`

**关键特征**：
- `Tensor(a!)`： 表示input参数会被**写入**（alias annotation）
- `-> ()`： 返回类型为 `void`（inplace操作不返回新tensor）
- Kernel参数类型: `at::Tensor&`（引用，非const）

**Dispatch Keys**：
- Meta: inplace shape inference（无输出tensor，input shape不变）
- PrivateUse1: NPU stub（空实现，不修改input）
- CPU: CPU stub（空实现，不修改input）

**来源**： 基于 `tests/cpp/rope_in_place.json`

**验证结果**：
```bash
# Schema 输出
xllm_ascend::rope_in_place(Tensor(a!) input, Tensor sin_cache, Tensor cos_cache) -> ()

# Python 验证
import ctypes, torch
ctypes.CDLL('libstub_kernel.so', ctypes.RTLD_GLOBAL)
op = torch.ops.xllm_ascend.rope_in_place
print(f'算子已注册: {op}')  # ✅ 成功
```

---

## 编译方式

### 方式1项目CMake集成构建（推荐）

```bash
# 配置项目（启用 BUILD_TESTS=ON）
cd torch-delegate
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -B build

# 构建 stub_kernel（随 tests 自动构建）
cmake --build build --target stub_kernel

# 查看输出
ls build/tests/cpp/stub_kernel/libstub_kernel.so
```

**优点**： 自动依赖管理，与测试基础设施集成

---

### 方式2独立CMake构建（脱离项目）

```bash
# 进入 stub_kernel 目录
cd tests/cpp/stub_kernel

# 创建独立构建目录
mkdir build && cd build

# 配置（必须指定 TORCH_PKG_DIR 和 TORCH_NPU_DIR）
# TORCH_PKG_DIR 指向 Python torch 包目录
# TORCH_NPU_DIR 指向 libtorch_npu 源码根目录
cmake .. \
    -DTORCH_PKG_DIR=/mnt/workspace/thj_delegate/lib/python3.12/site-packages/torch \
    -DTORCH_NPU_DIR=/mnt/workspace/gitCode/pytorch/libtorch_npu

# 构建
cmake --build .

# 输出在当前目录
ls libstub_kernel.so
```

**注意**：
- `TORCH_PKG_DIR` 必须指向 **Python torch包目录**（如 `/mnt/workspace/thj_delegate/lib/python3.12/site-packages/torch`）
- `TORCH_NPU_DIR` 必须指向 **libtorch_npu源码根目录**（如 `/mnt/workspace/gitCode/pytorch/libtorch_npu`）
- CMake会自动将源码父目录加入include路径，以解析 `NPUQueue.h` 中的 `#include "torch_npu/csrc/logging/LogContext.h"`
- 自动查找 `${TORCH_NPU_DIR}/lib/libtorch_npu.so` 并链接，提供 `NPUStream` 等运行时符号
- 将 `/path/to/libtorch` 替换为实际的libtorch安装路径

---

### 方式3 g++直接编译（无需CMake）

```bash
# 进入 stub_kernel 目录
cd tests/cpp/stub_kernel

# 设置 libtorch 路径
TORCH_PKG_DIR=/path/to/libtorch

# 直接编译
g++ -shared -fPIC -o libstub_kernel.so stub_kernel.cc \
    -std=c++17 \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I${TORCH_PKG_DIR}/include \
    -I${TORCH_PKG_DIR}/include/torch/csrc/api/include \
    -L${TORCH_PKG_DIR}/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,${TORCH_PKG_DIR}/lib

# 输出在当前目录
ls libstub_kernel.so
```

**注意**： 根据实际libtorch安装路径调整 `-I` 和 `-L` 参数。

---

## 使用方式

stub_kernel.so需要在 `FallbackCodegen` 调用 `findSchema` 之前加载，使其schema注册到libtorch dispatcher。推荐以下方式：

### 方式1 LD_PRELOAD（推荐）

```bash
LD_PRELOAD=/path/to/libstub_kernel.so ./torch_delegate_tests
```

### 方式2测试代码中dlopen

```cpp
#include <dlfcn.h>

void* handle = dlopen("/path/to/libstub_kernel.so", RTLD_NOW | RTLD_GLOBAL);
if (!handle) {
    fprintf(stderr, "Failed to load stub_kernel: %s\n", dlerror());
}
```

### 方式3 Python import（需构建为extension）

```python
# 构建 stub_kernel 为 Python extension（需 PyInit_stub_kernel 符号）
import torch
import stub_kernel  # 通过 TORCH_LIBRARY 静态初始化触发 schema 注册
```

---

## 验证方法

### 快速验证（Python）

```python
import ctypes
import torch

# 加载 stub_kernel.so（使用 RTLD_GLOBAL 注册 schema）
stub_path = '/path/to/libstub_kernel.so'
handle = ctypes.CDLL(stub_path, ctypes.RTLD_GLOBAL)
print(f'已加载: {stub_path}')

# 验证 schema 注册
op = torch.ops.xllm_ascend.rope_in_place
print(f'算子已注册: {op}')
print(f'成功: Schema 可通过 torch.ops 访问')
```

**预期输出**：
```text
已加载: /path/to/libstub_kernel.so
算子已注册: xllm_ascend.rope_in_place
成功: Schema 可通过 torch.ops 访问
```

---

### Schema注册检查（C++）

```cpp
#include <ATen/core/function_schema.h>

c10::OperatorName op_name("xllm_ascend::rope_in_place", "");
auto handle = torch::Dispatcher::singleton().findSchema(op_name);

if (handle.has_value()) {
    std::cout << "Schema found: " << handle->schema() << std::endl;
} else {
    std::cerr << "Schema NOT found for xllm_ascend::rope_in_place" << std::endl;
}
```

**预期输出**：
```text
Schema found: rope_in_place(Tensor input, Tensor sin_cache, Tensor cos_cache) -> Tensor
```

---

## 目录结构

```text
tests/cpp/stub_kernel/
├── stub_kernel.cc          # Schema 注册实现
├── CMakeLists.txt          # 独立 CMake 构建配置
└── README.md               # 本文件
```

**项目集成**：
- `cmake -DBUILD_TESTS=ON` 时自动构建
- 输出: `${BUILD_DIR}/tests/cpp/stub_kernel/libstub_kernel.so`
- 集成在 `tests/CMakeLists.txt`

---

## 故障排查

### Schema not found错误

```text
FallbackCodegen: schema not found for 'xllm_ascend::rope_in_place' with overload ''
```

**解决方案**：
1. 检查stub_kernel.so文件是否存在
2. 通过 `LD_PRELOAD` 或代码中 `dlopen` 加载stub_kernel.so
3. 检查dlopen是否成功（启用调试日志）:
   ```bash
   export TORCH_DELEGATE_LOG_LEVEL=DEBUG
   ./torch_delegate_tests
   ```

---

### ABI mismatch错误

```text
undefined symbol: _ZNK2at6Tensor... (ABI=0 vs ABI=1)
```

**解决方案**： 确保stub_kernel使用 `_GLIBCXX_USE_CXX11_ABI=1` 编译（与libtorch保持一致）

---

### dlopen权限错误

**解决方案**： 检查文件权限和路径可访问性:
```bash
ls -la /path/to/libstub_kernel.so
```

---

## 扩展其他算子

编辑 `stub_kernel.cc` 添加更多自定义算子:

```cpp
TORCH_LIBRARY(xllm_ascend, m) {
    m.def("rope_in_place(Tensor input, Tensor sin_cache, Tensor cos_cache) -> Tensor");
    // 在此处添加更多算子
    m.def("other_custom_op(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ascend, Meta, m) {
    m.impl("rope_in_place", [](...) { return at::empty_like(...); });
    m.impl("other_custom_op", [](...) { return at::empty_like(...); });
}
```

然后重新编译stub_kernel.so。

---

## 相关文件

- `tests/cpp/rope_in_place.json`： 定义算子结构的源JSON
- `csrc/fallback_codegen/fallback_codegen.cc`： Fallback codegen实现
- `tests/CMakeLists.txt`： 项目构建集成

---

## 验证结果总结

### 编译验证
- ✅ stub_kernel.so编译成功（485KB）
- ✅ 导出符号包含 `xllm_ascend::rope_in_place` schema注册
- ✅ ABI=1设置正确（与libtorch一致）

### Schema注册验证
```bash
python3 -c "
import ctypes, torch
handle = ctypes.CDLL('tests/cpp/stub_kernel/libstub_kernel.so', ctypes.RTLD_GLOBAL)
op = torch.ops.xllm_ascend.rope_in_place
print(f'算子: {op}')  # 输出: 算子: xllm_ascend.rope_in_place
"
```

### rope_in_place.json验证
- ✅ JSON解析成功（214KB, 21个nodes）
- ✅ rope_in_place node确认:
  - target: `torch.ops.xllm_ascend.rope_in_place.default`
  - inputs: 3（input, sin_cache, cos_cache）
  - outputs: 1
  - `ascend_ir.source`： MISSING（需要fallback codegen）
