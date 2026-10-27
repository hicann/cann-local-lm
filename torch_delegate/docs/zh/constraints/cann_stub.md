# CANN Stub设计文档

## 背景

项目中的模块需要调用CANN/ACL runtime API（`aclInit`、`aclrtMalloc` 等）。在没有NPU硬件的CI环境中，需要一个stub `libascendcl.so` 让测试解耦真实CANN硬件。

核心约束：stub的 `.so` 输出路径与真实CANN的相对路径一致，因此无论链接哪个，编译和打包产物完全相同——不需要区分"stub模式"和"生产模式"的编译开关，只需在运行时通过 `LD_LIBRARY_PATH` 决定加载哪个。

---

## 目录结构

stub属于测试基础设施，放在 `tests/cpp/cann_stub/` 而非 `csrc/`。

```text
tests/cpp/
├── cann_stub/
│   ├── ascendcl.cc            # libascendcl.so 对应的 stub 实现
│   └── cann_stub_control.h    # 测试专用控制接口（Reset/查询/错误注入）
└── test_cann_stub.cc          # GTest 单元测试
```

**文件命名规则**：一个 `.cc` 文件对应一个 `.so`，文件名与 `.so` 去掉 `lib` 前缀后的名称相同。当前只有 `libascendcl.so`，故只有 `ascendcl.cc`。未来若需要stub `libacl_op_compiler.so`，则新增 `acl_op_compiler.cc`。

---

## 共享库命名与路径

构建产物路径（`LIBRARY_OUTPUT_DIRECTORY`）：

```text
${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64/libascendcl.so
```

与真实CANN的相对路径对应：

```text
真实：<CANN_ROOT>/lib64/libascendcl.so
stub：<build>/tests/cpp/cann_stub/lib64/libascendcl.so
```

SONAME设置为 `libascendcl.so`（与真实库一致），通过 `OUTPUT_NAME "ascendcl"` 实现。

**只需一个 `.so`**：当前9个目标API均由真实CANN的 `libascendcl.so` 导出。

---

## CMake集成

### CANN_ROOT Cache变量

```cmake
set(CANN_ROOT "" CACHE PATH "CANN 安装根路径（含 include/ 和 lib64/ 符号链接）")
```

**CANN_ROOT校验逻辑**（必须非空且头文件存在，否则 `FATAL_ERROR`）：

- 若 `CANN_ROOT` 为空：报错，提示 `-DCANN_ROOT=<path>`
- 若 `CANN_ROOT` 非空但 `${CANN_ROOT}/include/acl/acl_rt.h` 不存在：报错，提示头文件路径不对

### stub target

```cmake
add_library(cann_stub SHARED tests/cpp/cann_stub/ascendcl.cc)
set_target_properties(cann_stub PROPERTIES
    OUTPUT_NAME "ascendcl"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64"
)
target_include_directories(cann_stub PRIVATE ${CANN_ROOT}/include)
target_compile_options(cann_stub PRIVATE ${DELEGATE_COMMON_FLAGS})
target_link_options(cann_stub PRIVATE ${DELEGATE_COMMON_FLAGS})
```

- `target_include_directories` 指向 `${CANN_ROOT}/include`（使用真实头文件，不拷贝任何类型定义）
- 遵守 `DELEGATE_COMMON_FLAGS`（`-Wall -Wextra -Werror` +可选ASAN）

### 测试executable链接

```cmake
target_link_libraries(torch_delegate_tests PRIVATE cann_stub ...)
```

### 运行时LD_LIBRARY_PATH注入

通过 `set_tests_properties` 将环境变量注入ctest，无需手动设置：

```cmake
set_tests_properties(torch_delegate_tests PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64:$ENV{LD_LIBRARY_PATH}"
)
```

**为何不需要 `USE_CANN_STUB` 开关**：stub与真实CANN的 `libascendcl.so` 具有相同的SONAME和ABI接口，链接stub编译出的模块产物与链接真实库完全等价。测试环境在运行时加载stub，生产环境在运行时加载真实库，编译阶段无需区分。

---

## 头文件策略

`ascendcl.cc` 直接 `#include <acl/acl_rt.h>`，使用真实CANN头文件中的类型定义，**不拷贝、不重新定义任何ACL类型**。`cann_stub_control.h` 同样包含此头文件以使用 `aclError` 等类型。

---

## Stub行为约定

| 函数 | 签名来源 | 返回值 | 副作用 |
|------|---------|--------|--------|
| `aclInit` | `acl_rt.h:858` | `ACL_SUCCESS` | 设置 `initialized_ = true` |
| `aclFinalize` | `acl_rt.h:870` | `ACL_SUCCESS` | 设置 `initialized_ = false` |
| `aclrtSetDevice` | `acl_rt.h:1284` | `ACL_SUCCESS` | 记录 `current_device_` |
| `aclrtResetDevice` | `acl_rt.h:1311` | `ACL_SUCCESS` | 清除 `current_device_`（置 -1） |
| `aclrtCreateStream` | `acl_rt.h:2280` | `ACL_SUCCESS` | `new` sentinel对象写入 `*stream`，加入 `live_streams_` |
| `aclrtDestroyStream` | `acl_rt.h:2328` | `ACL_SUCCESS` | `delete` sentinel，从 `live_streams_` 移除 |
| `aclrtSynchronizeStream` | `acl_rt.h:2356` | `ACL_SUCCESS` | 无操作 |
| `aclrtMalloc` | `acl_rt.h:1626` | `ACL_SUCCESS` | `malloc(size)` 写入 `*devPtr`，记入 `live_allocs_` |
| `aclrtFree` | `acl_rt.h:1869` | `ACL_SUCCESS` | `free(devPtr)`，从 `live_allocs_` 移除 |

**`aclrtMalloc` 使用真实 `malloc`**：有模块可能会实际读写张量数据，真实分配+ ASAN可检测越界和泄漏。

**`aclrtCreateStream` 分配sentinel对象**：给流句柄一个稳定的非空指针，ASAN可检测use-after-free。

---

## 全局状态结构（`CannStubState`）

单例，包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `initialized_` | `bool` | `aclInit`/`aclFinalize` 状态 |
| `current_device_` | `int32_t` | -1表示未设置 |
| `live_streams_` | `std::unordered_set<void*>` | 当前存活的流句柄 |
| `live_allocs_` | `std::unordered_map<void*, size_t>` | pointer → size |
| `state_mutex_` | `std::mutex` | 保护所有字段 |
| `next_error_` | `aclError` | 一次性错误注入，`ACL_SUCCESS` 表示不注入 |

---

## 测试控制接口（`cann_stub_control.h`）

仅供测试文件 `#include`，声明在 `cann_stub` 命名空间：

```cpp
namespace cann_stub {
    void Reset();                          // 重置全部状态，每个 test 的 SetUp() 调用
    bool IsInitialized();
    int32_t CurrentDevice();
    size_t LiveStreamCount();
    size_t LiveAllocCount();
    void SetNextError(aclError err);       // 一次性错误注入（下一次 API 调用返回该错误）
    std::vector<size_t> LiveAllocSizes();  // 当前所有存活分配的 size 快照
}
```

标准GTest fixture模式：

```cpp
class CannStubTest : public ::testing::Test {
protected:
    void SetUp()    override { cann_stub::Reset(); }
    void TearDown() override {
        EXPECT_EQ(cann_stub::LiveStreamCount(), 0u);
        EXPECT_EQ(cann_stub::LiveAllocCount(),  0u);
    }
};
```

---

## 验证方式

```bash
# 配置（CANN_ROOT 必须指定，否则 FATAL_ERROR）
TORCH_CMAKE_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -B build \
      -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
      -DCANN_ROOT=/home/sn/code/ascend/opensdk/cann/cann \
      -DENABLE_ASAN=ON

cmake --build build -j$(nproc)

# 运行测试（ctest 通过 set_tests_properties 自动注入 LD_LIBRARY_PATH）
cd build && ctest --output-on-failure
```

---

## 扩展 添加新stub API

使用 `/add-cann-stub-api <api_name>` command添加新API，该command会引导你完成：

1. 在真实CANN头文件中定位函数签名
2. 按行为策略表判断返回值和副作用
3. 在对应 `.cc` 文件中添加函数体
4. 在 `test_cann_stub.cc` 添加测试

详见 `.claude/commands/add-cann-stub-api.md`。
