# Fallback Codegen用户使用指南

本文档是Fallback Codegen的统一用户使用指南，面向需要在GE图模式中接入自定义算子的终端用户。涵盖从算子注册到执行的完整流程、环境变量配置、工作模式选择、参数类型支持和常见问题排查。

---

## 概述

### 什么是Fallback Codegen

Fallback Codegen是TorchDelegate的动态算子代码生成模块。当Python侧 `export_to_epair` 将PyTorch ExportedProgram转换为epair文件时，部分算子可能无法成功转换为Ascend IR（GE算子）。这些转换失败的节点会被标记为 `ascend_ir.source = ""`，需要在C++侧通过Fallback Codegen处理。

Fallback Codegen的作用：

1. **解析epair JSON**：识别转换失败的算子节点（`ascend_ir.source` 为空或缺失）
2. **查询算子schema**：从PyTorch Dispatcher获取算子的函数签名
3. **生成GE自定义算子代码**：根据schema自动生成C++代码和CMakeLists
4. **编译为共享库**：生成可动态加载的 `.so` 文件

### 适用场景

- 用户通过 `torch.library` 注册自定义算子
- 算子无GE入图交付件（没有 `REG_OP` 注册）
- 需要在GE图模式中透明执行用户NPU实现

### 三种算子场景对比

TorchDelegate根据算子是否携带 `ascend_ir` 标注、是否有GE入图交付件，将算子分为三种场景：

| 场景 | 描述 | ascend_ir | GE 交付件 | 构图方式 | 支持 |
|------|------|-----------|-----------|----------|---------|
| **场景 1** | 有完整入图交付件的算子 | `source="converter"`，ops 含完整 opdef | 有 | `NamedIoNodeBuilder` 直接构图 | ✅ 支持 |
| **场景 2** | Fallback 回调算子（NPU 执行） | `source=""`，`device="npu"`（缺省） | 无，由 Fallback Codegen 自动生成 | Codegen 生成 GE `EagerExecuteOp`（`REG_AUTO_MAPPING_OP`）→ 编译 → dlopen 注册 → 运行时回调 PyTorch Dispatcher（PrivateUse1） | ✅ 支持 |
| **场景 3** | CPU 算子（HostCpu 后端） | `source=""`，`device="cpu"` | 无，由 Fallback Codegen 自动生成 | Codegen 生成 GE `HostCpuExecuteOp + ShapeInferOp`（`REG_OP_BACKEND(..., kHostCPU)`）→ 编译 → dlopen 注册 → 运行时回调 PyTorch Dispatcher（CPU） | ✅ 支持 |

Fallback Codegen 同时处理**场景 2 与场景 3**，同一 `.so` 可包含两类算子。场景 3 的
Device 归属由导出期自动判定（`ascend_ir.device`），最低 CANN 版本 9.2.0
（详见
`docs/zh/features/cpu_npu_mixed_scheduling_hostcpu_20261027.md`）。


### 运行时调用链

```text
GE 图模式
  → GE 自定义算子 Execute
  → 创建 NPU at::Tensor（torch::from_blob，零拷贝）
  → PyTorch Dispatcher 调用
  → 用户算子 NPU 实现（aclnn、AscendC 等）
  → 结果写回 GE output tensor
```

---

## 前置条件

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| PyTorch | >= 2.9.0 | 提供Dispatcher、torch.library、torch.export |
| torch_npu | 与PyTorch配套 | 提供NPU设备后端、from_blob零拷贝 |
| CANN | >= 9.2.0 | 提供GeSession、ACL runtime、NamedIoNodeBuilder |
| g++ | 支持C++17 | 在线模式编译必需 |

### 用户需要提供

**1. PyTorch自定义算子实现（`.so` 文件）**

- 注册到PyTorch Dispatcher（`PrivateUse1` key）
- 提供Meta实现（用于 `torch.compile` 图模式形状推导）
- 算子实现使用NPU API（如aclnn、AscendC等）

**2. epair文件（torch_delegate导出）**

- 包含模型的描述JSON文件
- 包含fallback回aten需要的自定义算子名描述信息

---

## 快速开始

### 完整流程概览

```text
Step 1: Python 侧注册自定义算子
  ├── 定义算子 schema（torch.library.Library）
  ├── 注册 Meta 实现（register_fake）
  └── 注册 NPU 实现（PrivateUse1 dispatch key）

Step 2: 导出 epair 文件
  ├── torch.export.export 导出 ExportedProgram
  ├── export_to_epair 转换为 EpairContents
  └── epair_save 保存为 .epair 文件

Step 3: C++ Fallback Codegen 处理（自动触发）
  ├── GeGraphBuilder 加载 epair
  ├── FallbackCodegen 扫描 source="" 节点
  ├── 生成 GE 自定义算子代码
  ├── 编译为 .so（在线模式）
  └── dlopen 注册到 GE

Step 4: 执行推理
  ├── GeSession::CompileGraph 编译
  └── GeSession::RunGraphWithStreamAsync 执行
```

### 注册自定义算子

通过 `torch.library` 注册自定义算子，包含schema定义、Meta实现和NPU实现：

```python
import torch

# 定义算子 schema
lib = torch.library.Library("my_ops", "DEF")
lib.define("add_scalar(Tensor x, float alpha) -> Tensor")

# Meta 实现（用于图模式形状推导）
@torch.library.register_fake("my_ops::add_scalar")
def _add_scalar_meta(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.empty_like(x)

# NPU 实现（注册到 PrivateUse1 dispatch key）
@torch.library.impl("my_ops::add_scalar", "PrivateUse1")
def _add_scalar_npu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # 实际 NPU 实现（aclnn / AscendC 等），此处为占位
    return x + alpha
```

### 导出epair文件

```python
import torch
from torch.export import export
from torch_delegate import export_to_epair, epair_save

class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.my_ops.add_scalar(x, 0.5)

# 导出为 ExportedProgram
ep = export(MyModel(), (torch.randn(4, 4),))

# 转换为 EpairContents
# add_scalar 节点无 ascend_ir，source=""
contents = export_to_epair(ep)

# 查看转换统计
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")

# 保存为 epair 文件
epair_save(contents, "model.epair")
```

### C++加载执行

```cpp
#include "torch_delegate/epair_model_loader.h"

// 加载 epair 文件
td::EpairModelLoader loader;
td::ModelHandle model_handle;
td::Status status = loader.LoadEpair("model.epair", &model_handle);

// GeGraphBuilder 构建时自动触发 FallbackCodegen
// （需设置 TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1）
loader.CompileAndLoad(options, stream);

// 执行推理
std::vector<gert::Tensor> inputs, outputs;
loader.RunModelWithStreamAsync(stream, inputs, outputs);
```

### 完整示例代码

完整的端到端示例请参考：

- **[examples/fallback_inplace_optional_case/](../../../examples/fallback_inplace_optional_case/)** — 多Inplace +多可选参数场景
- **examples/fallback_e2e_test/** — A/B测试，验证完整aten类型支持
- **[examples/sym_int_case/](../../../examples/sym_int_case/)** — 简化e2e，验证非Tensor输入支持

---

## 工作模式

Fallback Codegen支持两种工作模式：在线模式和离线模式。

### 在线模式（默认）

- **行为**：生成代码文件，**立即编译**生成 `.so`，**动态加载**到进程
- **落盘**：默认使用临时目录（编译后清理），设置 `KEEP_ARTIFACTS=1` 后保留到 `OUTPUT_DIR`
- **产物**：
  - 默认（`KEEP_ARTIFACTS=0`）：仅内存中的 `.so` handle（临时文件被清理）
  - 调试（`KEEP_ARTIFACTS=1`）：`custom_op.cpp`（带g++命令注释）+ `CMakeLists.txt` + `libcust_opapi.so`
- **适用场景**：即时编译执行、单机环境、快速原型开发

### 离线模式

- **行为**：生成代码文件（`.cpp` + `CMakeLists.txt`），**不编译**
- **落盘**：文件必须落盘到 `OUTPUT_DIR`
- **产物**：
  - `custom_op.cpp`（GE自定义算子实现）
  - `CMakeLists.txt`（编译脚本）
- **适用场景**：跨机器部署、版本控制、手动修改代码、集成现有构建系统

### 行为矩阵

| 模式 | `OFFLINE` | `KEEP_ARTIFACTS` | `OUTPUT_DIR` | 落盘文件 | 编译行为 | 文件保留 |
|------|-----------|------------------|--------------|---------|---------|---------|
| **离线** | `1` | -（忽略） | 设置值或默认 `./output_dir` | cpp + CMakeLists.txt | 不编译 | 总是保留 |
| **在线（默认）** | `0`（默认） | `0`（默认） | -（忽略） | 无（临时目录） | g++编译+ dlopen | 清理临时文件 |
| **在线（调试）** | `0` | `1` | 设置值或默认 `./output_dir` | cpp（带g++注释） + CMakeLists.txt + .so | g++编译+ dlopen | 保留文件 |

### 模式选择建议

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 单机开发调试 | 在线模式 | 无需手动编译，快速迭代 |
| 跨机器部署 | 离线模式 | 编译环境和运行环境分离 |
| CI/CD流水线 | 离线模式 | 可版本控制生成代码 |
| 查看生成代码 | 离线模式 | 直接查看 .cpp文件 |
| 调试编译问题 | 在线调试模式 | 查看完整g++命令 |

---

## 环境变量参考

### 控制开关

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | 是否启用fallback codegen（`1`/`true`/`TRUE`，`0`/`false`/`FALSE` 禁用） | 启用 |

> **重要**：如果所有节点都成功转换（`ascend_ir.source = "converter"`），即使启用了环境变量，也不会触发codegen，不强制校验 `TORCH_PKG_DIR` 和 `TORCH_NPU_DIR`。

### 必需环境变量（触发时）

当有节点需要fallback codegen时，以下环境变量**必须设置**：

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `TORCH_PKG_DIR` | libtorch CMake配置目录（TorchConfig.cmake所在路径） | `/path/to/torch/share/cmake/Torch` |
| `TORCH_NPU_DIR` | torch_npu库目录（包含 `lib/libtorch_npu.so`） | `/path/to/libtorch_npu` |
| `ASCEND_HOME_PATH` | CANN安装路径（在线编译必需） | `/path/to/cann` |

### 工作模式控制

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `TORCH_DELEGATE_OFFLINE` | 工作模式（`1`=离线，`0`=在线） | `0`（在线） |
| `TORCH_DELEGATE_OUTPUT_DIR` | 文件落盘位置 | `./output_dir` |
| `TORCH_DELEGATE_KEEP_ARTIFACTS` | 在线模式下是否保留编译产物（`1`/`true`/`TRUE`） | `0`（不保留） |

### 环境变量查找方法

**查找TorchConfig.cmake**：

```bash
# Python whl 包
TORCH_CMAKE_DIR=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))")
export TORCH_PKG_DIR="${TORCH_CMAKE_DIR}"
```

**查找CANN路径**：

```bash
# 通常已通过 source 命令设置
source ~/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH  # 输出：/path/to/cann
```

### 优先级规则

环境变量读取遵循以下优先级：

**1. `OUTPUT_DIR`**：
```text
函数参数 > TORCH_DELEGATE_OUTPUT_DIR > 默认值 ./output_dir
```

**2. `OFFLINE`**：
```text
TORCH_DELEGATE_OFFLINE > 默认值 0（在线）
```

**3. `KEEP_ARTIFACTS`**（仅在线模式）：
```text
TORCH_DELEGATE_KEEP_ARTIFACTS > 默认值 0（不保留）
```

---

## 参数类型支持

### 支持的参数类型

Fallback Codegen支持21种 `InputKind` 参数类型，覆盖PyTorch Schema的全部常用参数类型：

| 类别 | InputKind | PyTorch类型 | GE input type |
|------|-----------|-------------|---------------|
| **Tensor类** | TENSOR | `Tensor` | `TensorType::ALL()` |
| | OPTIONAL_TENSOR | `Tensor?` | `.OPTIONAL_INPUT` |
| | TENSOR_LIST | `Tensor[]` | `.DYNAMIC_INPUT` |
| **标量类** | INT | `int` | `IntTensorType()` |
| | INT_OPT | `int?` | `IntTensorType()` |
| | FLOAT | `float` | `FloatTensorType()` |
| | FLOAT_OPT | `float?` | `FloatTensorType()` |
| | BOOL | `bool` | `BoolTensorType()` |
| | BOOL_OPT | `bool?` | `BoolTensorType()` |
| | SCALAR | `Scalar` | `TensorType::ALL()` |
| | SCALAR_OPT | `Scalar?` | `TensorType::ALL()` |
| **列表类** | INT_LIST | `int[]` | `IntTensorType()` |
| | INT_LIST_OPT | `int[]?` | `IntTensorType()` |
| | FLOAT_LIST | `float[]` | `FloatTensorType()` |
| | FLOAT_LIST_OPT | `float[]?` | `FloatTensorType()` |
| | BOOL_LIST | `bool[]` | `BoolTensorType()` |
| | BOOL_LIST_OPT | `bool[]?` | `BoolTensorType()` |
| | SCALAR_LIST | `Scalar[]` | `TensorType::ALL()` |
| **特殊类** | STRING | `str` | `StringTensorType()` |
| | STRING_OPT | `str?` | `StringTensorType()` |
| | SCALAR_TYPE | `ScalarType` | `IntTensorType()` |
| | SCALAR_TYPE_OPT | `ScalarType?` | `IntTensorType()` |

### value-dependency机制

非Tensor参数（标量类、列表类、特殊类）通过 **value-dependency** 机制在GE图中传递：

1. **注册阶段**：非Tensor参数通过 `InputsDataDependency` 注册为value-dependency input
2. **图执行前**：GE框架等待这些1元素tensor从device回读到host
3. **Execute时**：从host内存中读取标量值，组装为 `c10::IValue`

### 算子重载处理

Fallback Codegen支持三种算子重载版本：

| 重载类型 | target示例 | Execute输出处理 |
|---------|------------|-----------------|
| 标准版本 | `my_ops::add.default` | `MallocOutputTensor` 分配输出 |
| out版本 | `my_ops::add.out` | `GetInputTensor(out_index)` 获取输出 |
| 原地操作 | `my_ops::add_` | `MakeOutputRefInput` 输出引用输入 |

### 可选输入处理

可选参数通过 `as_none` 占位符机制处理：

```json
{
  "inputs": [
    {"as_tensor": "x"},
    {"as_none": ""},
    {"as_tensor": "y"}
  ]
}
```

未提供的可选参数使用schema中的默认值。

> 详细的参数类型映射、Execute代码生成、BuildStack/WriteBack实现请参考 [基于ATen IR回调的GE自定义算子自动生成方案设计](../features/aten_ir_callback_codegen_20260715.md)。

---

## 使用场景示例

### 模型全部转换成功（不触发codegen）

```bash
# 不设置任何环境变量
./build.sh

# 或设置了启用开关，但模型全部转换成功
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
./build.sh  # 不触发 codegen，不校验 TORCH_PKG_DIR/TORCH_NPU_DIR
```

### 离线模式（跨机器部署）

**适用场景**：编译环境和运行环境分离，需要在目标机器上手动编译。

```bash
# 1. 查找并设置必需环境变量
TORCH_CMAKE_DIR=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))")
export TORCH_PKG_DIR="${TORCH_CMAKE_DIR}"
export TORCH_NPU_DIR=/path/to/libtorch_npu

# 2. 设置 CANN 路径（离线模式可选，但建议设置用于 CMakeLists.txt）
source ~/Ascend/cann/set_env.sh

# 3. 配置输出目录（可选）
export TORCH_DELEGATE_OUTPUT_DIR=/path/to/output

# 4. 启用 fallback codegen（离线模式）
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=1  # 默认值，可省略

# 5. 运行程序，生成代码文件
./build.sh

# 产物：/path/to/output/custom_op.cpp + CMakeLists.txt
```

**后续在目标机器编译**：

```bash
cd /path/to/output
cmake -B build \
  -DTorch_DIR=$TORCH_PKG_DIR \
  -DTORCH_NPU_DIR=$TORCH_NPU_DIR \
  -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
cmake --build build -j4

# 产物：build/libcustom_op.so
```

### 在线模式（即时编译执行）

**适用场景**：单机环境，编译环境和运行环境相同。

```bash
# 1. 设置必需环境变量（在线模式必须设置 ASCEND_HOME_PATH）
TORCH_CMAKE_DIR=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0], 'share', 'cmake', 'Torch'))")
export TORCH_PKG_DIR="${TORCH_CMAKE_DIR}"
export TORCH_NPU_DIR=/path/to/libtorch_npu
source ~/Ascend/cann/set_env.sh  # 设置 ASCEND_HOME_PATH

# 2. 启用 fallback codegen（在线模式）
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=0

# 3. 运行程序，自动编译并加载
./build.sh

# 产物：内存中的 .so handle（临时文件被清理）
```

### 在线模式调试（保留编译产物）

**适用场景**：调试在线编译问题，需要查看生成的代码和g++命令。

```bash
# 1-2. 同场景 3 设置必需环境变量
...

# 3. 启用在线模式 + 保留产物
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=0
export TORCH_DELEGATE_KEEP_ARTIFACTS=1
export TORCH_DELEGATE_OUTPUT_DIR=/tmp/fallback_debug

# 4. 运行程序
./build.sh

# 产物（保留在 /tmp/fallback_debug）：
#   - custom_op.cpp（头部包含完整 g++ 命令）
#   - CMakeLists.txt
#   - libcust_opapi.so

# 5. 查看生成的 g++ 命令（用于调试）
head -10 /tmp/fallback_debug/custom_op.cpp

# 6. 手动重新编译（如果自动编译失败）
cd /tmp/fallback_debug
# 复制 custom_op.cpp 中注释的 g++ 命令，手动执行
g++ -shared -fPIC ... -o libcust_opapi.so custom_op.cpp
```

### 快速验证生成的代码

**适用场景**：想查看生成的代码，但不实际编译。

```bash
# 使用离线模式生成文件
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OFFLINE=1
export TORCH_DELEGATE_OUTPUT_DIR=/tmp/inspect_codegen

./build.sh

# 查看生成的代码
cat /tmp/inspect_codegen/custom_op.cpp
cat /tmp/inspect_codegen/CMakeLists.txt
```

---

## 代码生成流程

### 解析阶段

```cpp
// FallbackCodegen::GenerateFromModelJson
std::vector<OpTarget> targets = ParseModelJsonFromString(model_json);

// targets 仅包含 ascend_ir.source 为空的节点
if (targets.empty()) {
    // 没有需要 fallback 的节点，跳过 codegen
    return {success=true, skipped=true};
}
```

过滤逻辑：

```cpp
bool needs_fallback = false;
if (node.contains("ascend_ir") && node["ascend_ir"].is_object()) {
    const auto& ascend_ir = node["ascend_ir"];
    // 只对 source 缺失或为空的节点进行 fallback
    if (!ascend_ir.contains("source") ||
        (ascend_ir.contains("source") && ascend_ir["source"].get<std::string>() == "")) {
        needs_fallback = true;
    }
}
```

### Schema查询

```cpp
// 从 PyTorch Dispatcher 查询算子签名
c10::OperatorName op_name(full_op_name, overload);
auto handle = torch::Dispatcher::singleton().findSchema(op_name);
const auto& schema = handle->schema();
```

### 代码生成

生成文件：

- `custom_op.cpp`：GE自定义算子实现
- `CMakeLists.txt`：编译脚本

**生成的代码结构**：

- 每个算子一个 `EagerExecuteOp` 子类
- 辅助函数：`MapGeDtypeToTorch`、`ReadScalarInput`、`CreateNpuTensor`
- 注册宏：`REG_OP` + `REG_AUTO_MAPPING_OP`

### 编译阶段

```bash
cd output_dir
cmake -B build \
  -DTorch_DIR=$TORCH_PKG_DIR \
  -DTORCH_NPU_DIR=$TORCH_NPU_DIR \
  -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
cmake --build build -j4
```

生成产物：
- `output_dir/build/libcustom_op.so`

---

## 错误处理

### 错误类型

| 错误场景 | `success` | `skipped` | error_message |
|---------|-----------|-----------|---------------|
| JSON解析失败 | `false` | `false` | `"cannot parse model.json content: <parse_error>"` |
| 无需fallback（所有节点成功） | `true` | `true` | `"Skipped: no nodes need fallback codegen"` |
| 有需要fallback，但环境变量缺失 | `false` | `false` | `"TORCH_PKG_DIR and TORCH_NPU_DIR must be set when fallback codegen is needed"` |
| Schema未找到 | `false` | `false` | `"schema not found for '<op>' with overload '<overload>'"` |
| 编译失败 | `false` | `false` | `"compilation failed: <error>"` |

### GeGraphBuilder处理逻辑

在 `ge_graph_builder.cc` 中：

```cpp
auto codegen_result = FallbackCodegen::GenerateFromModelJson(graph_json);

if (!codegen_result.success) {
    // 添加 Error 级别 ValidationIssue，中断流程
    result.issues.push_back({
        severity: Error,
        code: "FALLBACK_CODEGEN_FAILED",
        message: codegen_result.error_message
    });
    result.ok = false;
    return;
} else if (codegen_result.skipped) {
    // 添加 Warning 级别 ValidationIssue，继续流程
    result.issues.push_back({
        severity: Warning,
        code: "FALLBACK_CODEGEN_SKIPPED",
        message: codegen_result.error_message
    });
}
```

### 常见错误与解决方案

| 错误消息 | 原因 | 解决方案 |
|---------|------|---------|
| `ASCEND_HOME_PATH not set` | 在线模式未设置CANN路径 | `source ~/Ascend/cann/set_env.sh` |
| `acl/acl.h: No such file` | g++找不到CANN头文件 | 检查 `ASCEND_HOME_PATH` 是否正确 |
| `cannot find -lascendcl` | g++找不到CANN库 | 检查 `ASCEND_HOME_PATH/lib64` 是否存在 |
| `libcust_opapi.so: undefined symbol` | ABI不兼容 | 使用CMake方法编译，或检查库版本匹配 |
| `basic_string::_M_construct null not valid` | 内部对象损坏（跨SO边界问题） | 在真实CANN环境运行，避免使用stub |
| `TORCH_PKG_DIR and TORCH_NPU_DIR must be set` | 有fallback节点但未设置环境变量 | 按第5节设置必需环境变量 |
| `schema not found for '<op>'` | 算子未注册到PyTorch Dispatcher | 检查 `torch.library.Library` 注册是否正确加载 |

---

## 调试指南

### 在线模式调试

如果在线编译失败，按以下步骤排查：

**1. 查看生成的g++命令**（设置 `KEEP_ARTIFACTS=1`）：

```bash
head -10 $OUTPUT_DIR/custom_op.cpp
```

**2. 手动执行g++命令**：

```bash
# 复制 custom_op.cpp 注释中的 g++ 命令
g++ -shared -fPIC ... -o test.so custom_op.cpp
```

**3. 使用CMake方法**（备选方案）：

```bash
cd $OUTPUT_DIR
cmake -B build -DTorch_DIR=$TORCH_PKG_DIR ...
cmake --build build -j4
```

**4. 检查环境变量**：

```bash
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
ls $ASCEND_HOME_PATH/include/acl/acl.h  # 应存在
ls $ASCEND_HOME_PATH/lib64/libascendcl.so  # 应存在
```

### 查看生成的代码

**离线模式**：

```bash
cat $OUTPUT_DIR/custom_op.cpp
cat $OUTPUT_DIR/CMakeLists.txt
```

**在线调试模式**（带g++命令注释）：

```bash
# 生成的 cpp 文件头部包含完整 g++ 命令
head -20 $OUTPUT_DIR/custom_op.cpp
```

### 生成文件说明

**离线模式生成的 `custom_op.cpp`**（不包含g++命令）：

```cpp
// AUTO-GENERATED by FallbackCodegen
#include <torch/torch.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <acl/acl.h>
...
```

**在线模式（调试）生成的 `custom_op.cpp`**（包含完整g++命令注释）：

```cpp
// AUTO-GENERATED by FallbackCodegen
//
// Online compilation command (g++):
//   g++ -shared -fPIC -std=c++17 -O2 \
//       -I/path/to/torch/include \
//       -I/path/to/cann/include \
//       -I/path/to/torch_npu/include \
//       -L/path/to/torch/lib \
//       -L/path/to/cann/lib64 \
//       -Wl,-rpath,... \
//       -Wl,--no-as-needed \
//       -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -lge_runner -lopapi \
//       -Wl,--as-needed \
//       -o libcust_opapi.so custom_op.cpp
//
// If g++ compilation fails, try CMake method:
//   cmake -B build && cmake --build build -j4
//
#include <torch/torch.h>
...
```

用户可以直接复制g++命令手动编译调试，或使用CMake方法作为备选方案。

---

## 性能考虑

### 编译开销

| 模式 | 首次加载 | 后续加载 |
|------|---------|---------|
| 离线模式 | 无编译开销（仅生成代码） | 需手动编译 |
| 在线模式 | 编译开销（约数秒） | 可缓存（`KEEP_ARTIFACTS=1`） |

### 编译缓存

在线模式下，如果 `OUTPUT_DIR` 已存在 `.so` 文件，不会重新编译（直接加载）。建议在生产环境设置 `KEEP_ARTIFACTS=1` 并指定固定的 `OUTPUT_DIR` 以利用缓存。

### 运行时开销

- **Dispatcher调用**：每次Execute通过 `op_handle.callBoxed` 调用，预计耗时 <1us
- **BuildStack**：遍历input_kinds构建调用栈，预计耗时 <100ns
- **WriteBack**：涉及D2D拷贝，耗时取决于输出tensor大小

---

## 注意事项

### 多人协作场景

- **默认行为**：不启用fallback codegen，不影响其他开发者
- **启用前准备**：确保已安装torch和torch_npu，并正确设置环境变量
- **CI/CD配置**：如需在CI中启用，建议在构建脚本中明确设置环境变量

### 安全性

- 生成的代码基于PyTorch schema，与ATen kernel ABI兼容
- 仅生成算子包装代码，不修改算子实现逻辑

### 在线模式依赖要求

**必需**：
- `ASCEND_HOME_PATH`：CANN安装路径（包含 `include/acl/acl.h`、`lib64/libascendcl.so`）
- `g++`：支持C++17的编译器

**建议**：
- 使用真实的CANN环境（`source ~/Ascend/cann/set_env.sh`）
- 避免在测试环境使用CANN stub（可能存在ABI兼容性问题）

---

## 示例项目

### fallback_inplace_optional_case

测试多Inplace参数+多可选参数场景。

- **测试特性**：
  - 多Inplace参数：`dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()`
  - 多可选参数：`multi_optional_op(Tensor x, float alpha=1.0, float beta=0.0, ...) -> Tensor`
- **目录**：`examples/fallback_inplace_optional_case/`
- **运行**：`cd examples/fallback_inplace_optional_case && bash run.sh`

### fallback_e2e_test

端到端A/B测试，验证Fallback Codegen完整aten类型支持。

- **Model A**：Tensor-only（回归基线）
- **Model B**：Full Type Spectrum（int/float/bool/int[]/Scalar/Tensor[]）
- **目录**：`examples/fallback_e2e_test/`
- **运行**：`cd examples/fallback_e2e_test && bash run.sh`

### sym_int_case

简化端到端测试，验证fallback机制对非tensor输入的支持。

- **测试模型**：SimpleMixedModel（动态/静态tensor +静态/动态非tensor + Converter + Custom算子）
- **目录**：`examples/sym_int_case/`
- **运行**：`cd examples/sym_int_case && bash run.sh`

---

## 相关文档

### 设计文档
- [Fallback特性设计文档](../features/fallback_codegen_design.md) — 初始设计（含实现演进说明）
- [整体设计文档](../features/overall_design_20260630.md) 第5节 — Fallback Codegen特性
- [基于ATen IR回调的GE自定义算子自动生成方案设计](../features/aten_ir_callback_codegen_20260715.md) — InputKind编码、BuildStack/WriteBack实现、用户接入规范

### 实现文档
- csrc/fallback_codegen/README.md — 模块实现文档

### 构建文档
- [构建指南](build.md) — TorchDelegate构建指南（含 `TORCH_NPU_DIR` 配置）
- [Epair打包指南](build_epair.md) — Python API使用指南
