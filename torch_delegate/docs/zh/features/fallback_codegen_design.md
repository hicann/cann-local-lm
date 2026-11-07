# Fallback Codegen特性设计文档

> [!NOTE]文档状态
> 本文为Fallback Codegen的初始设计稿，部分实现已演进。当前使用说明请参考[《Fallback Codegen用户指南》](../guide/fallback_codegen_guide.md)。

**状态**：已实现
**日期**：2026-05-15
**最后修订**：2026-07-20

---

> **实现演进说明**（2026-06-30更新）
>
> 本文档最初设计为独立CLI工具（`tools/fallback_codegen --epair ... --output-dir ...`）。
> 实际实现中，Fallback Codegen已集成到 `GeGraphBuilder::Build()` 流程中，
> 通过环境变量 `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1` 自动触发，
> 无需独立编译和调用CLI工具。
>
> 当前实现详情请参考：
> - `csrc/fallback_codegen/README.md`（实现/使用文档）
> - `docs/zh/guide/fallback_codegen_guide.md`（统一用户使用指南）
> - `docs/zh/features/overall_design_20260630.md` 第5节（Fallback Codegen特性）

---

## 背景与目标

### 问题场景

PyTorch `ExportedProgram` 中存在大量算子，torchair converter只能覆盖其中一部分。对于未覆盖的算子（包括用户自定义算子、`xllm_ascend` 等三方namespace算子），如果不做处理，整个GE图无法构建。

### 设计目标

通过Fallback Codegen，让converter未覆盖的算子能自动fallback到PyTorch Dispatcher调用，无需手动编写GE自定义算子代码：

1. **从 `ep_json_with_ascend_ir.json` 识别fallback节点**：节点 `ascend_ir.source` 为空或缺失
2. **基于PyTorch Schema自动生成GE自定义算子代码**：通过 `findSchemaOrThrow` 拿到schema，自动生成 `EagerExecuteOp` 子类
3. **在线编译并dlopen加载**：生成 `libcust_opapi.so`，`dlopen` 后注册到GE
4. **设备一致性**：通过 `at_npu::native::from_blob` 零拷贝创建NPU `at::Tensor`，数据始终在NPU设备上

### 触发位置

Fallback Codegen不是独立可执行文件，而是 `GeGraphBuilder` 构造函数中的一个步骤（`csrc/backend/ge_graph_builder.cc`，可搜索 `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` 定位）：

```cpp
const char *enable_codegen_env = std::getenv("TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN");
bool enable_codegen = true;
if (enable_codegen_env != nullptr) {
  enable_codegen = (std::string(enable_codegen_env) == "1");
}

if (enable_codegen) {
  auto codegen_result = torch_delegate::fallback::FallbackCodegen::GenerateFromModelJson(graph_json);
  // ...
}
```

环境变量 `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` 控制是否启用，未设置时默认启用。

---

## 整体架构

```text
┌─────────────────────────────────────────────────────────┐
│ Python 侧（torch_delegate.api）                          │
│                                                          │
│   ExportedProgram                                        │
│     ↓ torchair converter                                 │
│   ep_json_with_ascend_ir.json（带 ascend_ir 标注）       │
│     ↓ epair_save                                         │
│   model.epair（ZIP 归档）                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ C++ 侧（GeGraphBuilder 构造函数）                        │
│                                                          │
│   1. 从 archive 读取 ep_json_with_ascend_ir.json        │
│   2. FallbackCodegen::GenerateFromModelJson(json)        │
│      ├── 解析 JSON，识别 ascend_ir.source 为空的节点     │
│      ├── 对每个节点调用 Dispatcher findSchema 获取签名   │
│      ├── 生成 custom_op.cpp（EagerExecuteOp 子类 + REG_OP）│
│      ├── 生成 CMakeLists.txt（KEEP_ARTIFACTS=1 时落盘） │
│      ├── g++ 编译为 libcust_opapi.so                     │
│      └── dlopen 加载，schema 和 REG_OP 自动注册到 GE     │
│   3. 继续构建 GE Graph                                   │
└─────────────────────────────────────────────────────────┘
```

---

## epair文件格式

### epair文件结构

epair文件复用PT2 Archive格式（ZIP），新增两个文件：

```text
model.epair (ZIP 结构)
├── epair_manifest.json               ← epair 元数据（新增）
├── ep_json_with_ascend_ir.json       ← 带 ascend_ir 的 EP JSON（新增）
├── models/
│   └── model.json                    ← 标准 EP JSON（不含 ascend_ir）
└── data/
    ├── weights/
    │   ├── model_weights_config.json
    │   └── weight_0, weight_1, ...
    ├── constants/
    │   ├── model_constants_config.json
    │   └── tensor_0, tensor_1, ...
    └── model_sample_inputs.pt
```

文件名常量定义在 `csrc/common/epair_archive_constants.h:24-25`：
- `kAnnotatedModelJson = "extra/ep_json_with_ascend_ir.json"`
- `kAnnotatedManifestJson = "extra/epair_manifest.json"`

### 节点 `ascend_ir.source` 字段

`ep_json_with_ascend_ir.json` 中每个node对象可选包含 `ascend_ir` 字段，其中 `source` 标识转换来源：

| `source` 值 | 含义 | fallback处理 |
|-------------|------|---------------|
| `"converter"` | torchair converter成功转换为GE算子 | 不需要 |
| `""` 或字段缺失 | Mock节点 / converter未覆盖 | 需要Fallback Codegen |

`ParseTargetsFromJson`（`fallback_codegen.cc:400-463`）的过滤逻辑：
- 节点必须有 `ascend_ir` 字段且为object（placeholder、output等非计算节点没有该字段，跳过）
- `ascend_ir.source` 缺失或为空字符串 → 需要fallback
- 跳过 `torch.ops.aten._assert_tensor_metadata.default`（仅用于图结构校验，不参与计算）

### 算子target解析

FX图上target格式：`torch.ops.<namespace>.<op_name>.<overload>`

- `torch.ops.aten.topk.default` → namespace=aten, op_name=topk, overload=""
- `torch.ops.aten.add.Tensor` → namespace=aten, op_name=add, overload="Tensor"
- `torch.ops.xllm_ascend.rope_in_place.default` → namespace=xllm_ascend, op_name=rope_in_place, overload=""

`default` 会被转换为空字符串（PyTorch Dispatcher用空字符串表示默认overload）。

### Higher-Order Operator（HOP）展开

PyTorch的 `auto_functionalized_v2` 是处理inplace算子的HOP，真实算子名在 `inputs[0].arg.as_operator` 字段中。`ParseTargetFromJson`（`fallback_codegen.cc:319-398`）会检测并展开：

```json
{
  "target": "torch.ops.higher_order.auto_functionalized_v2",
  "inputs": [
    {"name": "", "arg": {"as_operator": "torch.ops.my_ops.dual_inplace_.default"}},
    {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
    ...
  ]
}
```

被替换为 `torch.ops.my_ops.dual_inplace_.default`，再走标准解析逻辑。

---

## 基于Schema的代码生成方案

### 方案概述

通过 `torch::Dispatcher::singleton().findSchema(op_name)` 获取算子schema，从中提取参数类型、alias_info、默认值等信息，自动生成GE自定义算子代码。

### SchemaInfo结构

`SchemaInfo`（`fallback_codegen.h:56-67`）记录从schema提取的关键信息：

```cpp
struct SchemaInfo {
    std::string full_op_name;   // namespace::op_name
    std::string overload;
    std::string op_name;
    std::string namespace_name;
    std::vector<SchemaArgInfo> arguments;
    std::vector<SchemaArgInfo> returns;
    bool is_inplace = false;    // 任一参数有 write alias
    bool is_out = false;        // overload == "out"
    bool has_variadic_input = false;
    int num_fixed_inputs = 0;
};
```

### 参数类型识别

`InputKind` 枚举覆盖PyTorch所有常见参数类型（`fallback_codegen.h:14-37`）：

- Tensor / Optional[Tensor] / List[Tensor]
- int / Optional[int] / List[int] / Optional[List[int]]
- float / Optional[float] / List[float] / Optional[List[float]]
- bool / Optional[bool] / List[bool] / Optional[List[bool]]
- Scalar / Optional[Scalar] / List[Scalar]
- str / Optional[str]
- ScalarType / Optional[ScalarType]

`GetSchemaInfo`（`fallback_codegen.cc:457-646`）通过 `c10::TypePtr` 的 `isSubtypeOf`、`castRaw<c10::OptionalType>`、`cast<c10::ListType>` 等方法精确识别每个参数类型，并记录：
- `has_default`：是否有默认值
- `default_value`：默认值（int/double/bool/string/int_list/double_list可直接提取，其他类型记为null）
- `alias_info`：alias标注（如 `Tensor(a!)` 表示写入别名，用于识别inplace操作）
- `is_out`：是否为out参数
- `is_variadic`：是否为可变参数（TensorList）

### GE算子标识符生成

`GenerateGeOpIdentifier`（`fallback_codegen.cc:298-308`）将 `namespace::op_name` 转为GE算子标识符：

- `aten::add` + `Tensor` → `aten__add_Tensor`
- `aten::relu_` + `` → `aten__relu_`
- `aten::add` + `out` → `aten__add_out`
- `my_ops::scale_add` + `` → `my_ops__scale_add`

格式：`namespace__op_overload`（`::` 替换为 `__`，overload用 `_` 连接）。

### REG_OP生成

`GenerateGeOpReg`（`fallback_codegen.cc:840-956`）根据schema生成 `REG_OP` 注册代码：

- **输入声明**：根据 `InputKind` 选择 `INPUT` / `OPTIONAL_INPUT` / `DYNAMIC_INPUT`
  - `TENSOR` / `OPTIONAL_TENSOR` → `TensorType::ALL()`
  - `INT` / `INT_OPT` / `INT_LIST` → `IntTensorType()`
  - `FLOAT` / `FLOAT_OPT` / `FLOAT_LIST` → `FloatTensorType()`
  - `BOOL` / `BOOL_OPT` / `BOOL_LIST` → `BoolTensorType()`
  - `STRING` / `STRING_OPT` → `StringTensorType()`
  - `SCALAR` / `SCALAR_OPT` / `SCALAR_LIST` → `TensorType::ALL()`
  - `SCALAR_TYPE` / `SCALAR_TYPE_OPT` → `IntTensorType()`
- **输出声明**：
  - 非inplace：从 `returns` 生成 `.OUTPUT()` / `.DYNAMIC_OUTPUT()`
  - inplace：跳过有write alias的return，改为从arguments中有write alias的参数生成 `.OUTPUT()`
- **属性**：`op_name`（String）、`overload`（String）、`input_kinds`（ListInt）、`num_outputs`（Int）、`is_inplace`（Bool）
- **InputsDataDependency**：所有非tensor输入（int/float/bool/scalar等）标记为数据依赖
- **REG_AUTO_MAPPING_OP**：自动映射注册，让GE走 `EagerExecuteOp::Execute` 路径

### Execute方法生成

`GenerateExecuteMethod`（`fallback_codegen.cc:988-1283`）生成 `EagerExecuteOp::Execute` 方法，核心流程：

1. **获取输入tensor**：对每个非variadic输入调用 `ctx->GetInputTensor(i)`
2. **out版本预分配输出**：从 `ctx->GetOutputTensor(i)` 拿到shape/dtype/format，调用 `ctx->MallocOutputTensor(i, ...)`
3. **构建args向量**：根据 `InputKind` 处理每个参数：
   - `TENSOR` / `OPTIONAL_TENSOR` → `CreateNpuTensor(ge_input)`（input为null时填 `c10::IValue()`）
   - `INT` / `INT_OPT` / `FLOAT` / `FLOAT_OPT` / `BOOL` / `BOOL_OPT` → `ReadScalarInput`（D2H拷贝后reinterpret_cast）
   - `INT_LIST` / `FLOAT_LIST` / `BOOL_LIST` → `ReadScalarInput` +循环构造vector
   - `SCALAR` / `SCALAR_OPT` → 根据 `GetDataType` 区分int64/float
   - `SCALAR_LIST` → 同上，循环构造vector
   - `STRING` / `STRING_OPT` → `ReadScalarInput` +构造 `std::string`
   - `SCALAR_TYPE` / `SCALAR_TYPE_OPT` → `ReadScalarInput` + `static_cast<at::ScalarType>`
   - `TENSOR_LIST`（variadic）→ 循环 `ctx->GetInputTensor(tensor_idx + i)`
   - 缺失且有默认值 → 使用schema提取的 `default_value`
   - 缺失且无默认值 → `c10::IValue()`（None）
4. **设置NPU stream**：`c10_npu::getStreamFromExternal(ge_stream, device_id)` + `NPUStreamGuard`
5. **调用算子**：`op_handle.callBoxed(args)`（非 `call`，因为 `callBoxed` 支持任意参数组合）
6. **写回输出**：
   - 非inplace：`args` 末尾的tensor作为输出，`MallocOutputTensor` + `aclrtMemcpyAsync`（D2D）
   - inplace且return为void：从saved tensor引用中提取结果，`MakeOutputRefInput` 让输出引用输入
   - inplace且有显式return：同非inplace路径

### Tensor转换（GE ↔ PyTorch）

`CreateNpuTensor`（`fallback_codegen.cc:814-822`，生成代码中的实现）使用 `at_npu::native::from_blob` 零拷贝创建NPU tensor：

```cpp
at::Tensor CreateNpuTensor(const gert::Tensor* ge_tensor) {
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < ge_tensor->GetShape().GetShape().GetDimNum(); i++) {
        sizes.push_back(ge_tensor->GetShape().GetShape().GetDim(i));
    }
    at::ScalarType torch_dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());
    auto options = at::TensorOptions().dtype(torch_dtype).device(at::kPrivateUse1);
    return at_npu::native::from_blob(const_cast<void*>(ge_tensor->GetAddr()), sizes, options);
}
```

**生命周期**：
- GE tensor内存由GE框架管理，`Execute` 方法执行期间有效
- `at::Tensor` 是栈上对象，函数返回后自动销毁，但不释放GE tensor内存
- `callBoxed` 期间 `at::Tensor` 保持有效，无需手动管理

### 标量输入读取（ReadScalarInput）

GE对custom op的Const节点数据放置在device内存（`kOnDeviceHbm`），无法直接 `GetAddr()` 读取，需要D2H拷贝：

```cpp
std::vector<uint8_t> ReadScalarInput(const gert::Tensor* tensor) {
    size_t byte_size = tensor->GetSize();
    std::vector<uint8_t> buf(byte_size);
    aclrtMemcpy(buf.data(), byte_size, tensor->GetAddr(), byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    return buf;
}
```

**原因**（代码注释 `fallback_codegen.cc:824-828`）：`custom_node_converter.cc` 硬编码了 `kOnDeviceHbm` placement，`kPlacementHost` 设置和 `HostInputs` 注册对custom op均无效。若GE未来修复此问题，可改为直接 `GetAddr()` 读取。

### Stream配合

GE Execute方法中获取GE stream，PyTorch算子使用同一stream，同步执行：

```cpp
aclrtStream ge_stream = static_cast<aclrtStream>(ctx->GetStream());
c10_npu::NPUStream npu_stream = c10_npu::getStreamFromExternal(ge_stream, device_id);
c10_npu::NPUStreamGuard guard(npu_stream);
op_handle.callBoxed(args);
```

**约束**：PyTorch算子必须在GE Execute方法中同步执行，不能异步。GE框架在Execute返回后统一同步stream。

---

## Inplace操作处理

### inplace识别

通过schema `alias_info` 识别：

```cpp
const auto &first_arg = schema.arguments().at(0);
if (first_arg.alias_info() != nullptr && first_arg.alias_info()->isWrite()) {
    info.is_inplace = true;
}
```

任一参数有 `isWrite()` alias都标记为inplace。

### inplace Execute生成

对于 `returns` 为空的inplace操作（如 `dual_inplace_(Tensor(a!) a, Tensor(b!) b, ...) -> ()`）：

1. **保存tensor引用**：`callBoxed` 会清空 `args`，提前保存 `args[i].toTensor()`
2. **调用算子**：`op_handle.callBoxed(args)`，PyTorch算子直接修改输入tensor的内存
3. **写回GE输入**：`aclrtMemcpyAsync` 从PyTorch tensor拷贝回GE input tensor（若地址不同）
4. **输出引用输入**：`ctx->MakeOutputRefInput(output_idx, input_idx)`，GE输出直接引用输入内存

### inplace REG_OP

```cpp
REG_OP(my_ops__dual_inplace_)
    .INPUT(a, TensorType::ALL())
    .INPUT(b, TensorType::ALL())
    .INPUT(scale_a, FloatTensorType())
    .INPUT(scale_b, FloatTensorType())
    .OUTPUT(a, TensorType::ALL())   // 输出名称与输入相同
    .OUTPUT(b, TensorType::ALL())
    .ATTR(op_name, String, "my_ops::dual_inplace_")
    ...
```

---

## 可选参数处理

### 缺失参数填充

当EP JSON中 `inputs` 数组长度小于schema参数数量时，缺失参数按以下顺序填充：

1. `as_none` 占位符 → 使用schema默认值或 `c10::IValue()`（None）
2. 尾部省略 → 使用schema默认值或 `c10::IValue()`（None）

### REG_OP全量声明

无论EP JSON中是否提供某个参数，`REG_OP` 都声明全量参数。GE Graph Builder为缺失的参数创建Const节点（使用默认值），保证 `Execute` 方法能从GE input读取到所有参数。

---

## 编译流程

### 在线编译（默认）

`GenerateFromTargets`（`fallback_codegen.cc:1422-1601`）的在线编译流程：

1. **创建临时目录**：`mkdtemp("/tmp/fbcg_XXXXXX")`
2. **生成 `custom_op.cpp`**：`EmitCustomOpCpp(schemas, gcc_cmd)`，文件头部包含完整g++命令注释
3. **生成 `CMakeLists.txt`**：`EmitCMakeLists(output_dir, torch_pkg_dir, torch_npu_dir)`（仅 `KEEP_ARTIFACTS=1` 时落盘）
4. **构建g++命令**：`BuildGccCommandFromPaths` 拼接 `-I`/`-L`/`-rpath` 等参数
5. **g++编译**：`CompileWithGcc(gcc_cmd)`，通过 `popen` 执行并捕获输出
6. **dlopen加载**：`dlopen(so_path, RTLD_NOW | RTLD_GLOBAL)`，让schema和REG_OP全局注册
7. **清理临时文件**：默认清理；`KEEP_ARTIFACTS=1` 时复制到 `${OUTPUT_DIR}/${ge_op_identifier}/`

### g++命令

`BuildGccCommandFromPaths`（`fallback_codegen.cc:1385-1432`）生成的命令格式：

```bash
g++ -shared -fPIC -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -g \
    -I${TORCH_PKG_DIR}/include \
    -I${TORCH_PKG_DIR}/include/torch/csrc/api/include \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/include/graph \
    -I${ASCEND_HOME_PATH}/include/register \
    -I${TORCH_NPU_DIR}/include \
    -I${TORCH_NPU_DIR}/.. \
    -L${TORCH_PKG_DIR}/lib \
    -L${TORCH_NPU_DIR}/lib \
    -L${ASCEND_HOME_PATH}/lib64 \
    -Wl,-rpath,${TORCH_PKG_DIR}/lib \
    -Wl,-rpath,${TORCH_NPU_DIR}/lib \
    -Wl,-rpath,${ASCEND_HOME_PATH}/lib64 \
    -Wl,--no-as-needed \
    -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -lge_runner -llowering -lopapi \
    -Wl,--as-needed \
    -o libcust_opapi.so custom_op.cpp
```

### 生成的CMakeLists.txt

`EmitCMakeLists`（`fallback_codegen.cc:1340-1462`）生成的CMakeLists.txt支持两种使用方式：

- **环境变量**：`TORCH_PKG_DIR`、`TORCH_NPU_DIR`、`ASCEND_HOME_PATH`
- **cmake变量**：`-DTORCH_PKG_DIR=...`、`-DTORCH_NPU_DIR=...`、`-DASCEND_HOME_PATH=...`

用于 `KEEP_ARTIFACTS=1` 时用户手动重新编译调试。

---

## 错误处理

### CodegenResult

```cpp
struct CodegenResult {
    bool success = false;
    bool skipped = false;       // 无节点需要 fallback 时为 true
    std::string error_message;
    std::string generated_cpp_path;     // KEEP_ARTIFACTS=1 时有效
    std::string generated_cmake_path;   // KEEP_ARTIFACTS=1 时有效
    void* so_handle = nullptr;          // dlopen handle
    std::vector<std::string> generated_op_names;
};
```

### 错误场景

| 场景 | `success` | `skipped` | `error_message` |
|------|-----------|-----------|-----------------|
| JSON解析失败 | `false` | `false` | `cannot parse model.json content: ...` |
| 无需fallback | `true` | `true` | `Skipped: no nodes need Fallback Codegen` |
| `TORCH_PKG_DIR` / `TORCH_NPU_DIR` 未设置 | `false` | `false` | `TORCH_PKG_DIR and TORCH_NPU_DIR must be set when Fallback Codegen is needed` |
| Schema未找到 | `false` | `false` | `schema not found for '...' with overload '...'` |
| g++编译失败 | `false` | `false` | `g++ compilation failed: ...` |
| dlopen失败 | `false` | `false` | `dlopen failed: ...` |

### GeGraphBuilder集成

`GeGraphBuilder` 构造函数中，codegen失败会直接 `return`（`init_status = FAILED`），中断图构建。codegen被跳过（`skipped=true`）时只打印日志，继续后续流程。

---

## 限制

1. **依赖PyTorch Dispatcher**：用户算子必须已注册到Dispatcher（通过 `TORCH_LIBRARY` / `torch.library`），否则 `findSchema` 失败
2. **依赖torch_npu**：生成代码使用 `at_npu::native::from_blob`、`c10_npu::NPUStream`，必须链接 `libtorch_npu.so`
3. **依赖CANN**：g++编译需要 `ASCEND_HOME_PATH` 指向CANN安装路径，运行时需要 `libascendcl.so`、`libge_runner.so`、`libopapi.so`
4. **性能开销**：通过Dispatcher `callBoxed` 调用有轻微开销；标量输入需要D2H拷贝
5. **同步执行**：PyTorch算子在GE Execute中同步执行，无法利用异步流

---

## 参考资源

- **环境变量**：[../guide/environment_variables.md](../guide/environment_variables.md) — TorchDelegate环境变量参考
- **epair格式**：[../guide/build_epair.md](../guide/build_epair.md) — epair文件结构和 `ascend_ir` 字段说明
- **构建指南**：[../guide/build.md](../guide/build.md) — TorchDelegate构建配置
- **源码**：
  - `csrc/fallback_codegen/fallback_codegen.h` — 公共接口和数据结构
  - `csrc/fallback_codegen/fallback_codegen.cc` — 实现
  - `csrc/backend/ge_graph_builder.cc` — 触发位置（`GeGraphBuilder` 构造函数）
- **测试**：`tests/cpp/test_fallback_codegen.cc` — 单元测试
- **示例**：`examples/fallback_inplace_optional_case/` — inplace +可选参数示例
