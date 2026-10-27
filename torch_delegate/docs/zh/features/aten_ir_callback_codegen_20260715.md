# 基于ATen IR回调的GE自定义算子自动生成方案

> [!NOTE]文档状态
> 本文为2026-07-15的设计稿，描述基于ATen IR回调的自定义算子生成方案。当前使用说明请参考[《Fallback Codegen用户指南》](../guide/fallback_codegen_guide.md)。

## 目的

本文档描述了基于ATen IR回调的GE自定义算子自动生成方案的设计需求和实现细节。目标读者为GE图引擎开发人员、torchair适配层开发人员以及自定义算子接入用户。

本方案旨在解决用户通过PyTorch自定义算子机制接入GE时需要重复编写交付件的问题，通过代码自动生成和Dispatcher回调机制，实现计算逻辑的自动复用。

## 范围

本文档涵盖以下内容：
- 方案背景和目标
- 三阶段设计流程（原型映射、Converter构图、Execute执行）
- 输入输出映射规则、值依赖注册、Schema元信息属性
- 用户接入规范和约束
- 框架实施优先级

本文档不包括以下内容：
- CPU自定义算子的支持方案（本方案仅面向NPU device场景）
- libtorch Dispatcher内部实现细节
- ATen IR到GE IR的完整转换规则（仅覆盖通过回调复用的部分）

---

## 总体概述

### 软件概述

#### 项目介绍

本项目是GE图引擎的一个新特性，用于简化用户自定义算子接入GE的流程。以往用户需要：
1. 实现converter（完成ATen IR → GE IR的转换）
2. 实现入GE图的交付件（通过 `custom_op.h` 中的接口实现 `Execute` 函数）

本方案通过自动代码生成和Dispatcher回调，消除交付件的重复编写工作。

#### 产品环境介绍

本产品是GE图引擎与torchair适配层协同工作的组件，在以下环境中运行：

**系统架构：**
```text
PyTorch (TORCH_LIBRARY 注册) → libtorch Dispatcher → ATen IR
                                           ↓
torchair (FX Graph追踪) → GE Converter → GE ComputeGraph
                                           ↓
GE Runtime (Execute) → Dispatcher回调 → PrivateUse1 实现 → NPU算子执行
```

**外部接口：**
1. PyTorch TORCH_LIBRARY/TORCH_LIBRARY_IMPL：用户通过此接口注册自定义算子
2. torchair converter：遍历FX Graph构建GE节点
3. libtorch Dispatcher：GE Execute通过此接口回调ATen IR

**相关产品：**
- libtorch：PyTorch C++前端，提供Dispatcher机制
- torchair：PyTorch到GE的适配层，负责图追踪和转换
- GE Runtime：图执行引擎，本方案主要涉及runtime/v2路径

---

### 竞品分析

#### 对比维度

主流深度学习框架自定义算子机制对比：

| 维度 | TensorRT | PyTorch | JAX | Inductor | TensorFlow | **GE本方案** |
|------|----------|---------|-----|----------|------------|---------------|
| **核心API** | IPluginV3 | TORCH_LIBRARY | custom_ops | custom_op | REGISTER_OP | **REG_OP + Dispatcher** |
| **注册机制** | Plugin Registry | Dispatcher | JIT装接 | Inductor后端 | Kernel Registry | **Operator Factory** |
| **Schema定义** | C++接口描述 | Python def字符串 | Python函数签名 | Python函数签名 | C++ OP_DEF | **ATen schema解析** |
| **Shape推导** | getOutputDimensions | Meta key | 抽象评估 | Symbolic tracing | ShapeFn | **Meta key（复用PyTorch）** |
| **执行模式** | Device Sink | Host Dispatch | XLA JIT | AOT编译 | TFRT/CPU | **Dispatcher回调** |
| **序列化** | serialize/deserialize | 无 | 无 | 无 | Bundle打包 | **OM文件（已有）** |
| **版本兼容** | Plugin Version | 无约束 | 无约束 | 无约束 | OpVersion | **待增强** |
| **主要语言** | C++ | Python/C++ | Python | Python | C++ | **C++ + Python（torchair）** |
| **适用场景** | 推理优化 | 训练+推理 | 科学计算 | PyTorch 2.x | 生产部署 | **在线推理（torchair）** |

#### 借鉴点分析

本方案的设计借鉴了以下竞品的优秀设计：

| 设计决策 | 参考竞品 | 借鉴原因 |
|----------|----------|----------|
| **Meta Key强制要求** | PyTorch | PyTorch强制要求Meta实现，保证编译期shape推导正确性。GE方案沿用此设计，确保静态shape场景不发生写越界。 |
| **Dispatcher多后端路由** | PyTorch | PyTorch Dispatcher可路由到不同后端（CPU/CUDA/PrivateUse1）。GE借鉴PrivateUse1 key设计，实现NPU后端适配。 |
| **Schema字符串定义** | PyTorch、JAX | Schema-driven设计简洁易维护。GE CodeGen解析ATen schema自动生成REG_OP，消除手动维护负担。 |
| **序列化机制** | TensorRT | TensorRT的serialize/deserialize支持引擎持久化。GE已有OM文件序列化机制，应进一步增强算子版本兼容。 |
| **OP_DEF + Kernel分离** | TensorFlow | TensorFlow的schema定义与Kernel实现分离。GE REG_OP + IMPL_OP设计类似，便于算子原型与实现独立演化。 |

#### 差异点分析

本方案与竞品的关键差异：

| 维度 | GE方案 | 竞品方案 | 差异原因 |
|------|----------|----------|----------|
| **适用场景** | 仅在线场景（torchair） | TensorRT：离线推理；PyTorch：训练+推理 | GE动态shape场景依赖Dispatcher回调，离线编译（atc）不支持运行时注册 |
| **执行模式** | Dispatcher回调（Host） | TensorRT：Task Sink（Device） | Dispatcher回调灵活支持动态shape，但引入Host调度开销；静态shape场景应保留Sink模式 |
| **序列化支持** | OM文件序列化算子原型 | TensorRT：完整序列化（含版本）；PyTorch：无序列化 | GE OM文件需增强算子版本兼容机制，参考TensorRT Plugin Version |
| **算子打包** | so_in_om（部分支持） | TensorFlow：完整Bundle打包 | GE应增强算子Bundle打包能力，支持算子依赖随模型分发 |
| **Meta验证** | 手动验证 | PyTorch：torch.compile自动验证 | GE应提供Meta实现验证工具，降低用户负担 |

#### 设计权衡说明

基于竞品分析，本方案的设计权衡：

1. **性能vs易用性**：选择易用性优先（Schema-driven + Dispatcher），牺牲部分性能（Host调度开销），适用于在线场景
2. **灵活性vs约束**：选择约束优先（强制Meta key），保证静态shape场景正确性
3. **编译期vs运行时**：选择运行时注册（依赖PyTorch Dispatcher），适用于在线场景；离线场景待后续支持

---

### 软件功能

本方案提供以下核心功能：

1. **原型自动生成**：CodeGen根据ATen schema自动生成GE算子原型（REG_OP）并注册到operator factory
2. **Converter自动构图**：torchair converter遍历FX Graph，自动构建GE节点并将schema信息编码为节点属性
3. **Execute自动执行**：GE内置通用Execute实现，通过读取节点属性动态构建Dispatcher调用

功能关系图：

```text
ATen Schema → [CodeGen] → REG_OP + 值依赖注册 + 元信息属性
                                    ↓
FX Graph → [Converter] → GE Graph（Const节点、合并算子）
                                    ↓
GE Execute → [BuildStack] → Dispatcher调用 → [WriteBack] → GE输出Tensor
```

### 设计约束

#### 用户必须提供的约束

1. **Schema定义**：通过 `TORCH_LIBRARY` 注册完整schema，数据类型只使用GE与PyTorch共同支持的类型
2. **Meta key实现**：必须注册Meta key，供torchair编译阶段符号形状推导
3. **PrivateUse1 key实现**：必须注册PrivateUse1 key的计算实现，供GE Execute通过Dispatcher回调
4. **so加载顺序**：`libcust_opapi.so` 必须在进入GE执行前加载

#### 框架自动完成的约束

1. **Schema驱动**：所有代码生成基于schema分析，而非硬编码
2. **零拷贝优先**：NPU场景下优先使用零拷贝方案
3. **统一承载模型**：所有非Tensor参数统一转为Tensor输入边，消除路径分裂

#### 不支持的场景

- 含 `Device` / `Layout` / `Generator` / `MemoryFormat` 参数的schema
- 含 `complex` 类型Scalar默认值（P0不支持）
- 含 `Tensor[]` 但无法静态确定输出数量上界
- schema解析失败的情况

### 假设和依赖关系

#### 本迭代的依赖

1. **libtorch Dispatcher**：依赖Dispatcher机制查找和调用ATen IR
2. **torchair converter**：依赖torchair提供FX Graph追踪和节点构建能力
3. **GE operator factory**：依赖GE提供算子注册机制（REG_OP/IMPL_OP）
4. **GE runtime/v2**：依赖GE runtime v2提供Execute执行环境

#### 假设

1. 用户已正确注册TORCH_LIBRARY和TORCH_LIBRARY_IMPL
2. Meta key实现的形状推导正确
3. PrivateUse1 key实现的计算逻辑正确
4. `libcust_opapi.so` 在运行时已加载

---

## 特性需求分析&设计

### 整体介绍

本方案通过三阶段设计实现自定义算子的自动接入：

| 阶段 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **步骤一：原型映射** | CodeGen根据schema生成REG_OP | ATen schema | REG_OP +值依赖注册+元信息属性 |
| **步骤二：Converter构图** | 遍历FX Graph构建GE节点 | FX Graph + schema | GE Graph（含Const节点、合并算子） |
| **步骤三：Execute执行** | 从gert::Tensor转torch API调用 | GE输入+元信息属性 | Dispatcher调用+ WriteBack |

流程图：
```text
[构建/加载阶段] CodeGen 扫描 TORCH_LIBRARY 注册的 ATen schema
                自动生成 REG_OP 代码并注册到 GE operator factory
                注册值依赖（InputsDataDependency）
                保存 schema 元信息为节点属性
    ↓
[图追踪阶段] torchair 遍历 FX Graph，对每个 call_function 节点调用 converter
    ↓
converter 处理参数：
  - Tensor 参数：直接连上游输出边
  - 非 Tensor 常量：创建 Const 节点（HOST 内存）
  - 非 Tensor 动态值：连上游算子输出边
  - 列表参数：生成合并算子
  - Optional None：可选输入不连边
    ↓
converter 在 GE 图中创建节点，写入元信息属性
    ↓
[运行时] GE Execute 被调用：
  1. 根据 input_kinds 判断每个参数类型
  2. 从输入边取值（Tensor 包装 / 标量提取）
  3. 通过 Dispatcher 调用 ATen IR（PrivateUse1 key）
  4. 将结果写回 GE 输出 Tensor
```

---

### 功能需求

#### 功能需求1原型映射规则（REG_OP生成）

**介绍**

CodeGen根据ATen schema自动生成GE算子原型，包括输入输出映射、值依赖注册、Schema元信息属性等。这是用户无需手动编写交付件的核心机制。

**输入**

- **来源**：用户通过TORCH_LIBRARY注册的ATen schema
- **数量**：根据用户注册的算子数量决定
- **度量单位**：schema条目
- **有效输入范围**：GE与PyTorch共同支持的ATen schema（排除Device/Layout/Generator/MemoryFormat参数）
- **时间要求**：构建阶段一次性生成

**处理**

**算子标识符与命名：**
- schema名 `my_ops::my_op` → REG_OP标识符 `my_ops__my_op`（`::` → `__`）
- schema名 `my_ops::add.Tensor` → REG_OP标识符 `my_ops__add_Tensor`（`.` → `_`）
- GE图中节点类型字符串仍保留schema名以保证dump可读性

**输入参数映射规则：**

核心设计：所有非Tensor参数 → Tensor输入边

| Schema参数 | REG_OP声明 | 值依赖 | dtype约束 |
|------------|------------|--------|-----------|
| `Tensor name` | `.INPUT(name, TensorType::ALL())` | 否 | 无约束 |
| `Tensor? name=None` | `.OPTIONAL_INPUT(name, TensorType::ALL())` | 否 | 无约束 |
| `Tensor[] name` | `.DYNAMIC_INPUT(name, TensorType::ALL())` | 否 | 无约束 |
| `int name`（无默认值）| `.REQUIRED_INPUT(name, IntTensorType())` | **是** | int64 |
| `int name=v`（有默认值）| `.OPTIONAL_INPUT(name, IntTensorType())` | **是** | int64 |
| `float name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, FloatTensorType())` | **是** | float32 |
| `bool name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, BoolTensorType())` | **是** | bool |
| `Scalar name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, TensorType::ALL())` | **是** | 多态（由tensor dtype表达） |
| `int[] name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, IntTensorType())` | **是** | 1-dim int tensor |
| `str name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, StringTensorType())` | **是** | string |
| `ScalarType name` | `.REQUIRED_INPUT/OPTIONAL_INPUT(name, IntTensorType())` | **是** | int64（枚举值） |

**必选/可选判断规则：**
- 无默认值的非Optional参数 → `.REQUIRED_INPUT`
- 有默认值或Optional参数 → `.OPTIONAL_INPUT`

**值依赖注册：**

非Tensor参数（除 `Tensor`、`Tensor?`、`Tensor[]` 外）都需要注册为值依赖：

```cpp
// REG_OP 示例
REG_OP(my_ops__my_op)
    .INPUT(x, TensorType::ALL())
    .REQUIRED_INPUT(dim, IntTensorType())           // int → 值依赖
    .OPTIONAL_INPUT(alpha, TensorType::ALL())       // Scalar → 值依赖
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(my_ops__my_op)

// IMPL_OP 中注册值依赖
IMPL_OP(my_ops__my_op)
    .InputsDataDependency({"dim", "alpha"})         // 非 Tensor 参数名列表
```

**Schema元信息属性：**

保留schema元信息为节点属性，供Execute使用：

```cpp
REG_OP(my_ops__my_op)
    // ... 输入输出声明 ...

    // Schema 元信息属性
    .ATTR(op_name, String, "my_ops::my_op")         // Dispatcher 查找
    .ATTR(input_kinds, ListInt, {...})              // 每个参数的类型编码
    .ATTR(num_outputs, Int, 1)                      // 输出数量
    .ATTR(is_inplace, Bool, false)                  // inplace 标记
    .OP_END_FACTORY_REG(my_ops__my_op)
```

| 属性 | 类型 | 用途 | 来源 |
|------|------|------|------|
| `op_name` | String | Dispatcher查找OperatorHandle | schema名 |
| `input_kinds` | ListInt | 每个参数的InputKind编码 | schema参数类型 |
| `num_outputs` | Int | WriteBack循环边界 | schema输出数 |
| `is_inplace` | Bool | WriteBack模式选择 | schema inplace标注 |

**InputKind编码定义：**

```cpp
enum InputKind {
    // Tensor 类（不走值依赖）
    TENSOR = 0,
    OPTIONAL_TENSOR = 1,
    TENSOR_LIST = 2,

    // 标量类（走值依赖）
    INT = 10,
    INT_OPT = 11,
    FLOAT = 12,
    FLOAT_OPT = 13,
    BOOL = 14,
    BOOL_OPT = 15,
    SCALAR = 16,
    SCALAR_OPT = 17,

    // 列表类（走值依赖）
    INT_LIST = 20,
    INT_LIST_OPT = 21,
    FLOAT_LIST = 22,
    BOOL_LIST = 23,

    // 特殊类（走值依赖）
    STRING = 30,
    STRING_OPT = 31,
    SCALAR_TYPE = 32,
    SCALAR_TYPE_OPT = 33,
};
```

**异常情况处理：**
- schema解析失败：跳过REG_OP生成，converter在运行时触发fallback
- 不支持的参数类型：跳过生成

**输出**

- **输出到何处**：GE operator factory（通过REG_OP注册）
- **数量**：一个算子对应一个REG_OP
- **度量单位**：算子原型
- **时序**：构建阶段一次性生成
- **有效输出范围**：符合GE算子注册规范的原型

---

#### 功能需求2 Converter构图转换

**介绍**

torchair converter遍历FX Graph，对每个自定义算子节点构建对应的GE节点，处理参数映射并写入元信息属性。

**输入**

- **来源**：torchair FX Graph（call_function节点）
- **数量**：根据FX Graph中自定义算子的数量决定
- **度量单位**：FX Node
- **有效输入范围**：已注册schema的自定义算子节点
- **时间要求**：图追踪阶段实时构建

**处理**

**参数处理规则：**

| FX args形态 | Converter行为 | GE图结构 |
|-------------|---------------|----------|
| Tensor（FX Node）| 直接连上游输出边 | 输入边连接上游节点 |
| 常量字面量 | 创建Const节点 | Const节点（HOST内存）连边 |
| FX Node（0-dim tensor）| 直接连上游输出边 | 输入边连接上游节点 |
| FX Node（SymInt输出，如 `aten.size`）| 连上游输出边 | 输入边连接 `aten.size` 等节点 |
| 列表参数（`int[]`）| 生成合并算子 | 合并算子输出连边 |
| None（Optional）| 不连边 | OPTIONAL_INPUT缺边 |
| 缺省（使用默认值）| 创建Const节点存默认值 | Const节点连边 |

**Const节点创建：**

```python
def create_const_node(value, input_kind):
    # 根据 input_kind 选择 dtype
    dtype_map = {
        INT: "int64",
        INT_OPT: "int64",
        FLOAT: "float32",
        FLOAT_OPT: "float32",
        BOOL: "bool",
        BOOL_OPT: "bool",
        SCALAR: infer_dtype_from_value(value),  # Python 类型推断
        STRING: "string",
        SCALAR_TYPE: "int64",  # 枚举值
    }

    dtype = dtype_map[input_kind]
    const_node = CreateConstNode(value, dtype, placement=HOST)
    return const_node
```

**Const节点特性：**
- value tensor存储在HOST内存
- dtype按schema类型或Python字面量类型选择
- Execute读取零开销（无D2H同步）

**列表参数处理：**

```python
def process_list_arg(args_list, input_kind):
    tensors = []
    for arg in args_list:
        if is_literal(arg):
            # 常量：创建 Const 节点
            tensors.append(create_const_node(arg, input_kind))
        elif is_fx_node(arg):
            # 上游节点：直接引用
            tensors.append(arg)

    # TODO: 生成合并算子，将多个 tensor 合为单个 tensor
    merge_node = CreateMergeOp(tensors, dtype=get_dtype(input_kind))
    return merge_node.output
```

**异常情况处理：**
- 合并算子尚未实现（P0 TODO）：当前跳过列表参数处理
- 不支持的参数类型：运行时fallback

**输出**

- **输出到何处**：GE ComputeGraph（节点）
- **数量**：一个FX Node对应一个GE Node
- **度量单位**：GE Node
- **时序**：图追踪阶段实时构建
- **有效输出范围**：符合GE图规范的节点

---

#### 功能需求3 Execute执行转换

**介绍**

GE Execute函数通过读取节点属性，动态构建Dispatcher调用栈，回调ATen IR实现计算逻辑，并将结果写回GE输出Tensor。

**输入**

- **来源**：GE runtime v2执行环境
- **数量**：每次执行一次调用
- **度量单位**：Execute函数调用
- **有效输入范围**：gert::Tensor输入（NPU device HBM内存）
- **时间要求**：运行时实时执行

**处理**

**整体流程：**

```cpp
graphStatus Execute(gert::EagerOpExecutionContext* ctx) {
    // 1. 获取元信息
    auto op_name = ctx->GetAttr<std::string>("op_name");
    auto input_kinds = ctx->GetAttr<std::vector<int>>("input_kinds");
    auto is_inplace = ctx->GetAttr<bool>("is_inplace");
    auto num_outputs = ctx->GetAttr<int>("num_outputs");

    // 2. 构建调用栈
    c10::Stack stack = BuildStack(ctx, input_kinds);

    // 3. Dispatcher 调用
    static auto& op_handle = GetCachedHandle(op_name);
    op_handle.callBoxed(&stack);

    // 4. WriteBack
    WriteBack(ctx, stack, is_inplace, num_outputs);

    return GRAPH_SUCCESS;
}
```

**BuildStack实现：**

```cpp
c10::Stack BuildStack(gert::EagerOpExecutionContext* ctx,
                      const std::vector<int>& input_kinds) {
    c10::Stack stack;
    int input_idx = 0;

    for (int kind : input_kinds) {
        auto* tensor = ctx->GetInputTensor(input_idx);

        // Optional 缺边处理
        if (tensor == nullptr) {
            if (IsOptional(kind)) {
                stack.emplace_back(c10::nullopt);
            } else {
                GELOGE("Required input %d is missing", input_idx);
                throw std::runtime_error("Required input missing");
            }
            input_idx++;
            continue;
        }

        // 根据类型提取值
        switch (kind) {
            case TENSOR:
            case OPTIONAL_TENSOR:
                stack.emplace_back(WrapNpuTensor(tensor));
                break;

            case INT:
            case INT_OPT:
                stack.emplace_back(ExtractInt(tensor));
                break;

            case FLOAT:
            case FLOAT_OPT:
                stack.emplace_back(ExtractFloat(tensor));
                break;

            case BOOL:
            case BOOL_OPT:
                stack.emplace_back(ExtractBool(tensor));
                break;

            case SCALAR:
            case SCALAR_OPT:
                stack.emplace_back(ExtractScalar(tensor));
                break;

            case INT_LIST:
            case INT_LIST_OPT:
                stack.emplace_back(ExtractIntList(tensor));
                break;

            // ... 其他类型处理
        }

        input_idx++;
    }

    return stack;
}
```

**Tensor包装：**

```cpp
at::Tensor WrapNpuTensor(const gert::Tensor* ge_tensor) {
    auto sizes = GetShape(ge_tensor->GetStorageShape());
    auto dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());
    int32_t device_id = GetCurrentDeviceId();

    return torch::from_blob(
        const_cast<void*>(ge_tensor->GetAddr()),
        sizes,
        [](void*) {},  // GE 管理内存，torch 不负责释放
        torch::TensorOptions()
            .dtype(dtype)
            .device(c10::Device(c10::DeviceType::PrivateUse1, device_id))
    );
}
```

**WriteBack实现：**

```cpp
void WriteBack(gert::EagerOpExecutionContext* ctx, c10::Stack& stack,
               bool is_inplace, int num_outputs) {
    for (int i = 0; i < num_outputs; ++i) {
        at::Tensor result = stack[i].toTensor();
        if (!result.is_contiguous()) {
            result = result.contiguous();
        }

        if (is_inplace && i == 0) {
            // inplace：结果写回输入 0
            const gert::Tensor* input_0 = ctx->GetInputTensor(0);
            aclrtMemcpy(const_cast<void*>(input_0->GetAddr()),
                        input_0->GetSize(),
                        result.data_ptr(), result.nbytes(),
                        ACL_MEMCPY_DEVICE_TO_DEVICE);
            ctx->MakeOutputRefInput(0, i);
        } else {
            // 非 inplace：分配输出并写入
            ge::DataType ge_dtype = MapTorchDtypeToGe(result.scalar_type());
            gert::Tensor* ge_out = ctx->MallocOutputTensor(
                i, result.sizes(), ge::FORMAT_ND, ge_dtype);
            aclrtMemcpy(ge_out->GetAddr(), ge_out->GetSize(),
                        result.data_ptr(), result.nbytes(),
                        ACL_MEMCPY_DEVICE_TO_DEVICE);
        }
    }
}
```

**异常情况处理：**
- Required input缺失：报错并抛异常
- Dispatcher调用失败：返回 `GRAPH_FAILED`
- 不支持的dtype：报错并抛异常

**输出**

- **输出到何处**：GE输出Tensor（gert::Tensor）
- **数量**：根据schema输出数量决定
- **度量单位**：gert::Tensor
- **时序**：运行时实时执行
- **有效输出范围**：连续内存布局、ND格式的tensor

---

### 非功能需求

#### 可维护性

1. **Schema驱动**：所有代码生成基于schema分析，修改schema后重新生成即可，无需手动维护
2. **通用Execute实现**：所有自定义算子共用同一Execute函数，通过属性配置行为，减少代码重复
3. **InputKind编码**：分段编码便于快速判断处理分支，易于扩展新类型

#### 可测试性

1. **单元测试**：BuildStack、WrapNpuTensor、WriteBack等函数可独立测试
2. **集成测试**：通过注册示例算子验证完整流程
3. **用户约束验证**：Meta key实现正确性、PrivateUse1 key实现正确性可通过测试用例验证

#### 可移植性

1. **依赖libtorch**：需要libtorch Dispatcher支持
2. **依赖torchair**：需要torchair converter支持
3. **依赖GE v2 runtime**：需要runtime v2执行环境

#### 平台化要求

本特性是GE的在线场景特性，涉及torchair适配层调用。不涉及芯片区分，但需要确认底层Dispatcher接口在各个平台下的支持能力。

#### 特性交叉分析

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| 静态Shape | **不适用** | 方案面向NPU device场景，通过Dispatcher回调ATen IR。静态shape场景（TaskSink模式）不涉及Dispatcher调用，Execute函数在Host端执行算子，而非TaskSink模式。 |
| 动态Shape | **适用** | Execute函数在运行时构建调用栈并通过Dispatcher调用ATen IR，这是动态shape场景的典型执行模式。方案设计支持动态shape推导（依赖用户提供的Meta key实现）。 |
| 动态Shape静态子图（图拆分） | **不适用** | 方案主要在在线场景使用，通过torchair converter构建GE图。静态子图（DavinciModelKernel）走v1 TaskSink路径，不涉及Dispatcher调用。 |
| 离线场景（atc编译） | **不适用** | 方案依赖torchair converter在运行时构建GE图，离线编译（atc）场景不适用。用户需要通过TORCH_LIBRARY注册算子，这是在线场景特有的。 |
| 在线场景（框架适配） | **适用** | 方案主要面向torchair框架适配，通过converter遍历FX Graph构建GE节点，通过Execute回调ATen IR。这是典型的在线场景。 |

---

### 性能

#### 模型编译时长

本特性涉及以下编译阶段：

1. **CodeGen（原型映射）**：扫描TORCH_LIBRARY注册的ATen schema，生成REG_OP。新增代码影响较小，主要是schema解析和代码生成，编译时长增加可控（预计 <1s）。

2. **torchair converter（构图）**：遍历FX Graph构建GE节点。新增代码为参数处理逻辑，对编译时长影响较小。

**影响评估：** 编译时长影响可控，不影响optimizeStage1、optimizeStage2、build、loadmodelonline等关键阶段。

#### OM大小和加载占用内存

本特性仅适用于在线场景，不涉及OM文件生成。加载时无额外内存占用。

#### 执行性能

**关键执行路径：**
1. BuildStack：构建调用栈，涉及遍历input_kinds和值提取
2. Dispatcher调用：通过 `op_handle.callBoxed` 调用ATen IR
3. WriteBack：结果写回，涉及 `aclrtMemcpy`

**性能影响评估：**
- BuildStack：每次执行遍历input_kinds，预计耗时 <100ns（小图）
- Dispatcher调用：依赖libtorch Dispatcher实现，预计耗时 <1us
- WriteBack：涉及D2D拷贝，耗时取决于输出tensor大小

**性能优化建议：**
1. OpCache：缓存 `op_handle` 避免重复查找（P2）
2. 避免BuildStack中的动态内存申请（按RT2约束）
3. 标量参数使用编译时常量类型（按用户接入规范）

---

### 接口设计

#### 新增/修改接口描述

本特性不新增GE对外接口，主要通过以下内部机制实现：

1. **REG_OP自动生成**：通过CodeGen自动生成算子原型
2. **通用Execute实现**：所有自定义算子共用同一Execute函数

用户接口为PyTorch TORCH_LIBRARY/TORCH_LIBRARY_IMPL，非GE接口。

#### 接口检查项

| **检查项** | **子检查项** | **是否涉及** |
| --- | --- | :--- |
| **接口说明** | 接口是否需要评审，评审需关注接口兼容和接口约束 | **不涉及**（无新增GE接口） |
| **接口说明** | 是否需要补充资料说明 | **涉及**（需补充用户接入规范文档） |
| **接口说明** | 接口文档中是否明确接口原型、功能、返回值等的说明 | **不涉及**（无新增GE接口） |
| **接口兼容** | 接口行为兼容性，修改前后行为是否发生变化 | **不涉及**（无修改GE接口） |
| **接口兼容** | 接口有无兼容性，新接口在老版本上是否能正常工作 | **不涉及**（无新增GE接口） |
| **接口兼容** | 接口是否涉及使用场景、调用时序等约束 | **涉及**（用户so加载顺序约束） |
| **接口约束** | 接口调用不满足约束条件时是否能清晰报错 | **涉及**（Required input缺失报错） |
| **接口约束** | 是否需要设计单独的测试用例 | **涉及**（需设计用户接入规范测试用例） |

---

### 软件设计

#### 关键数据结构

**InputKind编码：**
```cpp
enum InputKind {
    // Tensor 类（不走值依赖）
    TENSOR = 0,
    OPTIONAL_TENSOR = 1,
    TENSOR_LIST = 2,

    // 标量类（走值依赖）
    INT = 10,
    INT_OPT = 11,
    FLOAT = 12,
    FLOAT_OPT = 13,
    BOOL = 14,
    BOOL_OPT = 15,
    SCALAR = 16,
    SCALAR_OPT = 17,

    // 列表类（走值依赖）
    INT_LIST = 20,
    INT_LIST_OPT = 21,
    FLOAT_LIST = 22,
    BOOL_LIST = 23,

    // 特殊类（走值依赖）
    STRING = 30,
    STRING_OPT = 31,
    SCALAR_TYPE = 32,
    SCALAR_TYPE_OPT = 33,
};
```

**辅助函数：**
```cpp
bool IsOptional(InputKind kind);
bool IsValueDependency(InputKind kind);
```

#### 关键技术/算法

1. **Dispatcher回调机制**：通过 `op_handle.callBoxed(&stack)` 调用ATen IR
2. **值依赖注册**：非Tensor参数通过 `InputsDataDependency` 注册
3. **零拷贝Tensor包装**：通过 `torch::from_blob` 包装GE Tensor，避免数据拷贝

#### 流程设计

**三阶段流程：**

```text
阶段1: 原型映射
  ATen schema → CodeGen → REG_OP + 值依赖注册 + 元信息属性

阶段2: Converter 构图
  FX Graph → converter → GE Graph（Const节点、合并算子）

阶段3: Execute 执行
  GE输入 → BuildStack → Dispatcher调用 → WriteBack → GE输出
```

#### 对子模块的修改

本特性涉及以下子模块：

| 模块 | 修改内容 | 是否需要通知MDE |
|------|---------|-----------------|
| `graph_metadef/` | REG_OP自动生成机制 | **需要**（算子注册机制扩展） |
| `runtime/v2/` | Execute通用实现 | **需要**（v2执行器扩展） |
| torchair（外部组件） | Converter构图逻辑 | **需要**（torchair需同步更新） |

#### 错误处理

**系统错误**

- **内存分配失败**：WriteBack中 `MallocOutputTensor` 失败，返回 `GRAPH_FAILED`
- **Dispatcher查找失败**：`findSchemaOrThrow` 抛异常（需捕获并返回错误）

**接口错误**

- **Required input缺失**：返回 `GRAPH_FAILED`，日志记录缺失输入索引
- **不支持的dtype**：返回 `GRAPH_FAILED`，日志记录不支持的dtype
- **Dispatcher调用失败**：返回 `GRAPH_FAILED`

---

### 安全检查

#### 编码军规

参考 编码军规.md

#### 编码检查项

| **检查项** | **检查项说明** | **是否涉及** |
| --- | --- | --- |
| **是否涉及资源生命周期管理（如线程池、内存池等）** | 资源生命周期过长可能导致资源不足、响应变慢、系统崩溃等严重系统稳定性和可靠性问题 | **涉及**（OpCache生命周期需明确，建议进程级） |
| **是否创建新线程** | 1）新创建的线程需要拷贝ThreadLocalContext、ErrorContext等上下文信息。2）新创建线程需要设置线程名称 | **不涉及** |

---

### 兼容性检查

本特性仅适用于在线场景，不涉及OM文件生成，无老OM在新版本下执行的需求。

---

### DT设计

#### 测试边界

- **测试入口**：Execute函数调用
- **测试出口**：Dispatcher调用成功、WriteBack结果正确
- **打桩模块**：libtorch Dispatcher（可模拟调用）、gert::Tensor（可模拟输入输出）

#### 测试设计

**用例类型: UT、ST**

| **测试类别** | **关键测试项** | **测试方法** | **用例类型** |
| --- | --- | --- | --- |
| **功能** | BuildStack各类型提取正确性 | 构造不同InputKind输入，验证提取结果 | UT |
| **功能** | WrapNpuTensor零拷贝正确性 | 构造GE Tensor，验证包装后的torch Tensor地址一致 | UT |
| **功能** | WriteBack inplace/non-inplace模式 | 构造不同is_inplace属性，验证写回逻辑 | UT |
| **功能** | 完整流程（CodeGen → Converter → Execute） | 注册示例算子，验证端到端执行结果 | ST |
| **性能** | Execute执行性能 | 构造不同大小输入，测量执行耗时 | ST |
| **精度** | 算子计算精度 | 验证Dispatcher调用后的输出精度 | ST |
| **兼容性** | 不同schema形态支持 | 测试不同参数类型（Tensor/Scalar/列表等） | ST |
| **特性交叉** | 动态shape场景 | 测试动态shape输入的形状推导和执行 | ST |

#### 测试框架设计

当前测试框架（gtest）满足UT测试需求。ST测试需要集成torchair和libtorch环境。

---

### 验收标准

1. 用户通过TORCH_LIBRARY注册算子后，无需手动编写GE交付件
2. CodeGen能够正确生成REG_OP、值依赖注册、元信息属性
3. Converter能够正确构建GE节点（包括Const节点、合并算子）
4. Execute能够正确构建调用栈、回调Dispatcher、写回结果
5. UT/ST测试用例通过率100%
6. 执行性能符合预期（BuildStack <100ns，Dispatcher <1us）

---

## 附录

### 附录A数据类型映射

| GE DataType | PyTorch ScalarType |
|-------------|-------------------|
| DT_FLOAT | kFloat |
| DT_FLOAT16 | kHalf |
| DT_DOUBLE | kDouble |
| DT_INT8 | kInt8 |
| DT_INT16 | kInt16 |
| DT_INT32 | kInt32 |
| DT_INT64 | kInt64 |
| DT_UINT8 | kUInt8 |
| DT_BOOL | kBool |
| DT_BF16 | kBFloat16 |

---

### 附录B用户自定义算子接入规范

#### 功能正确性约束（必须满足）

**1. Schema定义**
通过 `TORCH_LIBRARY` 注册完整schema，数据类型只使用GE与PyTorch共同支持的类型。Inplace语义用 `Tensor(a!)` 显式标注；多输出在 `->` 后显式列出所有输出。

**2. Meta key实现（必须）**
必须注册Meta key，torchair FX图追踪阶段依赖Meta tensor做符号形状推导，缺失直接导致入图失败。Meta实现必须**正确推导输出的shape和dtype**，形状推导错误会导致静态shape场景下Execute写越界。Meta实现只做纯形状推导，不调用任何NPU/设备接口。

```cpp
TORCH_LIBRARY_IMPL(my_ops, Meta, m) {
    m.impl("my_op", [](const at::Tensor& x, const at::Tensor& y) {
        // 正确推导输出 shape 和 dtype，不做实际计算
        return at::empty(compute_output_shape(x, y),
                         x.options().dtype(compute_output_dtype(x, y)));
    });
}
```

**3. PrivateUse1 key实现（必须）**
必须注册PrivateUse1 key的计算实现，GE Execute通过Dispatcher命中此key；若仅注册Meta，运行时dispatch失败返回 `GRAPH_FAILED`。算子实现使用 `getCurrentNPUStream()` 获取当前stream；GE Execute在调用Dispatcher前已将GE stream桥接为当前NPU stream，算子无需额外处理。

**4. so加载顺序**
`libcust_opapi.so` 必须在进入GE执行前加载（`torch.ops.load_library` 或 `dlopen`）；未加载时 `findSchemaOrThrow` 直接抛异常。

#### 性能最佳推荐做法

**5. 标量参数使用编译时常量类型**
`alpha`、`dim`、`keepdim`、`scale` 等标量参数在schema中定义为 `float/int/bool` 类型，不要定义为 `Tensor`。`torch.compile` 追踪时会将其固化为常量，走Const节点HOST内存路径，Execute零开销读取。

```cpp
// 推荐：标量参数直接定义为 float
m.def("my_op(Tensor x, Tensor y, float alpha=1.0) -> Tensor");

// 不推荐：标量用 Tensor 传递
m.def("my_op(Tensor x, Tensor y, Tensor alpha) -> Tensor");
```

**6. 算子实现直接返回连续tensor**
PrivateUse1实现中，输出tensor应为连续内存布局（contiguous）。非连续tensor会在WriteBack时触发额外 `contiguous()` 拷贝。

**7. 输出使用ND格式**
算子输出使用标准多维数组格式（ND）。非ND格式不在本方案支持范围内。

#### 约束汇总

| 项目 | 类别 | 违反后果 |
|------|------|---------|
| 注册schema，数据类型在GE/PyTorch共同支持范围内 | 功能必须 | 不支持类型回退到手动实现路径 |
| Inplace用 `Tensor(a!)` 显式标注 | 功能必须 | inplace语义识别错误，内存管理异常 |
| Meta key实现，且形状推导正确 | 功能必须 | 缺失：FX图追踪失败；形状错误：Execute写越界 |
| PrivateUse1 key实现 | 功能必须 | 运行时dispatch失败，返回 `GRAPH_FAILED` |
| so在GE执行前加载 | 功能必须 | `findSchemaOrThrow` 抛异常 |
| 标量参数用 `float/int/bool`，不用 `Tensor` | 性能推荐 | 每次Execute引入D2H同步，打断NPU流水线 |
| 算子实现返回连续tensor | 性能推荐 | WriteBack触发额外 `contiguous()` 拷贝 |
| 输出使用ND格式 | 性能推荐 | 不在本方案支持范围，需格式转换 |

---

### 附录C框架实施优先级

| 优先级 | 功能 | 说明 |
|--------|------|------|
| P0 | REG_OP生成+值依赖注册 | 核心原型映射 |
| P0 | Converter基础构图（Tensor/标量） | 核心转换逻辑 |
| P0 | Execute BuildStack + Dispatcher调用 | 核心执行逻辑 |
| P0 | 异常处理+ Fallback接口 | 保证健壮性 |
| P1 | 列表参数合并算子 | 扩展参数类型 |
| P1 | String/ScalarType特殊类型处理 | 扩展参数类型 |
| P1 | Stream桥接 | NPU执行正确性 |
| P2 | OpCache性能优化 | 避免重复解析 |
| P2 | DFX（dump/profiling集成） | 可观测性 |
| P3 | 复杂schema支持（Tensor[]、多输出） | 扩大覆盖范围 |

---

### 附录D设计文档检查结果

#### 竞品分析检查

- [x] 竞品对比分析：已完成主流框架自定义算子机制对比分析
      - 对比维度覆盖：API设计、注册机制、Shape推导、执行模式、序列化、版本兼容
      - 分析框架数量：TensorRT、PyTorch、JAX、Inductor、TensorFlow（5个框架）
- [x] 借鉴点分析：已明确标注设计借鉴点和参考竞品
      - Meta Key强制要求 → 参考PyTorch
      - Dispatcher多后端路由 → 参考PyTorch PrivateUse1
      - Schema-driven CodeGen → 参考PyTorch/JAX
      - 序列化机制 → 参考TensorRT
      - OP_DEF + Kernel分离 → 参考TensorFlow
- [x] 差异点分析：已分析GE方案与竞品的差异
      - 差异原因已说明：GE仅适用于在线场景，依赖Dispatcher回调
      - 设计权衡已说明：易用性优先、约束优先、运行时注册
- [x] 设计权衡说明：已明确性能vs易用性、灵活性vs约束、编译期vs运行时的权衡决策

#### 跨特性交叉影响

- [x] 跨特性交叉影响：已按cross_feature_check.md逐场景分析，
      方案仅适用于在线场景（torchair框架适配），动态shape场景适用，
      静态shape、离线场景不适用，
      不涉及图拆分（动态shape静态子图）场景。

#### 关键特性设计原则和软件约束

- [x] 关键特性设计原则：
      1. 已加载 `graph_metadef.md`，方案遵循模块独立性原则（REG_OP生成作为算子注册扩展，不影响核心AscendIR结构）；
      2. 已加载 `rt2_runtime.md`，方案遵循RT2约束：
         - 加载时不修改计算图（CodeGen在构建阶段生成，不影响Lowering）
         - 执行时避免动态内存申请（BuildStack使用栈上 `c10::Stack`）
         - 支持多线程调用（Execute函数无共享资源）
      3. 方案无突破已有设计原则。

---

### 附录E TODO项

| 事项 | 说明 | 优先级 |
|------|------|--------|
| 合并算子具体实现 | Converter生成什么GE算子合并列表元素 | P0 |
| String tensor在Const节点存储格式 | GE String tensor的具体格式 | P1 |
| ScalarType枚举值映射表 | PyTorch ScalarType ↔ int64映射 | P1 |
| 值依赖注册完整示例 | InputsDataDependency的完整API用法 | P1 |
| Placement转换机制细节 | GE自动placement转换的具体流程 | P1 |
| OpCache性能优化 | 避免重复解析input_kinds | P2 |
