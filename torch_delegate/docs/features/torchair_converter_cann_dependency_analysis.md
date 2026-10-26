# 简介

本文分析 TorchAir Converter 导入和执行时触发的 CANN 依赖链，覆盖从 `torchair/__init__.py` 到单个 converter 函数体的完整依赖追踪。该分析作为 Lite converter 设计的基础依据。

# 一、Converter 转换流程

## 标准 TorchDelegate 转换流程

`record_ascend_ir=True` 时，TorchAir 对每个 FX 节点执行以下流程。已注册 converter 的运行时异常不进入 mock 分支，将直接终止转换。

```mermaid
flowchart TD
    Node[FX call_function 节点] --> Lookup{是否已有 converter}
    Lookup -->|是| Execute[执行已注册 converter]
    Lookup -->|否| Query[调用 get_registered_ir_def]
    Query --> Generate{能否生成 converter}
    Generate -->|是| Exec[exec 注册动态 converter]
    Exec --> Execute
    Generate -->|否| Mock[构造 mock GE 节点]
    Execute --> Record[记录 source=converter]
    Mock --> RecordMock[记录 source 为空字符串]
    RecordMock --> Fallback[由 C++ Fallback Codegen 处理]
```

## Converter 注册机制分析

先看一个真实的 converter 长什么样，以 `aten/add.py` 为例：

```python
@declare_supported([Support(F32(2, 2), F32(2, 1)), ...])
@register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
def conveter_aten_add_Tensor(self, other, *, alpha=1, meta_outputs=None):
    if alpha == 1:
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
        return ge.Add(self, other)
    else:
        self, other, alpha = dtype_promote(self, other, alpha, target_dtype=meta_outputs.dtype)
        return ge.AxpyV2(self, other, alpha)
```

一个 converter 文件由三部分组成：**注册声明**（`@register_fx_node_ge_converter` 把函数绑定到 `torch.ops.aten.add.Tensor`）、**支持声明**（`@declare_supported` 列出该算子支持的输入组合）、**转换函数体**（把 FX 节点翻译成 `ge.Add` 等 GE op）。

注册声明是本文关注的重点，它由 `fx2ge_converter.py` 的 `Converter` 类和工厂函数 `register_fx_node_ge_converter` 协作完成。工厂接收 `aten_op` 并返回一个 `Converter` 实例；当 `@register_fx_node_ge_converter(...)` 作为装饰器作用到函数上时，Python 会调用该实例的 `__call__` 方法，把注册逻辑执行在传入的 converter 函数上。流程如下：

```mermaid
flowchart TD
    Decorate["@register_fx_node_ge_converter(aten_op)"] --> Factory["返回 Converter 实例"]
    Factory --> Call["Converter.__call__(converter)"]
    Call --> Wrap["包装 converter 函数"]
    Wrap --> Meta{"签名含 meta_outputs?"}
    Meta -->|"是"| MTrue["require_meta = True"]
    Meta -->|"否"| MFalse["require_meta = False"]
    MTrue --> Attach["设置 op._ge_converter = 包装函数"]
    MFalse --> Attach
    Attach --> Ok{"设置成功?"}
    Ok -->|"是"| Done["返回 Converter 实例"]
    Ok -->|"否"| Global["回退到全局 _CONVERTERS 字典"]
    Global --> Done
```

注册最终落在两处：优先把 converter 挂到 `aten_op._ge_converter` 上（直接改 PyTorch 全局 `OpOverload` 对象）；若该对象不允许设属性，则回退到 `_CONVERTERS` 全局字典。

关键副作用：这条路会修改进程级全局状态（`OpOverload` 属性或 `_CONVERTERS`），影响面是持久的。Lite Context 初版沿用该注册机制，因此进程内 stub 注入后不能再切换到完整模式；需要两种模式时必须使用不同进程。

# 二、Converter 与 TorchAir 的实际依赖

分析基于 TorchAir revision `be0c45f60b305ef9ffa5f2f02feaab06fe80361f`。这里把依赖分为三个阶段，避免把“导入原生扩展”和“调用 CANN”混为一谈：

- 导入阶段：Python 加载模块、执行装饰器并加载 `_torchair.so`。
- 转换阶段：converter 把 FX 节点写成 GE protobuf；部分路径同时查询设备或 CANN IR。
- 后端接口：当前导出路径会提前初始化 Graph Engine，但不会加载、编译或执行 GE 图。

## 正常导入路径

Python 导入 `torchair._ge_concrete_graph.ge_converter.aten.mm` 时，会先执行父包 `torchair/__init__.py`。因此这不是“只导入 mm converter”：顶层包会加载编译器、GE API、converter 包和其他公开模块。

```mermaid
flowchart TD
    Request["import torchair...ge_converter.aten.mm"] --> Package["先执行 torchair/__init__.py"]

    Package --> Compiler["npu_fx_compiler / fx_summary"]
    Compiler --> FX["fx2ge_converter.py"]
    FX --> Native["torchair.core._torchair"]
    FX --> Backend["core/_backend.py"]

    Package --> ConverterPkg["ge_converter/__init__.py"]
    ConverterPkg --> Families["批量导入 prims 和 aten"]
    Families --> Leaf["aten/mm.py"]
    Leaf --> Utils["converter_utils.py"]
    Utils --> FX
    Utils --> GeApis["ge_apis.py"]
    Utils --> Compat["compat_ir.py"]
    Utils --> Custom["ge_custom.py"]

    Package --> GePkg["torchair.ge"]
    GePkg --> Custom
    GePkg --> GeApis
    GeApis --> RawOps["auto_generated_ge_raw_ops.py"]
    RawOps --> Compat
    Compat --> Native
    Custom --> Native
    Backend --> Native

    Package --> OptionalNpu["尝试 import torch_npu，可缺省"]
```

主要源码依据：

- `torchair/__init__.py:11-28` 导入编译器、`ge_converter`、`fx2ge_converter` 和 `torchair.ge`。
- `ge_converter/__init__.py:2-22` 导入固定 converter 组，并遍历 `prims`、`aten` 下的所有模块。
- `converter_utils.py:16-31` 同时导入 `ge_apis`、`compat_ir`、`fx2ge_converter`、`_ge_graph`、`check_platform` 和 `ge_custom`。
- `fx2ge_converter.py:34-35`、`compat_ir.py:5`、`ge_custom.py:7`、`core/_backend.py:18` 都会在模块加载时导入 `_torchair`。
- `ge_apis.py:14` 导入 `auto_generated_ge_raw_ops`，后者在 `auto_generated_ge_raw_ops.py:11` 导入 `compat_ir`，所以 `ge_apis` 也会间接加载 `_torchair`。

这条导入链说明了两个事实。第一，按标准包路径导入单个 converter 仍会执行 `torchair/__init__.py`，并触发 aten/prims 的批量注册。第二，converter 函数体即使只写 protobuf，其所在模块也不一定能在无 CANN 动态库的环境中直接导入，因为 `_torchair.so` 已经是导入时依赖。

## 导入时发生什么

导入阶段会发生：

- Python 动态加载 `_torchair.so`。该扩展链接 GE、ACL、Ascend metadef 和 HCCL 等动态库，因此系统必须能解析这些链接依赖。
- `ge_converter/__init__.py` 批量导入 aten/prims converter；各模块的装饰器随即执行，并修改 `OpOverload._ge_converter` 或 `_CONVERTERS`。
- `core/_backend.py:113-116` 注册 `atexit` 和 `SIGTERM`/`SIGINT` 处理函数。
- `torchair/__init__.py:38-44` 尝试导入 `torch_npu`。这里捕获 `ImportError` 和 `AttributeError`，所以 `torch_npu` 缺失不一定导致顶层导入失败。

导入阶段不会执行 converter 函数体，也不会因为 `check_platform.py` 被导入就查询设备。`is_arch35()` 内部的 `import torch_npu` 和 `get_device_name()` 只有调用该函数时才执行（`check_platform.py:2-8`）。同理，导入 `_torchair.so` 不等于已经调用 `GEInitialize`。

## Converter 执行路径

注册完成后，converter 在 FX 节点转换期间被调用。以 `aten.add.Tensor` 为例，`dtype_promote` 可能先插入 `Cast` 或 `Const`，随后调用 `ge.Add` 或 `ge.AxpyV2`。这些 generated GE API 获取当前 `GeGraph`，向 `GraphDef.op` 追加 `OpDef`，最后返回一个表示输出边的 `Tensor`。

```mermaid
flowchart LR
    subgraph B1["① 定位与准备"]
        direction TB
        FXNode["FX call_function 节点"]
        Lookup["查找已注册 converter"]
        Meta["meta_outputs 转为 TensorSpec"]
        FXNode --> Lookup
        Lookup --> Meta
    end
    subgraph B2["② 转换与 GE API"]
        direction TB
        Converter["执行 converter 函数体"]
        Helper["dtype/shape/layout 等转换逻辑"]
        GeCall["调用 generated GE API"]
        Converter --> Helper
        Helper --> GeCall
    end
    subgraph B3["③ 构图与输出"]
        direction TB
        Graph["get_default_ge_graph()"]
        OpDef["追加并填写 OpDef"]
        Output["返回 GE Tensor，并补充输出 meta"]
        Graph --> OpDef
        OpDef --> Output
    end
    B1 --> B2
    B2 --> B3
```

`ge.Add`（`auto_generated_ge_raw_ops.py:10728-10762`）和 `ge.MatMul`（`auto_generated_ge_raw_ops.py:34674-34721`）本身不执行数值计算，也不调用 GE Session。它们只写 protobuf。这里容易混淆的是：generated API 并非全部都只有 protobuf 行为。

### 转换阶段的额外查询

以下路径在 converter 执行期间会越过 protobuf 构图边界：

1. `custom_op()` 在 `ge_custom.py:15` 无条件调用 `_torchair.get_registered_ir_def(op_type)`。该接口查询已经注册到 GE 的算子 IR 信息，依赖 GE 和 CANN 环境。即使调用方显式传入 `inputs`、`outputs`、`attrs`，查询仍会先发生；`len(args)` 的判断位于查询之后。
2. `ge_op(..., ir=IrDef(...))` 在 `compat_ir.py:228-238` 调用 `_torchair.check_cann_compat()`。当前 generated GE API 中有 52 个此类调用点，分布在 50 个 generated wrapper 中。
3. `is_arch35()`、`torch.npu.matmul.allow_hf32` 和局部 `torch_npu` 导入会查询设备、平台配置或扩展 dtype。
4. `GeGraph()` 在 `_ge_graph.py:751-769` 创建 `TorchNpuGraph` 原生执行器对象。它尚未初始化 GE，但已经要求 `_torchair` 和其动态库可用。

## Graph Engine 初始化与图执行

TorchDelegate 导出 EPAIR 只需要生成 GE protobuf 和节点级 IR，不需要加载、编译或执行 GE 图。`_optimize_and_convert()` 中的 `converter.run(*new_input_values)` 是 FX interpreter 遍历图并调用 converter，不是 `GeGraph.run()`。

但当前 TorchAir 实现仍在导出路径中提前初始化 Graph Engine：

```mermaid
flowchart TD
    Export["TorchDelegate export_to_epair"] --> Convert["_optimize_and_convert"]
    Convert --> Construct["构造 GeConcreteGraph"]
    Construct --> Executor["GeGraph 创建 TorchNpuGraph"]
    Construct --> Init["InitializeGraphEngine"]
    Init --> GEInit["GEInitialize + aclrtSetDevice"]
    Convert --> Proto["生成 GraphDef 和 ascend_ir_map"]
    Proto --> Epair["写入 EPAIR，导出结束"]

    Proto -. "部署执行时才进入" .-> Load["GeGraph.load"]
    Load --> Compile["GeGraph.compile"]
    Compile --> Run["GeGraph.run"]
```

- `_optimize_and_convert()` 在 `ge_graph_ascend.py:309` 构造 `GeConcreteGraph`。
- `GeConcreteGraph.__init__` 在 `fx2ge_converter.py:964-990` 创建 `GeGraph`，随后调用 `initialize_graph_engine()`。
- `GeGraph.__init__` 在 `_ge_graph.py:751-769` 创建 `TorchNpuGraph` 原生执行器对象。
- `core/_backend.py:85-93` 把初始化请求传给 `_torchair.InitializeGraphEngine()`。
- `ge_graph_ascend.py` 不调用 `GeGraph.load/compile/run`；这三个方法只在部署执行路径中使用。

所以，图加载、编译和执行不影响 TorchDelegate 导出，但 Graph Engine 初始化仍然影响。它不是生成 EPAIR 所必需的步骤，只是 `GeConcreteGraph` 同时承担构图和执行职责带来的耦合。Lite Context 模式通过依赖打桩隔离构造期的原生执行器和初始化调用。

“不依赖 `torch_npu`”也不等于“不依赖 CANN”。`core/_backend.py:33-57` 在没有 `torch_npu` 时会使用环境变量和默认编译选项，随后仍通过 `_torchair.InitializeGraphEngine()` 调用 GE 和 ACL。

## Converter 源码依赖统计

本次统计以文件为单位，不把一个文件等同于一个 converter 函数。`ge_converter/` 下共有 954 个非 `__init__.py` Python 模块，其中 952 个包含 converter 注册；其余两个是 `converter_utils.py` 和 `experimental/hcom_allreduce.py`。

| 范围 | aten | custom | prims | 其他 | 合计 |
|---|---:|---:|---:|---:|---:|
| 非 `__init__.py` 模块 | 686 | 123 | 124 | 21 | 954 |
| 含注册的模块 | 686 | 123 | 124 | 19 | 952 |

对 954 个模块做静态扫描，得到以下集合。类别之间有重叠，不能直接相加。

| 集合 | 文件数 | 含义 |
|---|---:|---|
| 调用 `is_arch35()` | 25 | aten 10、custom 15；调用时导入 `torch_npu` 并查询设备名 |
| 局部导入 `torch_npu` | 19 | custom 17、experimental 1、lite 1；用途包括 dtype 和 CANN 版本查询 |
| 读取 `torch.npu.matmul.allow_hf32` | 2 | `fused_matmul.py`、`transpose_batchmatmul.py` |
| 调用 `custom_op()` | 5 | 每次调用都会通过 `get_registered_ir_def` 查询 GE 已注册算子 IR |
| 调用带 `IrDef` 的 generated GE API | 81 | 这些 API 可能执行 `check_cann_compat` |

前四类直接设备/schema 依赖共有 44 个去重后的模块，而不是各行之和 51。把 generated `IrDef` API 也纳入后，五类合计覆盖 96 个去重模块。其余 858 个模块没有命中这些已知模式，但这只能视为候选集，不能据此直接断言“完全不依赖 CANN”：模块导入仍会加载 `_torchair`，间接调用和后续版本新增路径也需要单独审计。

## Converter 使用 `torch_npu` 后做了什么

converter 中的 `torch_npu` 依赖不是同一种用途。源码中可以分成四类：判断目标架构、识别扩展 dtype、判断 CANN 版本和读取 HF32 配置。这些值都会改变最终写入的 GE 图，并非只用于日志或环境检查。

### 架构判断

`is_arch35()` 在 `check_platform.py:2-8` 内导入 `torch_npu`，调用 `torch_npu.npu.get_device_name()`，设备名包含 `Ascend910_95` 或 `Ascend950` 时返回 `True`。`ImportError` 被视为 `False`；设备查询本身的其他异常不会被捕获。

25 个 converter 模块共有 34 个调用点。调用结果主要影响以下构图行为：

| 行为 | 结果 | 涉及模块 |
|---|---|---|
| 切换 GE op 版本 | 在 V1/V2/V3/V4 或新旧算子之间选择，例如 `BNInfer`/`BatchNormV3`、`GroupNorm`/`GroupNormV2`、`QuantBatchMatmulV3`/`V4` | `aten/batch_norm_elemt.py`、`_native_batch_norm_legit.py`、`_native_batch_norm_legit_no_training.py`、`native_batch_norm_backward.py`、`native_group_norm.py`、`native_group_norm_backward.py`、`max_pool2d_with_indices.py`、`max_pool2d_with_indices_backward.py`、`std.py`；`custom/all_gather_matmul.py`、`matmul_reduce_scatter.py`、`quantmatmul.py`、`npu_moe_finalize_routing.py` |
| 限制算子支持范围 | 某些 converter 仅支持 arch35，或反过来在 arch35 上拒绝 | `custom/fused_matmul.py`、`quant_matmul_inplace_add.py`、`quant_grouped_matmul_inplace_add.py`、`flash_attention.py`；`aten/_native_batch_norm_legit.py` |
| 改变 op 属性和通信模式 | 选择 `ccu`/`ai_cpu`/`aiv`，或者决定某参数是否允许 | `custom/grouped_matmul_allto_allv.py`、`allto_allv_grouped_matmul.py`、`matmul_all_reduce.py`、`matmul_reduce_scatter_v2.py` |
| 改变 layout、shape 和输出含义 | 选择 NCHW/NCDHW，调整 pooling 属性长度，或决定是否额外计算/转换输出 | 上述 batch norm、group norm、max pool 模块 |
| 改变量化和 packed dtype 处理 | 启用 A16W4/MxA8W4，选择 INT4/FP4 的 bitcast、reshape 和输出 dtype | `custom/grouped_matmul.py`、`quantmatmul.py` |
| 选择算子组合 | 在 `Axpy`、`AxpyV2`、`Mul+Sub`、`Sxpy` 或 rotary 新旧接口之间选择 | `aten/rsub.py`、`custom/npu_rotary_mul.py`、`npu_rotary_mul_backward.py` |

几个具有代表性的差异：

- `max_pool2d_with_indices.py:44-57`：arch35 使用 V3 接口和二维空间属性；旧路径使用 V1、四维属性，并替换 indices 的处理方式。
- `native_group_norm.py:28-38`：arch35 的 `GroupNormV2` 直接返回 `rstd`；旧 `GroupNorm` 返回 variance，converter 还需构造 `Rsqrt(variance + eps)`。
- `quantmatmul.py:225-254`：arch35 构造 `QuantBatchMatmulV4`，旧平台构造 V3，两者的 scale、offset 和输出 dtype 处理不同。
- `flash_attention.py:46-47,116-117`：两个旧 flash-attention converter 在 arch35 上直接拒绝，而不是选择新 op。

Lite Context 模式不能把 torch_npu 缺失解释为 `is_arch35() == False`。架构结果必须来自目标设备配置，并保持与上述设备名判断一致。

### 扩展 dtype

19 个模块在函数体内导入 `torch_npu`，其中 15 个模块读取了 `torch_npu.*` 属性。实际使用的扩展 dtype 有五种：

| `torch_npu` dtype | PyTorch dtype 标识 | GE dtype | 读取后的操作 |
|---|---:|---|---|
| `hifloat8` | 290 | `DT_HIFLOAT8`（34） | 校验输入/输出 dtype，插入 `Bitcast`，或修改输出 descriptor |
| `float8_e8m0fnu` | 293 | `DT_FLOAT8_E8M0`（37） | 转换 dequant scale，或标注量化 scale 输出 |
| `float4_e2m1fn_x2` | 296 | `DT_FLOAT4_E2M1`（40） | 识别 packed FP4，扩展 shape、bitcast、reshape，并调整 descriptor |
| `float4_e1m2fn_x2` | 297 | `DT_FLOAT4_E1M2`（41） | 与 E2M1 一起识别 packed FP4，并选择对应 bitcast/repack 路径 |
| `int4` | 285 | `DT_INT4`（29） | 把 packed 权重 bitcast 为 INT4 |

PyTorch dtype 标识和 GE `DataType` 是两套编号。例如 `torch_npu.hifloat8` 对应 290，而 `DataType.DT_HIFLOAT8` 是 34。映射在 `_ge_graph.py:211-268` 中完成。因此不能把：

```python
x_dtype == torch_npu.float4_e2m1fn_x2
```

直接改成：

```python
x_dtype == DataType.DT_FLOAT4_E2M1
```

正确做法是先用 `torch_dtype_value_to_ge_type(x_dtype)` 转成 GE dtype，再与 `DataType.DT_FLOAT4_E2M1` 比较。只有已知输出必然是某种类型、仅需填写 protobuf descriptor 时，才适合直接使用 GE/protobuf dtype 常量。

这些 dtype 在 converter 中主要产生三类操作：

1. 输入校验和分支选择。`quant_lightning_indexer.py:52-60`、`kv_quant_sparse_flash_attention.py:47-55` 只接受 HIFLOAT8；`quantmatmul.py:98-106` 用 FP4 标识判断 MXFP4/A8W4 路径。
2. packed 数据展开。`quantmatmul.py:110-167`、`grouped_matmul.py:329-348`、`matmul_all_reduce.py:103-128` 会把最后一维乘 2（FP4）或乘 8（INT4），再执行 `Bitcast` 和 `Reshape`。
3. descriptor 修正。`npu_quant_scatter.py:43-68`、`quant_conv2d.py:48-82`、`all_gather_matmul_v2.py:140-141` 等在构图后显式填写 HIFLOAT8/FP4 输出 dtype。

有三处局部 `import torch_npu` 后没有读取任何 `torch_npu` 属性：

- `custom/allto_all_quant_matmul.py:99`
- `custom/allto_allv_quant_grouped_matmul.py:76`
- `custom/quant_matmul_allto_all.py:87`

这些 import 只带来加载副作用和环境检查效果。它们位于函数体内，发生时间晚于模块装饰器执行，不能用于注册 `torch.ops.npu` schema。Lite Context 可以通过 fake 隔离这些导入，但所需 operator schema 必须在当前进程中通过纯 Python 模块显式注册。

### CANN 版本判断

三个 converter 路径从 `torch_npu.npu.utils` 导入 `_is_gte_cann_version`。该函数进入 `torch_npu._C` 查询已安装的 CANN 组件版本，结果会改变 GE 图：

| 调用位置 | 判断 | 后续操作 |
|---|---|---|
| `custom/grouped_matmul.py:84-92` | CANN `>= 8.5.0` | 新版本用 `Bitcast(..., keep_dim=True)`；旧版本把最后一维乘 8，再 bitcast 和 reshape |
| `custom/grouped_matmul_finalize_routing.py:78-86` | CANN `>= 8.5.0` | 同样在两种 INT4 packed 表示之间选择 |
| `experimental/hcom_alltoall.py:87-103` | CANN `>= 8.2.RC1` | 计数均匀时使用 `HcomAllToAll`；否则使用 `HcomAllToAllV` |

Lite Context 模式可以从目标平台配置读取 CANN 兼容版本，但要保留 RC 版本排序和具体阈值。目标版本缺失时应报错，不能默认走旧版本分支。

### HF32 配置

两个 converter 读取 `torch.npu.matmul.allow_hf32`：

- `custom/fused_matmul.py:39-42`：先执行 arch35 判断，支持时把该布尔值写入 `FusedMatMul.enable_hf32`。
- `custom/transpose_batchmatmul.py:21-23`：直接把该布尔值写入 `TransposeBatchMatMul.enable_hf32`。

该值是用户配置，不应根据 SoC 自动推断。Lite Context 模式需要由目标平台配置显式提供，并原样写入 GE op 属性。

# 三、不依赖 torch_npu / CANN 的可复用范围

## 先区分“源码可复用”和“模块可直接导入”

TorchAir converter 的主体是 Python 构图逻辑，这部分可以复用；现有模块边界却把它和 `_torchair`、`_backend`、`GeGraph` 执行器放在同一条 import 链上。因此可复用的是代码能力，不是未经处理的 TorchAir 包导入方式。

直接执行下面的语句，并不能绕开 CANN：

```python
from torchair._ge_concrete_graph.ge_converter.aten.mm import *
```

Python 会先执行 `torchair/__init__.py`，随后通过多条路径加载 `_torchair.so`。在没有相应动态库的环境中，这一步就可能失败，converter 函数还没有机会执行。

## 可复用的构图部分

不考虑当前 import 耦合时，下面几类代码不需要计算设备，也不负责图编译和运行：

- converter 注册规则：operator 到 converter 的映射、`meta_outputs` 签名检查及包装逻辑。
- GE 数据表示：`DataType`、`Tensor`、`TensorSpec`、`GraphDef`、`OpDef` 和属性对象。
- 图栈和普通 generated GE API：取得当前图、追加 OpDef、连接输入输出并返回符号 Tensor。
- dtype、shape、layout、`SymInt` 等转换辅助函数。
- 未触发设备查询、`custom_op` 或 `IrDef` 兼容性检查的 converter 函数体。

这些代码完成的工作止于“生成 GE protobuf”。它们不包含 `GEInitialize`、`AddGraph`、`CompileGraph` 或 `RunGraph`。

## 不能原样复用的边界

### 包导入和注册

标准包导入会加载 `_torchair.so`、批量导入 converter，并修改 PyTorch 全局 `OpOverload`。Lite Context 初版在 TorchAir 导入前预置 `_torchair` stub，阻止真实扩展加载；converter 仍按标准包入口批量注册。该全局注册状态不做恢复，因此 Lite Context 与完整模式在进程内互斥。

### `GeGraph` 对象

`GeGraph` 同时保存 protobuf 图和 `_executor`。构图只需要前者，但现有构造函数无条件创建 `TorchNpuGraph`。无 CANN 模式需要一个不创建原生执行器的图对象，或为该构造依赖提供明确的隔离层；`load/compile/run` 不应进入 Lite 路径。

### CANN IR 查询

两条查询路径必须分开处理：

- `custom_op()` 总是通过 `get_registered_ir_def` 查询 GE 已注册算子 IR。Lite Context 模式不提供该查询能力，当前 5 个调用模块应标记为不支持，不能因为使用关键字形式的 `inputs/outputs/attrs` 就视为安全。
- 81 个模块会调用带 `IrDef` 的 generated GE API。这些 API 可能通过 `check_cann_compat` 查询当前 CANN 的算子原型。

无 CANN 环境不能伪造查询成功。`get_registered_ir_def` 不做离线替代，凡是依赖该接口的流程均不进入 Lite 支持范围。`check_cann_compat` 只有在目标 CANN 版本、可选输入和可选属性已经由离线清单验证时才可以跳过；否则也应将对应 converter 标记为不支持。

### 设备和 `torch_npu` 信息

25 个模块调用 `is_arch35()`，19 个模块局部导入 `torch_npu`，3 个路径查询 CANN 版本，另有 2 个模块读取 HF32 配置。替代方式要按语义拆开：架构、CANN 版本和 HF32 来自目标平台配置；扩展 dtype 通过稳定的 PyTorch dtype 标识到 GE dtype 映射处理。

尤其是当前 `is_arch35()` 判断设备名是否包含 `Ascend910_95` 或 `Ascend950`（`check_platform.py:2-8`），不能直接替换成 `soc_family == "Ascend910B"`。如需离线判断，平台配置必须使用与 TorchAir 相同的设备语义。

扩展 dtype 也不能直接替换成同名 `DataType` 常量，因为两边的数值域不同。应先调用等价于 `torch_dtype_value_to_ge_type()` 的映射；固定输出 descriptor 才可以直接填写 GE/protobuf dtype。

### 动态 converter 和 Graph Engine

未注册算子的自动 converter 生成依赖 `get_registered_ir_def` 查询 GE 已注册算子 IR。Lite Context 模式只使用已经注册且通过审计的 converter，不支持自动生成；调用 `custom_op()` 的 converter 因为依赖同一查询接口，也不支持。mock 节点属于完整转换流程的 fallback，同样不在“复用已注册 converter 生成 protobuf”的范围内。图加载、编译和执行不会出现在 EPAIR 导出流程中；构造 `GeConcreteGraph` 时触发的 GE 初始化由依赖打桩隔离。

## 可复用范围的判定方式

一个 converter 能在无 `torch_npu`、无 CANN 环境中复用，至少应满足：

1. 能在预置依赖打桩模块后完成注册，不加载真实 `_torchair.so`；其进程级注册状态只用于 Lite Context。
2. 执行路径只使用 protobuf 构图接口；若调用带 `IrDef` 的 generated API，已有与目标 CANN 版本匹配的离线兼容性结果。
3. 不读取本机设备状态。需要平台分支时，所需字段由目标平台配置显式提供。
4. 不调用图初始化、加载、编译和运行接口。

按当前静态扫描，858 个模块未命中已知的设备/schema 查询模式，可作为首轮候选集；这不是最终支持数量。最终清单还应按 converter 的实际调用图、目标 CANN 版本和源码 revision 逐项验证。

## Lite Context 模式的依赖替代

基于 revision `be0c45f60b305ef9ffa5f2f02feaab06fe80361f` 的原型验证表明，Lite Context 初版不必先抽取整套 converter 组件。其核心是在 TorchDelegate 当前进程中提供依赖打桩模块，隔离 converter 对 CANN 和 torch_npu 的功能依赖。依赖替代必须在导入任何 TorchAir 模块之前完成：

1. 在 `sys.modules` 预置 `torchair.core._torchair` stub，避免加载真实 `_torchair.so`；生命周期接口为空操作，图执行和 schema 查询接口明确报错。
2. 预置 torch_npu fake，并从目标平台配置提供设备型号、CANN 版本和 HF32 信息。
3. 按标准入口导入 TorchAir，沿用原有 aten/prims converter 注册机制。
4. 正常调用 Python `_optimize_and_convert()` 完成 protobuf 构图和 NodeIR 记录。

CPU `aten.add.Tensor` 原型在该环境中成功生成 `Data`、`Data`、`Add`、`NetOutput` 四个 OpDef，并记录 `add` 节点的 NodeIR；过程中没有加载真实 `_torchair.so`、`torch_npu` 或 CANN。

依赖打桩可以隔离普通已注册 converter 的导入、Graph Engine 初始化和执行器构造，但不能替代以下语义依赖：

- `custom_op()` 和自动 converter 生成需要真实 `get_registered_ir_def`，Lite 直接拒绝。
- 带 `IrDef` 的 generated API 需要 `check_cann_compat`；Lite Context 初版拒绝，后续只能使用按目标 CANN 版本生成的离线兼容清单。
- `is_arch35`、CANN 版本、HF32 和扩展 dtype 需要目标平台配置或稳定映射，不能由通用 stub 猜测。
- TorchAir 私有 import 和 `_optimize_and_convert()` 仍有版本耦合，必须锁定 revision 并执行差分测试。

Lite Context 初版采用进程内依赖打桩，不建立独立 Converter 注册表。由于模块缓存和 converter 注册状态不可完全恢复，同一进程不能混用 Lite Context 和完整模式。

# 结论

TorchAir converter 的普通构图函数主要生成 GE protobuf，但标准 TorchAir 包没有把这部分与原生后端隔离开。导入单个 converter 仍会先执行 `torchair/__init__.py`，加载 `_torchair.so` 并批量注册 converter；执行 converter 时，`custom_op`、部分 generated GE API 和设备条件分支还可能调用 `_torchair` 或 `torch_npu`。TorchDelegate 导出不加载、编译或执行 GE 图，但当前 `_optimize_and_convert` 会在构造 `GeConcreteGraph` 时提前初始化 Graph Engine，这部分也需要从 Lite 路径隔离。

因此，无 CANN 复用不能以“converter 函数体看起来只调用 `ge.*`”作为判定依据。可复用对象应限定为经过导入隔离和调用链审计的注册逻辑、protobuf 构图代码及 converter 函数体，并按 TorchAir revision 固定支持清单。
