# EpairModelLoader<a id="ZH-CN_TOPIC_TD_EPAIR_MODEL_LOADER"></a>

## 概览<a id="section_overview"></a>

`EpairModelLoader` 是TorchDelegate的核心C++类，封装了 `.epair` 归档文件解析、GE Graph构建、编译和执行的完整流程。该类使用PIMPL模式隔离实现细节，且对外ABI稳定。

同时，TorchDelegate提供 [Python API](python_api.md) 用于模型导出、保存和加载，支持将PyTorch `ExportedProgram` 转换为 `.epair` 文件。

头文件与库文件随whl分发，安装后位于 `<site-packages>/torch_delegate/inc/torch_delegate/` 与 `<site-packages>/torch_delegate/lib/`，CMake配置包位于 `<site-packages>/torch_delegate/lib/cmake/TorchDelegate/`。C++工程通过 `find_package(TorchDelegate)` 接入，前缀由 `torch-delegate-config --cmake-prefix`（或 `torch_delegate.cmake_prefix()`）获取；运行期需手动配置 `LD_LIBRARY_PATH`。详见 [构建指南](../guide/build.md)。

**数据流：**

```text
PyTorch Model
    │ torch.export.export()
    ▼
ExportedProgram
    │ [Python] export_to_epair()
    ▼
EpairContents
    │ [Python] epair_save()
    ▼
.epair 文件 (ZIP Archive)
    │ [C++] EpairModelLoader(archive_path)
    ▼
编译 & NPU 推理
    │ CompileAndLoad() → RunModel() / RunModelWithStreamAsync()
    ▼
推理结果
```

---

## C++ API<a id="section_cpp_api"></a>

### 类索引<a id="section_class_index"></a>

- **[Status状态码](#section_status)**
- **[EpairModelLoader构造函数](#section_constructor)**
- **[CompileAndLoad](#section_compile_and_load)**
- **[RunModel](#section_run_model_sync)**
- **[RunModelWithStreamAsync](#section_run_model)**
- **[GetGEGraph](#section_get_ge_graph)**

---

### Status状态码<a id="section_status"></a>

#### 头文件/库文件<a id="section_status_files"></a>

- 头文件：`#include <torch_delegate/td_types.h>`
- 库文件：`libtorch_delegate_backend.so`

#### 功能说明<a id="section_status_desc"></a>

`td::Status` 是 `int64_t` 的类型别名，用于表示TorchDelegate C++ API的返回值。

#### 状态码定义<a id="section_status_def"></a>

<a id="table_status"></a>
| 常量 | 值 | 说明 |
| --- | --- | --- |
| `td::SUCCESS` | 0 | 操作成功 |
| `td::FAILED` | 1 | 通用失败 |
| `td::INVALID_PARAM` | 2 | 非法参数 |
| `td::INTERNAL_ERROR` | 3 | 内部错误 |
| `td::UNIMPLEMENTED` | 4 | 功能未实现 |

---

### EpairModelLoader构造函数<a id="section_constructor"></a>

#### 头文件/库文件<a id="section_constructor_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

#### 功能说明<a id="section_constructor_desc"></a>

从 `.epair` 归档文件构造模型加载器。构造过程会立即解析归档内容并构建GE Graph，但不执行编译。初始化失败时内部记录错误状态，不抛出异常。

#### 函数原型<a id="section_constructor_proto"></a>

```cpp
explicit EpairModelLoader(const char *archive_path)
```

#### 参数说明<a id="section_constructor_params"></a>

<a id="table_constructor"></a>
| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| archive_path | 输入 | `.epair` 归档文件的路径，必须非空。支持相对路径和绝对路径，相对路径相对于进程当前工作目录（CWD）解析。 |

#### 约束说明<a id="section_constructor_constraints"></a>

- `archive_path` 必须指向有效的 `.epair` 文件（由 `epair_save()` 生成的ZIP归档），支持相对路径和绝对路径
- 构造函数不会抛出异常；初始化失败时，继续调用其他方法均返回非 `SUCCESS` 错误码（`INVALID_PARAM` 或 `FAILED`）

---

### CompileAndLoad<a id="section_compile_and_load"></a>

#### 头文件/库文件<a id="section_compile_and_load_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

#### 功能说明<a id="section_compile_and_load_desc"></a>

编译并加载模型到指定流，为图执行做准备，包括申请与管理图运行所需的内存及计算流等资源。若重复调用且模型已编译加载，接口直接返回 `SUCCESS`。

该接口执行前需要完成 `GEInitializeV2` 初始化。CompileAndLoad成功后，可使用 `RunModelWithStreamAsync` 接口执行推理。

#### 函数原型<a id="section_compile_and_load_proto"></a>

```cpp
Status CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream)
```

#### 参数说明<a id="section_compile_and_load_params"></a>

<a id="table_compile_and_load"></a>
| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| options | 输入 | 编译选项，key和value类型为 `ge::AscendString`。一般情况下可传入空 `{}`。支持的配置项参见 [options参数说明](https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/options_parameters_description.md)。 |
| stream | 输入 | ACL流指针，通过 `aclrtCreateStream` 创建。若与 `RunModelWithStreamAsync` 配合使用，建议传入有效值且与后续执行使用的流保持一致；若非同一条流，需在加载后调用 `aclrtSynchronizeStream` 完成流同步。可为 `nullptr`，此时使用内部默认流。 |

#### 返回值说明<a id="section_compile_and_load_ret"></a>

<a id="table_compile_and_load_ret"></a>
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`：编译和加载均成功<br>`td::FAILED`：初始化失败（归档解析或Graph构建失败）/ Session创建失败 / AddGraph / CompileGraph / LoadGraph任一失败<br>`td::INVALID_PARAM`：构造函数传入null路径 |

#### 约束说明<a id="section_compile_and_load_constraints"></a>

- 调用前必须完成 `GEInitializeV2` 初始化
- `stream` 与后续 `RunModelWithStreamAsync` 使用的流推荐为同一条；若非同一条，需在 `CompileAndLoad` 后对流调用 `aclrtSynchronizeStream` 完成同步
- 该方法包含编译和加载两个步骤，编译阶段不分配模型运行所需的内存资源，加载阶段负责内存及计算流等资源的申请与管理
- 编译后的模型生命周期与 `EpairModelLoader` 实例绑定，析构时自动清理

---

### RunModel<a id="section_run_model_sync"></a>

#### 头文件/库文件<a id="section_run_model_sync_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

#### 功能说明<a id="section_run_model_sync_desc"></a>

同步运行模型，输出执行结果。接口阻塞直到图执行完成，返回时输出数据已就绪，无需再进行流同步。输入张量数据可位于Host侧或Device侧；输出张量调用方无需分配内存，执行完成后由GE自动分配Host内存赋值返回。该接口需在 `CompileAndLoad` 成功后调用，内部封装GE同步接口 `RunGraph`。

#### 函数原型<a id="section_run_model_sync_proto"></a>

```cpp
Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)
```

#### 参数说明<a id="section_run_model_sync_params"></a>

<a id="table_run_model_sync"></a>
| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 输入张量列表，每个 `gert::Tensor` 需正确设置 `Shape`、`DataType`、`Format`，数据可位于Host侧或Device侧（Device侧通过 `aclrtMalloc` 分配）；SymInt/SymFloat/SymBool 输入必须位于Host侧，否则返回 `INVALID_PARAM`。若数据位于Host侧而图在Device上执行，输入进入数据队列时会对每个输入做一次内存申请和拷贝，输入较大或较多时可能存在性能瓶颈。 |
| outputs | 输出 | 输出张量列表。调用方无需分配内存空间，传入空vector即可，执行完成后由GE自动分配Host内存并赋值返回。 |

#### 返回值说明<a id="section_run_model_sync_ret"></a>

<a id="table_run_model_sync_ret"></a>
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`：运行成功<br>`td::FAILED`：初始化失败（归档解析或Graph构建失败）/ 模型未加载（`CompileAndLoad` 未调用或失败） / `RunGraph` 失败 / 与 `RunModelWithStreamAsync` 混用（同一实例已调用过异步接口，直接报错拒绝）<br>`td::INVALID_PARAM`：构造函数传入null路径 / SymInt/SymFloat/SymBool 输入传入Device侧tensor或输入个数不足 |

#### 约束说明<a id="section_run_model_sync_constraints"></a>

- 必须先调用 `CompileAndLoad` 成功，该接口不会自动完成加载
- 接口同步执行，阻塞直到图执行完成后返回，返回后可直接读取输出，无需调用 `aclrtSynchronizeStream`
- 本接口与 `RunModelWithStreamAsync` 互斥，同一模型不可混用同步与异步接口执行；若同一实例已调用过 `RunModelWithStreamAsync`，本接口将直接返回 `FAILED` 并打印明确日志提示互斥（GE层 `GeSession` 亦有执行模式检查，本接口在 `EpairModelLoader` 层提前拦截给出友好报错）
- 同一实例不支持多线程并发调用，请勿在多个线程中并发使用同一实例
- SymInt/SymFloat/SymBool 输入在图中为Host侧Data节点，对应的输入tensor必须位于Host侧，本接口在执行前校验placement，不满足时返回 `INVALID_PARAM` 并打印明确日志
- 输入Tensor数据位于Device侧时，其Device侧存储地址须32字节对齐，否则可能出现未定义错误
- 输入内存需在调用前由调用方申请，调用后由调用方负责释放
- 输出内存由GE自动分配在Host侧并赋值返回，调用方无需预分配

---

### RunModelWithStreamAsync<a id="section_run_model"></a>

#### 头文件/库文件<a id="section_run_model_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

#### 功能说明<a id="section_run_model_desc"></a>

异步运行模型，输出执行结果。输入和输出张量数据均位于Device侧。该接口需在 `CompileAndLoad` 成功后调用。

#### 函数原型<a id="section_run_model_proto"></a>

```cpp
Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs)
```

#### 参数说明<a id="section_run_model_params"></a>

<a id="table_run_model"></a>
| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| stream | 输入 | 指定图在哪个Stream上运行。 |
| inputs | 输入 | 输入张量列表，每个 `gert::Tensor` 需正确设置 `Shape`、`DataType`、`Format`，数据位于Device侧（通过 `aclrtMalloc` 分配）；SymInt/SymFloat/SymBool 输入必须位于Host侧，否则返回 `INVALID_PARAM`。 |
| outputs | 输出 | 输出张量列表。调用方传入空vector即可，GE使用内置allocator分配Device内存，生命周期与图一致，需在图卸载前（析构 `EpairModelLoader` 前）通过 `outputs.clear()` 主动释放；若传入已分配内存的Tensor，则GE使用该内存存放输出，需要保证申请的输出内存足够写入模型输出数据。 |

#### 返回值说明<a id="section_run_model_ret"></a>

<a id="table_run_model_ret"></a>
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`：运行成功<br>`td::FAILED`：初始化失败（归档解析或Graph构建失败）/ 模型未加载（`CompileAndLoad` 未调用或失败） / `RunGraphWithStreamAsync` 失败 / 与 `RunModel` 混用（同一实例已调用过同步接口，直接报错拒绝）<br>`td::INVALID_PARAM`：构造函数传入null路径 / SymInt/SymFloat/SymBool 输入传入Device侧tensor或输入个数不足 |

#### 约束说明<a id="section_run_model_constraints"></a>

- 必须先调用 `CompileAndLoad` 成功，该接口不会自动完成加载
- 输入和输出Tensor的Device侧存储地址均须32字节对齐，否则可能出现未定义错误
- SymInt/SymFloat/SymBool 输入在图中为Host侧Data节点，对应的输入tensor必须位于Host侧，本接口在执行前校验placement，不满足时返回 `INVALID_PARAM` 并打印明确日志
- 输入内存需在调用前申请，调用后由调用方负责释放
- 若 `outputs` 传空，输出内存由GE内置allocator分配，生命周期与图一致，需在图卸载前通过 `outputs.clear()` 主动释放；当前不支持外置allocator
- 接口异步返回，需通过 `aclrtSynchronizeStream` 对stream完成同步后再读取输出；若希望调用返回后即可直接读取输出（同步语义），可使用 [RunModel](#section_run_model_sync) 接口
- 本接口与 [RunModel](#section_run_model_sync) 互斥，同一模型不可混用同步与异步接口执行；若同一实例已调用过 `RunModel`，本接口将直接返回 `FAILED` 并打印明确日志提示互斥

---

### GetGEGraph<a id="section_get_ge_graph"></a>

#### 头文件/库文件<a id="section_get_ge_graph_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

#### 功能说明<a id="section_get_ge_graph_desc"></a>

获取内部GE Graph的副本。该方法通过 `CopyFrom` 拷贝返回内部构建的GE Graph，调用方可自行将该Graph添加到 `ge::GeSession` 中执行，实现更细粒度的控制（如自行管理Session生命周期、添加多个Graph等）。

#### 函数原型<a id="section_get_ge_graph_proto"></a>

```cpp
Status GetGEGraph(ge::Graph *graph) const
```

#### 参数说明<a id="section_get_ge_graph_params"></a>

<a id="table_get_ge_graph"></a>
| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| graph | 输出 | 输出参数，必须非空。本接口拷贝内部Graph副本返回，调用方获得返回Graph的所有权。 |

#### 返回值说明<a id="section_get_ge_graph_ret"></a>

<a id="table_get_ge_graph_ret"></a>
| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | td::Status | `td::SUCCESS`：获取成功<br>`td::FAILED`：初始化失败（归档解析或Graph构建失败）/ 内部Graph为空 / 拷贝失败<br>`td::INVALID_PARAM`：构造函数传入null路径 |

#### 约束说明<a id="section_get_ge_graph_constraints"></a>

- 构造函数必须成功执行（模型初始化成功）
- `graph` 指针必须非空，必须是有效的Graph对象
- 返回的Graph是副本，调用方拥有所有权
- **输入shape可能与EP JSON不一致**：内部GE Graph构建完成后会执行SymPack优化Pass（合并冗余Pack节点、清理死Data、重排Data index），此时图中Data节点的数量、index、shape均可能与EP JSON中的原始输入定义不同。若需获取优化后实际的输入规格（Data节点数量、index、shape、dtype、placement），应通过 `ge::Graph::GetAllNodes()` 遍历 `type == "Data"` 的节点并读取其 `index` 属性和 `TensorDesc`，或通过GE Dump工具导出图结构查看，**不应直接按EP JSON的输入签名构造 `RunGraph` 的输入**。关于SymPack优化Pass对输入结构的影响及运行时输入适配机制，参见 [SymPack合并优化与AssertTensorMetadata过滤](../features/sympack_merge_20260710.md)

---
## 相关文档<a id="section_references"></a>

- [Python API参考](python_api.md) — Python侧完整API文档
- [构建指南](../guide/build.md) — 环境配置与编译指导
- [epair格式说明](../guide/build_epair.md) — `.epair` 文件结构和用户指南
