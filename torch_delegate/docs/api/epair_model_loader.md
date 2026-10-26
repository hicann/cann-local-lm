# EpairModelLoader<a name="ZH-CN_TOPIC_TD_EPAIR_MODEL_LOADER"></a>

## 概览<a name="section_overview"></a>

`EpairModelLoader` 是 TorchDelegate 的核心 C++ 类，封装了 `.epair` 归档文件解析、GE Graph 构建、编译和执行的完整流程。该类使用 PIMPL 模式隔离实现细节，且对外 ABI 稳定。

同时，TorchDelegate 提供 [Python API](python_api.md) 用于模型导出、保存和加载，支持将 PyTorch `ExportedProgram` 转换为 `.epair` 文件。

头文件与库文件随 whl 分发，安装后位于 `<site-packages>/torch_delegate/inc/torch_delegate/` 与 `<site-packages>/torch_delegate/lib/`，CMake 配置包位于 `<site-packages>/torch_delegate/lib/cmake/TorchDelegate/`。C++ 工程通过 `find_package(TorchDelegate)` 接入，前缀由 `torch-delegate-config --cmake-prefix`（或 `torch_delegate.cmake_prefix()`）获取；运行期需手动配置 `LD_LIBRARY_PATH`。详见 [构建指南](../build.md)。

**数据流：**

```
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

## C++ API<a name="section_cpp_api"></a>

### 类索引<a name="section_class_index"></a>

- **[Status 状态码](#status-状态码)**
- **[EpairModelLoader 构造函数](#epairmodelloader-构造函数)**
- **[CompileAndLoad](#compileandload)**
- **[RunModel](#runmodel)**
- **[RunModelWithStreamAsync](#runmodelwithstreamasync)**
- **[GetGEGraph](#getgegraph)**

---

### Status 状态码<a name="section_status"></a>

## 头文件/库文件<a name="section_status_files"></a>

- 头文件：`#include <torch_delegate/td_types.h>`
- 库文件：`libtorch_delegate_backend.so`

## 功能说明<a name="section_status_desc"></a>

`td::Status` 是 `int64_t` 的类型别名，用于表示 TorchDelegate C++ API 的返回值。

## 状态码定义<a name="section_status_def"></a>

<a name="table_status"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="25%" id="mcps1.1.3.1.1">常量</th><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.3.1.2">值</th><th class="cellrowborder" valign="top" width="65%" id="mcps1.1.3.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.3.1.1"><code>td::SUCCESS</code></td><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.3.1.2">0</td><td class="cellrowborder" valign="top" width="65%" headers="mcps1.1.3.1.3">操作成功</td></tr>
<tr><td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.3.1.1"><code>td::FAILED</code></td><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.3.1.2">1</td><td class="cellrowborder" valign="top" width="65%" headers="mcps1.1.3.1.3">通用失败</td></tr>
<tr><td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.3.1.1"><code>td::INVALID_PARAM</code></td><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.3.1.2">2</td><td class="cellrowborder" valign="top" width="65%" headers="mcps1.1.3.1.3">非法参数</td></tr>
<tr><td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.3.1.1"><code>td::INTERNAL_ERROR</code></td><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.3.1.2">3</td><td class="cellrowborder" valign="top" width="65%" headers="mcps1.1.3.1.3">内部错误</td></tr>
<tr><td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.3.1.1"><code>td::UNIMPLEMENTED</code></td><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.3.1.2">4</td><td class="cellrowborder" valign="top" width="65%" headers="mcps1.1.3.1.3">功能未实现</td></tr>
</tbody></table>

---

### EpairModelLoader 构造函数<a name="section_constructor"></a>

## 头文件/库文件<a name="section_constructor_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

## 功能说明<a name="section_constructor_desc"></a>

从 `.epair` 归档文件构造模型加载器。构造过程会立即解析归档内容并构建 GE Graph，但不执行编译。初始化失败时内部记录错误状态，不抛出异常。

## 函数原型<a name="section_constructor_proto"></a>

```
explicit EpairModelLoader(const char *archive_path)
```

## 参数说明<a name="section_constructor_params"></a>

<a name="table_constructor"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="9.68%" id="mcps1.1.4.1.2">输入/输出</th><th class="cellrowborder" valign="top" width="77.54%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">archive_path</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输入</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3"><code>.epair</code> 归档文件的路径，必须非空。支持相对路径和绝对路径，相对路径相对于进程当前工作目录（CWD）解析。</td></tr></tbody></table>

## 约束说明<a name="section_constructor_constraints"></a>

- `archive_path` 必须指向有效的 `.epair` 文件（由 `epair_save()` 生成的 ZIP 归档），支持相对路径和绝对路径
- 构造函数不会抛出异常；初始化失败时，继续调用其他方法均返回非 `SUCCESS` 错误码（`INVALID_PARAM` 或 `FAILED`）

---

### CompileAndLoad<a name="section_compile_and_load"></a>

## 头文件/库文件<a name="section_compile_and_load_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

## 功能说明<a name="section_compile_and_load_desc"></a>

编译并加载模型到指定流，为图执行做准备，包括申请与管理图运行所需的内存及计算流等资源。若重复调用且模型已编译加载，接口直接返回 `SUCCESS`。

该接口执行前需要完成 `GEInitializeV2` 初始化。CompileAndLoad 成功后，可使用 `RunModelWithStreamAsync` 接口执行推理。

## 函数原型<a name="section_compile_and_load_proto"></a>

```
Status CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream)
```

## 参数说明<a name="section_compile_and_load_params"></a>

<a name="table_compile_and_load"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="9.68%" id="mcps1.1.4.1.2">输入/输出</th><th class="cellrowborder" valign="top" width="77.54%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">options</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输入</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">编译选项，key 和 value 类型为 <code>ge::AscendString</code>。一般情况下可传入空 <code>{}</code>。支持的配置项参见 <a href="https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/cpp/ge/options_params/options_parameters_description.md">options参数说明</a>。</td></tr>
<tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">stream</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输入</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">ACL 流指针，通过 <code>aclrtCreateStream</code> 创建。若与 <code>RunModelWithStreamAsync</code> 配合使用，建议传入有效值且与后续执行使用的流保持一致；若非同一条流，需在加载后调用 <code>aclrtSynchronizeStream</code> 完成流同步。可为 <code>nullptr</code>，此时使用内部默认流。</td></tr></tbody></table>

## 返回值说明<a name="section_compile_and_load_ret"></a>

<a name="table_compile_and_load_ret"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.14%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="10.04%" id="mcps1.1.4.1.2">类型</th><th class="cellrowborder" valign="top" width="77.82%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.14%" headers="mcps1.1.4.1.1">-</td><td class="cellrowborder" valign="top" width="10.04%" headers="mcps1.1.4.1.2">td::Status</td><td class="cellrowborder" valign="top" width="77.82%" headers="mcps1.1.4.1.3"><code>td::SUCCESS</code>：编译和加载均成功<br><code>td::FAILED</code>：初始化失败（归档解析或 Graph 构建失败）/ Session 创建失败 / AddGraph / CompileGraph / LoadGraph 任一失败<br><code>td::INVALID_PARAM</code>：构造函数传入 null 路径</td></tr></tbody></table>

## 约束说明<a name="section_compile_and_load_constraints"></a>

- 调用前必须完成 `GEInitializeV2` 初始化
- `stream` 与后续 `RunModelWithStreamAsync` 使用的流推荐为同一条；若非同一条，需在 `CompileAndLoad` 后对流调用 `aclrtSynchronizeStream` 完成同步
- 该方法包含编译和加载两个步骤，编译阶段不分配模型运行所需的内存资源，加载阶段负责内存及计算流等资源的申请与管理
- 编译后的模型生命周期与 `EpairModelLoader` 实例绑定，析构时自动清理

---

### RunModel<a name="section_run_model_sync"></a>

## 头文件/库文件<a name="section_run_model_sync_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

## 功能说明<a name="section_run_model_sync_desc"></a>

同步运行模型，输出执行结果。接口阻塞直到图执行完成，返回时输出数据已就绪，无需再进行流同步。输入张量数据可位于 Host 侧或 Device 侧；输出张量调用方无需分配内存，执行完成后由 GE 自动分配 Host 内存赋值返回。该接口需在 `CompileAndLoad` 成功后调用，内部封装 GE 同步接口 `RunGraph`。

## 函数原型<a name="section_run_model_sync_proto"></a>

```
Status RunModel(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)
```

## 参数说明<a name="section_run_model_sync_params"></a>

<a name="table_run_model_sync"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="9.68%" id="mcps1.1.4.1.2">输入/输出</th><th class="cellrowborder" valign="top" width="77.54%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">inputs</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输入</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">输入张量列表，每个 <code>gert::Tensor</code> 需正确设置 <code>Shape</code>、<code>DataType</code>、<code>Format</code>，数据可位于 Host 侧或 Device 侧（Device 侧通过 <code>aclrtMalloc</code> 分配）。若数据位于 Host 侧而图在 Device 上执行，输入进入数据队列时会对每个输入做一次内存申请和拷贝，输入较大或较多时可能存在性能瓶颈。</td></tr>
<tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">outputs</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输出</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">输出张量列表。调用方无需分配内存空间，传入空 vector 即可，执行完成后由 GE 自动分配 Host 内存并赋值返回。</td></tr></tbody></table>

## 返回值说明<a name="section_run_model_sync_ret"></a>

<a name="table_run_model_sync_ret"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.14%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="10.04%" id="mcps1.1.4.1.2">类型</th><th class="cellrowborder" valign="top" width="77.82%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.14%" headers="mcps1.1.4.1.1">-</td><td class="cellrowborder" valign="top" width="10.04%" headers="mcps1.1.4.1.2">td::Status</td><td class="cellrowborder" valign="top" width="77.82%" headers="mcps1.1.4.1.3"><code>td::SUCCESS</code>：运行成功<br><code>td::FAILED</code>：初始化失败（归档解析或 Graph 构建失败）/ 模型未加载（<code>CompileAndLoad</code> 未调用或失败） / <code>RunGraph</code> 失败 / 与 <code>RunModelWithStreamAsync</code> 混用（同一实例已调用过异步接口，直接报错拒绝）<br><code>td::INVALID_PARAM</code>：构造函数传入 null 路径</td></tr></tbody></table>

## 约束说明<a name="section_run_model_sync_constraints"></a>

- 必须先调用 `CompileAndLoad` 成功，该接口不会自动完成加载
- 接口同步执行，阻塞直到图执行完成后返回，返回后可直接读取输出，无需调用 `aclrtSynchronizeStream`
- 本接口与 `RunModelWithStreamAsync` 互斥，同一模型不可混用同步与异步接口执行；若同一实例已调用过 `RunModelWithStreamAsync`，本接口将直接返回 `FAILED` 并打印明确日志提示互斥（GE 层 `GeSession` 亦有执行模式检查，本接口在 `EpairModelLoader` 层提前拦截给出友好报错）
- 同一实例不支持多线程并发调用，请勿在多个线程中并发使用同一实例
- 输入 Tensor 数据位于 Device 侧时，其 Device 侧存储地址须 32 字节对齐，否则可能出现未定义错误
- 输入内存需在调用前由调用方申请，调用后由调用方负责释放
- 输出内存由 GE 自动分配在 Host 侧并赋值返回，调用方无需预分配

---

### RunModelWithStreamAsync<a name="section_run_model"></a>

## 头文件/库文件<a name="section_run_model_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

## 功能说明<a name="section_run_model_desc"></a>

异步运行模型，输出执行结果。输入和输出张量数据均位于 Device 侧。该接口需在 `CompileAndLoad` 成功后调用。

## 函数原型<a name="section_run_model_proto"></a>

```
Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs)
```

## 参数说明<a name="section_run_model_params"></a>

<a name="table_run_model"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="9.68%" id="mcps1.1.4.1.2">输入/输出</th><th class="cellrowborder" valign="top" width="77.54%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">stream</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输入</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">指定图在哪个 Stream 上运行。</td></tr>
<tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">inputs</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输入</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">输入张量列表，每个 <code>gert::Tensor</code> 需正确设置 <code>Shape</code>、<code>DataType</code>、<code>Format</code>，数据位于 Device 侧（通过 <code>aclrtMalloc</code> 分配）。</td></tr>
<tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">outputs</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输出</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">输出张量列表。调用方传入空 vector 即可，GE 使用内置 allocator 分配 Device 内存，生命周期与图一致，需在图卸载前（析构 <code>EpairModelLoader</code> 前）通过 <code>outputs.clear()</code> 主动释放；若传入已分配内存的 Tensor，则 GE 使用该内存存放输出，需要保证申请的输出内存足够写入模型输出数据。</td></tr></tbody></table>

## 返回值说明<a name="section_run_model_ret"></a>

<a name="table_run_model_ret"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.14%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="10.04%" id="mcps1.1.4.1.2">类型</th><th class="cellrowborder" valign="top" width="77.82%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.14%" headers="mcps1.1.4.1.1">-</td><td class="cellrowborder" valign="top" width="10.04%" headers="mcps1.1.4.1.2">td::Status</td><td class="cellrowborder" valign="top" width="77.82%" headers="mcps1.1.4.1.3"><code>td::SUCCESS</code>：运行成功<br><code>td::FAILED</code>：初始化失败（归档解析或 Graph 构建失败）/ 模型未加载（<code>CompileAndLoad</code> 未调用或失败） / <code>RunGraphWithStreamAsync</code> 失败 / 与 <code>RunModel</code> 混用（同一实例已调用过同步接口，直接报错拒绝）<br><code>td::INVALID_PARAM</code>：构造函数传入 null 路径</td></tr></tbody></table>

## 约束说明<a name="section_run_model_constraints"></a>

- 必须先调用 `CompileAndLoad` 成功，该接口不会自动完成加载
- 输入和输出 Tensor 的 Device 侧存储地址均须 32 字节对齐，否则可能出现未定义错误
- 输入内存需在调用前申请，调用后由调用方负责释放
- 若 `outputs` 传空，输出内存由 GE 内置 allocator 分配，生命周期与图一致，需在图卸载前通过 `outputs.clear()` 主动释放；当前不支持外置 allocator
- 接口异步返回，需通过 `aclrtSynchronizeStream` 对 stream 完成同步后再读取输出；若希望调用返回后即可直接读取输出（同步语义），可使用 [RunModel](#runmodel) 接口
- 本接口与 [RunModel](#runmodel) 互斥，同一模型不可混用同步与异步接口执行；若同一实例已调用过 `RunModel`，本接口将直接返回 `FAILED` 并打印明确日志提示互斥

---

### GetGEGraph<a name="section_get_ge_graph"></a>

## 头文件/库文件<a name="section_get_ge_graph_files"></a>

- 头文件：`#include <torch_delegate/epair_model_loader.h>`
- 库文件：`libtorch_delegate_backend.so`

## 功能说明<a name="section_get_ge_graph_desc"></a>

获取内部 GE Graph 的副本。该方法通过 `CopyFrom` 拷贝返回内部构建的 GE Graph，调用方可自行将该 Graph 添加到 `ge::GeSession` 中执行，实现更细粒度的控制（如自行管理 Session 生命周期、添加多个 Graph 等）。

## 函数原型<a name="section_get_ge_graph_proto"></a>

```
Status GetGEGraph(ge::Graph *graph) const
```

## 参数说明<a name="section_get_ge_graph_params"></a>

<a name="table_get_ge_graph"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="9.68%" id="mcps1.1.4.1.2">输入/输出</th><th class="cellrowborder" valign="top" width="77.54%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1">graph</td><td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2">输出</td><td class="cellrowborder" valign="top" width="77.54%" headers="mcps1.1.4.1.3">输出参数，必须非空。本接口拷贝内部 Graph 副本返回，调用方获得返回 Graph 的所有权。</td></tr></tbody></table>

## 返回值说明<a name="section_get_ge_graph_ret"></a>

<a name="table_get_ge_graph_ret"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="12.14%" id="mcps1.1.4.1.1">参数名</th><th class="cellrowborder" valign="top" width="10.04%" id="mcps1.1.4.1.2">类型</th><th class="cellrowborder" valign="top" width="77.82%" id="mcps1.1.4.1.3">说明</th></tr></thead>
<tbody><tr><td class="cellrowborder" valign="top" width="12.14%" headers="mcps1.1.4.1.1">-</td><td class="cellrowborder" valign="top" width="10.04%" headers="mcps1.1.4.1.2">td::Status</td><td class="cellrowborder" valign="top" width="77.82%" headers="mcps1.1.4.1.3"><code>td::SUCCESS</code>：获取成功<br><code>td::FAILED</code>：初始化失败（归档解析或 Graph 构建失败）/ 内部 Graph 为空 / 拷贝失败<br><code>td::INVALID_PARAM</code>：构造函数传入 null 路径</td></tr></tbody></table>

## 约束说明<a name="section_get_ge_graph_constraints"></a>

- 构造函数必须成功执行（模型初始化成功）
- `graph` 指针必须非空，必须是有效的 Graph 对象
- 返回的 Graph 是副本，调用方拥有所有权
- **输入 shape 可能与 EP JSON 不一致**：内部 GE Graph 构建完成后会执行 SymPack 优化 Pass（合并冗余 Pack 节点、清理死 Data、重排 Data index），此时图中 Data 节点的数量、index、shape 均可能与 EP JSON 中的原始输入定义不同。若需获取优化后实际的输入规格（Data 节点数量、index、shape、dtype、placement），应通过 `ge::Graph::GetAllNodes()` 遍历 `type == "Data"` 的节点并读取其 `index` 属性和 `TensorDesc`，或通过 GE Dump 工具导出图结构查看，**不应直接按 EP JSON 的输入签名构造 `RunGraph` 的输入**。关于 SymPack 优化 Pass 对输入结构的影响及运行时输入适配机制，参见 [SymPack 合并优化与 AssertTensorMetadata 过滤](../features/20260710v1_sympack-merge-and-assert-tensor-metadata.md)

---
## 相关文档<a name="section_references"></a>

- [Python API 参考](python_api.md) — Python 侧完整 API 文档
- [构建指南](../build.md) — 环境配置与编译指导
- [epair 格式说明](../build_epair.md) — `.epair` 文件结构和用户指南
