---
upstream: 基础需求和约束.md
upstream_version: v0.9
spec_version: 0.1.0
status: draft
module: c_api
depends_on: pipeline+engine_executor/SPEC.md (v0.4.0)
---

# C API 规格（SPEC）

> **模块定位**：C API 是 Engine 的稳定 ABI 边界层，封装 C++ API（`EngineExecutor` / `Context` / `RequestHandle`），以 `extern "C"` 函数 + opaque handle 形式对外暴露。不替代 C++ API，是为跨语言绑定（Python / Rust / Go / Java）和服务化 wrapper 提供**编译器无关、ABI 稳定**的入口。

---

## 第 1 章 定稿目的

定义 C API 的 What / Why：提供什么能力、为什么需要这层、与 C++ API 的边界关系。How（opaque handle 实现、类型映射、内存生命周期）见 DESIGN。

## 第 2 章 范围

### 2.1 纳入

- C 头文件 `c_api.h` 中所有 `extern "C"` 函数签名与类型声明
- 状态码、枚举值与 C++ API 的 1:1 映射
- 内存管理约定（库分配 / 调用方释放）
- 命名约定与 ABI 版本策略

### 2.2 不纳入

- C++ API 内部实现细节（PIMPL / PipelineExecutorBase / Scheduler 等）——见 `docs/pipeline+engine_executor/`
- 底层运行时（IModelManager）——见 `docs/model_executor/`
- 分词器内部（TokenizerBase / BPE / SP）——见 `docs/tokenizer/`
- 多模态 Prompt 的 C 适配（ModalityEmbedding / Tensor handle）——首期范围外，后续扩展
- Python / Rust / Go binding 的具体实现——C API 就绪后独立做

## 第 3 章 期望

在 C++ API 之上补充一层 C ABI 稳定边界：

- **ABI 稳定**：所有导出符号经 `extern "C"` 声明，符号名不随 C++ 编译器版本变化（无 name mangling），跨 GCC / Clang / MSVC 兼容。
- **跨语言绑定**：Python（ctypes / cffi）、Rust（bindgen）、Go（cgo）、Java（JNI）可直接调用 C API，无需 C++ 编译器。
- **零侵入**：C API 是 C++ API 的薄封装，不改变 C++ API 的任何签名或行为；C++ API 保持独立可用。
- **异常安全**：C 边界禁止异常传播，所有 C++ → C 跨界均经 `try/catch(...)` 守卫，异常映射为 `CANN_LM_ENGINE_FAILURE`。
- **内存清晰**：`Result.token_ids` 和 `Result.text` 由库内 `malloc` 分配，调用方经 `CANN_LMEngine_ResultDestroy` 释放；opaque handle 经配对 `Create` / `Destroy` 管理。

## 第 4 章 功能需求

### 4.1 行为规格

C API 的行为语义**与 C++ API 完全一致**（见 `pipeline+engine_executor/SPEC.md` 第 4 章），差异仅在接口表达形式（C 函数 + opaque handle vs C++ 类成员）。

#### 4.1.1 Executor 生命周期

| 行为 | C 函数 | 对应 C++ | 语义 |
|------|--------|----------|------|
| 路径加载构造 | `CANN_LMEngine_ExecutorCreate` | `EngineExecutor(path, path, opts)` | 成功返回 `SUCCESS` + `*out` 指向有效 handle；失败返回 `FAILURE`，`*out = NULL` |
| Buffer 加载构造 | `CANN_LMEngine_ExecutorCreateFromBuffer` | `EngineExecutor(artifact, artifact, opts)` | 同上 |
| 析构 | `CANN_LMEngine_ExecutorDestroy` | `~EngineExecutor()` | 释放 Executor 及其持有的 Context 表；传 NULL 为安全 no-op |

#### 4.1.2 Context 管理

| 行为 | C 函数 | 对应 C++ | 语义 |
|------|--------|----------|------|
| 创建 | `CANN_LMEngine_ContextCreate` | `CreateContext(Context*)` | 成功 `*out` 收到 `ContextId`（非 0），`SUCCESS`；失败返回 C++ 对应状态码，`*out = 0` |
| 销毁 | `CANN_LMEngine_ContextDestroy` | `DestroyContext(ContextId)` | 幂等（已销毁返回 `SUCCESS`） |
| 取消 | `CANN_LMEngine_ContextCancel` | `CancelContext(ContextId)` | 取消该 Context 上所有在飞请求；线程安全 |

> **Context 表示**：C API 直接用 `CANN_LMEngine_ContextId`（`uint64_t`）标识 Context，不引入额外的 opaque Context handle。Context 的内部状态（KV 句柄 / 历史等）由 Executor wrapper 持有的 `Context` C++ 对象承载。

#### 4.1.3 推理

| 行为 | C 函数 | 对应 C++ | 语义 |
|------|--------|----------|------|
| 同步生成 | `CANN_LMEngine_ExecutorGenerate` | `Generate(Prompt, Context, SamplingParams, Result*)` | 阻塞至完成；`Result` 由库分配内部指针 |
| 异步生成 | `CANN_LMEngine_ExecutorGenerateAsync` | `GenerateAsync(...)` | 立即返回 `Request` handle；callback 为 NULL 时非流式 |

> **Prompt**：首期 C API 仅接受 `const char* prompt_text`（纯文本）。多模态 embedding 的 C 适配（需 Tensor handle 设计）留待后续版本。

> **SamplingParams**：C 结构体用指针 + count 表达数组字段（`stop_token_ids` / `stop_strings`），避免 C++ `std::vector`。

#### 4.1.4 Request 查询与控制

| 行为 | C 函数 | 对应 C++ | 语义 |
|------|--------|----------|------|
| 查询状态 | `CANN_LMEngine_RequestGetStatus` | `RequestHandle::GetStatus()` | 返回 `RequestStatus` 枚举；NULL handle 返回 `QUEUED`（安全默认） |
| 是否终态 | `CANN_LMEngine_RequestIsDone` | `RequestHandle::IsDone()` | 返回 `int`（0 = 非终态，1 = 终态） |
| 单请求取消 | `CANN_LMEngine_RequestCancel` | `RequestHandle::Cancel()` | 幂等；NULL handle 返回 `SUCCESS` |
| 阻塞等待 | `CANN_LMEngine_RequestWait` | `RequestHandle::Wait(Result*)` | 终态语义同 C++：`SUCCESS`/`FAILURE` 填 `Result`，`CANCELED` 不填 |
| 非阻塞取 | `CANN_LMEngine_RequestTryGetResult` | `RequestHandle::TryGetResult(Result*)` | 未终态返回 `INVALID_STATE` |
| 析构 | `CANN_LMEngine_RequestDestroy` | `~RequestHandle()` | 释放 handle；传 NULL 为安全 no-op |

#### 4.1.5 Result 内存管理

| 行为 | C 函数 | 语义 |
|------|--------|------|
| 释放 | `CANN_LMEngine_ResultDestroy` | `free(result->token_ids)` + `free(result->text)`，置 NULL；对已清零 Result 为安全 no-op |

> **所有权约定**：`CANN_LMEngine_Result` 是栈分配的 POD struct（调用方声明），但其内部 `token_ids` 和 `text` 指针由库 `malloc` 分配。调用方在使用完毕后**必须**调用 `ResultDestroy` 释放，否则内存泄漏。`finish_reason` 是值类型，无需释放。

### 4.2 流式回调

C API 用 C 函数指针 + `user_data` 替代 C++ `std::function<void(TokenId, std::string)>`：

```c
typedef void (*CANN_LMEngine_StreamCallback)(
    CANN_LMEngine_TokenId  token,
    const char*            text,
    size_t                text_len,
    void*                 user_data
);
```

- `text` 指针仅在回调调用期间有效（指向 C++ `std::string` 内部缓冲），回调返回后可能失效——调用方不应保存该指针。
- 回调在 pipeline 内部线程触发，调用方回调内**不应**阻塞或调用 Engine 同步 API（同 C++ API 约束）。
- `text_len` 保证等于 `strlen(text)`（text 为 null-terminated）。

## 第 5 章 非功能需求

| 指标 | 要求 | 证据来源 |
|------|------|----------|
| ABI 稳定 | **必须**：所有导出符号为 `extern "C"`，符号名不随编译器版本变化 | C API 设计约束 |
| 头文件纯净 | **必须**：`c_api.h` 仅 include `<stddef.h>` / `<stdint.h>`，不依赖任何 C++ 标准库头 | C API 设计约束 |
| 异常安全 | **必须**：C 边界禁止异常传播，所有跨界经 `try/catch(...)` → `FAILURE` | C API 设计约束 |
| 内存安全 | **必须**：`Result` 内部指针配对 `ResultDestroy`；opaque handle 配对 `Destroy`；NULL 参数安全处理 | C API 设计约束 |
| 命名一致 | **必须**：所有函数名以 `CANN_LMEngine_` 前缀开头，按 `Executor` / `Context` / `Request` / `Result` 分类 | C API 设计约束 |
| 状态码一致 | **必须**：C `int` 状态码值与 C++ `Status` 常量值 1:1 对应（0–5） | C API 设计约束 |
| 线程安全 | 同 C++ API：`GenerateAsync` 可多线程并发；同一 ContextId 并发返回 `INVALID_STATE` | pipeline+engine_executor SPEC §5 |
| 延迟 | C API 封装开销**应该**可忽略（一层函数指针转发 + 类型转换） | — |

## 第 6 章 约束与依赖

### 6.1 行为约束

| 约束 | 要求 | 证据来源 |
|------|------|----------|
| ABI 稳定 | 导出符号经 `extern "C"`；符号集合仅增不改不删（向后兼容） | C API 核心约束 |
| 异常边界 | C 函数内 `try/catch(...)` 捕获所有 C++ 异常，映射 `FAILURE`；C 侧无异常概念 | C API 核心约束 |
| 内存约定 | `Result` 内部指针库分配 + 调用方释放；handle 配对 Create/Destroy | C API 核心约束 |
| 头文件隔离 | `c_api.h` 不 include C++ 头；不暴露 C++ 类型（`std::string` / `std::vector` 等） | C API 核心约束 |
| 行为等价 | C API 行为语义与 C++ API 完全一致；差异仅在表达形式 | pipeline+engine_executor SPEC |

### 6.2 错误分类表

| 失败场景 | 返回状态码 | 说明 |
|----------|-----------|------|
| 参数为 NULL | `INVALID_ARGUMENT` (2) | `executor` / `prompt_text` / `out` 等必填参数为 NULL |
| Context 不存在 | `NOT_FOUND` (4) | `context_id` 在 Executor 的 Context 表中未找到 |
| 同一 Context 并发 | `INVALID_STATE` (3) | 同 C++ API |
| 构造失败 | `FAILURE` (1) | 模型加载失败等内部错误 |
| C++ 异常逃逸 | `FAILURE` (1) | `try/catch(...)` 兜底 |
| 取消 | `CANCELED` (5) | 同 C++ API |
| 正常完成 | `SUCCESS` (0) | 同 C++ API |

### 6.3 软件依赖

- C11 编译器（调用方）
- C++17 编译器（库编译侧，`c_api.cpp` 用 C++ 编译）
- 链接 `libcann_lm_engine.a`（同一静态库，同时含 C++ API 和 C API 符号）

## 第 7 章 向后兼容性

### 7.1 ABI 版本策略

- 当前版本为 C API v1（首版）
- **向后兼容原则**：已有函数签名和类型定义不修改不删除；新功能以新增函数 / 新增类型 / 新增枚举值方式扩展
- 如需破坏性变更，新建 v2 头文件（`c_api_v2.h`）或版本前缀（`CANN_LMEngine2_`），不影响 v1 消费者

### 7.2 与 C++ API 版本关系

- C API 跟随 C++ API 版本演进，但 C API 的 ABI 不因 C++ API 的实现变化而破坏
- C++ API 新增方法不自动出现在 C API 中——C API 需显式新增对应 C 函数

## 第 8 章 验证用例

| 用例 ID | Given / When / Then |
|---------|---------------------|
| TC_C_001 | **Given** 纯 C 编译器（`cc`）**When** 编译 `c_api_smoke.c` 并链接 `libcann_lm_engine.a` **Then** 编译链接成功，运行 exit 0 |
| TC_C_002 | **Given** `NULL` 参数 **When** 调用 `ExecutorCreate(NULL, NULL, NULL, NULL)` **Then** 返回 `INVALID_ARGUMENT` |
| TC_C_003 | **Given** 有效路径 **When** 调用 `ExecutorCreate` **Then** 返回 `SUCCESS`，`*out` 非空 |
| TC_C_004 | **Given** stub 实现 **When** 调用 `ContextCreate` **Then** 返回 `FAILURE`（stub 预期） |
| TC_C_005 | **Given** 不存在的 `context_id` **When** 调用 `ExecutorGenerate` **Then** 返回 `NOT_FOUND` |
| TC_C_006 | **Given** 零初始化 `Result` **When** 调用 `ResultDestroy` **Then** 安全 no-op，内部指针保持 NULL |
| TC_C_007 | **Given** `NULL` handle **When** 调用 `RequestGetStatus(NULL)` **Then** 返回 `QUEUED`（安全默认） |
| TC_C_008 | **Given** `NULL` handle **When** 调用 `RequestCancel(NULL)` **Then** 返回 `SUCCESS`（幂等） |
| TC_C_009 | **Given** `NULL` handle **When** 调用 `ExecutorDestroy(NULL)` **Then** 安全 no-op |
| TC_C_010 | **Given** C API 产生 `Result` **When** 读取 `result.text` 和 `result.token_ids` **Then** 数据有效；调用 `ResultDestroy` 后指针变 NULL |

### 8.1 验证环境

- 编译器：GCC / Clang（C + C++ 混合编译）
- 构建系统：CMake（`LANGUAGES C CXX`）
- 验证方式：`c_api_smoke.c`（纯 C 编译的 smoke test，`-Wall -Wextra -Wpedantic`）

---

## 变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 0.1.0 | 2026-08-14 | 初版：C API 规格定稿。定义 15 个 `extern "C"` 函数（Executor 3 + Context 3 + Inference 2 + Request 6 + Result 1）、opaque handle 类型、POD struct 类型、状态码 1:1 映射、内存管理约定（库分配 + 调用方 ResultDestroy）、命名约定（`CANN_LMEngine_` 前缀按 Executor/Context/Request/Result 分类）、ABI 版本策略、10 条验证用例。联动 pipeline+engine_executor SPEC v0.4.0、基础 v0.9。 |
| 0.1.1 | 2026-08-14 | 目录重组：代码路径 `c_api/` → `api/c/`，C++ API `api/` → `api/cpp/`。§3 文件路径、§4 代码示例 include 路径同步。功能不变，ABI 不变。 |
