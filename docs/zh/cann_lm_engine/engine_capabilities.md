---
module: cann_lm_engine
status: draft
depends_on: c_api/SPEC.md, pipeline+engine_executor/SPEC.md
---

# Engine 能力清单（以 C API 为入口）

> **文档定位**：罗列 Engine 当前（截至 2026-11）已支持的能力，统一以 C API（`cann_lm/engine/api/c/c_api.h`）为入口视角说明——每个能力点对应哪个 C 函数 / 结构体字段 / meta_json 配置，以及关键行为语义。接口签名细节见 `c_api/SPEC.md`；流水线内部机制见 `pipeline+engine_executor/` 与 `lm_engine_inference_flow.md`。
>
> ⚠ C ABI 处于 pre-release，布局尚无兼容承诺（头文件 ABI CHANGELOG 为准）；本文随代码演进维护。

---

## 1. C API 总览

| 分类 | C 函数 | 语义 |
|------|--------|------|
| Executor 生命周期 | `CANN_LMEngine_ExecutorCreate` / `ExecutorDestroy` | 构造（单一入口，路径/buffer 合一）与析构 |
| Context 管理 | `CANN_LMEngine_ContextCreate` / `ContextDestroy` / `ExecutorSetContext` / `ContextCancel` | 会话创建 / 销毁 / 显式激活 / 级联取消 |
| 推理 | `CANN_LMEngine_ExecutorGenerate` / `ExecutorGenerateAsync` | 同步生成 / 异步生成（含流式回调） |
| 请求管理 | `RequestGetStatus` / `RequestIsDone` / `RequestCancel` / `RequestWait` / `RequestTryGetResult` / `RequestDestroy` | 异步请求的观测、等待、取消与释放 |
| 结果内存 | `CANN_LMEngine_ResultDestroy` | 释放库分配的 `token_ids` / `text` |

关键约定：

- **状态码**：`SUCCESS=0` / `FAILURE=1` / `INVALID_ARGUMENT=2` / `INVALID_STATE=3` / `NOT_FOUND=4` / `CANCELED=5`，与 C++ `Status` 常量 1:1。
- **内存归属**：`Result.token_ids` / `Result.text` 由库内 `malloc` 分配，调用方经 `ResultDestroy` 释放（零初始化/重复销毁均为安全 no-op）；opaque handle（Executor/Request）经配对 Create/Destroy 管理。
- **Result 填充契约**：仅当返回 `SUCCESS` 或 `FAILURE` 时 `Result` 携带（部分）内容，其余错误码下 `Result` 已被清零。
- **异常安全**：C 边界无异常穿越，所有跨界 `try/catch(...)` 兜底为 `FAILURE` 并落日志。
- **请求归属**（ADR-0002）：`Generate` / `GenerateAsync` 无 context 参数，请求一律归属 Executor **当前活动 Context**；无活动 Context → `INVALID_STATE`。
- **线程安全**：`CancelContext` / `RequestCancel` 可与该 Context 在飞请求并发调用；流式回调在 pipeline 内部线程触发，回调内不应阻塞或调 Engine 同步 API。

---

## 2. 模型加载（ExecutorCreate）

### 2.1 构件描述：`CANN_LMEngine_ModelArtifact`

| 字段 | 能力 |
|------|------|
| `prefill_omc_path` / `prefill_omc_buf` | prefill 图（路径 / 内存 buffer 二选一，**buffer 优先**） |
| `decode_omc_path` / `decode_omc_buf` | decode 图（同上） |
| `prefill_weight` / `decode_weight` | prefill / decode 权重路径（图与权重分离，`NULL` → 空串） |
| `meta_json_path` / `meta_json_buf` | meta_json 配置（路径 / buffer，buffer 优先） |
| `isEnableSharedWeight` | 是否使能 prefill/decode 权重共享绑定（`false` = decode 不绑定，`share_weights_model_id` 保持 0） |

- **PD 分离双图**：prefill / decode 双执行器各 Load 各自图位与权重（SR20260903001 合并构件形态）。
- **单一构造入口**：路径与内存 buffer 两种形态合一，`ExecutorCreateFromBuffer` 已并入。
- **加载失败即终止**：图位已配置而加载失败（含权重 InitWeights，如设备端 NPU 配额耗尽）→ 构造内终止初始化并完成卸载，`*out` 保持 NULL，失败码原样上抛——不交出半初始化实例。
- **无模型形态**：双图位全空时构造成功，生命周期 API 可用、推理报错（stub/host 部署形态）。

### 2.2 引擎选项：`CANN_LMEngine_EngineOptions`

| 字段 | C++ 缺省 | 能力 |
|------|----------|------|
| `default_chunk_size` | 1024 | prefill 切块大小（token 数）；实际取值链：meta `model_config.prefill_len[0]` > 本字段（>0 才采信）> 1024 |
| `max_concurrent_prefill` | 1 | 并发 prefill 上限；当前调度实现尚未消费该值（预留字段，行为恒单并发 prefill） |
| `max_batch_size_override` | 0 | decode 批容量覆盖；实际取值链：meta `model_config.batch_num` > 本字段（>0 才采信）> 4 |
| `tokenizer_type` | 空=BPE | tokenizer 算法选择："BPE" \| "Qwen" \| "SPM"；未知值 WARN 后回退 BPE |
| `tokenizer_path` | 空 | tokenizer.json 词表产物路径；非空时优先于 meta_json `tokenizer_config.tokenizer_file`；不可加载 → tokenizer 置空（Generate 快速失败） |

> C 侧无缺省初始化机制：结构体零初始化（`{0}`）时 int 字段为 0，按"未设置"走上述回退链。

---

## 3. 会话与 KV Cache（Context 管理）

| 能力 | 入口 | 语义 |
|------|------|------|
| 创建会话 | `ContextCreate` | 每 Context 独立一套 KV Cache 池 + 缓存台账（prompt 前缀复用记账）；KV tensor 在此点实建（all-or-nothing，任一层失败 Context 不诞生） |
| 显式激活 | `ExecutorSetContext` | 切换活动 Context（ADR-0002）：未注册 → `NOT_FOUND`；Executor 全局有在飞请求 → `INVALID_STATE`；目标即当前活动 → 幂等 `SUCCESS`。切换链锁内原子完成（模板池转发登记 + 双执行器 KV 工位刷新 + assembler 记账同步） |
| 销毁会话 | `ContextDestroy` | 释放对应 KV 句柄；幂等（已销毁再调返回 `SUCCESS`）；有在飞请求 → `INVALID_STATE`（先 Cancel）；销毁活动 Context → 允许，回无活动态（不隐式 fallback） |
| 级联取消 | `ContextCancel` | 取消该 Context 上所有在飞请求，可与在飞请求并发调用 |

支撑的 KV Cache 能力面：

- **层类型感知**：`meta_json model_config.layer_types` 支持 `full_attention` / `linear_attention` / `sliding_attention`（gemma4 滑窗层入 KV 池、末维 = sliding_window；Qwen3.5 线性层走独立 LinearStatePool）。
- **容量/形态**：`kv_cache_max_len`、`num_attention_kv_heads`、`num_attention_head_dims`、`kv_cache_type`（缺省/未知回退 FLOAT16）、`sliding_window`。
- **多轮前缀复用（PromptCache 缓存台账）**：同 Context 下一轮请求与台账 LCP 前缀比对，命中段免重算续写（客户端每轮发全量文本）；失败/取消请求按快照回退台账（R10）；**linear_attention 模型整体禁用该 feature（R3 门控）**。
- **权重共享**：`isEnableSharedWeight = true` 时 prefill/decode 权重共享绑定。

---

## 4. 推理（Generate / GenerateAsync）

### 4.1 两种调用形态

| 入口 | 形态 | 行为 |
|------|------|------|
| `ExecutorGenerate` | 同步 | 阻塞至完成，`CANN_LMEngine_Result` 直出完整结果 |
| `ExecutorGenerateAsync` | 异步 | 立即返回 `CANN_LMEngine_Request` 句柄；decode 池满时入队等待（不返回失败） |

Prompt 输入当前为**纯文本** `const char* prompt_text`（多模态 embedding 的 C 适配未在 C ABI 暴露）。

### 4.2 采样能力：`CANN_LMEngine_SamplingParams`

| 旋钮 | 关闭值 | 能力 |
|------|--------|------|
| `temperature` | 1.0 | logits 温度缩放；0 经钳制退化为贪心 |
| `top_k` | 0 | top-k 截断（只留概率最高 k 个） |
| `top_p` | 1.0 | 核采样（累计概率截尾，保留最高概率项） |
| `seed` | 0 | 非 0 时 per-call 建 seeded RNG，可复现采样 |
| `repetition_penalty` | 1.0 | >1 对本请求**已生成** token 降权（不含 prompt/跨轮历史） |

映射规则：默认参数 → 贪心（含 argmax 快路径）；`temperature != 1` 或 `top_k > 0` 或 `top_p ∈ (0,1)` 任一成立 → 概率采样；越界旋钮值视为关闭。logits 支持 FLOAT32 / FLOAT16。

### 4.3 终止语义：`CANN_LMEngine_RequestOptions`

| 字段 | 缺省 | 能力 |
|------|------|------|
| `max_tokens` | 0/负 → 1024 | 生成 token 数上限 |
| `stop_token_ids` + count | NULL | 用户显式停止 token 集（归因优先） |
| `stop_strings` + count | NULL | 停止字符串：命中即终止（`FinishReason::StopString`），并从输出文本中**剥离停止部分** |

停止检查链（先到先停）：`stop_token_ids` → meta `tokenizer_config.eos_token_id`（eos 命中 → `StopToken`；`token_ids` 含 eos、`text` 不含、不入台账）→ `stop_strings` → `max_tokens`。

`FinishReason` 枚举：`MaxTokens` / `StopToken` / `StopString` / `Canceled` / `Error`。

### 4.4 流式回调

`CANN_LMEngine_StreamCallback`：异步生成时逐 token 触发（token id + 增量文本），回调内 `text` 仅在回调期间有效。callback 传 NULL = 非流式。

### 4.5 调度执行能力（内部，经构造参数/meta_json 配置）

- **chunked prefill**：长 prompt 按 chunk size 切块（优先级链：meta `prefill_len[0]` > `default_chunk_size` > 1024），prefill 与 decode 交错。
- **连续批调度（continuous batching）**：FIFO 任务队列 + decode 池（容量 = `max_batch_size`：meta `batch_num` > override > 4）+ iteration 级组批，终态请求即时出池补位。
- **取消**：请求级（`RequestCancel` / `Scheduler.Cancel`）与 Context 级两级；取消即时生效（当前 iteration 后退出池，部分 token 丢弃）。

---

## 5. 异步请求管理（Request Handle）

| 入口 | 能力 |
|------|------|
| `RequestGetStatus` | 请求状态机：`Queued` → `Prefilling` → `Decoding` → `Completed` / `Canceled` / `Failed` |
| `RequestIsDone` | 终态判断（1/0） |
| `RequestCancel` | 取消单请求（NULL 容忍，返回 SUCCESS） |
| `RequestWait` | 阻塞等待至完成并取 `Result` |
| `RequestTryGetResult` | 非阻塞取结果（未完成时不覆盖 out） |
| `RequestDestroy` | 释放请求句柄 |

---

## 6. 分词能力（Tokenizer）

- **算法**：BPE / Qwen / SPM（经 `EngineOptions.tokenizer_type` 选择，取代原硬编码 BPE；未知值 WARN 回退 BPE）。
- **词表来源优先级**：`EngineOptions.tokenizer_path` > meta_json `tokenizer_config.tokenizer_file`；未供给 → tokenizer 置空，`Generate` 快速失败。
- **special token 语义**：meta_json `tokenizer_config` 的 `bos/eos/special_tokens` + 跨段 `model_config.pad_token_id`；eos 停止判定与词表加载成败解耦（special 通道独立）。
- **词表缓存（可选性能开关）**：`CANN_LM_TOKENIZER_BINCACHE` env 显式开启 tokenizer.json → .bincache sidecar 加速装载（默认关，见 perf_optimize.md §3.4）。

---

## 7. 可观测性与调试

- **DFX 采集（DfxCollector）**：forward 计时（timing）与 IO dump 能力，随 forward 驱动采集。
- **张量调试 dump（env 开关，默认关）**：`CANN_LM_DEBUG_DUMP_TENSORS` / `CANN_LM_DEBUG_DUMP_KV` / `CANN_LM_DEBUG_DUMP_DIR`（缺省 `/data/local/tmp/dump/`）。
- **结构化日志**：`CANN_LLM_LOG*` 系列宏（错误/告警/信息分级；异常兜底与初始化失败均落日志）。
- **性能埋点**：prefill 整体耗时（首 chunk 起步 → 末块耗尽）、PromptCache 复用/换绑日志（`promptcache.rebind`）等。

---

## 8. 跨语言绑定

- **Python**：`cann_lm/engine/python/cann_lm`（ctypes 直连本 C ABI，提供 `Engine` / `Context` / `Request` / `Result` 包装与 CLI/chat 入口）；库加载惰性，`CANN_LM_NDK_LIB_DIR` 可注入宿主 stub。
- 其余语言（Rust/Go/Java）按 c_api/SPEC.md 的绑定策略独立开展。

---

## 9. 已知限制与不承诺项

- Prompt 仅纯文本；多模态 `ModalityEmbedding` 未出 C ABI。
- `ContextOptions` / `ConfigureContext` 为空占位（C ABI 未暴露运行期调参入口）。
- `PrecomputedTopK`（NPU 预算 top-k 加速路径）为算法层内部通道，不对 C ABI 暴露。
- 缓存台账当前单 batch（单台账×单工位）；多 batch 台账为遗留事项（tmp_docs/AR202610211451.md §7）。
- C ABI pre-release：结构体布局（含 `ModelArtifact` 尾部追加字段）不承诺兼容，外部绑定须按当前头文件重新生成。
- 采样 logits 暂支持 FLOAT32 / FLOAT16；模型输出 dtype 超出该集时采样报 `INVALID_ARGUMENT`。

---

## 10. 典型调用序列（速查）

```c
// 1) 构造（双图 + 双权重 + meta_json）
CANN_LMEngine_ModelArtifact art = {0};
art.prefill_omc_path = ".../prefill.omc";
art.decode_omc_path  = ".../decode.omc";
art.prefill_weight   = ".../prefill.weight";
art.decode_weight    = ".../decode.weight";
art.meta_json_path   = ".../meta.json";
CANN_LMEngine_EngineOptions opts = {0};
CANN_LMEngine_Executor* eng = NULL;
CANN_LMEngine_ExecutorCreate(&art, &opts, &eng);   // 失败 → eng 保持 NULL

// 2) 建会话并激活
CANN_LMEngine_ContextId ctx = 0;
CANN_LMEngine_ContextCreate(eng, &ctx);
CANN_LMEngine_ExecutorSetContext(eng, ctx);

// 3a) 同步生成（采样 + 终止选项）
CANN_LMEngine_SamplingParams sp = { .temperature = 0.7f, .top_k = 50, .top_p = 0.9f };
CANN_LMEngine_RequestOptions ro = { .max_tokens = 256 };
CANN_LMEngine_Result res;
CANN_LMEngine_ExecutorGenerate(eng, "你好", &sp, &ro, &res);
// res.token_ids / res.text / res.finish_reason
CANN_LMEngine_ResultDestroy(&res);

// 3b) 异步 + 流式
CANN_LMEngine_Request* req = NULL;
CANN_LMEngine_ExecutorGenerateAsync(eng, "你好", &sp, &ro, &req, on_token, user_data);
CANN_LMEngine_RequestWait(req, &res);              // 或 TryGetResult 轮询
CANN_LMEngine_ResultDestroy(&res);
CANN_LMEngine_RequestDestroy(req);

// 4) 收尾
CANN_LMEngine_ContextCancel(eng, ctx);             // 可选：级联取消在飞请求
CANN_LMEngine_ContextDestroy(eng, ctx);
CANN_LMEngine_ExecutorDestroy(eng);
```

> 多轮对话：同一 Context 内客户端每轮发**全量**对话文本，引擎经缓存台账自动前缀复用；切换会话先 `ExecutorSetContext`（要求 Executor 空闲）。
