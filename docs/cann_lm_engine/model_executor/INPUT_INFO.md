# ModelExecutor 模型输入信息

> 来源：backend `IBuiltModel::GetInputDescs()` + om 图输入定义 + 代码实现 + 用户确认。
> 本文档是 ModelExecutor 输入填充设计的**事实来源**（SPEC/DESIGN 引用本文档）。

## 1. 模型输入 tensor 组成

模型 om 图有多个输入 tensor，由 `IBuiltModel::GetInputDescs()` 返回 `vector<TensorDesc>`。每个 `TensorDesc` 含 `name`、`dataType`、`format`、`dims`、`size`。

ModelExecutor 在 Load 时据 descs 经 `TensorManager::Instance().CreateTensor(desc, modelId, &out)` 申请 `input_tensors_`（`vector<shared_ptr<ITensorBuffer>>`）。

**工位排序未定**——每个 omc 可能不同，需要从 `TensorDesc.name` 中确认索引。

### 1.1 输入 tensor 工位（6 个，全部在 input_tensors_ 中）

| # | 名称 | 类型 | 维度 | 填充者 | 填充时机 | 说明 |
|---|------|------|------|--------|---------|------|
| 1 | input_embed | **按 TensorDesc.dataType**（随路径变化，见 §2.1） | `[batch, seqLen, hidden]`（TokenId 路径为 `[batch, seqLen]`） | TextEmbedding::FillInputs（SR20260828001 接线，Scheduler 两输入准备点） | Forward 前 | embedding 输入（tokenId 路径走模型内部 embedding） |
| 2 | embed_scales | FLOAT32 | PerToken：`batch × seqLen × 1`；PerGroup：`batch × seqLen × groups` | embedding 模块 | 输入准备时 | embedding 反量化参数（2026-08-29 已实现，SR20260828001） |
| 3 | attention_mask | FLOAT32 | `batch * 1 * seqLen * kvCacheLen` | InputAssembler | Forward 内 | 注意力矩阵：可见=0，不可见=-FLT_MAX |
| 4 | position_ids | INT32 | `batch * seqLen` | InputAssembler | Forward 内 | 当前 tokenId 对应位置信息（位置编码） |
| 5 | new_kv_cache_pos | INT32 | `[batch, seqLen]` | InputAssembler | Forward 内 | 指导模型生成下一 kvCache 的位置（per-slot），可能绕回 |
| 6 | kvCache | — | `[numLayers, max_batch, max_seqLen, 2(K&V), num_heads, head_dim]` | 共享池 | — | KV Cache 池 buffer（经 TensorManager 申请，作为 input_tensors_ 元素） |

### 1.2 输出 tensor

| # | 名称 | 类型 | 维度 | 填充者 |
|---|------|------|------|--------|
| 1 | logits | FLOAT32 | `[batch, vocabSize]` | ModelManager::Run 填充 |

## 2. 各工位填充规则

### 2.1 input_embed

- **填充者**：`TextEmbedding::FillInputs`（SR20260828001 §2.2 接线点——Scheduler 的 prefill/decode 两输入准备点调用；经 C-02 只读 accessor `InputTensors()`/`InputNameIndex()` 定位工位）
- **写入类型**：以该工位 `TensorDesc.dataType` 为准（D-06），非文件存储类型：
  - TokenId 路径：INT32 / INT64（token id 值写入）
  - Embedding 路径：FLOAT16 / FLOAT32 / INT8（低精度行值展开转换，不乘 scale——反量化在 NPU）
- **数据来源**：
  - embedding 路径：TextEmbedding host gather 低精度行（InMemory 内存拷贝 / InFlash 按行 pread）
  - tokenId 路径：tokenId 序列，走模型内部 embedding（om 图承担查表）
- **SetInputData 现状**：保留为"外部直填通道"API（memcpy 语义不变），SR20260828001 接线后仓内零调用
- **注意**：真实后端需按 batch 维拼装；当前 stub 仅写首个 slot（M5 遗留）

### 2.2 embed_scales

- **填充者**：embedding 模块（`TextEmbedding::FillInputs` Embedding 路径，scale_type ≠ None 时）
- **说明**：embedding 反量化参数，输入准备时填入；scale 恒内存态（模块 Load 期加载）
- **维度**：PerToken `batch × seqLen × 1`；PerGroup `batch × seqLen × groups`（groups = hidden/group_size）
- **类型**：FLOAT32（scale 存储为 Fp16 时模块做值展开转换）
- **状态**：**已实现**（2026-08-29，SR20260828001；此前"搁置"记录已过时）

### 2.3 attention_mask

- **填充者**：InputAssembler
- **维度**：`batch * 1 * seqLen * kvCacheLen`（4 维，第 2 维固定为 1）
  - prefill：`batch=1, seqLen=chunk_len, kvCacheLen=cursor_before + chunk_len`
  - decode：`batch=active_count, seqLen=1, kvCacheLen=max(cursor_i + 1)`
- **填充规则**（causal）：
  - `mask[b][0][i][j] = 0.0` if `j <= cursor_b_i`（当前位置及之前，可见）
  - `mask[b][0][i][j] = -FLT_MAX` if `j > cursor_b_i`（之后，遮蔽）
- **padding 屏蔽**：decode batch 内不同 cursor 的 slot 用 `max_kv_len = max(cursor_i + 1)` 取 max；短 slot 尾部置 -FLT_MAX
- **数据类型**：FLOAT32

### 2.4 position_ids

- **填充者**：InputAssembler
- **维度**：`batch * seqLen`
- **推导规则**（位置从 KV 池获取）：
  - prefill：`pos[i] = cursor_start + i`（i=0..chunk_len-1）
  - decode：`pos[b][0] = cursor_b`（各 active slot 的当前光标值）
- **数据类型**：INT32

### 2.5 new_kv_cache_pos

- **填充者**：InputAssembler（位置信息从 KV 池获取）
- **维度**：`[batch, seqLen]`（与 position_ids 同维，但语义不同）
- **说明**：指导模型生成下一 kvCache 的位置——当前 token 写入 KV 池的位置
- **与 position_ids 的区别**（不可合并，分别填入不同工位）：
  - `position_ids`：`batch × seqLen`，用于位置编码
  - `new_kv_cache_pos`：`batch × seqLen`，用于 KV 池写入位置指导，**可能绕回**
  - 两者当前实现取相同值（均为各 slot cursor 推导），但语义独立、工位不同，不可合并
- **绕回逻辑**：归 KV 池管理——KV 池管理光标位置，通过 KV 池获取下一 KV 位置（含绕回），提供给 InputAssembler
- **填充时序**：光标在 `Advance` **前**读取（先从 KV 池获取位置，填入 tensor，再推进光标）

### 2.6 kvCache

- **填充者**：共享 KV 池（`KvCachePool`）
- **申请方式**：经 `TensorManager::Instance().CreateTensor(desc, modelId, &out)` 申请——**作为 input_tensors_ 数组元素**，随其他输入一起提交给模型推理
- **绑定方式**：Load 时据 `IBuiltModel::GetInputDescs()` 中名为 "kvCache" 的 desc 申请 ITensorBuffer；该 buffer 底层指向 KV 池的共享内存（layer-major 排布）
- **说明**：prefill/decode 两实例共享同一池零拷贝；池内存由 TensorManager 分配

## 3. 填充流程（Forward 内部）

```
Forward(opts):
  1. FillInputs(...)                  → 填 input_embed（+ Embedding 路径的 embed_scales）；
                                        Scheduler 输入准备点调 TextEmbedding::FillInputs
                                        （stub 时代 name_index 为空 → NOT_FOUND 忽略，同旧 SetInputData 行为）
  2. AssemblePrefill/Decode           → 从 KV 池获取光标位置 → 填 position_ids + new_kv_cache_pos + attention_mask → 推进光标
  3. mm->Run(input_tensors_, output_tensors_)  → MM 读全部 inputs（含 kvCache）、填 output[0]（logits）
  4. Logits()                         → 返回 output_tensors_[0]
```

### 3.1 prefill 路径（`chunk_len > 1`）

```
cursor_before = pool->NextPosition(slot)           // 从 KV 池获取
position_ids = [cursor_before, cursor_before+1, ..., cursor_before+chunk_len-1]
new_kv_cache_pos = position_ids（[1, chunk_len]，相同值；batch=1）
kvCacheLen = cursor_before + chunk_len
mask[0][0][i][j] = 0.0 if j <= cursor_before+i else -FLT_MAX
pool->Advance(slot, chunk_len)                     // 推进光标（KV 池管理绕回）
```

### 3.2 decode 路径（`chunk_len == 1`）

```
cursors[i] = pool->NextPosition(slots[i])          // 从 KV 池获取（含绕回）
max_kv_len = max(cursors[i] + 1)
position_ids[i][0] = cursors[i]
new_kv_cache_pos[i][0] = cursors[i]                // per-slot [batch, 1]
mask[i][0][0][j] = 0.0 if j <= cursors[i] else -FLT_MAX
pool->Advance(slots[i], 1)                        // 各 active slot 推进光标
```

## 4. name → index 映射

Load 时建立 `TensorDesc.name → input_tensors_ index` 映射表。InputAssembler 按 name 查找对应 tensor 填充（不再硬编码 `[1]`/`[2]`）。

```cpp
// ModelExecutor Load 后建立
std::unordered_map<std::string, size_t> input_name_index_;
for (size_t i = 0; i < in_descs.size(); ++i) {
    input_name_index_[in_descs[i].name] = i;
}
```

InputAssembler 接收 name→index 映射，按 name 查找 `position_ids` / `new_kv_cache_pos` / `attention_mask` 工位。

## 5. TensorDesc → ITensorBuffer 映射

| TensorDesc 字段 | ITensorBuffer 接口 | 说明 |
|----------------|-------------------|------|
| `name` | `GetDesc().name` | om 图输入张量名（如 "input_embed"） |
| `dataType` | `GetDesc().dataType` | FLOAT16 / FLOAT32 / INT32（backend `cann_llm_engine::DataType`） |
| `format` | `GetDesc().format` | ND（本模型所有输入为 ND 格式） |
| `dims` | `GetDesc().dims` | int64_t 维度数组 |
| `size` | `GetSize()` | 字节大小（0=动态 shape，TensorBufferImpl 默认分配 1024） |

## 6. 已知缺口（关联遗留项）

| 遗留 # | 关联输入 | 问题 |
|---------|---------|------|
| #4 | 所有 | `TensorDesc.size==0` 时分配 1024 字节（动态 shape） |
| #5 | 所有 | `malloc` 未初始化 → 未填区域为堆垃圾 |
| #6 | input_embed | `SetInputData` 不检查 src null（现仓内零调用，仅外部直填通道保留） |
| M4 | input_embed | prefill/decode 靠 chunk_len 启发式区分 |
| M5 | input_embed | decode batch>1 仅写首个 slot |
| — | kvCache | 需确认 KV 池 buffer 如何与 ITensorBuffer 关联（池内存指向） |

## 7. 版本

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1 | 2026-08-26 | 初版：6 输入工位 + 填充规则 + 流程 + name→index 映射 |
| 0.2 | 2026-08-26 | 用户确认：kvCache 也是 input_tensors_ 元素（经 TensorManager 申请）；position_ids 与 new_kv_cache_pos 不可合并；attention_mask 4 维（第 2 维=1）；绕回归 KV 池；embed_scales 由 embedding 模块填（搁置） |
| 0.3 | 2026-08-26 | review v0.13-M1 确认：`new_kv_cache_pos` 维度由 1D `[seqLen]` 改为 `[batch, seqLen]`（per-slot）。decode 各 slot 各自 cursor（非共享 cursors[0]）。§1.1/§2.5/§3.1/§3.2 同步。 |
| 0.4 | 2026-08-29 | U-4 订正（随 TextEmbedding 实现/SR20260828001 落地）：input_embed 类型改"按 TensorDesc.dataType 随路径变化"（旧记 FLOAT16 为 stub 占位）；embed_scales 维度 PerGroup 补 `×groups`（旧记 `×1` 仅 PerToken 成立）、状态由"搁置"改"已实现"；input_embed 填充者由 SetInputData 改为 TextEmbedding::FillInputs（Scheduler 两输入准备点接线）；§3 流程步骤 1 与 §6 遗留表同步。 |
