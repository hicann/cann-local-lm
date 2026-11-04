# Gemma4 E2B-IT导出方案：静态形状与动态形状

> **说明**：本文档描述Gemma4 E2B-IT的两种导出方案：
> - **静态形状方案**（第1-13节）：使用固定的prefill/decode方法，显式管理state tensor
> - **动态形状方案**（第14节）：使用单一方法，通过dynamic shape支持可变序列长度
>
> 两种方案均已在MMLU评测中验证，可根据实际需求选择。

## 目的

本文定义Gemma4 E2B-IT text-only的两种导出与运行方案。模型cache语义仅
存在于Python adapter；通用backend/runtime只管理具名、常驻device的state tensor。

目标不是重写runtime的K/V管理，而是在导出层将Gemma4的cache显式化为
ExecuTorch方法边界，复用通用的EPAIR、GE AOT、PTE/PTD和
device-resident state流程。

## 模型与状态事实

目标模型为Gemma4 E2B-IT的text-only路径，其decoder包含35层：

```text
35 decoder layers
31 DynamicCache layers (sliding_attention 和 full_attention)
2 shared attention layers (sliding_attention 和 full_attention)
hidden_size=1536
num_attention_heads=8
num_key_value_heads=1
head_dim=256 (sliding_attention)
global_head_dim=512 (full_attention)
sliding_window=512
num_kv_shared_layers=20
```

DynamicCache layers使用标准K/V cache；shared attention layers在最后20层共享
sliding_attention和full_attention的K/V cache。Transformers cache object是导出期
实现细节，不能作为ExecuTorch runtime的跨方法接口。

## 方案选择

### 采用：导出层显式state adapter

`Gemma4StaticStateAdapter` 接收全部state tensor，重建仅在本次forward内有效的
Transformers `DynamicCache` 和shared KV dictionary，并返回logits和更新后的state tensor。
通用backend通过 `state_io_contract` 将它们标记为 `kv_cache`。

该方案保留Transformers权重与前向语义，同时使runtime只承担device tensor的分配、
namespace隔离和inout生命周期管理。

### 不采用：runtime感知Gemma4 cache

不在C++ runtime引入Gemma4、Transformers、K/V或shared attention专用分支。这样可避免
runtime与Python模型实现耦合，也保持已有state contract可复用于其他模型。

### 不采用：直接导出Transformers StaticCache

`StaticCache` 可辅助eager/export，但其内部状态未稳定地成为ExecuTorch user input/output。
它可能被lifted为constant或落入内部copy路径，无法满足backend-owned mutable state、
多session隔离与逐容量验证要求。

## State契约

adapter将全部Gemma4 cache tensor显式定义为方法输入和输出：

```text
cache.layer_<n>.key/value              kv_cache
shared.sliding_attention.key/value     kv_cache
shared.full_attention.key/value        kv_cache
```

prefill重置state namespace。decode从同一namespace读取上一轮state并写入下一轮
state。runtime不得依赖Gemma4或Transformers cache class。

### DynamicCache K/V State

每个DynamicCache layer产生：

```text
cache.layer_<layer_idx>.key
cache.layer_<layer_idx>.value
```

shape为：

```text
[batch, num_key_value_heads, max_seq_len, head_dim]
```

其中 `head_dim` 根据layer type选择：
- sliding_attention: `head_dim=256`
- full_attention: `global_head_dim=512`

role为 `kv_cache`。prefill写入prompt的K/V；decode读取已有prefix并写入当前token
位置。state的物理shape始终使用 `max_seq_len`，而不是当前运行轮次的token数。

### Shared Attention K/V State

最后20层共享sliding_attention和full_attention的K/V cache，产生：

```text
shared.sliding_attention.key
shared.sliding_attention.value
shared.full_attention.key
shared.full_attention.value
```

shape为：

```text
[batch, num_key_value_heads, max_seq_len, head_dim]
```

其中sliding_attention使用 `head_dim=256`，full_attention使用 `global_head_dim=512`。

role为 `kv_cache`。prefill写入全部prompt的K/V；decode读取已有prefix并写入当前
token位置。

### State更新策略

Gemma4的decode阶段存在两类state更新模式：

1. **DynamicCache layers**：每层独立更新自己的K/V cache。
2. **Shared attention layers**：最后20层共享sliding/full attention的K/V cache，
   因此只有最后更新该cache的layer会写入shared state。

adapter在 `_static_states_from_outputs` 中根据 `state_specs` 的 `kind` 字段区分
`cache` 和 `shared`，从 `past_key_values` 和 `shared_kv_states` 中提取对应的K/V tensor。

## 方法边界与运行时位置

artifact包含固定prefill method和单token decode method：

```text
prefill(input_ids, input_pos, states...) -> logits, next_states...
decode(input_ids, input_pos, states...) -> logits, next_states...
```

当前 `decode_seq_len` 固定为1。调用方先执行 `prefill`，随后每轮调用同一个 `decode`，
并把当前绝对位置写入 `input_pos`。full/sliding attention state使用固定容量，mask
根据运行时位置屏蔽未写入区域，不需要为每个cache长度生成独立方法。

Gemma4的Transformers cache update需要显式传递位置。adapter在runtime decode模式
下从 `input_pos` 获取当前cache长度，函数式更新固定容量cache和shared K/V state，
保证因果mask、K/V prefix与position ID和实际生成长度一致。

## 导出兼容与Copy

### ExecuTorch兼容配置

导出入口使用 `aten_edge_compile_config()`：

```text
旧版 ExecuTorch: _use_edge_ops=False
新版 ExecuTorch: _skip_dim_order=True，随后 EdgeOpOverload -> ATen
未知 API: 导出前失败
```

进入EPAIR conversion前，系统通过底层ATen overload恢复已知EdgeOpOverload。
identity `_clone_dim_order` 改写为 `aten.clone`；非identity dim-order layout明确拒绝。

### 文本路径模型裁剪

Gemma4 checkpoint同时包含视觉和音频tower。纯文本导出不会调用这两类输入，因此导出
入口在模型加载后将 `vision_tower` 和 `audio_tower` 置空，降低主机内存峰值。该操作不
改变语言模型权重和文本计算路径。

```python
mm = getattr(model, "model", model)
for name in ("vision_tower", "audio_tower"):
    if hasattr(mm, name) and getattr(mm, name) is not None:
        setattr(mm, name, None)
```

### Eager Attention

Gemma4默认使用 `scaled_dot_product_attention` (SDPA)，但在NPU上可能产生数值问题。
adapter在加载模型时指定 `attn_implementation="eager"`，使用手动的matmul + softmax实现：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
```

### Copy语义

Gemma4 state update可能生成 `aten.copy.default`。系统必须保留该算子，并由受支持的
TorchAir和TorchDelegate copy路径转换。将其改写为 `aten.alias` 会丢失copy的storage
语义，因此不正确。

## 数据流

导出阶段：

```text
Gemma4 E2B-IT checkpoint
  -> 释放未使用的 vision/audio tower（纯文本路径）
  -> Gemma4StaticStateAdapter (attn_implementation="eager")
  -> torch.export(prefill/decode)
  -> ExecuTorch transform/decomposition
  -> EdgeOpOverload -> ATen compatibility lowering
  -> AscendBackend.preprocess_multimethod
  -> EPAIR
  -> GE Offline_compile
  -> model.pte + ascend.weights.package.ge_aot.ptd
```

运行阶段：

```text
prefill input_ids/input_pos
  -> runtime resolves reset state namespace
  -> GE execute
  -> logits copied to host; next state remains device-resident
  -> greedy token
  -> decode with the same namespace and runtime input_pos
```

`executorch_npu_stateful_runner session` 必须在同一进程和同一Module中完成prefill与
后续decode，退出前释放state namespace。

## Artifact契约

backend核心artifact为：

```text
model.pte
ascend.weights.package.ge_aot.ptd
```

`generation_config.json` 是应用层sidecar。其schema version 1记录
`prefill_seq_len`、`max_seq_len` 和 `max_new_tokens`，不属于backend artifact ABI。

## 实现边界

当前实现只覆盖Gemma4 E2B-IT text-only static-state路径，包含：

1. state spec和zero-state构造。
2. real checkpoint eager prefill/decode。
3. 固定shape的 `torch.export`。
4. multimethod prefill/decode打包。
5. EPAIR、GE AOT、PTE/PTD和stateful runner流程。
6. 文本路径tower裁剪和eager attention配置。

当前不包含：

1. vision token或image输入。
2. dynamic shape。
3. `decode_seq_len > 1` 的chunked decode。
4. Gemma4专用runtime C++ 逻辑。
5. 以padding或截断兼容错误的prompt长度。

## 测试与验收

### 单元测试

测试至少覆盖：

1. DynamicCache/shared attention layer分别生成正确state spec。
2. state名称、role、dtype与shape稳定。
3. 不同 `max_seq_len` 下固定容量state和运行时位置contract一致。
4. `generation_config.json` schema、容量与prompt长度校验。
5. ExecuTorch legacy/current compatibility configuration与未知API拒绝。
6. 文本路径tower裁剪不影响语言模型输出。

### 真实checkpoint验收

真实模型验收分层执行：

1. tokenizer prompt token数与 `prefill_seq_len` 精确匹配。
2. CPU eager prefill/decode logits有限，greedy token可生成。
3. `torch.export` 后全部state位于方法输入/输出边界，不能被隐藏为lifted constant。
4. EPAIR全量转换、GE Offline_compile、PTE/PTD序列化成功。
5. NPU session在同一namespace中完成 `prefill -> decode`，并逐轮更新 `input_pos`。
6. NPU token ID与同一adapter的CPU greedy baseline一致。
7. NPU输出无NaN/Inf。

## 风险与已知性能问题

### Transformers内部实现变化

adapter必须将对Transformers内部cache field、layer type与mask hook的依赖集中在
`examples/gemma4/export/adapter.py`，并以state-spec测试约束。

### State内存规模

DynamicCache K/V state与shared attention K/V state都随层数和 `max_seq_len` 增长。
导出和验证应先使用较小的 `max_seq_len`，再逐步扩大；不能把小容量artifact的成功外推为长上下文能力。

### ExecuTorch Decomposition性能

真实Gemma4 static-state图包含大量显式state input（34个state tensor）。当前观察到
`ExportedProgram.run_decompositions()` 会进入PyTorch AOT Autograd的
functionalization、alias metadata和symbolic-shape interpreter路径。该阶段可能成为导出
瓶颈，尚未进入EPAIR或GE。优化必须保持state input/output与copy语义，不得以
`aten.alias` 替代 `aten.copy.default`。

### SDPA数值稳定性

Gemma4默认的 `scaled_dot_product_attention` 在NPU上可能产生数值问题。必须使用
`attn_implementation="eager"` 配置，使用手动的matmul + softmax实现。

## 与Qwen3.5的差异

### Cache结构

| 特性 | Qwen3.5 | Gemma4 |
|------|---------|--------|
| Cache类型 | DynamicCache + LinearCache | DynamicCache + Shared KV |
| State类型 | kv_cache + mutable_state | kv_cache only |
| Linear attention | 有（conv_states + recurrent_states） | 无 |
| Shared attention | 无 | 有（sliding + full） |
| 层数 | 24 | 35 |

### State绑定策略

Qwen3.5使用简单的state绑定策略，因为所有state都在decode阶段更新。

Gemma4使用3-pass状态绑定策略：

1. **Pass 1: Exclusive dependency matching**
   - 匹配只依赖当前state input的output
   - 适用于prefill阶段，每个state output只依赖对应的state input

2. **Pass 2: Fewest dependencies matching**
   - 匹配依赖当前state input且依赖其他state input最少的output
   - 适用于decode阶段，state output可能累积依赖多个state input

3. **Pass 3: Unused state detection**
   - 检测state input是否被任何state output使用
   - 如果未使用，设置 `next_output_arg = -1` 表示state未更新
   - 适用于decode阶段的shared attention states

### Anchor机制

Gemma4的prefill阶段存在一个问题：某些state input（如shared attention states）在
prefill时完全被替换，不会读取previous state。这导致 `torch.export` 可能将这些state
input优化掉。

解决方案是在state output上添加anchor：

```python
anchor = sum(s.flatten()[0] * 0 for s in states)
next_states = tuple(ns + anchor for ns in next_states)
```

这会在state output上添加对所有state input的依赖，但不会改变数值（因为anchor = 0）。
`torch.export` 会保留所有state input，确保delegated subgraph包含完整的state I/O。

## 3-Pass状态绑定算法详解

### 问题背景

ExecuTorch的 `to_edge_transform_and_lower` 会将原始导出图转换为delegated subgraph，
其中state output的顺序可能与原始导出图不同。我们需要在delegated subgraph中正确识别
每个state output。

### Pass 1: Exclusive Dependency Matching

**目标**：匹配只依赖当前state input的output。

**算法**：

```python
exclusive_matches = [
    output_name
    for output_name in output_names
    if output_name not in used_output_names
    and _matches_state_metadata(nodes.get(output_name), binding)
    and source_input_name in dependency_cache[output_name]
    and not (dependency_cache[output_name] - {source_input_name}) & state_input_names
]
```

**适用场景**：
- Prefill阶段：每个state output只依赖对应的state input
- 简单模型：state output之间没有交叉依赖

**示例**：

```text
states_0 (input) -> aten_cat_0 (output)  # 只依赖 states_0
states_1 (input) -> aten_cat_1 (output)  # 只依赖 states_1
```

### Pass 2: Fewest Dependencies Matching

**目标**：匹配依赖当前state input且依赖其他state input最少的output。

**算法**：

```python
dep_matches = [
    output_name
    for output_name in output_names
    if output_name not in used_output_names
    and _matches_state_metadata(nodes.get(output_name), binding)
    and source_input_name in dependency_cache[output_name]
]
if dep_matches:
    output_name = min(
        dep_matches,
        key=lambda name: len(dependency_cache[name] & state_input_names),
    )
```

**适用场景**：
- Decode阶段：state output可能累积依赖多个state input
- 复杂模型：state output之间存在交叉依赖

**示例**：

```text
states_0 (input) -> aten_cat_0 (output)  # 依赖 states_0, states_1
states_1 (input) -> aten_cat_1 (output)  # 依赖 states_0, states_1, states_2
```

Pass 1无法匹配（因为都有交叉依赖），Pass 2选择依赖最少的output。

### Pass 3: Unused State Detection

**目标**：检测state input是否被任何state output使用。

**算法**：

```python
state_is_used = any(
    source_input_name in dependency_cache[output_name]
    and _matches_state_metadata(nodes.get(output_name), binding)
    for output_name in output_names
)
if not state_is_used:
    binding["next_output_arg"] = -1  # sentinel: no output
```

**适用场景**：
- Decode阶段的shared attention states：某些state在decode时不更新

**示例**：

```text
states_30 (shared.sliding_attention.key) -> 无 output 依赖它
states_31 (shared.sliding_attention.value) -> 无 output 依赖它
```

设置 `next_output_arg = -1`，runtime保持state不变。

### 为什么需要3-Pass策略

1. **Pass 1优先**：exclusive dependency是最可靠的匹配方式，避免误匹配。
2. **Pass 2补充**：处理decode阶段的累积依赖问题。
3. **Pass 3兜底**：处理未更新的state，避免强制映射导致错误。

这种分层策略确保了在不同场景下都能正确绑定state output，同时保持向后兼容性。

## 动态形状方案

除了上述静态形状方案外，Gemma4 E2B-IT还支持动态形状导出方案。该方案使用单一方法
（而非分离的prefill/decode），通过 `torch.export.Dim` 支持可变序列长度。

### 方案概述

动态形状方案的核心思想是：

1. **单一方法**：导出一个统一的 `model` 方法，同时处理prefill和decode
2. **动态序列长度**：使用 `torch.export.Dim` 定义序列长度维度，范围为 `[2, MAX_SEQ_LEN]`
3. **内部KV cache**：使用 `Gemma4CustomKVCache` 管理KV cache，通过 `cumulative_length`
   控制prefill/decode行为
4. **无需显式state I/O**：KV cache作为模型内部状态，不作为方法的输入/输出

### 导出配置

```python
class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, cache: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.cache = cache

    def forward(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        for layer_cache in self.cache.layers:
            layer_cache.cumulative_length.copy_(cache_position[0])
        return self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        ).logits

# 动态形状定义
sequence_length = torch.export.Dim(
    "sequence_length", min=2, max=MAX_SEQ_LEN
)
dynamic_shapes = {
    "input_ids": {1: sequence_length},
    "cache_position": {0: sequence_length},
}
```

### 方法签名

动态形状方案只导出一个方法：

```text
model(input_ids, cache_position) -> logits
```

- `input_ids`: shape `[batch, seq_len]`，`seq_len` 为动态维度
- `cache_position`: shape `[seq_len]`，与 `input_ids` 的 `seq_len` 一致
- `logits`: shape `[batch, seq_len, vocab_size]`

### 运行时行为

运行时通过 `cache_position` 区分prefill和decode：

- **Prefill**：`cache_position` 从0开始，长度为prompt token数
- **Decode**：`cache_position` 从prompt长度开始，每次增加1

KV cache由 `Gemma4CustomKVCache` 内部管理，通过 `cumulative_length` 记录当前序列长度。
每次forward时，将 `cache_position[0]` 复制到 `cumulative_length`，模型自动读取和更新
对应位置的KV cache。

### 与静态形状方案的对比

| 特性 | 静态形状方案 | 动态形状方案 |
|------|-------------|-------------|
| 方法数量 | 2个（prefill + decode） | 1个（model） |
| 序列长度 | 固定（prefill_seq_len, decode_seq_len=1） | 动态 [2, MAX_SEQ_LEN] |
| State管理 | 显式state I/O（34个tensor） | 内部KV cache |
| State绑定 | 需要3-pass算法 | 无需绑定 |
| Anchor机制 | 需要 | 不需要 |
| 编译时间 | 较长（两个方法） | 较短（一个方法） |
| 部署复杂度 | 较高（需管理两个方法） | 较低（单一方法） |
| 性能 | 略高（固定形状优化） | 略低（动态形状开销） |

### 适用场景

**静态形状方案适用于**：
- 需要精细控制state生命周期
- 需要多session隔离
- 需要逐容量验证
- 对性能要求极高

**动态形状方案适用于**：
- 快速原型验证
- 简化部署流程
- 不需要复杂的state管理
- 对编译时间敏感

### MMLU评测验证

动态形状方案已在MMLU评测中验证：

- **NPU准确率**：48.87%（6863/14042）
- **CPU准确率**：48.92%（6870/14042）
- **NPU与CPU差异**：0.05%（高度一致）

评测使用 `MAX_SEQ_LEN=4096`，支持MMLU数据集中的所有样本长度。

### 实现细节

动态形状方案的关键实现文件：

- `examples/gemma4/export/pt2.py`：PT2导出脚本
- `examples/gemma4/export/pte.py`：PTE导出脚本
- `examples/common/cache.py`：`Gemma4CustomKVCache` 实现

导出流程：

```text
Gemma4 E2B-IT checkpoint
  -> Gemma4CustomKVCache (内部管理 KV cache)
  -> torch.export with dynamic shapes
  -> ExecuTorch transform/decomposition
  -> AscendBackend.preprocess_multimethod
  -> EPAIR
  -> GE Offline_compile
  -> model.pte + ascend.weights.package.ge_aot.ptd
```

运行流程：

```text
input_ids + cache_position
  -> runtime loads model
  -> GE execute with dynamic shape
  -> logits copied to host
  -> greedy token
  -> next iteration with updated cache_position
```

### 限制与注意事项

1. **序列长度范围**：`min=2` 是torch.export的要求，不能为1
2. **最大序列长度**：受NPU内存和编译时间限制，建议不超过4096
3. **KV cache内存**：随 `MAX_SEQ_LEN` 线性增长，需提前规划
4. **编译时间**：动态形状编译比固定形状慢，首次导出可能需要30-60分钟
5. **运行时性能**：动态形状会引入少量性能开销，但对MMLU评测影响可忽略
