# Qwen3.5导出设计：动态Prefill与固定Decode

## 目的

本文定义Qwen3.5 text-only模型在ExecuTorch和Ascend backend上的导出、状态管理与
运行时契约。设计目标是将prefill和decode封装到同一个PTE中，共享一份权重数据，
并通过state namespace在多次调用之间持久化模型状态。

模型导出为两个 `ExportedProgram`：动态序列长度的 `prefill` 和固定单token的
`decode`。两者使用相同的模型前向语义，差异由输入shape和导出时的初始状态决定。

## 模型与状态事实

Qwen3.5 text-only decoder是混合注意力架构，由 `config.layer_types` 指定每层类型：

```text
full_attention
linear_attention
```

两类层的状态语义不同：

- Full Attention保存固定容量的K/V cache，并维护逻辑序列长度。
- Linear Attention保存convolution state和recurrent state。
- 物理状态容量在导出时确定，逻辑长度由运行时位置决定。
- Batch size当前固定为1。

Transformers Cache对象只属于导出侧模型抽象，不作为跨method的公开Python接口。
跨prefill/decode的持久化状态由ExecuTorch state contract和Ascend state namespace
承担。

## 方案选择

### 采用：导出侧Cache Module与通用state bridge

Cache作为导出Wrapper的Module State注册。Functionalization将发生mutation的
buffer提升为方法边界上的state input/output，再由backend将其转换为device-resident
state。该方案保持Transformers模型前向语义，同时避免runtime感知Qwen3.5的具体
Cache类。

Prefill和decode使用独立的导出状态：

- Prefill从空的Linear Attention state开始。
- Decode以“已有前序状态”为初始导出条件。
- 运行时使用同一state namespace将prefill产生的状态交给decode。

### 不采用：Runtime内置Qwen3.5 Cache分支

C++ runtime不识别Qwen3.5的layer type、K/V cache、convolution state或recurrent
state。runtime只处理通用的state binding、生命周期、namespace隔离和device buffer。

### 不采用：将Cache展平为应用层输入输出

Cache不作为调用方需要手动维护的State Tuple，也不暴露为普通 `EValue`。这样可以让
状态保持在device，避免每轮decode在host和device之间搬运大容量cache。

## State契约

### Full Attention State

每个Full Attention层维护：

```text
key/value:
  [batch, num_key_value_heads, max_context_length, head_dim]

cumulative_length:
  scalar int64
```

K/V使用固定物理容量，通过位置更新写入对应区域。`cumulative_length` 表示当前请求
的逻辑长度，不改变K/V的物理shape。

### Linear Attention State

每个Linear Attention层维护：

```text
conv_state:
  [batch, conv_dim, linear_conv_kernel_dim]

recurrent_state:
  [batch, linear_num_value_heads,
   linear_key_head_dim, linear_value_head_dim]
```

`has_previous_state` 是导出期确定的控制状态。Prefill图将其固定为空状态路径，decode
图将其固定为已有状态路径，避免在图中依赖Python层面的 `S == 1` 分支。

### State Owner与生命周期

每个Cache Tensor只能有一个Module注册所有者。Cache layer的公开引用可以指向该
所有者，但不能重复注册同一个底层Tensor，避免AOT Functionalization产生不明确的
alias关系。

状态生命周期如下：

```text
state namespace 创建
  -> prefill 读取初始 state 并写入 next state
  -> current/next 交换
  -> decode 读取前一轮 state 并写入 next state
  -> 每轮执行后交换
  -> 请求结束时释放 namespace
```

## 方法边界

### Prefill

```text
prefill(
  input_ids:      int64[1, S],
  cache_position: int64[S]
) -> logits[1, S, vocab]
```

动态约束为：

```text
4 <= S <= max_seq_len - 1
```

Prefill的 `cache_position` 从 `0` 到 `S-1`。Full Attention使用其第一个位置更新
逻辑长度；Linear Attention从空状态开始处理整个prompt。

### Decode

```text
decode(
  input_ids:      int64[1, 1],
  cache_position: int64[1]
) -> logits[1, 1, vocab]
```

Decode固定处理一个token。第一次decode的位置为 `S`，后续位置依次递增。decode
使用与prefill相同的state namespace，但使用decode图对应的state contract。

### 调用顺序

```text
prefill(prompt_ids, positions=0..S-1) -> token_0
decode(token_0, position=S)           -> token_1
decode(token_1, position=S+1)         -> token_2
...
```

Prefill和后续decode必须在同一执行上下文中使用同一namespace。不同请求必须使用
互相隔离的namespace。

## 导出策略

### Prefill导出

Prefill使用symbolic sequence dimension。示例输入只用于确定导出图和状态容量，不能
将示例长度误认为运行时固定长度。动态维度的下界为4，是当前Linear Attention
convolution kernel的最小有效序列长度。

上界使用 `max_seq_len - 1`，为最大上下文位置保留边界空间，避免导出约束将边界值识别
为越界情况。

### Decode导出

Decode使用 `[1, 1]` 的输入shape和 `[1]` 的位置shape。Linear Attention的已有状态
路径在导出时确定，运行时不依赖Python条件分支选择状态路径。

### PTE组织

两个method使用同一个 `acl_soc_name`，角色分别为 `prefill` 和 `decode`，并使用
一致的delegate identity：

```text
method_name
delegate_index
role
acl_soc_name
```

Runtime必须校验manifest、当前method和delegate identity一致，防止错误的多method
artifact被加载到错误的执行上下文。

## 导出兼容与状态转换

### Functionalization

已注册Cache Buffer的mutation在图签名中表现为：

```text
InputKind.BUFFER
OutputKind.BUFFER_MUTATION
```

状态匹配依据ExportedProgram Graph Signature的身份，不依赖FX名称、Tensor shape
或lowering后的target名称。

### CANN State Bridge

```text
Buffer Mutation
  -> delegated graph state input/output
  -> manifest state_current/state_next
  -> device-resident state namespace
  -> 每次执行交换 current/next
```

自动生成的device-only state使用：

```text
reset_policy = never
reset_owner = false
host_visibility = device_only
```

首次执行没有显式initializer时，runtime以全零state初始化。对于动态state，若ACL
descriptor查询到的input/output size为0，runtime使用state cache记录的实际容量，
确保GE执行不会收到零字节buffer。

### 动态输入容量

动态普通输入的device buffer按模型descriptor要求的容量分配，但H2D只复制当前
实际Tensor的数据长度。这样既满足GE对动态binding容量的要求，也避免从host读取
超出实际输入的数据。

## 数据流

导出阶段：

```text
Qwen3.5 text-only checkpoint
  -> Cache Module 与模型前向组合
  -> torch.export(prefill/decode)
  -> ExecuTorch edge transform/decomposition
  -> Ascend partitioning
  -> GE offline compilation
  -> 一个 PTE + 一个共享 PTD
```

运行阶段：

```text
prefill input_ids/cache_position
  -> state namespace 解析与初始 state
  -> GE execute
  -> logits 返回 host，Cache state 保持 device-resident
  -> decode input_ids/cache_position
  -> 同一 namespace 继续执行
```

## Artifact契约

核心artifact为：

```text
model.pte
ascend.weights.package.ge_aot.ptd
```

PTE包含 `prefill` 和 `decode` 两个method。PTD保存共享权重数据，不应为两个method
重复生成权重副本。设备SoC、method identity、state contract和artifact manifest必须
彼此一致。

## 实现边界

当前设计覆盖：

1. text-only Qwen3.5；
2. Full Attention与Linear Attention混合Cache；
3. batch size为1；
4. prefill动态序列长度；
5. 单token decode；
6. device-resident state和namespace隔离；
7. prefill/decode共用PTE和PTD。

当前不覆盖：

1. vision或multimodal输入；
2. batch size大于1；
3. decode sequence length大于1的chunked decode；
4. 动态扩展物理Cache容量；
5. MMLU评测适配与准确率结论。

## 测试与验收

### 结构验收

至少应验证：

1. Cache layer类型与 `layer_types` 一致；
2. 每个Cache Tensor只有一个Module owner；
3. prefill和decode的Graph Signature保留所需state mutation；
4. prefill的动态范围为 `[4, max_seq_len - 1]`；
5. decode输入shape为 `[1, 1]` 和 `[1]`；
6. 两个method的manifest identity与compile spec一致；
7. PTE只引用一份共享PTD权重。

### 运行验收

应覆盖：

1. prefill长度4、8、32、128；
2. 同一namespace的连续decode；
3. 不同namespace之间的状态隔离；
4. 动态state ACL size为0时的buffer fallback；
5. 不同请求并发生成输入时的文件和namespace隔离；
6. 输出logits有效且生成token可解析；
7. MMLU评测流程完成后，再增加loglikelihood一致性验收。

## 风险与已知限制

### 动态shape边界

序列长度小于4不满足当前Linear Attention convolution kernel的约束。达到物理
`max_seq_len` 边界时还可能触发位置更新和导出shape约束，因此当前设计将动态上界设为
`max_seq_len - 1`。

### State容量

Cache物理容量随导出配置固定，并随层数、head数、head dimension和最大上下文长度
增长。小容量artifact的成功不能外推为更长上下文的成功。

### 多method identity

Prefill和decode必须使用正确的method/delegate identity。任何manifest、compile
spec或运行时method不一致都应拒绝加载，而不是尝试执行。

### 评测语义

当前decode路径验证的是生成功能和token输出，不等价于MMLU的选项loglikelihood
评测。MMLU需要单独定义选项评分、请求隔离和结果聚合协议。

## 与Gemma4的差异

| 特性 | Qwen3.5 | Gemma4 |
| --- | --- | --- |
| Cache结构 | Full Attention + Linear Attention | DynamicCache + Shared KV |
| Linear state | convolution state + recurrent state | 无 |
| Shared attention | 无 | 有 |
| Prefill | 动态，`4 <= S <= max_seq_len-1` | 由Gemma4方案单独定义 |
| Decode | 固定单token | 固定单token |
| State形式 | Cache Module mutation，runtime通用bridge | 显式static-state adapter |
| 物理Cache | 导出时固定容量 | 导出时固定容量 |
| MMLU | 尚未完成 | 由Gemma4方案单独验收 |

Qwen3.5不需要将Gemma4的shared attention state、显式state spec或adapter专用
逻辑引入runtime。两者共享的是通用的ExecuTorch state bridge、PTE/PTD组织和
state namespace生命周期。
