<!--
人类可读版设计文档。内容与 DESIGN.md 等价，但以叙述为主、编号缩写为辅。
机器可读的权威版本仍为 ./DESIGN.md（含完整 ID 交叉引用）。
v0.1: 对应 SPEC SR20260828001 初稿（含 U-1 已解决：TokenId 路径表在权重中）。
v0.2: 外部审查（+2，2026-08-29）放行条件修订——Scheduler 接线点、U-8 通道登记、
      OHOS CMake 注册及若干 L 级订正；技术决策（D-01~D-11）不变。
-->

# TextEmbedding 模块设计（人类可读版）

简体中文 | [English](../../../en/cann_lm_engine/embedding/DESIGN_HUMAN.md)

> 配套：`SPEC.md`（行为规格）、`DESIGN.md`（机器可读的权威设计，含完整编号交叉引用）。
> 本文档是 DESIGN.md 的叙述式重写，方便通读；技术决策完全一致。

## 这个模块是做什么的

一句话：**它是推理引擎里的"embedding 搬运工"——把文件系统上的低精度 embedding 表和反量化参数加载进来（或干脆留在文件里按需读），在前向之前按 token id 查出对应的行，把结果按 om 图要求的类型填进模型的输入工位。它只搬字节、只做数值表示的展开，永远不做反量化——乘 scale 那一步在 NPU 上。**

它夹在 `pipelineExecutor`（上层调度）和 om 图输入工位（`input_embed` / `embed_scales`）之间，干三件事：

1. **加载（Load）**：读 embedding 表文件和 scale 文件。表可以整块读进内存（InMemory），也可以不进内存、留个文件句柄每次现读（InFlash）。
2. **查表（Gather）**：给一组 token id，按行偏移把表里对应的低精度行拷出来，同时把每个 token 的反量化参数（scale）也收集出来。
3. **填工位（FillInputs）**：把查到的行做类型转换后写进 `input_embed` 工位，把 scale 写进 `embed_scales` 工位——写入类型以工位自己的 `TensorDesc` 为准，不是以文件存储类型为准。

它是 engine 内部的 C++ 组件，**只**被 pipeline / model_executor 消费，不对外暴露任何绑定。

## 理解本模块的四个关键点

### 1. host 只搬运，反量化永远在 NPU

这是所有决策的地基。embedding 表是低精度的（INT8 等），配上 scale 才能还原成高精度。本模块在 host 侧做的事——拷行、收集 scale、类型转换——全是**字节搬运和数值表示展开**：INT8 的 5 变成 FLOAT16 的 5.0，值不变，不乘任何东西。乘 scale（真正的反量化）发生在 om 图内的 NPU 上。所以 host 侧的 gather 与存储类型无关（纯 stride 拷贝），类型转换与量化无关（纯值展开）——这两条"无关"让实现非常薄。

### 2. 双路径：表在不在权重里，决定本模块管多少事

- **tokenId 路径**（om 图内置 embedding）：**表已经包含在模型权重里随图加载了**，本模块什么都不用加载，`FillInputs` 只把 token id 填进 `input_embed` 工位就完事——甚至不要求先 Load。NPU 在图内自己查表、自己反量化。
- **embedding 路径**（om 图不含内置 embedding）：本模块的完整主场——Load 表 → host 查表 → 填 `input_embed` + `embed_scales` → NPU 反量化。

走哪条路不是本模块自己决定的，是 model_executor 按 om 图有没有内置 embedding 决定的（那是 model_executor 设计里早已接受的 D-07）。

### 3. 填什么类型，工位说了算

用户在 Load 时指定的 `EmbeddingType`（Fp32/Fp16/Int8）描述的是**文件在磁盘上的字节布局**；而实际写进 `input_embed` 工位的数据类型，取自该工位 `TensorDesc.dataType`（om 图声明什么就是什么）。两者不一致时，模块在填充前做类型转换（逐元素值展开）。同型就直拷。不支持的组合（INT4、BF16 这些）明确拒绝，返回 `INVALID_ARGUMENT`。

### 4. 量化粒度：PerToken 与 PerGroup

scale 文件的形状由初始化参数 `quantMode` 决定：

- **PerToken**：每个词表项（表的一行）一个 scale，文件里就是 `[vocab]` 个数。
- **PerGroup**：一行 embedding 按 hidden 维切成若干组（组大小 = `GroupSize`，必填且必须整除 hidden_dim），**每组各一个 scale**——一个 tokenId 对应多个反量化参数。文件形状 `[vocab × (hidden/group_size)]`，token v 的第 g 组 scale 在下标 `v×groups+g`。

相应地，Gather 收集 out_scales 时：PerToken 每个 token 收 1 个，PerGroup 每个 token 收 `groups` 个。

## 结构：一个公开类，两个内部帮手

对外只有 `TextEmbedding` 一个具体类（当初 brainstorm 在"多态层次 vs 单类标签分派"里选了后者——因为 host 侧逻辑跟类型无关，多态没有落点）。内部拆两个不出公开头的帮手：

- **TableSource（M-02）**：表数据源策略。统一接口就一个 `ReadRow(id, dst)`——给我 token id，我把那一行字节放进你的 buffer。两个实现：
  - `MemoryTableSource`：整表在内存 buffer 里，ReadRow 就是 memcpy。
  - `FlashTableSource`：只有 fd 和行宽，ReadRow 是一次 `pread`（按 `id × row_bytes` 偏移位置读，不动文件游标，天然可多线程并发）。
  - 这是"预留策略化钩子"的第一落点：将来 INT4 需要位解包之类差异化逻辑，就在这层扩。
- **TypeConverter（M-03）**：纯函数集，一张支持矩阵（存储类型 × 工位 dtype），同型 memcpy、异型值展开、其余拒绝。无状态无依赖。

三个都落在 `core/component/text_embedding.{h,cpp}` 一个文件对里。同头文件的 `ModalityEntry`（多模态占位条目）是隔壁 Preprocessor 的职责，原地保留不动。

## 两个加载方式的取舍

| | InMemory（默认） | InFlash |
|---|---|---|
| 表常驻内存 | ≈ 表文件字节 | 0（只有 scale + 一个 fd） |
| gather 成本 | memcpy | 每行一次 pread 系统调用 |
| 适合 | 吞吐优先（服务器 ACL 侧） | 内存受限（端侧 OHOS） |

两者是**运行期配置**（`EmbeddingConfig.load_mode`），不是编译期二选一——同一个构建产物都能跑。scale 无论哪种方式都恒为内存态：它只有 vocab（或 vocab×groups）个标量，比表小得多，而且每次 gather 都要按下标随机访问，放文件里不划算。

InFlash 的代价是文件生命周期：Load 之后表文件必须一直在那（模块持有 fd 到析构）。文件中途被截断或替换，gather 时 `pread` 短读，返回 `FAILURE`。热更新不支持——有需求再说（SPEC 未决项 U-7）。

## 接口

```cpp
Status Load(const EmbeddingConfig& config);

Status Gather(const TokenId* ids, size_t count,
              std::shared_ptr<ITensorBuffer> out_embed,
              std::shared_ptr<ITensorBuffer> out_scales) const;   // 可并发

Status FillInputs(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                  const std::unordered_map<std::string, size_t>& name_index,
                  EmbeddingPath path,
                  const TokenId* ids, size_t count);

// 访问器：Table() / Scale() / TableType() / LoadMode() / Loaded()
```

`EmbeddingConfig`：两个路径、两种类型、量化粒度 + group_size、vocab/hidden（来自 ModelConfig）、load_mode。

关于 `FillInputs` 的参数——它拿的是 model_executor 的输入工位数组（`input_tensors_`）和名字→下标映射（`input_name_index_`）。为此 `ModelExecutor` 新增两个**只读** accessor（`InputTensors()` / `InputNameIndex()`），是加法变更，不动既有接口。本模块只写名为 `input_embed` 和 `embed_scales` 的工位，越界即缺陷——跟 InputAssembler 写的另外三个工位（position_ids / new_kv_cache_pos / attention_mask）井水不犯河水。

## 一次完整的前向（embedding 路径）

```
init 期:   pipeline 拿 ModelConfig（meta_json 反序列化，属 ModelConfig 属主的既有规划）
             → EmbeddingConfig::FromModelConfig(mc, &config)   ← U-8 已决议：路线 b
             → table_path 非空才 text_embedding_.Load(config)
           （InMemory：整表+scale 读入内存；InFlash：open+fstat 校验+留 fd）
           （table_path 为空 = tokenId 部署形态，跳过 Load——meta_json 反序列化落地前
             ModelConfig 字段缺省即为空，与现状行为兼容）

每步前向:  输入准备期——具体发生在 Scheduler::Step 内的两个输入准备点
           （prefill：从队列取下个 chunk；decode：Preprocessor.PrepareDecodeStep 各 active 请求 last token）：
             text_embedding_->FillInputs(executor.InputTensors(),
                                          executor.InputNameIndex(),
                                          path, ids, count)
               ├─ Embedding 路径: 逐 token ReadRow（memcpy 或 pread）→ 类型转换 → 写 input_embed
               │                 （ids 取自 chunk / decode 输入，count = 行数）
               └─ 按 token 收集 scale → 写 embed_scales
           （同时 InputAssembler 填另外三个工位；KV 池工位 Load 时已绑定）
           executor.Forward(opts) → NPU 反量化 + 图计算 → logits
```

要说清楚"输入准备期"到底在哪：**RunLoop 本身不填输入**，它只循环调 `scheduler_.Step()`；现行代码里 `input_embed` 的唯一写入方是 Scheduler 里的两处 `SetInputData` 调用（prefill 一个、decode 一个）。所以接线就落在这两个点上：`Scheduler` 构造函数新增一个 `TextEmbedding*` 非拥有指针（和它已经持有的 `Preprocessor*` 一个路数，由 pipeline 传 `&text_embedding_` 进去），两个输入准备点把 `SetInputData(chunk)` 换成 `FillInputs(...)`——tokenId 路径也一并换掉（模块顺带做 INT32/INT64 值转换，取代裸 memcpy）。换完之后 `SetInputData` 在仓内零调用，留作"外部直填通道"API 不删。**同一 iteration 内 `input_embed` 只允许被写一次**——`SetInputData` 和 `FillInputs` 不许先后都上，这是硬规则（否则后写覆盖先写，行为看调用顺序）。

tokenId 路径就简单了：`FillInputs(..., TokenId, ids, count)` 把 id 转成工位 dtype（INT32/INT64）写进 `input_embed`，其余交给图内权重。

## 并发模型（以及为什么没有锁）

Load 之后，模块的所有成员（表 buffer 或 fd、scale buffer、配置快照、行宽）全部只读。只读数据并发访问不需要锁；InFlash 的 `pread` 是位置读，多个线程共用一个 fd 各读各的偏移，POSIX 保证安全。所以这个模块**没有锁**——契约只有两条：Load 别并发调（init 期天然串行）、Load 和查询别交叉。`Gather` 是 const 的可以随便并发；`FillInputs` 写的是工位 buffer，本来就活在输入准备期的单线程域里。

## 错误处理

全部走 `infra/status.h` 六个粗码，不抛异常：

- **INVALID_ARGUMENT**：文件缺失/尺寸不对、PerGroup 参数非法、scale 配置自相矛盾（type=None 却给了路径）、token id 越界（整批拒绝不部分产出）、输出容量不足、NULL 入参、不支持的类型转换组合。
- **INVALID_STATE**：embedding 路径没 Load 就 Gather/FillInputs（tokenId 路径豁免）。
- **NOT_FOUND**：name→index 里没有 `input_embed`（或该有 scale 时没有 `embed_scales`）。
- **FAILURE**：InFlash 运行期文件损坏（pread 短读/I/O 错）；TensorManager 申请 buffer 失败。

Load 失败保持未加载态，修好配置可以重 Load，不会半加载。

## 测试怎么打

GoogleTest，落 `tests/cpp/engine/`，UT target `ut_text_embedding`。核心思路：**NPU 侧反量化不在 UT 范围**（后端依赖），UT 只验 host 侧三件事——加载、查表、填工位。SPEC 第 8 章有完整 TC 清单（TC_001–TC_302），几个值得点名的：

- **InFlash 与 InMemory 一致性**（TC_008/009）：同一表文件两种模式加载，gather/填充结果逐字节一致——这是 InFlash 正确性的锚。
- **类型转换矩阵**：每个支持的组合逐元素验值（含负数 INT8、Fp16 非规约数），每个不支持的组合验拒绝。
- **mock 工位 bundle**：直接用 `CreateTensorBuffer` 按 desc 造 input bundle + name_index，不用 mock TensorManager（桩后端无副作用，走真实单例即可）。
- **并发**（TC_301/302）：4 线程并发 Gather，结果与串行一致；InFlash 共享 fd 并发 pread 互不干扰。
- **grep 断言**：`text_embedding.{h,cpp}` 里不得出现具体后端类名。

## 与现状的衔接（迁移说明）

现在的 `text_embedding.h/.cpp` 是 stub era 桩：`LoadAndBind()` 恒返 FAILURE。本次原地重写：`LoadAndBind` 删除（全仓 grep 确认零价值调用后），换成 `Load/Gather/FillInputs` 三件套。上下游基本不动：`PipelineExecutorFactory::create` 仍按值传 `TextEmbedding`，`Preprocessor` 仍持非拥有指针，`ModalityEntry` 原位保留；**唯一的例外是 `Scheduler` 构造签名新增 `TextEmbedding*`**（接线需要，见上文"输入准备期"段）。UT 注册时记得主 CMakeLists 和 `ohos/CMakeLists.txt` **两边都要挂**——OHOS 侧是显式逐 target 复制的，主链加了 OHOS 不加，鸿蒙上就跑不到这个测试（仓里 `ut_sampler`/`ut_request_handle` 已经这么漂过一次，别再添一个）。

有一处现状要跟着订正：`INPUT_INFO.md`（model_executor 输入工位的事实来源）里 `embed_scales` 记的维度 `batch×seqLen×1` 只对 PerToken 成立（PerGroup 应为 `×groups`），`input_embed` 记的 FLOAT16 是 stub 占位（实际按 TensorDesc、随路径变）。实现落地时随 PR 一起订正（SPEC 未决项 U-4）。

## 留了哪些口子（未决项速览）

- **U-2**：vocab/hidden 的来源字段名，实现时和 ModelConfig 对齐（入参形态已定）。
- **U-3**：INT4/BF16 本版本不支持——出现真实需求就是 M-02/M-03 策略化升级的触发条件。
- **U-5**：吞吐/延迟/内存预算都是预算目标，实现期微基准校准。
- **U-6**：没有 scale 的模型，`embed_scales` 工位跳过不填（当前取法）；首个无 scale 模型接入时再确认。
- **U-7**：InFlash 文件不支持热更新，损坏走 FAILURE。
- ~~**U-8**（2026-08-29 外审新增）~~：`EmbeddingConfig` 从用户配置到 pipeline 的通道——**已决议（2026-08-29 用户拍板）路线 b**：meta_json → `ModelConfig` 扩展 embedding 段（plain 字符串/整数字段，序列化友好）→ `EmbeddingConfig::FromModelConfig` 组装（字符串→枚举的解析校验在本模块，依赖方向合法）。L1 `EngineOptions`/C ABI 一动不动。vocab/hidden 也一并落 ModelConfig（U-2 的来源就此定了）。
- 新增一条实现期复核项：表/scale buffer 经 `TensorManager::CreateTensor` 申请时 `modelId=0` 哨兵的语义，真实后端实装时确认 host buffer 不受 modelId 绑定影响。
