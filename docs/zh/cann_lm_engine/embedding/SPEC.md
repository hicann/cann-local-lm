# SR20260828001 — CANN-LM Engine TextEmbedding 模块

简体中文 | [English](../../../en/cann_lm_engine/embedding/SPEC.md)

## 第 1 章 概述

本规格定义 `cann_lm/engine/core/component/text_embedding` 下 C++ embedding 模块的业务行为。该模块为 CANN-LM 推理引擎提供 embedding 表的加载、host 侧查表（gather）、以及模型输入工位填充能力：把文件系统上的低精度 embedding 表与反量化 scale 参数加载进内存，或（InFlash 模式）让表留在文件系统、按需直接读取；在 host 侧按下标 gather 低精度行，并将结果按 om 图 `TensorDesc` 声明的类型填入模型输入工位（`input_embed` / `embed_scales`）。反量化（dequant）**永远在 NPU 侧完成**，本模块在 host 侧不做任何反量化数学运算。该模块是 engine 内部 C++ 组件，仅被 pipeline / model_executor 层消费，**不对外暴露**任何跨语言或外部接口。

需求编号：**SR20260828001**。软件层次：**framework**（engine core 可复用库，含向后兼容性章节）。

读者应在 30 秒内理解：本 spec 规定一个"加载低精度表 + host gather + 主动填工位"、双路径（tokenId / embedding）、支持 perToken / perGroup 两种量化粒度、engine 内部消费的 C++ embedding 模块的行为契约。

## 第 2 章 术语表

| 术语 | 定义 |
|------|------|
| embedding 表 | `[vocab_size, hidden_dim]` 的二维矩阵，第 v 行是词表项 v 的 embedding 向量，以低精度类型存储于文件系统 |
| 存储类型 | embedding / scale 数据在**文件系统中的字节布局类型**（`EmbeddingType` / `EmbeddingScaleType`），由用户在 Load 时指定 |
| 填充类型 | 实际写入模型输入工位的数据类型，由 om 图 `TensorDesc.dataType` 决定；与存储类型不一致时模块**必须**在填充前做类型转换 |
| scale / 反量化参数 | 低精度 embedding 还原为高精度所需的系数；与 embedding 表分离存放 |
| PerToken | 量化粒度：每个词表项（表的一行）对应**一个** scale |
| PerGroup | 量化粒度：一行 embedding 按 `group_size` 分组，**每组**各一个 scale；一个 tokenId 对应多个反量化参数 |
| group_size | PerGroup 模式下每组的元素个数（hidden 维方向）；PerGroup 模式必填 |
| gather / 查表 | 按 tokenId 下标从表中拷贝对应行的操作；host 侧 gather 为纯按下标拷贝（stride 拷贝），不反量化 |
| tokenId 路径 | om 图内置 embedding 查表的推理路径：**embedding 表已包含在模型权重中随图加载**，图内完成 gather + dequant，输入工位装 token id；不经本模块传表 |
| embedding 路径 | om 图不含内置 embedding 的推理路径：host 侧 gather 低精度行填 `input_embed`，NPU 侧 dequant |
| InMemory / InFlash | 表加载方式：InMemory = 整表一次性读入 host 内存；InFlash = 表留在文件系统、不占常驻内存，gather/填充时按行偏移直接从文件读取 |
| 工位 / 输入工位 | 模型 om 图的输入 tensor 槽位（`input_embed`、`embed_scales` 等），见 `docs/model_executor/INPUT_INFO.md` |
| name→index 映射 | `TensorDesc.name` → `input_tensors_` 下标的映射表，model_executor Load 时建立 |
| ITensorBuffer | backend 抽象张量容器（`backend/api/tensor_buffer.h`）：`GetData()` / `GetSize()` / `GetDesc()` |
| TensorDesc | om 图张量元数据快照：`name` / `dataType` / `format` / `dims` / `size` |
| 类型转换 | host 侧数据表示变更（如 INT8 值展开为 FLOAT16），**不乘 scale**；与反量化（乘 scale，NPU 侧）严格区分 |
| D-07 | model_executor 设计已接受决策：tokenId 走模型内部 embedding，embedding 表由 pipeline 调 embedding 模块（C-02）加载，model_executor 不重新实现查表 |
| C-02 | model_executor 设计中 pipeline→embedding 模块的调用契约编号 |
| RSS | 进程驻留物理内存集，用于衡量表加载内存占用 |

## 第 3 章 背景

### 3.1 现状

`cann_lm/engine/core/component/text_embedding/text_embedding.h/.cpp` 当前为 stub era 桩实现：`TextEmbedding::LoadAndBind()` 直接返回 `FAILURE`，头注释标注"多态性待 docs/embedding DESIGN"。同文件还存放 `ModalityEntry`（多模态占位符定位条目，归 Preprocessor 消费的另一职责）。`INPUT_INFO.md` 记录 `embed_scales` 工位填充者为 embedding 模块（当前搁置）。用户需求原始输入见 `docs/embedding/proposal.md`。

### 3.2 问题

- **无表加载能力**：om 图不含内置 embedding 的模型（embedding 路径）无法获得输入 embedding；`embed_scales` 工位无人填充。
- **量化部署受阻**：低精度（INT8 等）embedding 表 + scale 分离存放的部署形态，缺少 host 侧加载与 gather 通道。
- **契约缺失**：`LoadAndBind()` 桩签名与真实需求（路径、类型、量化粒度、工位填充）不匹配，无行为规格可依。

### 3.3 期望与已确认设计决策

模块职责经 brainstorm（2026-08-27）与用户三轮拍板（2026-08-28）确定为：

| 维度 | 结论 |
|------|------|
| 职责定位 | **双路径**：tokenId 路径（om 图内置 embedding，表已包含在权重中随图加载，模块仅填 token id）｜embedding 路径（host gather 低精度行，NPU dequant） |
| 反量化位置 | **永远 NPU 侧**；host 只读低精度表 + scale 并填工位，host 不做反量化数学 |
| 文件格式 | embedding = raw 2D `[vocab, hidden]` row-major 连续；scale = raw 1D（形状由量化粒度决定，见 4.1.1） |
| 量化粒度 | 初始化输入参数 `quantMode`：**PerToken**（每词表项一个 scale）或 **PerGroup**（一个 tokenId 的 embedding 对应多个反量化参数，必填 `GroupSize`） |
| 表加载方式 | 可选：**InMemory**（整表读入 host 内存，默认）或 **InFlash**（表不读入内存，gather/填充时按行偏移直接从文件读取） |
| 结构方案 | **单类 + 类型标签分派**（不做多态层次；host gather 类型无关）；将来某类型出现差异 host 逻辑（如 INT4 解包）时再升级为策略化/多态，接口形状不变 |
| 表的交付方式 | **无 BindToModel**：表/scale 不作权重绑定。tokenId 路径无需传表（表在权重中）；embedding 路径表/scale 作为**输入工位数据**由本模块主动填入 |
| 填工位归属 | **模块主动填**（方案 B）：`FillInputs(...)` 依赖 model_executor 的 input bundle + name→index 映射 |
| 填充类型 | 实际填入类型按目标工位 `TensorDesc.dataType` 获取；用户指定的是文件系统存储类型；不匹配时填充前**必须**实现类型转换 |
| gather 输出 | `Gather` **同时输出** gathered 低精度行（`out_embed`）与按 tokenId 收集的 per-token scale（`out_scales`） |

### 3.4 模块边界

- **本模块做**：表/scale 文件加载（InMemory 整表入内存 / InFlash 留在文件按行读）、host gather（含 per-token scale 收集）、类型转换、输入工位填充（含 tokenId 路径的 token id 填充）。
- **本模块不做**：反量化（NPU）、tokenId 路径的表交付（**表已包含在模型权重中随图加载**，不经本模块）、attention_mask/position_ids 等其他工位填充（InputAssembler）、多模态占位符替换（Preprocessor，经 `ModalityEntry`）、表内存的底层申请策略（经 ITensorBuffer 抽象）、InFlash 表文件的热更新。
- **上下游基本不动，除一处接线变更**：`PipelineExecutorFactory::create` 仍按值传 `TextEmbedding`；`Preprocessor` 仍持非拥有 `TextEmbedding*`；**`Scheduler` 构造签名需新增 `TextEmbedding*` 非拥有指针**（`input_embed` 工位在现行代码中的唯一写入方是 `Scheduler` 内两个 `SetInputData` 调用点；embedding/tokenId 两路径的工位填充均由此接入 `FillInputs`，详见 DESIGN §2.2 接线点）；仅 `LoadAndBind()` 桩被新接口取代。

## 第 4 章 功能需求

### 4.1 行为规格

#### 4.1.1 初始化与加载（Load）

- **前置条件**：调用方（pipeline，M-02 于 init 期）持有完整 `EmbeddingConfig`（含表加载方式 `load_mode`）。config 经 **meta_json `embedding_config` 段 → pipeline 内联解析步骤直调 `EmbeddingConfig::FromSection`** 通道传入（SR20260903001 取代 U-8 路线 b——通道部分取代、解析语义维持，2026-09-04 用户拍板）；组装映射见 DESIGN C-01 §2。
- **处理逻辑**：
  - 模块**必须**以配置中的 `table_path` 读取 embedding 表，字节布局为 raw 2D `[vocab_size, hidden_dim]` row-major 连续。读取方式由 `load_mode` 决定：
    - **InMemory**（默认）：整表一次性读入 host 侧 `ITensorBuffer`。
    - **InFlash**：表**不读入内存**——Load 仅以文件元数据校验（打开文件、校验字节大小、记录行 stride = `hidden_dim × elem_size`），保持文件打开状态；后续 `Gather` / `FillInputs` 按行偏移（`id × stride`）直接从文件做位置读取（`pread` 类语义）。
  - 模块**必须**按 `scale_type ≠ None` 时以 `scale_path` 读取 scale 文件至 host 侧 `ITensorBuffer`。**scale 恒为内存态**（不适用 InFlash——scale 体量为 `vocab` 或 `vocab × groups` 个标量，远小于表，且每次 gather 均需按下标随机访问）。scale 元素个数由量化粒度决定：
    - **PerToken**：`vocab_size` 个元素（第 v 个元素是表第 v 行的 scale），文件字节大小 = `vocab_size × scale_elem_size`。
    - **PerGroup**：`vocab_size × groups` 个元素 row-major（token v 的第 g 组 scale 位于下标 `v × groups + g`），其中 `groups = hidden_dim / group_size`；文件字节大小 = `vocab_size × groups × scale_elem_size`。
  - 模块**必须**在 PerGroup 模式校验 `group_size` 已提供、`group_size > 0`、`group_size ≤ hidden_dim` 且 `hidden_dim % group_size == 0`，任一不满足返回 `INVALID_ARGUMENT`。
  - 模块**必须**校验文件字节大小（InMemory 实读、InFlash `stat` 元数据）与按上述规则计算的期望值一致，不一致返回 `INVALID_ARGUMENT`。
  - 模块**必须**支持 `scale_type = None`（scale 文件不存在的部署形态）：`scale_path` 为空、不加载 scale、后续 `Gather` 的 `out_scales` 不产出数据。
  - 模块**必须**在 `scale_type = None` 且 `scale_path` 非空时返回 `INVALID_ARGUMENT`（防呆：配置自相矛盾）。
  - 模块**必须**在文件缺失 / 不可读时返回 `INVALID_ARGUMENT`，不抛异常。
  - 模块**应该**在加载成功后记录表规模（vocab_size / hidden_dim / 类型 / 量化粒度 / 加载方式）供访问器查询。
- **vocab_size / hidden_dim 来源**：作为 Load 配置入参由调用方提供（预期来自 ModelConfig；见未决项 U-2）。
- **tokenId 路径与 Load 的关系**：tokenId 路径的表已包含在模型权重中随图加载，本模块的 `Load` **仅服务 embedding 路径**；tokenId 路径的 `FillInputs` 不要求先 Load（见 4.1.3）。

#### 4.1.2 查表（Gather）

- **前置条件**：模块已成功 Load；调用方提供 `ids[count]` 与两个输出 buffer。
- **处理逻辑**：
  - 模块**必须**按每个 tokenId 从表中拷贝对应的低精度行（`hidden_dim × elem_size` 字节，stride 拷贝）到 `out_embed`，布局为 `[count, hidden_dim]` 连续。拷贝为**纯字节搬运**，与存储类型无关、**不反量化**。
    - **InMemory**：从内存表 buffer 拷贝。
    - **InFlash**：按行偏移 `ids[i] × stride` 从表文件做位置读取（`pread` 类语义，不移动文件游标、可多线程并发）。
  - 模块**必须**同时按 tokenId 收集 scale 到 `out_scales`（scale 恒为内存态，见 4.1.1）：
    - PerToken：`count` 个元素（第 i 个元素 = 表第 `ids[i]` 行的 scale）。
    - PerGroup：`count × groups` 个元素（token i 的第 g 组 scale 位于下标 `i × groups + g`）。
  - 模块**必须**在任一 `ids[i] ≥ vocab_size` 时返回 `INVALID_ARGUMENT`（整批拒绝，不部分产出）。
  - 模块**必须**在 `out_embed` / `out_scales` 容量（`GetSize()`）不足时返回 `INVALID_ARGUMENT`。
  - 模块**必须**在未 Load 时返回 `INVALID_STATE`。
  - 模块**必须**在 InFlash 读取失败（I/O 错误 / 短读）时返回 `FAILURE`（文件在 Load 后被截断/替换等运行期损坏归此路径）。
  - 模块**应该**保证 `Gather` 为 `const` 只读操作，可在多线程并发调用（见第 5 章）。
  - 模块**可以**对 InFlash 批量随机读做访存局部性优化（如按文件偏移排序后顺序读），但**不得**改变 `out_embed` 的行序语义。
  - `scale_type = None` 时模块**必须**跳过 `out_scales` 写入并返回 `SUCCESS`。

#### 4.1.3 工位填充（FillInputs）

- **前置条件**：调用方（model_executor / pipeline）提供 input bundle（`vector<shared_ptr<ITensorBuffer>>`）与 name→index 映射；Embedding 路径还要求模块已成功 Load。
- **处理逻辑**：
  - 模块**必须**支持两条填充路径，由 `path` 参数选择：
    - **Embedding 路径**（`EmbeddingPath::Embedding`）：
      - 模块**必须**对 `ids` 内部执行 4.1.2 的 gather（行 + per-token scale；InMemory / InFlash 均支持）。
      - 模块**必须**将 gathered 行填入 `input_embed` 工位：**实际写入类型以该工位 `TensorDesc.dataType` 为准**；与存储类型不一致时，模块**必须**在填充前完成类型转换（转换语义见 4.1.4）。
      - 模块**必须**将收集的 scale 填入 `embed_scales` 工位（`scale_type = None` 时跳过，见未决项 U-6）：PerToken 填 `[count]`，PerGroup 填 `[count × groups]`。
    - **TokenId 路径**（`EmbeddingPath::TokenId`）：
      - **embedding 表已包含在模型权重中随图加载**，本模块不经手表/scale；模块**必须**将 token id 填入 `input_embed` 工位（按工位 `TensorDesc.dataType`，支持 INT32/INT64，其余返回 `INVALID_ARGUMENT`），om 图内完成 gather + dequant。
      - TokenId 路径**不要求模块先 Load**：未加载状态下 `FillInputs(TokenId, ...)` 仍可正常填充 token id。
  - 模块**必须**经 name→index 映射定位工位；`input_embed` 缺失返回 `NOT_FOUND`；Embedding 路径下 `scale_type ≠ None` 且 `embed_scales` 缺失返回 `NOT_FOUND`。
  - 模块**必须**在 Embedding 路径未 Load 时返回 `INVALID_STATE`。
- **路径选择归属**：由 model_executor 按 om 是否内置 embedding 决定（D-07 既有逻辑），非本模块自决。

#### 4.1.4 类型转换

- **语义**：类型转换是 host 侧**数据表示变更**（如 INT8 的整数值展开为 FLOAT16 的同值浮点），**不乘 scale**——乘 scale 属于反量化，永远在 NPU 侧。
- **转换矩阵**（存储类型 → 工位 TensorDesc 类型）：
  - 同型：直接字节拷贝。
  - INT8 → FLOAT16 / FLOAT32；FLOAT16 → FLOAT32：逐元素值展开。
  - scale 存储 Fp16 → `embed_scales` 工位 FLOAT32：逐元素值展开。
  - **其余组合（含 INT4、涉及 INT4/BF16 的任意转换）本版本不支持**，返回 `INVALID_ARGUMENT`（见未决项 U-3）。
- 模块**必须**在转换目的 buffer 容量不足时返回 `INVALID_ARGUMENT`。

#### 4.1.5 双路径数据流（端到端）

- **Embedding 路径**：`Load(表+scale，InMemory 或 InFlash)` → `FillInputs(Embedding, ids)`：内部 gather 低精度行（内存拷贝或文件按行读取）→ 类型转换 → 填 `input_embed`；收集 scale → 填 `embed_scales` → NPU 反量化 → 图后续计算。
- **TokenId 路径**：embedding 表**已包含在模型权重中随图加载**（不经本模块）→ `FillInputs(TokenId, ids)`：token id 填 `input_embed`（无需先 Load）→ NPU gather + dequant。

#### 4.1.6 错误处理总则

- 全部失败路径返回 `infra/status.h` 的 `Status` 错误码（`INVALID_ARGUMENT` / `INVALID_STATE` / `NOT_FOUND` / `FAILURE`），不抛异常跨层。
- NULL 指针入参（`ids == nullptr` 且 `count > 0` 等）返回 `INVALID_ARGUMENT`；空 `shared_ptr` 输出 buffer（`Gather` 的 `out_embed` / `out_scales`）与工位 bundle 内空 buffer（`FillInputs`）同样返回 `INVALID_ARGUMENT`。
- 加载失败后模块**必须**保持未加载状态（可重新 Load），不得半加载。
- InFlash 模式下表文件在 Load 后发生的运行期损坏（截断 / I/O 错误 / 短读）在 `Gather` / `FillInputs` 时返回 `FAILURE`。

### 4.2 接口需求

```cpp
// —— 类型（engine 内部，非 backend DataType 的子集映射）——
enum class EmbeddingType     { Fp32, Fp16, Int8, Int4 };   // 表的文件存储类型
enum class EmbeddingScaleType{ None, Fp32, Fp16 };         // scale 的文件存储类型
enum class EmbeddingQuantMode{ PerToken, PerGroup };       // scale 粒度
enum class EmbeddingPath     { TokenId, Embedding };       // FillInputs 路径选择
enum class EmbeddingLoadMode { InMemory, InFlash };        // 表加载方式：整表入内存 / 留在文件按行读

struct EmbeddingConfig {
    std::string           table_path;   // embedding 表文件（raw [vocab, hidden] row-major）
    std::string           scale_path;   // scale 文件；scale_type=None 时须为空
    EmbeddingType         table_type;
    EmbeddingScaleType    scale_type;
    EmbeddingQuantMode    quant_mode;
    size_t                group_size;   // PerGroup 必填；PerToken 忽略
    size_t                vocab_size;   // 表行数（来源 ModelConfig，见 U-2）
    size_t                hidden_dim;   // hidden 维长度（来源 ModelConfig，见 U-2）
    EmbeddingLoadMode     load_mode = EmbeddingLoadMode::InMemory;  // 表加载方式（scale 恒为内存态）

    /// 组装入口（SR20260903001：取代 U-8 路线 b 的 FromModelConfig——D-12 直接参数化）：
    /// embedding_config 段 6 字符串 + 跨段 vocab/hidden → EmbeddingConfig（字符串→枚举解析校验）。
    /// weights 为空 → SUCCESS 且 table_path 为空（调用方据此不 Load，tokenId 部署形态）。
    static Status FromSection(const std::string& weights, const std::string& dequant_scale,
                              const std::string& type, const std::string& scale_type,
                              const std::string& quant_mode, const std::string& load_mode,
                              size_t vocab_size, size_t hidden_dim, EmbeddingConfig* out);
};

class TextEmbedding {
public:
    TextEmbedding() = default;

    /// 4.1.1：加载低精度表 +（可选）scale。InMemory 整表入 host 内存；
    /// InFlash 仅校验文件元数据并保持文件打开，gather 时按行偏移直读文件。
    Status Load(const EmbeddingConfig& config);

    /// 4.1.2：embedding 路径 host 查表（按下标拷贝低精度行 + 收集 scale，不反量化）。
    /// InMemory 从内存拷贝；InFlash 按行偏移 pread。
    Status Gather(const TokenId* ids, size_t count,
                  std::shared_ptr<ITensorBuffer> out_embed,
                  std::shared_ptr<ITensorBuffer> out_scales) const;

    /// 4.1.3：主动填模型输入工位（方案 B：依赖 input bundle + name→index）。
    /// TokenId 路径不要求先 Load（表在模型权重中）；Embedding 路径要求已 Load。
    Status FillInputs(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                      const std::unordered_map<std::string, size_t>& name_index,
                      EmbeddingPath path,
                      const TokenId* ids, size_t count);

    /// 访问器（pipeline / UT 用）。
    std::shared_ptr<ITensorBuffer> Table() const;       // 未加载或 InFlash 返回 nullptr（InFlash 无整表内存映像）
    std::shared_ptr<ITensorBuffer> Scale() const;       // 未加载或 None 返回 nullptr（恒内存态）
    EmbeddingType                  TableType() const;
    EmbeddingLoadMode              LoadMode() const;
    bool                           Loaded() const;
};
```

- `ModalityEntry` 保留于同头文件（多模态占位，归 Preprocessor 的另一职责，本 spec 不约束其行为）。
- 旧桩 `LoadAndBind()` 由 `Load` 取代并删除（见第 7 章）。

### 4.3 时序

```
pipeline init (M-02, 仅 embedding 路径需要):
  TextEmbedding emb;
  emb.Load(config)                     ← 表(InMemory 整表入内存 / InFlash 仅开文件+校验) + scale(恒内存态)

每步前向（输入准备期，实际调用方为 Scheduler 的 prefill/decode 两个输入准备点；InputAssembler 填其余工位之前/并行）:
  emb.FillInputs(input_tensors_, input_name_index_, path, ids, count)
      ├─ Embedding 路径: gather(内存拷贝/文件按行读) → 类型转换 → 填 input_embed + embed_scales
      └─ TokenId 路径:   填 token id（表在模型权重中，无需 Load）

UT / 调试直查:
  emb.Gather(ids, count, out_embed, out_scales)
```

时序约束：`Load` 先于 embedding 路径的一切 `Gather` / `FillInputs`；tokenId 路径的 `FillInputs` 无 Load 前置。`FillInputs` 与 Forward 的相对时序归 model_executor 输入准备流程（`INPUT_INFO.md` §3），本模块不感知。InFlash 模式下表文件在模块生命周期内**必须**保持可访问（Load 打开、析构关闭）。

## 第 5 章 非功能需求

| 维度 | 指标 | 测试条件 | 证据来源 |
|------|------|----------|----------|
| gather 吞吐 | InMemory host gather 达到连续 memcpy 带宽的合理比例（预算目标，无实测基线，未决 U-5） | aarch64，INT8 表 `[65536, 4096]`，batch gather 测时 | 预算目标 |
| InFlash gather 延迟 | 单批 gather（≤ max_seqLen 行）文件读取延迟可接受（预算目标，无实测基线，未决 U-5；批量随机读可做偏移排序优化，见 4.1.2） | aarch64，同表规模 InFlash 模式，对比 InMemory 基线 | 预算目标 |
| 表加载内存 | InMemory 常驻 ≈ 表 + scale 文件字节大小（各一份 host buffer，无额外拷贝）；**InFlash 常驻 ≈ scale 字节 + 文件描述符**（表零常驻） | 加载后 RSS 测量 | 设计约束（单份常驻 / 零常驻） |
| 线程安全 | 同一实例 `Gather`（const）可并发（InFlash 经 `pread` 类位置读，共享 fd 亦可并发）；`Load` 不可并发、`Load` 与 `Gather`/`FillInputs` 不并发 | 并发用例 TC_3xx | 设计约束 |
| 错误可观测 | 全部失败路径返回错误码，无静默吞错、无异常逃逸 | 异常 TC_1xx | 行为契约 |
| 可移植 | g++ 13 / C++17，aarch64（Ascend 宿主）通过 | CI 编译 + GoogleTest | 环境事实 |

自检：每项可测量、可验证；无 N/A 项；未决项登记见文末。

## 第 6 章 约束与依赖

- **硬件依赖**：host 侧纯 CPU（文件读取 + memcpy + 类型转换）；无 NPU 算子依赖（反量化在图内）。
- **软件依赖**：C++17（g++ 13 已验证）、GoogleTest（UT）。不依赖 ACL/NDK 具体后端类名——仅经 `backend/api/tensor_buffer.h` 的 `ITensorBuffer` / `TensorDesc` 抽象交互（与 model_executor v0.4 "后端编译时确定"的原则一致）。
- **跨模块交互**：
  - `model_executor`：消费 `FillInputs`（输入准备期）；提供 input bundle 与 name→index；路径选择（D-07）。
  - `pipeline`：M-02 于 init 期调 `Load`；`PipelineExecutorFactory::create` 按值传 `TextEmbedding`（接口不破坏）；`Scheduler` 持非拥有 `TextEmbedding*` 并于输入准备期调 `FillInputs`（构造签名新增指针参数，见 3.4 与 DESIGN §2.2 接线点）。
  - `Preprocessor`：持非拥有 `TextEmbedding*`；`ModalityEntry` 消费方。
  - `INPUT_INFO.md`：6 工位事实来源；`embed_scales` 填充者=本模块（本次落地）。
- 自检：无隐式依赖；明确不依赖动态库热卸载、不做 NPU 内存管理。

## 第 7 章 向后兼容性

- 本模块为 engine 内部 C++ 组件，不对外暴露接口，无外部 API 二进制兼容负担。
- **内部接口变更**：`LoadAndBind()` 桩被 `Load(const EmbeddingConfig&)` 取代。当前全仓对该符号的消费为零价值（桩恒返 `FAILURE`），替换无行为回归风险；`PipelineExecutorFactory` / `Preprocessor` 的持有与传值方式不变；`Scheduler` 构造签名新增 `TextEmbedding*`（见 3.4，接线变更）。
- **行为对齐**：`embed_scales` 工位从"搁置"变为"本模块填充"（embedding 路径），属 INPUT_INFO 既定填充者的落地，非语义变更。
- 自检：framework 层保留本章；无外部接口兼容负担。

## 第 8 章 验证标准

> 编号：TC_0xx 正常 / TC_1xx 异常 / TC_2xx 边界 / TC_3xx 并发。UT 落 `tests/cpp/engine/`，GoogleTest；NPU 侧反量化结果不在 host UT 范围（后端依赖），UT 只验 host 侧加载 / 查表 / 转换 / 填工位。

### 正常路径（TC_0xx）

**TC_001 PerToken 加载成功**（关联 4.1.1；类别：正常路径）
- Given INT8 表文件 `[vocab=16, hidden=8]` + Fp32 scale 文件 `[16]` + PerToken 配置
- When `Load(config)`
- Then 返回 SUCCESS；`Table()` 字节大小 = 16×8×1；`Scale()` 字节大小 = 16×4；`Loaded()` 为真

**TC_002 PerGroup 加载成功**（关联 4.1.1；类别：正常路径）
- Given INT8 表 `[16, 8]` + Fp32 scale `[16 × 4]`（group_size=2，groups=4）+ PerGroup 配置
- When `Load(config)`
- Then 返回 SUCCESS；`Scale()` 字节大小 = 16×4×4

**TC_003 PerToken gather 正确性**（关联 4.1.2；类别：正常路径）
- Given TC_001 已加载，ids = {3, 0, 15}
- When `Gather(ids, 3, out_embed, out_scales)`
- Then `out_embed` 逐字节等于表中第 3/0/15 行拼接；`out_scales` 等于 scale[3]/scale[0]/scale[15]

**TC_004 PerGroup gather 的 scale 收集**（关联 4.1.2；类别：正常路径）
- Given TC_002 已加载，ids = {5}
- When `Gather(&id, 1, out_embed, out_scales)`
- Then `out_scales` 的 4 个元素依次等于 scale 文件中下标 `5×4+0..3` 的值

**TC_005 Embedding 路径 FillInputs 填对工位**（关联 4.1.3；类别：正常路径）
- Given 已加载（INT8 存储），mock input bundle 含 `input_embed`（desc FLOAT16 `[3, 8]`）与 `embed_scales`（desc FLOAT32 `[3]`），name→index 映射
- When `FillInputs(inputs, name_index, EmbeddingPath::Embedding, ids={3,0,15}, 3)`
- Then `input_embed` 数据为表中 3/0/15 行的 INT8 值展开成 FLOAT16；`embed_scales` 为 scale[3]/scale[0]/scale[15] 的 FLOAT32

**TC_006 TokenId 路径 FillInputs（无需 Load）**（关联 4.1.3；类别：正常路径）
- Given mock `input_embed` desc INT32 `[3]`，模块**未加载**
- When `FillInputs(..., EmbeddingPath::TokenId, ids={3,0,15}, 3)`
- Then `input_embed` 数据 = {3, 0, 15} 的 INT32 表示（表在模型权重中，不经本模块）

**TC_010 TokenId 路径 INT64 工位**（关联 4.1.3；类别：正常路径）
- Given mock `input_embed` desc INT64 `[3]`，模块**未加载**
- When `FillInputs(..., EmbeddingPath::TokenId, ids={3,0,15}, 3)`
- Then `input_embed` 数据 = {3, 0, 15} 的 INT64 表示（uint32→INT64 值展开）

**TC_007 同型免转换直填**（关联 4.1.4；类别：正常路径）
- Given 存储类型 FLOAT16、工位 desc FLOAT16
- When FillInputs（Embedding 路径）
- Then 数据逐字节等于 gathered 原始行（无转换路径命中）

**TC_008 InFlash 加载与 gather 正确性**（关联 4.1.1/4.1.2；类别：正常路径）
- Given TC_001 同一表文件与配置，但 `load_mode = InFlash`
- When `Load(config)` 后 `Gather(ids={3,0,15}, 3, out_embed, out_scales)`
- Then 返回 SUCCESS；`Table()` 为 nullptr、`LoadMode()` 为 InFlash、`Loaded()` 为真；`out_embed`/`out_scales` 与 TC_003（InMemory）逐字节一致

**TC_009 InFlash 模式 FillInputs（Embedding 路径）**（关联 4.1.3；类别：正常路径）
- Given InFlash 已加载（INT8 存储），mock `input_embed` desc FLOAT16 + `embed_scales` desc FLOAT32
- When `FillInputs(..., EmbeddingPath::Embedding, ids, 3)`
- Then 两工位数据与 TC_005（InMemory 同配置）逐字节一致

### 异常路径（TC_1xx）

**TC_101 表文件缺失**（关联 4.1.1）→ `Load` 返回 `INVALID_ARGUMENT`，`Loaded()` 为假，可重新 Load 成功。
**TC_102 文件尺寸不符**（关联 4.1.1）→ 表或 scale 字节数与期望不符，`INVALID_ARGUMENT`。
**TC_103 PerGroup 参数非法**（关联 4.1.1）→ group_size=0 / 未提供 / 不整除 hidden / scale_path 空，各返回 `INVALID_ARGUMENT`。
**TC_104 scale 配置自相矛盾**（关联 4.1.1）→ `scale_type=None` 且 `scale_path` 非空 → `INVALID_ARGUMENT`。
**TC_105 tokenId 越界**（关联 4.1.2）→ 任一 id ≥ vocab → `INVALID_ARGUMENT`，`out_embed` 不被部分写入。
**TC_106 未加载先查**（关联 4.1.2/4.1.3）→ `Gather` 与 `FillInputs(Embedding, ...)` 返回 `INVALID_STATE`；`FillInputs(TokenId, ...)` 不受影响（见 TC_006）。
**TC_107 工位缺失**（关联 4.1.3）→ name→index 无 `input_embed` 或（Embedding 路径且 scale_type≠None 时）无 `embed_scales` → `NOT_FOUND`。
**TC_108 不支持的转换组合**（关联 4.1.4）→ 存储 INT4 → 工位 FLOAT16 → `INVALID_ARGUMENT`。
**TC_109 输出容量不足**（关联 4.1.2/4.1.4）→ `out_embed`/`out_scales` GetSize 偏小 → `INVALID_ARGUMENT`。
**TC_110 NULL 入参**（关联 4.1.6）→ `ids=nullptr, count>0` → `INVALID_ARGUMENT`。
**TC_111 InFlash 运行期文件损坏**（关联 4.1.6）→ InFlash 已加载后截断表文件，`Gather` / `FillInputs`（Embedding 路径）返回 `FAILURE`。
**TC_112 空 buffer 入参**（关联 4.1.6）→ `Gather` 传空 `shared_ptr` 的 `out_embed`/`out_scales`；`FillInputs` 传含空 buffer 的工位 → `INVALID_ARGUMENT`，无解引用崩溃。
**TC_113 FromModelConfig 未知枚举字符串**（关联 4.1.1/4.2，U-8 路线 b）→ `ModelConfig.embedding_table_type = "fp8"` 等未知值 → `INVALID_ARGUMENT`；`embedding_table_path` 为空 → SUCCESS 且 `table_path` 为空（不加载信号）。

### 边界（TC_2xx）

**TC_201 count=0**（关联 4.1.2）→ `Gather(ids, 0, ...)` 返回 SUCCESS，不写任何字节。
**TC_202 scale_type=None**（关联 4.1.1/4.1.2）→ 无 scale 文件加载成功；Gather 跳过 `out_scales` 返回 SUCCESS；FillInputs 跳过 `embed_scales`（处置见 U-6）。
**TC_203 vocab 边界 id**（关联 4.1.2）→ id = vocab-1 成功、id = vocab 拒绝。
**TC_204 加载失败后可恢复**（关联 4.1.6）→ 失败 Load 后修正配置再 Load 成功。

### 并发（TC_3xx）

**TC_301 并发 Gather（InMemory）**（关联第 5 章）→ 4 线程对同一已加载实例并发 Gather 相同/不同 ids，结果与串行一致，无崩溃（可选 TSan）。
**TC_302 并发 Gather（InFlash）**（关联第 5 章）→ 4 线程共享同一 InFlash 实例（共享 fd）并发 Gather，结果与串行一致，无崩溃；位置读取不得相互干扰游标。

## 未决项登记

| ID | 未决项 | 类别 | 阻塞 spec 放行 | 处置阶段 |
|----|--------|------|----------------|----------|
| ~~U-1~~ | ~~TokenId 路径下整表/scale 进图的通道~~ **已解决（2026-08-28 用户确认）**：tokenId 路径无需额外传入 embedding 表——**表已包含在模型权重中随图加载**，本模块只填 token id | 系统边界 | 否（已解决） | 已并入 4.1.1/4.1.3/4.1.5 |
| U-2 | `vocab_size` / `hidden_dim` 的权威来源（本 spec 假设为 Load 入参、由 ModelConfig 传入）。U-8 决议后来源已定 = `ModelConfig` 扩展字段（与 embedding 通道一并落地），仅余字段命名/类型对齐 | 接口契约 | 否 | 实现期随 ModelConfig 扩展落地 |
| U-3 | INT4 / BF16 存储类型：backend `DataType` 无对应枚举、INT4 需位解包——本版本不支持 | 功能范围 | 否（枚举占位、转换矩阵显式排除） | 出现真实需求时升级结构方案（多态/策略化） |
| ~~U-4~~ | ~~`INPUT_INFO.md` 订正~~ **已完成（2026-08-29 随实现落地，INPUT_INFO v0.4）**：`embed_scales` 维度已补 PerGroup `×groups`；`input_embed` dtype 已改"按 TensorDesc，随路径变化"；填充者/流程表述同步 FillInputs 接线 | 文档一致性 | 否（已完成） | 已闭环 |
| U-5 | gather 吞吐 / InFlash 读取延迟 / 加载内存预算无实测基线 | 验收标准 | 否（预算目标） | 实现期微基准 |
| U-6 | `scale_type=None` 时 `embed_scales` 工位处置（不填 vs 填 1.0） | 行为契约 | 否（本 spec 取"跳过不填"） | 首个无 scale 模型接入时确认 |
| U-7 | InFlash 模式表文件生命周期：模块生命周期内须保持可访问（4.3 已约束）；文件被替换/热更新场景是否需要支持 | 部署约束 | 否（本 spec 不支持热更新，损坏归 TC_111 的 FAILURE 路径） | 出现热更新需求时扩展 |
| ~~U-8~~ | ~~`EmbeddingConfig` 的用户→pipeline 传递通道未决议~~ **已解决（2026-08-29 用户拍板，路线 b）；后被 SR20260903001（2026-09-04 用户拍板）部分取代**：通道改为 meta_json `embedding_config` 段 → pipeline 内联解析步骤直调 `EmbeddingConfig::FromSection`（8 参数直入，D-12；ModelConfig 不再承载 embedding 扩展字段）；vocab/hidden 跨段取自 model_config 段。解析与校验语义维持本 spec 口径 | 接口契约 | 否（已解决） | 4.1.1 与 DESIGN C-01/F-01 已同步 |

**阻塞级未决项清零**：U-1 已解决（表在权重中）；U-8 已解决（通道走 meta_json→ModelConfig，2026-08-29）；U-2/U-4 为对齐类（U-2 来源已定仅余字段对齐）；U-3/U-5/U-6/U-7 为范围/预算/部署类。无信任/授权边界、无安全红线相关未决项。**全部未决项均不再阻塞进入实现。**

## 一致性自检（生成期）

- 接口契约覆盖：4.2 每个语义接口均可追溯至 4.1 行为（Load↔4.1.1、Gather↔4.1.2、FillInputs↔4.1.3、访问器↔4.1.1）。✓
- 测试覆盖：4.1 每条"必须"在第 8 章有 TC（加载/查表/填工位/转换/错误/边界/并发）。✓
- 术语一致：全文术语在第 2 章有定义（含 PerToken/PerGroup/存储类型/填充类型的区分）。✓
- 时序一致：4.3 时序与 4.1 及 INPUT_INFO §3 无矛盾。✓
- 层次适配：framework 层含第 7 章。✓
- 决策可追溯：3.3 决策表逐条对应 brainstorm 会话（2026-08-27）与用户确认（2026-08-28）。✓
