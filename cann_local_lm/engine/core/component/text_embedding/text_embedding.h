#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// TextEmbedding（v0.12 → v2.0：stub 桩 LoadAndBind 原地重写，SR20260828001）。
// 权威设计：docs/embedding/DESIGN.md（v0.3）。行为规格：docs/embedding/SPEC.md。
//
// 单类 façade（D-03）：Load（表/scale 加载，InMemory/InFlash 运行期配置）/
// Gather（host 查表 + scale 收集，纯字节搬运）/ FillInputs（主动填 input_embed
// 与 embed_scales 工位，tokenId/embedding 双路径）。
// host 只搬运与值展开，不反量化——乘 scale 永远在 NPU（D-01）。
// InMemory 表/scale 一律存于堆内存（unique_ptr<char[]>）——TensorBuffer 仅用于
// 模型输入/输出工位，不得作通用 host 大容量存储（TensorBuffer 内存对 CPU 可能不可写）。
// 内部帮手 TableSource（M-02）/ 类型转换（M-03）为 .cpp 内 detail 实现细节，
// 不出现在本公开头。
// ═══════════════════════════════════════════════════════════════════════════

#include "infra/status.h"
#include "api/cpp/types.h"
#include "core/tensor_manager.h" // ITensorBuffer/TensorDesc/DataType/TensorManager/ModelConfig

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cann_local_lm::engine {

/// M-02 行读取策略（仅前向声明；类定义在 .cpp 的 detail 命名空间，公开头无 API 面）。
namespace detail {
class TableSource;
} // namespace detail

/// 多模态 embedding 在 token 序列中的占位符定位条目（Preprocessor 用于占位符替换）。
/// 物理同文件的另一职责，本模块不约束其行为。
struct ModalityEntry {
    size_t position;                          // 占位符在 token 序列中的下标
    std::shared_ptr<ITensorBuffer> embedding; // 对应多模态 embedding（来自 Prompt，已算好）
};

/// 表的文件存储类型（描述磁盘字节布局；Int4 占位，U-3 本版本不支持）。
enum class EmbeddingType { Fp32, Fp16, Int8, Int4 };

/// scale 的文件存储类型。
enum class EmbeddingScaleType { None, Fp32, Fp16 };

/// scale 量化粒度（D-07）：PerToken = 每词表项一个；PerGroup = 每组一个。
enum class EmbeddingQuantMode { PerToken, PerGroup };

/// FillInputs 路径选择（选择权归 model_executor D-07，非本模块自决）。
enum class EmbeddingPath { TokenId, Embedding };

/// embedding 输入工位名（omc 导出名歧义仲裁）：
/// 引擎规范名 input_embed，omc 常见导出别名 embeddings（om 侧不改名，引擎
/// 统一在此解析；FillInputs/维度修正/SetInputData/debug inject 四消费点共用）。
inline constexpr char kEmbedSlotName[] = "input_embed";
inline constexpr char kEmbedSlotAlias[] = "embeddings";

/// name→index 解析 embedding 输入工位：规范名优先、别名兜底（两者并存取规范名）；
/// 均缺 → name_index.end()（各消费点自行按各自口径报错）。
inline std::unordered_map<std::string, std::size_t>::const_iterator FindEmbedSlot(
    const std::unordered_map<std::string, std::size_t>& name_index)
{
    auto it = name_index.find(kEmbedSlotName);
    return it != name_index.end() ? it : name_index.find(kEmbedSlotAlias);
}

/// om 输入 desc 工位名匹配（维度修正等按名遍历场景）：规范名或别名。
inline bool IsEmbedSlotName(const std::string& name) { return name == kEmbedSlotName || name == kEmbedSlotAlias; }

/// embedding scale 输入工位名：引擎规范名 embed_scales，omc 转换器导出别名
/// embeddings_scale（gemma4 c42 实名，解析同 FindEmbedSlot 口径）。
inline constexpr char kEmbedScaleSlotName[] = "embed_scales";
inline constexpr char kEmbedScaleSlotAlias[] = "embeddings_scale";

/// name→index 解析 embedding scale 输入工位：规范名优先、别名兜底
/// （两者并存取规范名）；均缺 → name_index.end()。
inline std::unordered_map<std::string, std::size_t>::const_iterator FindEmbedScaleSlot(
    const std::unordered_map<std::string, std::size_t>& name_index)
{
    auto it = name_index.find(kEmbedScaleSlotName);
    return it != name_index.end() ? it : name_index.find(kEmbedScaleSlotAlias);
}

/// om 输入 desc 工位名匹配（维度修正按名遍历场景）：规范名或别名。
inline bool IsEmbedScaleSlotName(const std::string& name)
{
    return name == kEmbedScaleSlotName || name == kEmbedScaleSlotAlias;
}

/// gemma4 per-token identity 查表的两个模型输入工位名。工位名正拼
/// per_layer_token_identity_scale（老转换器时代契约拼写 ..._sacle 已弃用，
/// 须以新转换器重导出图）。meta 键亦仅规范拼写 token_identity_scale_type
/// ——读取处见 pipeline_executor.cpp ParseEmbeddingSection。
inline constexpr char kTokenIdentitySlotName[] = "per_layer_token_identity";
inline constexpr char kTokenIdentityScaleSlotName[] = "per_layer_token_identity_scale";

/// name→index 解析 identity scale 输入工位；缺（name_index 无该工位）→
/// name_index.end()。
inline std::unordered_map<std::string, std::size_t>::const_iterator FindIdentityScaleSlot(
    const std::unordered_map<std::string, std::size_t>& name_index)
{
    return name_index.find(kTokenIdentityScaleSlotName);
}

/// 表加载方式（D-08，运行期配置非编译期）：InMemory 整表入内存 / InFlash 留文件按行 pread。
/// scale 无论何种方式恒为内存态（D-09）。
enum class EmbeddingLoadMode { InMemory, InFlash };

/// Load 配置（4.1.1）。来源：ModelConfig 经 FromModelConfig 组装（U-8 路线 b）。
struct EmbeddingConfig {
    std::string table_path; // 合并单文件（四段布局，见下）；空 = 不加载（tokenId 部署形态）
    EmbeddingType table_type = EmbeddingType::Fp32;
    EmbeddingScaleType scale_type = EmbeddingScaleType::None;
    EmbeddingQuantMode quant_mode = EmbeddingQuantMode::PerToken;
    size_t group_size = 0; // PerGroup 必填；PerToken 忽略
    size_t vocab_size = 0; // 来源 ModelConfig（U-2）
    size_t hidden_dim = 0; // 来源 ModelConfig（U-2）
    EmbeddingLoadMode load_mode = EmbeddingLoadMode::InMemory;

    // ── per_layer_token_identity（gemma4）──
    /// 单文件四段布局（段内行优先，无头部、无段间 padding）：
    ///   [主表 vocab×hidden][identity 表 vocab×W][主 scale][identity scale]
    /// 各段偏移/长度均由本结构字段计算；identity/identity scale 段是否读取由
    /// pipeline 依图工位判定（以图为准）：图带工位 ⇒ 从对应偏移读取。
    /// meta 键 token_identity_type：identity 表读取类型；缺省 fp16。
    EmbeddingType token_identity_type = EmbeddingType::Fp16;
    /// meta 键 token_identity_scale_type：identity scale 读取类型；缺省 fp16；
    /// 仅图带 scale 工位时被消费——pipeline 依图回填 None = 不读。
    EmbeddingScaleType token_identity_scale_type = EmbeddingScaleType::Fp16;
    /// 非来自 meta_json：pipeline 自工位 dims 尾维连乘推导后回填。Load 与
    /// LoadTokenIdentity 均以本字段计算段偏移（唯一入口，两方法恒同源）。
    size_t token_identity_row_width = 0;

    /// embedding_config 段 7 字符串（解析步骤提取）+ model_config 跨段
    /// vocab/hidden → EmbeddingConfig。字符串→枚举解析与校验在此。
    /// weights 为空 → out->table_path 为空（"不加载"信号，tokenId 部署形态，
    /// 调用方据此不调 Load）。
    /// 主表与 identity 解析互相独立：weights 空仍解析 identity 读取类型
    /// （identity 数据与主表同一文件，启用与否由图工位判定）；identity host
    /// 查表只服务 Embedding 部署形态（TokenId 形态 identity 由图内自理，
    /// pipeline InitEmbedding_ 门控）。
    /// 错误：未知枚举 → INVALID_ARGUMENT（解析步骤转段降级）。
    static Status FromSection(const std::string& weights,
                              const std::string& type,                // "fp32"|"fp16"|"int8"|"int4"
                              const std::string& scale_type,          // ""|"none"|"fp32"|"fp16"
                              const std::string& quant_mode,          // ""|"per_token"|"per_group"
                              const std::string& load_mode,           // ""|"in_memory"|"in_flash"
                              const std::string& identity_type,       // ""|"fp32"|"fp16"|"int8"|"int4"
                              const std::string& identity_scale_type, // ""|"fp32"|"fp16"
                              size_t vocab_size, size_t hidden_dim, EmbeddingConfig* out);
};

/// engine 内部 embedding 模块（pipeline/model_executor/UT 消费；不对外暴露）。
/// 不可复制（表 fd 所有权语义清晰）；可移动（移动后原对象置未加载态）。
class TextEmbedding {
public:
    TextEmbedding(); // .cpp 定义（pimpl 惯例：source_ 不完整类型，特殊成员全部出编译单元）
    ~TextEmbedding();

    TextEmbedding(const TextEmbedding&) = delete;
    TextEmbedding& operator=(const TextEmbedding&) = delete;
    TextEmbedding(TextEmbedding&&) noexcept;
    TextEmbedding& operator=(TextEmbedding&&) noexcept;

    /// 4.1.1：加载主表 +（可选）主 scale 段（单文件四段布局：scale 段位于
    /// 主表段与 identity 表段之后）。InMemory 整表 mmap 入映射页；
    /// InFlash 仅校验文件元数据并保持文件打开，gather 时按行偏移直读文件；
    /// scale 恒内存态（D-09）。语义：同步；调用方保证不并发、不与
    /// Gather/FillInputs 并发（契约见 DESIGN §2.4）。失败保持未加载态（可重 Load）。
    /// 错误：文件缺失/尺寸不符（含 scale 段读不足）/PerGroup 参数非法 → INVALID_ARGUMENT
    Status Load(const EmbeddingConfig& config);

    /// 4.1.2：embedding 路径 host 查表（按下标拷贝低精度行 + 收集 scale，不反量化）。
    /// 语义：const 只读、线程安全（可并发）。scale_type=None 时跳过 out_scales 写入。
    /// 错误：未 Load → INVALID_STATE；ids 越界 / 容量不足 / NULL（含 ids=nullptr、
    ///       out_embed 为空 shared_ptr；scale 需要时 out_scales 为空）→ INVALID_ARGUMENT；
    ///       InFlash 读失败（I/O/短读）→ FAILURE
    Status Gather(const TokenId* ids, size_t count, std::shared_ptr<ITensorBuffer> out_embed,
                  std::shared_ptr<ITensorBuffer> out_scales) const;

    /// 4.1.3：主动填模型输入工位（依赖 input bundle + name→index，D-05）。
    /// TokenId 路径不要求先 Load（表在模型权重中，D-02）；Embedding 路径要求已 Load。
    ///       写入类型以工位 TensorDesc.dataType 为准（D-06），不匹配经值展开转换。
    /// 工位名：input_embed 优先、omc 导出别名 embeddings 兜底（FindEmbedSlot）。
    /// 错误：input_embed/embeddings 均缺名 → NOT_FOUND；(Embedding 路径且 scale_type≠None)
    ///       embed_scales 缺名 → NOT_FOUND；Embedding 路径未 Load → INVALID_STATE；
    ///       转换组合不支持 / 容量不足 / NULL（含 ids=nullptr 且 count>0、工位 buffer 为空）
    ///       → INVALID_ARGUMENT
    Status FillInputs(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                      const std::unordered_map<std::string, size_t>& name_index, EmbeddingPath path, const TokenId* ids,
                      size_t count);

    /// gemma4 per_layer_token_identity 表加载：与 Load 同构的第二个表源
    /// （独立状态，不混用 loaded_）。identity 表取自 config.table_path 主表段
    /// 之后；identity scale（token_identity_scale_type ≠ None 时）取自主 scale
    /// 段之后（PerToken 恒 vocab 项；None 由 pipeline 依图回填 = 不读取）。
    /// 行宽 W 取 config.token_identity_row_width（pipeline 回填，非 meta 字段）。
    /// 全文件总长按四段和精确判等（identity 表为末段时不含后段）。失败保持
    /// identity 未加载态。
    /// 错误：文件缺失/段尺寸不符（短/超长）/Int4（U-3）→ INVALID_ARGUMENT
    Status LoadTokenIdentity(const EmbeddingConfig& config);

    // ── 访问器（pipeline / UT 用；均不抛异常）──
    // 无 Table()/Scale() 存储暴露接口：TensorBuffer 仅用于模型输入/输出工位，
    // 表数据消费一律走 FillInputs / FillTokenIdentitySlot_（Gather 亦经其行搬运）。
    EmbeddingType TableType() const;    // 未加载 → Fp32（哨兵）+ Loaded() 判别
    EmbeddingLoadMode LoadMode() const; // 未加载 → InMemory（哨兵）
    bool Loaded() const;
    bool IdentityLoaded() const;

private:
    /// FillInputs 内部 helper（§5.3）：identity 未启用 → SUCCESS（零行为变化）；
    /// 启用则查两工位按 token 落位（scale PerToken）。仅 Embedding 路径分派时
    /// 执行（TokenId 部署形态不查表，identity 由图内自理）；错误原样上抛
    /// （Scheduler 现行只 WARN，不阻断）。
    Status FillTokenIdentitySlot_(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                                  const std::unordered_map<std::string, size_t>& name_index, const TokenId* ids,
                                  size_t count);

    std::unique_ptr<detail::TableSource> source_; // InMemory/InFlash 二态
    std::unique_ptr<char[]> scale_data_;          // scale 恒内存态（D-09）；堆内存非 TensorBuffer
    EmbeddingConfig cfg_{};                       // 加载成功的配置快照（Load 后只读）
    size_t row_bytes_ = 0;                        // hidden × 存储元素大小
    size_t groups_ = 1;                           // PerGroup = hidden/group_size；PerToken 恒 1
    bool loaded_ = false;

    // ── per_layer_token_identity：与主表并列的第二表源（generate 独立，不混用上方状态）──
    std::unique_ptr<detail::TableSource> identity_source_; // InMemory/InFlash 二态
    std::unique_ptr<char[]> identity_scale_data_;          // PerToken scale，恒内存态（D-09）；堆内存
    EmbeddingConfig identity_cfg_{}; // LoadTokenIdentity 成功后的配置快照（含行宽 W）
    size_t identity_row_bytes_ = 0;  // 行宽 W × 存储元素大小
    bool identity_loaded_ = false;
};

} // namespace cann_local_lm::engine
