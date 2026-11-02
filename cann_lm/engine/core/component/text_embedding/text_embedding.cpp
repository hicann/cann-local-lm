// ═══════════════════════════════════════════════════════════════════════════
// TextEmbedding 实现（SR20260828001，DESIGN v0.3）。
// 内部帮手（不出公开头）：
//   detail::ConvertRows/ConvertScales/ConversionSupported —— M-03 纯函数转换
//   detail::MmapTableSource / detail::FlashTableSource —— M-02 表源二态
//   （InMemory=mmap 只读映射 L2 / InFlash=open+pread 直读）
// host 只搬运与值展开，不反量化（D-01）。
// 通用 dtype 工具（fp16 位级转换/dtype 字节数与简名/格式化打印）在
// utils/data_type_utils.h，本模块不再自带。
// ═══════════════════════════════════════════════════════════════════════════
#include "core/component/text_embedding/text_embedding.h"

#include "infra/log.h"
#include "utils/data_type_utils.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace cann_lm::engine {

namespace detail {

// ── M-03：存储元素大小 ──────────────────────────────────────────────────────
size_t StorageElemSize(EmbeddingType t)
{
    switch (t) {
        case EmbeddingType::Fp32:
            return 4;
        case EmbeddingType::Fp16:
            return 2;
        case EmbeddingType::Int8:
            return 1;
        case EmbeddingType::Int4:
            return 0; // U-3：不支持
    }
    return 0;
}

size_t ScaleElemSize(EmbeddingScaleType t)
{
    switch (t) {
        case EmbeddingScaleType::Fp32:
            return 4;
        case EmbeddingScaleType::Fp16:
            return 2;
        case EmbeddingScaleType::None:
            return 0;
    }
    return 0;
}

/// 工位/tokenId 工位的元素大小（全 dtype 覆盖，未知 → 0）。
size_t SlotElemSize(DataType t) { return DataTypeElemSize(t); }

// ── M-03：转换矩阵（SPEC §4.1.4 唯一权威的代码落点）────────────────────────
/// 同型 = 直接字节拷贝；Fp16→FLOAT32、Int8→FLOAT16/FLOAT32 = 值展开；
/// 其余（含 Int4，U-3）不支持。
bool ConversionSupported(EmbeddingType src, DataType dst)
{
    switch (src) {
        case EmbeddingType::Fp32:
            return dst == DataType::FLOAT32;
        case EmbeddingType::Fp16:
            return dst == DataType::FLOAT16 || dst == DataType::FLOAT32;
        case EmbeddingType::Int8:
            return dst == DataType::INT8 || dst == DataType::FLOAT16 || dst == DataType::FLOAT32;
        case EmbeddingType::Int4:
            return false; // U-3
    }
    return false;
}

/// 行数据转换：纯值展开（不乘 scale，D-01）。同型 memcpy。
Status ConvertRows(const void* src, EmbeddingType src_type, void* dst, DataType dst_type, size_t elem_count)
{
    if (src == nullptr || dst == nullptr)
        return Status(Status::INVALID_ARGUMENT);
    if (!ConversionSupported(src_type, dst_type))
        return Status(Status::INVALID_ARGUMENT);

    if (src_type == EmbeddingType::Fp32) { // 同型 memcpy
        std::memcpy(dst, src, elem_count * 4);
    } else if (src_type == EmbeddingType::Fp16) {
        if (dst_type == DataType::FLOAT16) {
            std::memcpy(dst, src, elem_count * 2);
        } else { // Fp16 → FLOAT32 值展开
            auto* s = static_cast<const fp16_t*>(src);
            auto* d = static_cast<float*>(dst);
            for (size_t i = 0; i < elem_count; ++i)
                d[i] = static_cast<float>(s[i]);
        }
    } else { // Int8
        if (dst_type == DataType::INT8) {
            std::memcpy(dst, src, elem_count);
        } else if (dst_type == DataType::FLOAT16) {
            auto* s = static_cast<const int8_t*>(src);
            auto* d = static_cast<fp16_t*>(dst);
            for (size_t i = 0; i < elem_count; ++i) {
                d[i] = fp16_t(static_cast<float>(s[i]));
            }
        } else { // Int8 → FLOAT32
            auto* s = static_cast<const int8_t*>(src);
            auto* d = static_cast<float*>(dst);
            for (size_t i = 0; i < elem_count; ++i)
                d[i] = static_cast<float>(s[i]);
        }
    }
    return Status();
}

/// scale 转换：目标 = 工位类型（FLOAT32/FLOAT16，D-06 工位为准）。
/// 同型 memcpy；Fp16→FLOAT32 值展开；FLOAT32→FLOAT16 经软件转换（位级精确）。
Status ConvertScales(const void* src, EmbeddingScaleType src_type, void* dst, DataType dst_type, size_t elem_count)
{
    if (src == nullptr || dst == nullptr)
        return Status(Status::INVALID_ARGUMENT);
    if (dst_type != DataType::FLOAT32 && dst_type != DataType::FLOAT16) {
        return Status(Status::INVALID_ARGUMENT);
    }
    if (src_type == EmbeddingScaleType::Fp32) {
        if (dst_type == DataType::FLOAT32) {
            std::memcpy(dst, src, elem_count * 4);
        } else { // Fp32 → FLOAT16
            auto* s = static_cast<const float*>(src);
            auto* d = static_cast<fp16_t*>(dst);
            for (size_t i = 0; i < elem_count; ++i)
                d[i] = fp16_t(s[i]);
        }
    } else if (src_type == EmbeddingScaleType::Fp16) {
        if (dst_type == DataType::FLOAT16) {
            std::memcpy(dst, src, elem_count * 2); // 同型
        } else {                                   // Fp16 → FLOAT32 值展开
            auto* s = static_cast<const fp16_t*>(src);
            auto* d = static_cast<float*>(dst);
            for (size_t i = 0; i < elem_count; ++i)
                d[i] = static_cast<float>(s[i]);
        }
    } else {
        return Status(Status::INVALID_ARGUMENT);
    }
    return Status();
}

// ── 调试：查表结果打印的量纲上限（FillInputs 落工位后调用；纯读，不改行为）──
// 首 kDumpRows 个 token 各详印前 kDumpCols 维，防 hidden/count 大时刷屏。
constexpr size_t kDumpRows = 4; // 最多详印前 4 个 token
constexpr size_t kDumpCols = 8; // 每 token 最多详印前 8 维

// ── M-02：表源（统一 ReadRow 行读取契约，类型无关纯字节搬运）──────────────
// InMemory 载体为 mmap 只读映射（L2：RAII munmap，RowPtr 直指映射页；段偏移
// 即映射 offset，校验序 open→fstat 段尾→mmap→madvise(MADV_SEQUENTIAL)，
// 短文件在 map 前拒绝，无 SIGBUS 窗口）——TensorBuffer 仅用于模型输入/输出
// 工位，不作 host 大容量存储（TensorBuffer 内存对 CPU 可能不可写，作存储会
// 静默读 0）。
class TableSource {
public:
    virtual ~TableSource() = default;
    /// 把第 id 行 row_bytes 字节写入 dst。InFlash 短读/I/O 错 → FAILURE。
    virtual Status ReadRow(TokenId id, void* dst) const = 0;
    virtual size_t RowBytes() const = 0;
    /// InMemory：返回映射页中该行只读指针（供 FillInputs 免 staging 直转）；
    /// InFlash：nullptr（须走 ReadRow + 栈上 staging）。
    virtual const void* RowPtr(TokenId id) const = 0;
};

/// mmap 只读表源（L2）：生命周期持映射（RAII munmap），行 0 数据位于映射内
/// map_head_ 处。主表段 file_offset=0、identity 段 = 主表段之后，两段各自
/// 独立 mmap（L1 并行下两映射互不干扰；缺页即读盘，MADV_SEQUENTIAL 交内核
/// 顺序预读）。mmap offset 须页对齐：映射自段起点的页对齐下界起，段头
/// 页内残量由 map_head_ 吸收（RowPtr = map + map_head_ + id*row_bytes）。
class MmapTableSource final : public TableSource {
public:
    /// 段几何校验 + 建映射：fstat size ≥ file_offset + 段长（require_exact
    /// 时取 ==）→ mmap(PROT_READ|MAP_PRIVATE，页对齐下界)。
    /// 截断拒绝在 map 前完成（无 SIGBUS 窗口；运行中截断部署期不存在，R-2）。
    static Status Create(const std::string& path, uint64_t file_offset, size_t row_bytes, size_t rows,
                         bool require_exact, std::unique_ptr<TableSource>* out)
    {
        if (out == nullptr || row_bytes == 0 || rows == 0) {
            return Status(Status::INVALID_ARGUMENT);
        }
        // rows * row_bytes 溢出防护（调用方已防，双扣）
        if (rows > std::numeric_limits<std::size_t>::max() / row_bytes) {
            return Status(Status::INVALID_ARGUMENT);
        }
        const size_t len = rows * row_bytes;
        const uint64_t seg_end = file_offset + len; // 调用方已防回绕（双扣）
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            return Status(Status::INVALID_ARGUMENT);
        struct stat sb {};
        const uint64_t fsize = (::fstat(fd, &sb) == 0) ? static_cast<uint64_t>(sb.st_size) : 0;
        const bool size_ok = require_exact ? (fsize == seg_end) : (fsize >= seg_end);
        if (::fstat(fd, &sb) != 0 || !size_ok) {
            ::close(fd);
            return Status(Status::INVALID_ARGUMENT); // 截断/超长：map 前拒绝
        }
        // 页对齐下界映射：map_off ≤ file_offset < map_off + PAGE；段尾页内
        // 越出由 mmap 向上取整页覆盖（访问仍限 [map_head_, map_head_+len)，
        // 段后页脚越读不会发生——RowPtr 恒以行界为界）
        const long page = ::sysconf(_SC_PAGESIZE);
        const uint64_t map_off = (page > 0) ?
                                     (file_offset / static_cast<uint64_t>(page)) * static_cast<uint64_t>(page) :
                                     file_offset;
        const size_t map_head = static_cast<size_t>(file_offset - map_off);
        const size_t map_len = map_head + len;
        void* map = ::mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(map_off));
        ::close(fd); // 映射建立后 fd 即可关（映射生命周期独立于 fd）
        if (map == MAP_FAILED)
            return Status(Status::FAILURE);
#ifdef MADV_SEQUENTIAL
        ::madvise(map, map_len, MADV_SEQUENTIAL); // 内核顺序预读替代 stream 两跳
#endif
        out->reset(new MmapTableSource(map, map_len, map_head, row_bytes));
        return Status();
    }
    ~MmapTableSource() override
    {
        if (map_ != nullptr)
            ::munmap(map_, len_);
    }
    MmapTableSource(const MmapTableSource&) = delete;
    MmapTableSource& operator=(const MmapTableSource&) = delete;

    Status ReadRow(TokenId id, void* dst) const override
    {
        std::memcpy(dst, RowPtr(id), row_bytes_);
        return Status();
    }
    size_t RowBytes() const override { return row_bytes_; }
    const void* RowPtr(TokenId id) const override
    {
        return static_cast<const char*>(map_) + map_head_ + static_cast<size_t>(id) * row_bytes_; // 直指映射页内段头
    }

private:
    // 私有构造：建源唯经静态工厂 Create（校验序/madvise 集中于工厂）
    MmapTableSource(void* map, size_t len, size_t map_head, size_t row_bytes)
        : map_(map), len_(len), map_head_(map_head), row_bytes_(row_bytes)
    {}

    void* map_;       // RAII：Create 建立映射、析构 munmap
    size_t len_;      // 映射长（map_head + 段长，向上页取整后）
    size_t map_head_; // 段头在映射内的页内偏移（file_offset % PAGE）
    size_t row_bytes_;
};

class FlashTableSource final : public TableSource {
public:
    FlashTableSource(int fd, size_t row_bytes, uint64_t file_offset = 0)
        : fd_(fd), row_bytes_(row_bytes), file_offset_(file_offset)
    {}
    ~FlashTableSource() override
    {
        if (fd_ >= 0)
            ::close(fd_);
    }
    FlashTableSource(const FlashTableSource&) = delete;
    FlashTableSource& operator=(const FlashTableSource&) = delete;

    Status ReadRow(TokenId id, void* dst) const override
    {
#ifdef _WIN32
        // host 无 pread;InFlash 路径在 host 回归不用(tokenId 路径)。
        (void)id;
        (void)dst;
        return Status(Status::FAILURE);
#else
        const uint64_t off = file_offset_ + static_cast<uint64_t>(static_cast<size_t>(id) * row_bytes_);
        ssize_t n = ::pread(fd_, dst, row_bytes_, static_cast<off_t>(off));
        if (n < 0 || static_cast<size_t>(n) != row_bytes_) {
            return Status(Status::FAILURE); // 运行期损坏（截断/I/O），U-7
        }
        return Status();
#endif
    }
    size_t RowBytes() const override { return row_bytes_; }
    const void* RowPtr(TokenId) const override { return nullptr; }

private:
    int fd_; // RAII：Load 打开、析构关闭（fd 生命周期 = 模块生命周期）
    size_t row_bytes_;
    /// 文件内段偏移（主表恒 0；identity 源 = 主表段之后，合并文件布局——Q12）。
    uint64_t file_offset_;
};

} // namespace detail

// ═══════════════════════ M-01 façade ═══════════════════════════════════════

TextEmbedding::TextEmbedding() = default;
TextEmbedding::~TextEmbedding() = default;

TextEmbedding::TextEmbedding(TextEmbedding&& other) noexcept
    : source_(std::move(other.source_)),
      scale_data_(std::move(other.scale_data_)),
      cfg_(std::move(other.cfg_)),
      row_bytes_(other.row_bytes_),
      groups_(other.groups_),
      loaded_(other.loaded_),
      identity_source_(std::move(other.identity_source_)),
      identity_scale_data_(std::move(other.identity_scale_data_)),
      identity_cfg_(std::move(other.identity_cfg_)),
      identity_row_bytes_(other.identity_row_bytes_),
      identity_loaded_(other.identity_loaded_)
{
    other.loaded_ = false; // 移动后原对象置未加载态（DESIGN M-01 §3）
    other.identity_loaded_ = false;
}

TextEmbedding& TextEmbedding::operator=(TextEmbedding&& other) noexcept
{
    if (this != &other) {
        source_ = std::move(other.source_);
        scale_data_ = std::move(other.scale_data_);
        cfg_ = std::move(other.cfg_);
        row_bytes_ = other.row_bytes_;
        groups_ = other.groups_;
        loaded_ = other.loaded_;
        other.loaded_ = false;

        identity_source_ = std::move(other.identity_source_);
        identity_scale_data_ = std::move(other.identity_scale_data_);
        identity_cfg_ = std::move(other.identity_cfg_);
        identity_row_bytes_ = other.identity_row_bytes_;
        identity_loaded_ = other.identity_loaded_;
        other.identity_loaded_ = false;
    }
    return *this;
}

Status EmbeddingConfig::FromSection(const std::string& weights, const std::string& dequant_scale,
                                    const std::string& type, const std::string& scale_type,
                                    const std::string& quant_mode, const std::string& load_mode,
                                    const std::string& identity_type, const std::string& identity_scale_type,
                                    size_t vocab_size, size_t hidden_dim, EmbeddingConfig* out)
{
    if (out == nullptr)
        return Status(Status::INVALID_ARGUMENT);
    *out = EmbeddingConfig{};
    out->vocab_size = vocab_size;
    out->hidden_dim = hidden_dim;

    // ── 主表（embedding_weights）：空 = "不加载"信号（tokenId 部署形态），
    //    主表字段不解析；identity 读取类型仍须解析（配置层独立——合并布局下
    //    identity 数据与主表同一文件，启用由图工位判定，Q12；TokenId 形态下
    //    仅为可解析，host 查表/加载由 pipeline 按 Loaded() 门控跳过，Q11）──
    if (weights.empty()) {
        // 主表空但 dequant_scale 非空：沿旧口径静默忽略（保持既有行为）
    } else {
        out->table_path = weights;
        out->scale_path = dequant_scale;

        // table_type：table_path 非空时必填（"int4" 可解析，Load 期拒绝——U-3）
        if (type == "fp32") {
            out->table_type = EmbeddingType::Fp32;
        } else if (type == "fp16") {
            out->table_type = EmbeddingType::Fp16;
        } else if (type == "int8") {
            out->table_type = EmbeddingType::Int8;
        } else if (type == "int4") {
            out->table_type = EmbeddingType::Int4;
        } else {
            return Status(Status::INVALID_ARGUMENT); // 未知/缺失（TC_113）
        }

        // scale_type：空串与 "none" 均为 None（缺省友好）
        if (scale_type.empty() || scale_type == "none") {
            out->scale_type = EmbeddingScaleType::None;
        } else if (scale_type == "fp32") {
            out->scale_type = EmbeddingScaleType::Fp32;
        } else if (scale_type == "fp16") {
            out->scale_type = EmbeddingScaleType::Fp16;
        } else {
            return Status(Status::INVALID_ARGUMENT);
        }

        // quant_mode：空串与 "per_token" 均 PerToken
        if (quant_mode.empty() || quant_mode == "per_token") {
            out->quant_mode = EmbeddingQuantMode::PerToken;
        } else if (quant_mode == "per_group") {
            out->quant_mode = EmbeddingQuantMode::PerGroup;
        } else {
            return Status(Status::INVALID_ARGUMENT);
        }

        // load_mode：空串与 "in_memory" 均 InMemory
        if (load_mode.empty() || load_mode == "in_memory") {
            out->load_mode = EmbeddingLoadMode::InMemory;
        } else if (load_mode == "in_flash") {
            out->load_mode = EmbeddingLoadMode::InFlash;
        } else {
            return Status(Status::INVALID_ARGUMENT);
        }

        out->group_size = 0; // schema 无组大小组段（per_group 恒在 Load 期拒绝，TC_209）
    }

    // ── per_layer_token_identity 读取类型（合并文件布局——Q12：
    //    identity 数据并入 embedding_weights 文件主表段之后、scale 段并入
    //    embedding_dequant_scale 文件主 scale 段之后；"是否读取"由 pipeline
    //    依图工位判定（以图为准，B-10），本节只解析读取类型；与主表启用与否
    //    互相独立）──
    // identity_type：缺省 fp16（Q10；"int4" 可解析，LoadTokenIdentity 期拒绝——U-3）
    if (identity_type.empty() || identity_type == "fp16") {
        out->token_identity_type = EmbeddingType::Fp16;
    } else if (identity_type == "fp32") {
        out->token_identity_type = EmbeddingType::Fp32;
    } else if (identity_type == "int8") {
        out->token_identity_type = EmbeddingType::Int8;
    } else if (identity_type == "int4") {
        out->token_identity_type = EmbeddingType::Int4;
    } else {
        return Status(Status::INVALID_ARGUMENT);
    }

    // identity_scale_type：scale 段读取类型（正拼键，meta 仅此一拼写）；缺省 fp16。
    // 仅图带 scale 工位时被消费（pipeline 依图回填 None = 不读——图缺工位
    // 属正常形态不数值化报错）；"none" 不再表达"不启用"（启用信号已随独立
    // 路径键一并删除）→ 非法值（段降级）
    if (identity_scale_type.empty() || identity_scale_type == "fp16") {
        out->token_identity_scale_type = EmbeddingScaleType::Fp16;
    } else if (identity_scale_type == "fp32") {
        out->token_identity_scale_type = EmbeddingScaleType::Fp32;
    } else {
        return Status(Status::INVALID_ARGUMENT);
    }
    return Status();
}

Status TextEmbedding::Load(const EmbeddingConfig& config)
{
    CANN_LLM_LOGI("embedding.load: start path=%s, vocab_size=%zu, hidden_dim=%zu, "
                  "load_mode=%s, table_type=%d",
                  config.table_path.c_str(), config.vocab_size, config.hidden_dim,
                  config.load_mode == EmbeddingLoadMode::InMemory ? "InMemory" : "InFlash",
                  static_cast<int>(config.table_type));

    // ── 参数校验（全部先于任何资源获取）──
    if (config.vocab_size == 0 || config.hidden_dim == 0 || config.table_path.empty()) {
        CANN_LLM_LOGE("embedding.load: invalid args vocab_size=%zu, hidden_dim=%zu, table_path='%s'", config.vocab_size,
                      config.hidden_dim, config.table_path.c_str());
        return Status(Status::INVALID_ARGUMENT); // TC_101/TC_103 类
    }
    if (config.table_type == EmbeddingType::Int4) {
        CANN_LLM_LOGE("embedding.load: Int4 not supported (U-3)");
        return Status(Status::INVALID_ARGUMENT); // U-3
    }
    if (config.scale_type == EmbeddingScaleType::None && !config.scale_path.empty()) {
        CANN_LLM_LOGE("embedding.load: scale_type=None but scale_path='%s' is set", config.scale_path.c_str());
        return Status(Status::INVALID_ARGUMENT); // TC_104
    }
    const size_t elem = detail::StorageElemSize(config.table_type);
    // C1: overflow guard — hidden_dim * elem must not wrap (external ModelConfig input)
    if (elem != 0 && config.hidden_dim > std::numeric_limits<std::size_t>::max() / elem) {
        CANN_LLM_LOGE("embedding.load: overflow hidden_dim=%zu * elem=%zu", config.hidden_dim, elem);
        return Status(Status::INVALID_ARGUMENT);
    }
    const size_t row_bytes = config.hidden_dim * elem;

    size_t groups = 1;
    if (config.quant_mode == EmbeddingQuantMode::PerGroup) {
        if (config.group_size == 0 || config.group_size > config.hidden_dim ||
            config.hidden_dim % config.group_size != 0) {
            CANN_LLM_LOGE("embedding.load: invalid PerGroup group_size=%zu, hidden_dim=%zu", config.group_size,
                          config.hidden_dim);
            return Status(Status::INVALID_ARGUMENT); // TC_103
        }
        groups = config.hidden_dim / config.group_size;
    }

    // C1: overflow guard — vocab_size * row_bytes must not wrap
    if (row_bytes != 0 &&
        static_cast<uint64_t>(config.vocab_size) > std::numeric_limits<std::uint64_t>::max() / row_bytes) {
        CANN_LLM_LOGE("embedding.load: overflow vocab_size=%zu * row_bytes=%zu", config.vocab_size, row_bytes);
        return Status(Status::INVALID_ARGUMENT);
    }
    const uint64_t expected_table = static_cast<uint64_t>(config.vocab_size) * row_bytes;
    const size_t scale_elem = detail::ScaleElemSize(config.scale_type);
    const uint64_t scale_count = config.scale_type == EmbeddingScaleType::None ?
                                     0 :
                                     static_cast<uint64_t>(config.vocab_size) * groups;
    const uint64_t expected_scale = scale_count * scale_elem;

    CANN_LLM_LOGI("embedding.load: expected_table=%lu bytes, expected_scale=%lu bytes",
                  static_cast<unsigned long>(expected_table), static_cast<unsigned long>(expected_scale));

    // ── 表（InMemory mmap 只读映射 / InFlash open+fstat 校验）──
    std::unique_ptr<detail::TableSource> source;
    if (config.load_mode == EmbeddingLoadMode::InMemory) {
        // mmap 段源（L2）：fstat "≥ 主表段" 放行（Q12 identity 段后随——超长
        // 放行）/ 短文件 map 前拒绝；identity 段完整性与精确总长由
        // LoadTokenIdentity 期校验兜底。RowPtr 直指映射页，FillInputs 免 staging。
        Status st = detail::MmapTableSource::Create(config.table_path, /*file_offset=*/0, row_bytes, config.vocab_size,
                                                    /*require_exact=*/false, &source);
        if (!st.ok()) {
            CANN_LLM_LOGE("embedding.load: mmap table source failed st=%d, expected=%lu, path='%s'", st.code,
                          static_cast<unsigned long>(expected_table), config.table_path.c_str());
            return st; // TC_101/TC_102
        }
    } else {
        int fd = ::open(config.table_path.c_str(), O_RDONLY);
        if (fd < 0) {
            CANN_LLM_LOGE("embedding.load: cannot open(InFlash) table file '%s'", config.table_path.c_str());
            return Status(Status::INVALID_ARGUMENT); // TC_101
        }
        struct stat sb {};
        // 合并文件布局（Q12）：" ≥ 主表段" 即放行（identity 段可后随；行偏移自文件头
        // 起，base=0，后段不触碰）；总长精确校验移至 LoadTokenIdentity 期
        if (::fstat(fd, &sb) != 0 || static_cast<uint64_t>(sb.st_size) < expected_table) {
            CANN_LLM_LOGE("embedding.load: table file size mismatch fstat_size=%lu, expected=%lu, path='%s'",
                          static_cast<unsigned long>(sb.st_size), static_cast<unsigned long>(expected_table),
                          config.table_path.c_str());
            ::close(fd);
            return Status(Status::INVALID_ARGUMENT); // TC_102
        }
        // fd 所有权随即移交 FlashTableSource（后续任一失败路径 RAII 关闭）
        source = std::make_unique<detail::FlashTableSource>(fd, row_bytes);
    }

    // ── scale（恒内存态，D-09；堆内存非 TensorBuffer）──
    std::unique_ptr<char[]> scale_data;
    if (config.scale_type != EmbeddingScaleType::None) {
        std::ifstream f(config.scale_path, std::ios::binary);
        if (!f) {
            CANN_LLM_LOGE("embedding.load: cannot open scale file '%s'", config.scale_path.c_str());
            return Status(Status::INVALID_ARGUMENT);
        }
        scale_data.reset(new (std::nothrow) char[static_cast<size_t>(expected_scale)]);
        if (!scale_data) {
            CANN_LLM_LOGE("embedding.load: cannot alloc scale buffer bytes=%lu",
                          static_cast<unsigned long>(expected_scale));
            return Status(Status::FAILURE);
        }
        f.read(scale_data.get(), static_cast<std::streamsize>(expected_scale));
        // 合并文件布局（Q12）：主 scale 段之后可跟随 identity scale 段
        // （per_layer_token_identity_scale）—— 不以 EOF 取整；仅校验主段完整
        if (static_cast<uint64_t>(f.gcount()) != expected_scale) {
            CANN_LLM_LOGE("embedding.load: scale file size mismatch read=%lu, expected=%lu, path='%s'",
                          static_cast<unsigned long>(f.gcount()), static_cast<unsigned long>(expected_scale),
                          config.scale_path.c_str());
            return Status(Status::INVALID_ARGUMENT); // TC_102（过短）
        }
    }

    // ── 提交（此前所有失败路径均保持未加载态）──
    CANN_LLM_LOGI("embedding.load: success");
    source_ = std::move(source);
    scale_data_ = std::move(scale_data);
    cfg_ = config;
    row_bytes_ = row_bytes;
    groups_ = groups;
    loaded_ = true;
    return Status();
}

Status TextEmbedding::LoadTokenIdentity(const EmbeddingConfig& config, size_t row_width)
{
    CANN_LLM_LOGI("embedding.load_identity: start path=%s, vocab_size=%zu, row_width=%zu, "
                  "load_mode=%s, identity_type=%d",
                  config.table_path.c_str(), config.vocab_size, row_width,
                  config.load_mode == EmbeddingLoadMode::InMemory ? "InMemory" : "InFlash",
                  static_cast<int>(config.token_identity_type));

    // ── 参数校验（全部先于任何资源获取；失败保持 identity 未加载态）──
    // 合并文件布局（Q12）：identity 数据位于 config.table_path 主表段之后，
    // segment 偏移与段大小均须主表几何（hidden_dim / table_type）参与计算
    if (config.vocab_size == 0 || config.hidden_dim == 0 || row_width == 0 || config.table_path.empty()) {
        CANN_LLM_LOGE("embedding.load_identity: invalid args vocab_size=%zu, hidden_dim=%zu, "
                      "row_width=%zu, table_path='%s'",
                      config.vocab_size, config.hidden_dim, row_width, config.table_path.c_str());
        return Status(Status::INVALID_ARGUMENT);
    }
    if (config.token_identity_type == EmbeddingType::Int4) {
        CANN_LLM_LOGE("embedding.load_identity: Int4 not supported (U-3)");
        return Status(Status::INVALID_ARGUMENT);
    }
    const bool want_scale = config.token_identity_scale_type != EmbeddingScaleType::None;
    if (want_scale && (config.scale_type == EmbeddingScaleType::None || config.scale_path.empty())) {
        // 图带 scale 工位但主表无 scale 文件 → 无处读取（调用方 InitEmbedding_ 依图
        // 校验应已致命；此处防御 UT/其他调用方误用，对齐 TC_104 口径）
        CANN_LLM_LOGE("embedding.load_identity: identity scale requested but main scale "
                      "absent (scale_type=%d, scale_path='%s')",
                      static_cast<int>(config.scale_type), config.scale_path.c_str());
        return Status(Status::INVALID_ARGUMENT);
    }

    // ── 段几何（Q12：offset/size 全部可计算，无文件头）──
    const size_t main_elem = detail::StorageElemSize(config.table_type);
    // C1: overflow guard — 主表段 hidden × elem、vocab × 主行字节数
    if (main_elem != 0 && config.hidden_dim > std::numeric_limits<std::size_t>::max() / main_elem) {
        CANN_LLM_LOGE("embedding.load_identity: overflow hidden_dim=%zu * elem=%zu", config.hidden_dim, main_elem);
        return Status(Status::INVALID_ARGUMENT);
    }
    const uint64_t main_table = static_cast<uint64_t>(config.vocab_size) * config.hidden_dim * main_elem;

    const size_t elem = detail::StorageElemSize(config.token_identity_type);
    // C1: overflow guard — row_width * elem 不得回绕（row_width 来自工位 desc，外部输入）
    if (elem != 0 && row_width > std::numeric_limits<std::size_t>::max() / elem) {
        CANN_LLM_LOGE("embedding.load_identity: overflow row_width=%zu * elem=%zu", row_width, elem);
        return Status(Status::INVALID_ARGUMENT);
    }
    const size_t identity_row_bytes = row_width * elem;
    // C1: overflow guard — vocab_size * row_bytes 不得回绕
    if (identity_row_bytes != 0 &&
        static_cast<uint64_t>(config.vocab_size) > std::numeric_limits<std::uint64_t>::max() / identity_row_bytes) {
        CANN_LLM_LOGE("embedding.load_identity: overflow vocab_size=%zu * row_bytes=%zu", config.vocab_size,
                      identity_row_bytes);
        return Status(Status::INVALID_ARGUMENT);
    }
    const uint64_t identity_table = static_cast<uint64_t>(config.vocab_size) * identity_row_bytes;
    // C1: overflow guard — 主表段 + identity 段不得回绕
    if (identity_table > std::numeric_limits<std::uint64_t>::max() - main_table) {
        CANN_LLM_LOGE("embedding.load_identity: overflow main=%lu + identity=%lu",
                      static_cast<unsigned long>(main_table), static_cast<unsigned long>(identity_table));
        return Status(Status::INVALID_ARGUMENT);
    }
    const uint64_t expected_table = main_table + identity_table;
    const uint64_t identity_offset = main_table;

    // scale PerToken（Q4）：每词表项 1 个系数，无组概念。主 scale 段大小按 PerToken
    // 计算（vocab 项；PerGroup 主表在 Load 期已拒——TC_209，不可达此处）
    const size_t main_scale_elem = detail::ScaleElemSize(config.scale_type);
    const uint64_t main_scale = want_scale ? static_cast<uint64_t>(config.vocab_size) * main_scale_elem : 0;
    const size_t scale_elem = detail::ScaleElemSize(config.token_identity_scale_type);
    const uint64_t expected_scale = want_scale ? static_cast<uint64_t>(config.vocab_size) * scale_elem : 0;
    const uint64_t scale_total = main_scale + expected_scale;

    CANN_LLM_LOGI("embedding.load_identity: identity_offset=%lu bytes, identity_table=%lu "
                  "bytes, expected_scale=%lu bytes (total_file=%lu)",
                  static_cast<unsigned long>(identity_offset), static_cast<unsigned long>(identity_table),
                  static_cast<unsigned long>(expected_scale), static_cast<unsigned long>(expected_table));

    // ── 表（InMemory 自偏移 mmap 只读映射 / InFlash open+fstat 校验；同主表二态）──
    std::unique_ptr<detail::TableSource> source;
    if (config.load_mode == EmbeddingLoadMode::InMemory) {
        // mmap 段源（L2）：总长精确校验前移到 map 前（require_exact——
        // 原串行版"读毕须恰为文件尾"（短/超长均拒，B-10 致命）语义逐字保留）；
        // require_exact=false ident 段不存在后段，超长即文件与图不匹配。
        Status st = detail::MmapTableSource::Create(config.table_path, identity_offset, identity_row_bytes,
                                                    config.vocab_size, /*require_exact=*/true, &source);
        if (!st.ok()) {
            CANN_LLM_LOGE("embedding.load_identity: mmap identity source failed st=%d "
                          "(expected=%lu, main=%lu, identity=%lu, path='%s')",
                          st.code, static_cast<unsigned long>(expected_table), static_cast<unsigned long>(main_table),
                          static_cast<unsigned long>(identity_table), config.table_path.c_str());
            return st; // 短/超长均拒绝（B-10 致命）
        }
    } else {
        int fd = ::open(config.table_path.c_str(), O_RDONLY);
        if (fd < 0) {
            CANN_LLM_LOGE("embedding.load_identity: cannot open(InFlash) table file '%s'", config.table_path.c_str());
            return Status(Status::INVALID_ARGUMENT);
        }
        struct stat sb {};
        // 总长精确校验：主表段 + identity 段（identity 启用时图有填充需求，
        // 多/缺一字节均说明文件与图不匹配）
        if (::fstat(fd, &sb) != 0 || static_cast<uint64_t>(sb.st_size) != expected_table) {
            CANN_LLM_LOGE("embedding.load_identity: table file size mismatch fstat_size=%lu, "
                          "expected=%lu (main=%lu + identity=%lu), path='%s'",
                          static_cast<unsigned long>(sb.st_size), static_cast<unsigned long>(expected_table),
                          static_cast<unsigned long>(main_table), static_cast<unsigned long>(identity_table),
                          config.table_path.c_str());
            ::close(fd);
            return Status(Status::INVALID_ARGUMENT);
        }
        // 行偏移基数 = 主表段大小（合并文件内 identity 段从主表段之后起）
        source = std::make_unique<detail::FlashTableSource>(fd, identity_row_bytes, identity_offset);
    }

    // ── scale 段（PerToken，恒内存态——D-09；堆内存非 TensorBuffer；主 scale 段之后偏移读取）──
    std::unique_ptr<char[]> scale_data;
    if (want_scale) {
        std::ifstream f(config.scale_path, std::ios::binary);
        if (!f) {
            CANN_LLM_LOGE("embedding.load_identity: cannot open scale file '%s'", config.scale_path.c_str());
            return Status(Status::INVALID_ARGUMENT);
        }
        // 主 scale 段之后即 identity scale 段；读毕须恰为文件尾
        f.seekg(static_cast<std::streamoff>(main_scale));
        if (!f) {
            CANN_LLM_LOGE("embedding.load_identity: cannot seek to identity scale segment "
                          "(offset=%lu, path='%s')",
                          static_cast<unsigned long>(main_scale), config.scale_path.c_str());
            return Status(Status::INVALID_ARGUMENT);
        }
        scale_data.reset(new (std::nothrow) char[static_cast<size_t>(expected_scale)]);
        if (!scale_data) {
            CANN_LLM_LOGE("embedding.load_identity: cannot alloc scale buffer bytes=%lu",
                          static_cast<unsigned long>(expected_scale));
            return Status(Status::FAILURE);
        }
        f.read(scale_data.get(), static_cast<std::streamsize>(expected_scale));
        if (static_cast<uint64_t>(f.gcount()) != expected_scale || f.peek() != std::char_traits<char>::eof()) {
            CANN_LLM_LOGE("embedding.load_identity: identity scale segment size mismatch "
                          "(read=%lu, expected=%lu, file_expected=%lu, path='%s')",
                          static_cast<unsigned long>(f.gcount()), static_cast<unsigned long>(expected_scale),
                          static_cast<unsigned long>(scale_total), config.scale_path.c_str());
            return Status(Status::INVALID_ARGUMENT);
        }
    }

    // ── 提交（此前所有失败路径均保持 identity 未加载态）──
    CANN_LLM_LOGI("embedding.load_identity: success");
    identity_source_ = std::move(source);
    identity_scale_data_ = std::move(scale_data);
    identity_cfg_ = config;
    identity_cfg_.token_identity_row_width = row_width; // 加载期回填（非 meta 字段）
    identity_row_bytes_ = identity_row_bytes;
    identity_loaded_ = true;
    return Status();
}

Status TextEmbedding::Gather(const TokenId* ids, size_t count, std::shared_ptr<ITensorBuffer> out_embed,
                             std::shared_ptr<ITensorBuffer> out_scales) const
{
    if (!loaded_)
        return Status(Status::INVALID_STATE); // TC_106
    if (count == 0)
        return Status(); // TC_201
    if (ids == nullptr)
        return Status(Status::INVALID_ARGUMENT); // TC_110
    if (!out_embed || out_embed->GetData() == nullptr || out_embed->GetSize() < count * row_bytes_) {
        return Status(Status::INVALID_ARGUMENT); // TC_109/TC_112
    }

    const size_t scale_elem = detail::ScaleElemSize(cfg_.scale_type);
    const bool has_scale = cfg_.scale_type != EmbeddingScaleType::None;
    if (has_scale &&
        (!out_scales || out_scales->GetData() == nullptr || out_scales->GetSize() < count * groups_ * scale_elem)) {
        return Status(Status::INVALID_ARGUMENT); // TC_109/TC_112
    }

    // 先全量校验 ids 再拷贝（整批拒绝，不部分产出——TC_105）
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= cfg_.vocab_size)
            return Status(Status::INVALID_ARGUMENT);
    }

    auto* embed_dst = static_cast<char*>(out_embed->GetData());
    for (size_t i = 0; i < count; ++i) {
        Status st = source_->ReadRow(ids[i], embed_dst + i * row_bytes_);
        if (!st.ok())
            return st; // TC_111（FAILURE）
    }

    if (has_scale) {
        const char* scale_src = scale_data_.get();
        auto* scale_dst = static_cast<char*>(out_scales->GetData());
        const size_t row_scale_bytes = groups_ * scale_elem;
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(scale_dst + i * row_scale_bytes, scale_src + static_cast<size_t>(ids[i]) * row_scale_bytes,
                        row_scale_bytes);
        }
    }
    return Status();
}

Status TextEmbedding::FillInputs(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                                 const std::unordered_map<std::string, size_t>& name_index, EmbeddingPath path,
                                 const TokenId* ids, size_t count)
{
    if (count == 0)
        return Status();
    if (ids == nullptr)
        return Status(Status::INVALID_ARGUMENT); // TC_110

    // 工位解析：规范名 input_embed 优先，omc 导出别名 embeddings 兜底
    // （问题 1 方案 b——om 实名见 10100915_debug.log [6] name=embeddings）。
    auto it = FindEmbedSlot(name_index);
    if (it == name_index.end() || it->second >= inputs.size()) {
        return Status(Status::NOT_FOUND); // TC_107
    }
    const auto& embed_slot = inputs[it->second];
    if (!embed_slot || embed_slot->GetData() == nullptr) {
        return Status(Status::INVALID_ARGUMENT); // TC_112
    }
    const DataType slot_type = embed_slot->GetTensorDesc().dataType;

    // ── TokenId 路径：表在模型权重中，本模块无需 Load（D-02）──
    if (path == EmbeddingPath::TokenId) {
        if (slot_type != DataType::INT32 && slot_type != DataType::INT64) {
            return Status(Status::INVALID_ARGUMENT);
        }
        const size_t slot_elem = detail::SlotElemSize(slot_type);
        if (embed_slot->GetSize() < count * slot_elem) {
            return Status(Status::INVALID_ARGUMENT);
        }
        if (slot_type == DataType::INT32) {
            auto* dst = static_cast<int32_t*>(embed_slot->GetData());
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<int32_t>(ids[i]);
        } else {
            auto* dst = static_cast<int64_t*>(embed_slot->GetData());
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<int64_t>(ids[i]);
        }
        // per_layer_token_identity：TokenId 部署形态不查表（identity 由图内
        // 自理，host 零触碰——docs/gemma4_token_identity.md §5.3）
        return Status();
    }

    // ── Embedding 路径 ──
    if (!loaded_)
        return Status(Status::INVALID_STATE); // TC_106
    // per_layer_token_identity 落位（§5.3 唯一调用点，Q11）：embed 填充前执行，失败早退
    {
        Status identity_status = FillTokenIdentitySlot_(inputs, name_index, ids, count);
        if (!identity_status.ok())
            return identity_status;
    }
    if (!detail::ConversionSupported(cfg_.table_type, slot_type)) {
        return Status(Status::INVALID_ARGUMENT); // TC_108
    }
    const size_t slot_elem = detail::SlotElemSize(slot_type);
    // C1: overflow guard — count * hidden_dim * slot_elem must not wrap (would bypass the capacity check)
    if (cfg_.hidden_dim != 0 && count > std::numeric_limits<std::size_t>::max() / cfg_.hidden_dim) {
        return Status(Status::INVALID_ARGUMENT);
    }
    const size_t need_rows = count * cfg_.hidden_dim;
    if (slot_elem != 0 && need_rows > std::numeric_limits<std::size_t>::max() / slot_elem) {
        return Status(Status::INVALID_ARGUMENT);
    }
    if (embed_slot->GetSize() < need_rows * slot_elem) {
        return Status(Status::INVALID_ARGUMENT); // TC_109
    }

    // scale 工位（scale_type ≠ None 时必须存在；类型以工位为准——D-06，
    // FLOAT32/FLOAT16 皆可，经值展开/软件转换写入。nch_3b 模型工位为 FLOAT16）
    std::shared_ptr<ITensorBuffer> scale_slot;
    DataType scale_slot_type = DataType::FLOAT32; // scale_type=None 时未用
    if (cfg_.scale_type != EmbeddingScaleType::None) {
        // 工位解析：规范名 embed_scales 优先，omc 导出别名 embeddings_scale 兜底
        auto sit = FindEmbedScaleSlot(name_index);
        if (sit == name_index.end() || sit->second >= inputs.size()) {
            return Status(Status::NOT_FOUND); // TC_107
        }
        scale_slot = inputs[sit->second];
        scale_slot_type = scale_slot ? scale_slot->GetTensorDesc().dataType : DataType::RESERVED;
        const size_t scale_slot_elem = detail::SlotElemSize(scale_slot_type);
        if (!scale_slot || scale_slot->GetData() == nullptr ||
            (scale_slot_type != DataType::FLOAT32 && scale_slot_type != DataType::FLOAT16) || scale_slot_elem == 0 ||
            scale_slot->GetSize() < count * groups_ * scale_slot_elem) {
            return Status(Status::INVALID_ARGUMENT);
        }
    }

    // 先全量校验 ids 再写首行（工位不被部分污染——F-03 §4 铁律，TC_105）
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= cfg_.vocab_size)
            return Status(Status::INVALID_ARGUMENT);
    }

    // 逐行流式：InMemory 免 staging 直转（RowPtr）；InFlash 经栈上局部 staging
    auto* dst = static_cast<char*>(embed_slot->GetData());
    const size_t row_slot_bytes = cfg_.hidden_dim * slot_elem;
    std::vector<char> staging; // 仅 InFlash 使用；栈帧内分配释放（无 mutable 成员，§2.4）
    for (size_t i = 0; i < count; ++i) {
        const void* src_row = source_->RowPtr(ids[i]);
        if (src_row != nullptr) {
            Status st = detail::ConvertRows(src_row, cfg_.table_type, dst + i * row_slot_bytes, slot_type,
                                            cfg_.hidden_dim);
            if (!st.ok())
                return st;
        } else {
            if (staging.empty())
                staging.resize(row_bytes_);
            Status st = source_->ReadRow(ids[i], staging.data());
            if (!st.ok())
                return st; // TC_111（FAILURE）
            st = detail::ConvertRows(staging.data(), cfg_.table_type, dst + i * row_slot_bytes, slot_type,
                                     cfg_.hidden_dim);
            if (!st.ok())
                return st;
        }
    }

    // scale 收集 → 转换（按工位类型）→ 写 embed_scales
    if (cfg_.scale_type != EmbeddingScaleType::None) {
        const size_t scale_elem = detail::ScaleElemSize(cfg_.scale_type);
        const size_t row_scale_bytes = groups_ * scale_elem;
        std::vector<char> raw(count * row_scale_bytes);
        const char* scale_src = scale_data_.get();
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(raw.data() + i * row_scale_bytes, scale_src + static_cast<size_t>(ids[i]) * row_scale_bytes,
                        row_scale_bytes);
        }
        Status st = detail::ConvertScales(raw.data(), cfg_.scale_type, scale_slot->GetData(), scale_slot_type,
                                          count * groups_);
        if (!st.ok())
            return st;
    }

    // 调试：打印查表结果（input_embed 工位刚写入的行向量；含 id 便于对位）
    {
        const size_t cols = std::min(cfg_.hidden_dim, detail::kDumpCols);
        CANN_LLM_LOGD("embedding.fill_inputs: Embedding lookup done, count=%zu, hidden=%zu, "
                      "dtype=%s, table_type=%d",
                      count, cfg_.hidden_dim, DataTypeName(slot_type), static_cast<int>(cfg_.table_type));
        const size_t rows = std::min(count, detail::kDumpRows);
        for (size_t i = 0; i < rows; ++i) {
            CANN_LLM_LOGD("embedding.fill_inputs: token[%zu] id=%u, head=[%s]", i, ids[i],
                          FormatElems(dst, slot_type, i * cfg_.hidden_dim, cols).c_str());
        }
        if (cfg_.scale_type != EmbeddingScaleType::None) {
            const size_t sc = std::min(groups_, detail::kDumpCols);
            CANN_LLM_LOGD("embedding.fill_inputs: token[0] scales(head %zu)=[%s]", sc,
                          FormatElems(scale_slot->GetData(), scale_slot_type, 0, sc).c_str());
        }
    }
    return Status();
}

// ── per_layer_token_identity 落位（§5.3；沿 Embedding 路径落位风格）──
Status TextEmbedding::FillTokenIdentitySlot_(const std::vector<std::shared_ptr<ITensorBuffer>>& inputs,
                                             const std::unordered_map<std::string, size_t>& name_index,
                                             const TokenId* ids, size_t count)
{
    if (!identity_loaded_)
        return Status(); // 未启用（零行为变化，旧图/未配置）

    const EmbeddingConfig& ic = identity_cfg_;

    // 工位查找（工位缺失 = 图不带该输入，非本能力管辖 → NOT_FOUND）
    auto it = name_index.find(kTokenIdentitySlotName);
    if (it == name_index.end() || it->second >= inputs.size()) {
        return Status(Status::NOT_FOUND);
    }
    const auto& slot = inputs[it->second];
    if (!slot || slot->GetData() == nullptr) {
        return Status(Status::INVALID_ARGUMENT);
    }
    const DataType slot_type = slot->GetTensorDesc().dataType;
    if (!detail::ConversionSupported(ic.token_identity_type, slot_type)) {
        return Status(Status::INVALID_ARGUMENT); // TC_108 口径
    }
    const size_t slot_elem = detail::SlotElemSize(slot_type);
    const size_t width = ic.token_identity_row_width;
    // C1: overflow guard — count × W × slot_elem 不得回绕（绕过容量校验）
    if (width != 0 && count > std::numeric_limits<std::size_t>::max() / width) {
        return Status(Status::INVALID_ARGUMENT);
    }
    const size_t need_elems = count * width;
    if (slot_elem != 0 && need_elems > std::numeric_limits<std::size_t>::max() / slot_elem) {
        return Status(Status::INVALID_ARGUMENT);
    }
    if (slot->GetSize() < need_elems * slot_elem) {
        return Status(Status::INVALID_ARGUMENT); // TC_109 口径
    }

    // scale 工位（PerToken：每 token 1 个系数，Q4；类型以工位为准——D-06，
    // FLOAT32/FLOAT16 皆可，复用 ConvertScales 值展开/软件转换写入）
    std::shared_ptr<ITensorBuffer> scale_slot;
    DataType scale_slot_type = DataType::FLOAT32; // scale 未启用时未用
    const bool has_scale = ic.token_identity_scale_type != EmbeddingScaleType::None;
    if (has_scale) {
        // 工位解析：per_layer_token_identity_scale（唯一名）
        auto sit = FindIdentityScaleSlot(name_index);
        if (sit == name_index.end() || sit->second >= inputs.size()) {
            return Status(Status::NOT_FOUND);
        }
        scale_slot = inputs[sit->second];
        scale_slot_type = scale_slot ? scale_slot->GetTensorDesc().dataType : DataType::RESERVED;
        const size_t scale_slot_elem = detail::SlotElemSize(scale_slot_type);
        if (!scale_slot || scale_slot->GetData() == nullptr ||
            (scale_slot_type != DataType::FLOAT32 && scale_slot_type != DataType::FLOAT16) || scale_slot_elem == 0 ||
            scale_slot->GetSize() < count * scale_slot_elem) {
            return Status(Status::INVALID_ARGUMENT);
        }
    }

    // 先全量校验 ids 再写首行（工位不被部分污染——TC_105 铁律，整批拒绝）
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= ic.vocab_size)
            return Status(Status::INVALID_ARGUMENT);
    }

    // 逐行流式：InMemory 免 staging 直转（RowPtr）；InFlash 经栈上局部 staging
    auto* dst = static_cast<char*>(slot->GetData());
    const size_t row_slot_bytes = width * slot_elem;
    std::vector<char> staging; // 仅 InFlash 使用；栈帧内分配释放
    for (size_t i = 0; i < count; ++i) {
        const void* src_row = identity_source_->RowPtr(ids[i]);
        if (src_row != nullptr) {
            Status st = detail::ConvertRows(src_row, ic.token_identity_type, dst + i * row_slot_bytes, slot_type,
                                            width);
            if (!st.ok())
                return st;
        } else {
            if (staging.empty())
                staging.resize(identity_row_bytes_);
            Status st = identity_source_->ReadRow(ids[i], staging.data());
            if (!st.ok())
                return st;
            st = detail::ConvertRows(staging.data(), ic.token_identity_type, dst + i * row_slot_bytes, slot_type,
                                     width);
            if (!st.ok())
                return st;
        }
    }

    // scale 收集 → 转换（按工位类型）→ 写 per_layer_token_identity_scale（PerToken）
    if (has_scale) {
        const size_t scale_elem = detail::ScaleElemSize(ic.token_identity_scale_type);
        std::vector<char> raw(count * scale_elem);
        const char* scale_src = identity_scale_data_.get();
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(raw.data() + i * scale_elem, scale_src + static_cast<size_t>(ids[i]) * scale_elem, scale_elem);
        }
        Status st = detail::ConvertScales(raw.data(), ic.token_identity_scale_type, scale_slot->GetData(),
                                          scale_slot_type, count);
        if (!st.ok())
            return st;
    }

    // 调试：打印查表结果（per_layer_token_identity 工位刚写入；含 id 便于对位）
    {
        const size_t cols = std::min(width, detail::kDumpCols);
        CANN_LLM_LOGD("embedding.fill_identity: identity lookup done, count=%zu, width=%zu, "
                      "dtype=%s, type=%d",
                      count, width, DataTypeName(slot_type), static_cast<int>(ic.token_identity_type));
        const size_t rows = std::min(count, detail::kDumpRows);
        for (size_t i = 0; i < rows; ++i) {
            CANN_LLM_LOGD("embedding.fill_identity: token[%zu] id=%u, head=[%s]", i, ids[i],
                          FormatElems(dst, slot_type, i * width, cols).c_str());
        }
        if (has_scale) {
            CANN_LLM_LOGD("embedding.fill_identity: token[0] scale=[%s]",
                          FormatElems(scale_slot->GetData(), scale_slot_type, 0, 1).c_str());
        }
    }
    return Status();
}

// （无 Table()/Scale() 存储暴露接口：TensorBuffer 仅用于模型输入/输出工位，
//  表数据消费一律走 FillInputs / FillTokenIdentitySlot_。）

EmbeddingType TextEmbedding::TableType() const
{
    return loaded_ ? cfg_.table_type : EmbeddingType::Fp32; // 哨兵 + Loaded() 判别
}

EmbeddingLoadMode TextEmbedding::LoadMode() const
{
    return loaded_ ? cfg_.load_mode : EmbeddingLoadMode::InMemory; // 哨兵
}

bool TextEmbedding::Loaded() const { return loaded_; }

bool TextEmbedding::IdentityLoaded() const { return identity_loaded_; }

} // namespace cann_lm::engine
