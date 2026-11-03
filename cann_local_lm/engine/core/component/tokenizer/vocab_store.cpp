#include "core/component/tokenizer/tokenizer.h"

#include "infra/log.h"

#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace cann_local_lm::engine {

// ── tokenizer.json SAX 解析器（D-09：nlohmann 流式，控制大文件内存）────────
// 仅捕获模块所需字段：model.type/byte_fallback/unk_token、model.vocab、
// model.merges、added_tokens[]、normalizer(Replace 型)。其余子树跳过。
// 注：Qwen 实物 vocab 为 dict（id 连续 0..N-1）、added_tokens 追加式（id > base）；
// gemma4 实物 added_tokens 为 base 内联式（id 落在 base 范围内，属既有条目的
// special 标记）——两种形态统一按 overlay 处理（added 优先，允许同 id 同内容）。
namespace {

class TokenizerJsonSax {
public:
    // nlohmann SAX 接口
    bool null()
    {
        if (state_ == State::kSkip) {
            ExitSkipOnScalar();
            return true;
        }
        // 已识别键的 null 值（"normalizer": null、"unk_token": null 等）→ 统一回退父态
        switch (state_) {
            case State::kRootKeyAdded:
            case State::kRootKeyModel:
            case State::kRootKeyNormalizer:
                state_ = State::kRoot;
                break;
            case State::kAddedId:
            case State::kAddedContent:
            case State::kAddedSpecial:
                state_ = State::kAddedElem;
                break;
            case State::kModelType:
            case State::kModelByteFallback:
            case State::kModelUnk:
                state_ = State::kModel;
                break;
            case State::kVocabValue:
                state_ = State::kVocab;
                break;
            case State::kNormalizerType:
            case State::kNormalizerContent:
                state_ = State::kNormalizer;
                break;
            case State::kNormalizerPattern:
                state_ = State::kNormalizerPatternObj;
                break;
            default:
                break;
        }
        return true;
    }
    bool boolean(bool v)
    {
        if (state_ == State::kSkip) {
            ExitSkipOnScalar();
            return true;
        }
        if (state_ == State::kAddedSpecial) {
            added_special_ = v;
            state_ = State::kAddedElem;
        } else if (state_ == State::kModelByteFallback) {
            byte_fallback_ = v;
            state_ = State::kModel;
        }
        return true;
    }
    bool number_integer(std::int64_t v) { return HandleNumber(v); }
    bool number_unsigned(std::uint64_t v) { return HandleNumber(static_cast<std::int64_t>(v)); }
    bool number_float(double, const std::string&) { return true; }
    bool string(std::string& v)
    {
        if (state_ == State::kSkip) {
            ExitSkipOnScalar();
            return true;
        }
        switch (state_) {
            case State::kModelType:
                model_type_ = v;
                state_ = State::kModel;
                break;
            case State::kModelUnk:
                unk_token_ = v;
                state_ = State::kModel;
                break;
            case State::kAddedContent:
                added_content_ = v;
                state_ = State::kAddedElem;
                break;
            case State::kMergesItem:
                merges_raw_.push_back(v);
                break; // 字符串格式 "A B"
            case State::kMergesPair:
                merge_pair_.push_back(v);
                break; // 数组格式 ["A","B"]
            case State::kNormalizerType:
                normalizer_type_ = v;
                state_ = State::kNormalizer;
                break;
            case State::kNormalizerContent:
                normalizer_content_ = v;
                state_ = State::kNormalizer;
                break;
            case State::kNormalizerPattern:
                normalizer_pattern_ = v;
                state_ = State::kNormalizerPatternObj;
                break;
            default:
                break; // 其余字符串（version 等）丢弃
        }
        return true;
    }
    bool binary(nlohmann::json::binary_t&) { return true; }
    bool start_object(std::size_t)
    {
        if (state_ == State::kSkip) {
            ++skip_depth_;
            return true;
        }
        switch (state_) {
            case State::kRootKeyAdded:
                state_ = State::kAddedElem;
                break;
            case State::kRootKeyModel:
                state_ = State::kModel;
                break;
            case State::kRootKeyNormalizer:
                state_ = State::kNormalizer;
                break;
            case State::kNormalizerPatternObj:
                break; // pattern 对象内层，保持
            default:
                break;
        }
        return true;
    }
    bool key(std::string& k)
    {
        if (state_ == State::kSkip)
            return true;
        switch (state_) {
            case State::kRoot:
                if (k == "added_tokens")
                    state_ = State::kRootKeyAdded;
                else if (k == "model")
                    state_ = State::kRootKeyModel;
                else if (k == "normalizer")
                    state_ = State::kRootKeyNormalizer;
                else if (!k.empty())
                    EnterSkip(); // version/truncation/padding/pre_tokenizer/... 跳过
                break;
            case State::kAddedElem:
                if (k == "id")
                    state_ = State::kAddedId;
                else if (k == "content")
                    state_ = State::kAddedContent;
                else if (k == "special")
                    state_ = State::kAddedSpecial;
                else
                    EnterSkip(); // lstrip/rstrip/single_word/normalized 等跳过
                break;
            case State::kModel:
                if (k == "type")
                    state_ = State::kModelType;
                else if (k == "byte_fallback")
                    state_ = State::kModelByteFallback;
                else if (k == "unk_token")
                    state_ = State::kModelUnk;
                else if (k == "vocab")
                    state_ = State::kVocab;
                else if (k == "merges")
                    state_ = State::kMerges;
                else
                    EnterSkip(); // ignore_merges/dropout/continuing_subword 等跳过
                break;
            case State::kVocab:
                vocab_key_ = k;
                state_ = State::kVocabValue;
                break;
            case State::kNormalizer:
                if (k == "type")
                    state_ = State::kNormalizerType;
                else if (k == "content")
                    state_ = State::kNormalizerContent;
                else if (k == "pattern")
                    state_ = State::kNormalizerPatternObj;
                else
                    EnterSkip();
                break;
            case State::kNormalizerPatternObj:
                if (k == "String")
                    state_ = State::kNormalizerPattern;
                else
                    EnterSkip(); // Regex 类型 pattern（本模块不实现）跳过
                break;
            default:
                break;
        }
        return true;
    }
    bool end_object()
    {
        if (state_ == State::kSkip) {
            ExitSkipOnEnd();
            return true;
        }
        switch (state_) {
            case State::kAddedElem:
                if (added_id_ >= 0) {
                    added_tokens_.emplace_back(added_content_, added_id_);
                    if (added_special_)
                        special_ids_.push_back(added_id_);
                }
                added_id_ = -1;
                added_content_.clear();
                added_special_ = false;
                state_ = State::kRootKeyAdded; // 数组内继续或由 end_array 收尾
                break;
            case State::kVocab:
                state_ = State::kModel;
                break;
            case State::kModel:
                state_ = State::kRoot;
                break;
            case State::kNormalizer:
                state_ = State::kRoot;
                break;
            case State::kNormalizerPatternObj:
                state_ = State::kNormalizer;
                break;
            default:
                break;
        }
        return true;
    }
    bool start_array(std::size_t)
    {
        if (state_ == State::kSkip) {
            ++skip_depth_;
            return true;
        }
        if (state_ == State::kMerges)
            state_ = State::kMergesItem;
        else if (state_ == State::kMergesItem) {
            // 数组格式 merges：元素为 ["A","B"] 二元组（tokenizers≥0.20 保存形态）
            merge_pair_.clear();
            state_ = State::kMergesPair;
        }
        // kRootKeyAdded（added_tokens 数组）保持，元素对象由 start_object 处理
        return true;
    }
    bool end_array()
    {
        if (state_ == State::kSkip) {
            ExitSkipOnEnd();
            return true;
        }
        if (state_ == State::kMergesPair) {
            if (merge_pair_.size() == 2) {
                // 数组格式直接产出 pair（F-04：免拼接，token 含空格也不失真）
                merge_pairs_.emplace_back(std::move(merge_pair_[0]), std::move(merge_pair_[1]));
            }
            merge_pair_.clear();
            state_ = State::kMergesItem;
        } else if (state_ == State::kMergesItem) {
            state_ = State::kModel;
        } else if (state_ == State::kRootKeyAdded) {
            state_ = State::kRoot;
        }
        return true;
    }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&)
    {
        parse_ok_ = false;
        return false;
    }

    // 结果访问
    bool parse_ok() const { return parse_ok_; }
    const std::vector<std::pair<std::string, std::int64_t>>& added_tokens() const { return added_tokens_; }
    const std::vector<std::int64_t>& special_ids() const { return special_ids_; }
    const std::vector<std::string>& merges_raw() const { return merges_raw_; } // 字符串格式
    const std::vector<std::pair<std::string, std::string>>& merge_pairs() const
    {
        return merge_pairs_; // 数组格式
    }
    const std::vector<std::pair<std::string, std::int64_t>>& vocab_pairs() const { return vocab_pairs_; }
    const std::string& model_type() const { return model_type_; }
    bool byte_fallback() const { return byte_fallback_; }
    const std::string& unk_token() const { return unk_token_; }
    const std::string& normalizer_type() const { return normalizer_type_; }
    const std::string& normalizer_pattern() const { return normalizer_pattern_; }
    const std::string& normalizer_content() const { return normalizer_content_; }

private:
    bool HandleNumber(std::int64_t v)
    {
        if (state_ == State::kSkip) {
            ExitSkipOnScalar();
            return true;
        }
        if (state_ == State::kAddedId) {
            added_id_ = v; // int64 原样保存，范围校验在 load 期（F-01）
            state_ = State::kAddedElem;
        } else if (state_ == State::kVocabValue && !vocab_key_.empty()) {
            vocab_pairs_.emplace_back(vocab_key_, v);
            state_ = State::kVocab;
        }
        return true;
    }

    // 进入跳过态：未识别键的子树整体丢弃。
    // 退出规则（外审 F-05 修复）：仅 end_object/end_array 递减深度；
    // 标量事件在 depth==0 时（值为标量）立即退出，depth>0（嵌套容器内）忽略——
    // 若标量也递减，嵌套对象内部的标量会提前耗尽深度导致误退（外审实证）。
    void EnterSkip()
    {
        skip_return_ = state_;
        skip_depth_ = 0;
        state_ = State::kSkip;
    }
    void ExitSkipOnScalar()
    { // 未识别键的值为标量：消费即退出
        if (skip_depth_ == 0)
            state_ = skip_return_;
    }
    void ExitSkipOnEnd()
    { // 容器结束：递减深度，归零退出
        if (skip_depth_ > 0)
            --skip_depth_;
        if (skip_depth_ == 0)
            state_ = skip_return_;
    }

    enum class State {
        kRoot,
        kRootKeyAdded,
        kRootKeyModel,
        kRootKeyNormalizer,
        kAddedElem,
        kAddedId,
        kAddedContent,
        kAddedSpecial,
        kModel,
        kModelType,
        kModelByteFallback,
        kModelUnk,
        kVocab,
        kVocabValue,
        kMerges,
        kMergesItem,
        kMergesPair,
        kNormalizer,
        kNormalizerType,
        kNormalizerContent,
        kNormalizerPatternObj,
        kNormalizerPattern,
        kSkip,
    };
    State state_ = State::kRoot;
    State skip_return_ = State::kRoot;
    std::size_t skip_depth_ = 0;
    bool parse_ok_ = true;

    std::int64_t added_id_ = -1;
    std::string added_content_;
    bool added_special_ = false;
    std::vector<std::pair<std::string, std::int64_t>> added_tokens_;
    std::vector<std::int64_t> special_ids_;

    std::string vocab_key_;
    std::vector<std::pair<std::string, std::int64_t>> vocab_pairs_;
    std::vector<std::string> merges_raw_;
    std::vector<std::pair<std::string, std::string>> merge_pairs_; // 数组格式（免拼接）
    std::vector<std::string> merge_pair_;                          // 数组格式 merges 的当前二元组

    std::string model_type_;
    bool byte_fallback_ = false;
    std::string unk_token_;
    std::string normalizer_type_, normalizer_pattern_, normalizer_content_;
};

} // namespace

// ── L4 .bincache sidecar（perf_optimize.md §3.4）──────────────────────────────
// 二进制缓存：紧邻 tokenizer.json 的 "<json>.bincache"。收益 = 二次启动起
// 绕过 SAX+hash 构建（262K 词表 ~870ms → 顺序 read + 容器批量填充 ~百 ms 级）。
// 正确性由回退保证：一致性键（magic+version+源 size+mtime）任一不符、或读取
// 中途任何失败 ⇒ 整包丢弃走 JSON 全量（行为 = 现状）。缓存装载走与 JSON 完全
// 相同的构建+校验管道（仅输入源不同），使结构合法性（F-01 上界/双向映射冲突
// 拒/merges 交叉校验）在缓存面同样成立——被篡改/位漂的缓存最坏以失败告终回
// 退，不产 polluted 字表。first-run 落盘失败（只读分区）静默——第二次启动
// 照走 JSON。
// 运维开关（2026-10-30 反转为默认关，opt-in）：env `CANN_LM_TOKENIZER_
// BINCACHE` 显式开启缓存面——设置且非 "0" 才试命中+落盘；未设/显式 "0" =
// JSON 直载（= L4 合入前的基线行为：sidecar 不读不写，部署目录零副作用）。
// 归因：缓存收益仅在"目录持久、多次启动"形态成立——"开机整目录还原"形态
// 下缓存每次被抹，miss 态反而比无缓存多付 Serialize 16.5MB 写（真机实测
// miss 1267ms > 基线 925.6ms，perf_optimize.md §6.4），且默认开会令未受益
// 部署凭空多出 sidecar 文件。故默认走 JSON 直载，确知目录持久的部署显式开。

namespace {

// 头部 POD 定长、字段显式逐段 r/w（不做整 struct memcpy——规避 padding 布局
// 与字节序假设落盘；段间无 alignment 要求，r/w 全部经 helper）。
constexpr char kCacheMagic[8] = {'C', 'L', 'M', 'V', 'B', 'I', 'N', '1'};
constexpr std::uint32_t kCacheVersion = 1;

/// 缓存面总开关（env CANN_LM_TOKENIZER_BINCACHE，默认关/opt-in）：设置且
/// 非"0"即开——运维侧 export 任意非空值即开，显式 ="0" 关闭。取值语义与
/// debug_tensor_dump 的"非空即设"对齐（"0"为显式关）。每次 load 各查一次
/// 不缓存——load 一生一次，无热路径开销。
bool BincacheEnabled()
{
    const char* v = std::getenv("CANN_LM_TOKENIZER_BINCACHE");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

// 源文件一致性键（mtime 以 st_mtime 秒粒度——同秒内改写由 size 兜底；部署
// 产物 mtime 由镜像/OTA 整目录替换，粒度足够）
bool SourceKey(const std::filesystem::path& json_path, std::uint64_t* size, std::int64_t* mtime)
{
    struct stat sb {};
    if (::stat(json_path.c_str(), &sb) != 0)
        return false;
    // fit：fs::file_size() 返回 uintmax_t（host 64 位），st_size 为有符号——
    // 双向入宽符号统一后再写缓存头
    *size = static_cast<std::uint64_t>(sb.st_size);
    *mtime = static_cast<std::int64_t>(sb.st_mtime);
    return true;
}

bool WriteU32(std::ostream& out, std::uint32_t v)
{
    return static_cast<bool>(out.write(reinterpret_cast<const char*>(&v), 4));
}
bool ReadU32(std::istream& in, std::uint32_t* v) { return static_cast<bool>(in.read(reinterpret_cast<char*>(v), 4)); }
bool WriteI64(std::ostream& out, std::int64_t v)
{
    return static_cast<bool>(out.write(reinterpret_cast<const char*>(&v), 8));
}
bool ReadI64(std::istream& in, std::int64_t* v) { return static_cast<bool>(in.read(reinterpret_cast<char*>(v), 8)); }
bool WriteStr(std::ostream& out, const std::string& s)
{
    if (s.size() > 0xFFFFFFFFu)
        return false;
    if (!WriteU32(out, static_cast<std::uint32_t>(s.size())))
        return false;
    return static_cast<bool>(out.write(s.data(), static_cast<std::streamsize>(s.size())));
}
bool ReadStr(std::istream& in, std::string* s)
{
    std::uint32_t n = 0;
    if (!ReadU32(in, &n))
        return false;
    s->assign(n, '\0');
    if (n == 0)
        return true;
    in.read(s->data(), static_cast<std::streamsize>(n));
    return static_cast<std::size_t>(in.gcount()) == static_cast<std::size_t>(n);
}

/// 键位回填（load 落缓存成功路径）：头部 offset 12（8 magic + 4 version）起
/// 12 字（u32 lo + u32 hi + i64 mtime）。半写窗口安全：truncate 起步首 8 字
/// 即零 → magic 失配拒；本函数执行前（键位为零）对任何真实源恒失配拒。
void PlugKey(const std::filesystem::path& cache_path, std::uint64_t size, std::int64_t mtime)
{
    std::ofstream f(cache_path, std::ios::binary | std::ios::in);
    if (!f)
        return;
    f.seekp(12, std::ios::beg);
    WriteU32(f, static_cast<std::uint32_t>(size));
    WriteU32(f, static_cast<std::uint32_t>(size >> 32));
    WriteI64(f, mtime);
}

} // namespace

Status VocabStore::Serialize(const std::filesystem::path& cache_path) const
{
    // 落盘失败静默（load 分流点 (void)Serialize——只读分区/权限缺失为常态
    // 部署形态，缓存纯增益，不改变加载结果）。公开重载仅写数据段（键位补
    // 写由 load 分流点 TakeKey+PatchKey 完成——Serialize 不持 json 路径）。
    std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
    if (!out)
        return Status::FAILURE;

    out.write(kCacheMagic, 8);
    WriteU32(out, kCacheVersion);
    WriteU32(out, 0); // source_size_lo 键位（load 分流点回填，见 PlugKey）
    WriteU32(out, 0); // source_size_hi
    WriteI64(out, 0); // source_mtime 键位
    WriteU32(out, static_cast<std::uint32_t>(tok2id_.size()));
    WriteU32(out, static_cast<std::uint32_t>(id2tok_.size()));
    WriteU32(out, static_cast<std::uint32_t>(merges_.size()));
    WriteU32(out, static_cast<std::uint32_t>(added_tokens_.size()));
    WriteU32(out, static_cast<std::uint32_t>(special_strip_ids_.size()));

    for (const auto& [lit, id] : tok2id_) {
        if (!WriteStr(out, lit))
            return Status::FAILURE;
        if (!WriteU32(out, static_cast<std::uint32_t>(id)))
            return Status::FAILURE;
    }
    for (const auto& s : id2tok_) {
        if (!WriteStr(out, s))
            return Status::FAILURE;
    }
    for (const auto& [a, b] : merges_) {
        if (!WriteStr(out, a))
            return Status::FAILURE;
        if (!WriteStr(out, b))
            return Status::FAILURE;
    }
    for (const auto& [lit, id] : added_tokens_) {
        if (!WriteStr(out, lit))
            return Status::FAILURE;
        if (!WriteU32(out, static_cast<std::uint32_t>(id)))
            return Status::FAILURE;
    }
    for (const int id : special_strip_ids_) {
        if (!WriteU32(out, static_cast<std::uint32_t>(id)))
            return Status::FAILURE;
    }
    if (!WriteStr(out, model_type_))
        return Status::FAILURE;
    if (!out.write(reinterpret_cast<const char*>(&byte_fallback_), 1)) {
        return Status::FAILURE;
    }
    if (!WriteStr(out, unk_token_))
        return Status::FAILURE;
    if (!out.write(reinterpret_cast<const char*>(&has_replace_normalizer_), 1)) {
        return Status::FAILURE;
    }
    if (!WriteStr(out, replace_normalizer_.first))
        return Status::FAILURE;
    if (!WriteStr(out, replace_normalizer_.second))
        return Status::FAILURE;
    out.flush();
    return out.good() ? Status::SUCCESS : Status::FAILURE;
}

Status VocabStore::TryLoadFromCache(const std::filesystem::path& json_path, const std::filesystem::path& cache_path)
{
    // 键校验失败/文件缺失/任何读失败 → FAILURE（调用方回退 JSON 全量）。
    std::uint64_t src_size = 0;
    std::int64_t src_mtime = 0;
    if (!SourceKey(json_path, &src_size, &src_mtime))
        return Status::FAILURE;

    std::ifstream in(cache_path, std::ios::binary);
    if (!in)
        return Status::FAILURE;

    char magic[8] = {};
    if (!in.read(magic, 8) || std::memcmp(magic, kCacheMagic, 8) != 0) {
        return Status::FAILURE;
    }
    std::uint32_t version = 0;
    if (!ReadU32(in, &version) || version != kCacheVersion)
        return Status::FAILURE;
    std::uint64_t size_key = 0;
    if (!ReadU32(in, reinterpret_cast<std::uint32_t*>(&size_key))) {
        return Status::FAILURE;
    }
    {
        std::uint32_t hi = 0;
        if (!ReadU32(in, &hi))
            return Status::FAILURE;
        size_key |= static_cast<std::uint64_t>(hi) << 32;
    }
    if (size_key != src_size)
        return Status::FAILURE;
    std::int64_t mtime_key = 0;
    if (!ReadI64(in, &mtime_key) || mtime_key != src_mtime) {
        return Status::FAILURE;
    }

    // 计数（读入后与实际段长一致由"读到流尾"兜底——段计数不足/超出时流读
    // 失败即回退，不做全量前读校验）
    std::uint32_t vocab_num = 0, id2tok_num = 0, merge_num = 0, added_num = 0, special_num = 0;
    if (!ReadU32(in, &vocab_num) || !ReadU32(in, &id2tok_num) || !ReadU32(in, &merge_num) || !ReadU32(in, &added_num) ||
        !ReadU32(in, &special_num)) {
        return Status::FAILURE;
    }
    if (vocab_num == 0 || id2tok_num == 0)
        return Status::FAILURE; // 空词表非法

    // 局部构建（与 JSON 完全相同的校验管道——F-02/F-01/merges 交叉校验）
    std::unordered_map<std::string, int> tok2id;
    std::vector<std::string> id2tok;
    std::vector<std::pair<std::string, std::string>> merges;
    std::vector<std::pair<std::string, int>> added_tokens;
    std::vector<int> special_strip_ids;
    std::string model_type, unk_token;
    bool byte_fallback = false;
    bool has_replace_normalizer = false;
    std::pair<std::string, std::string> replace_normalizer;

    constexpr std::int64_t kMaxReasonableTokenId = 10'000'000;

    tok2id.reserve(vocab_num);
    for (std::uint32_t i = 0; i < vocab_num; ++i) {
        std::string lit;
        std::uint32_t id32 = 0;
        if (!ReadStr(in, &lit) || !ReadU32(in, &id32))
            return Status::FAILURE;
        if (id32 > static_cast<std::uint32_t>(kMaxReasonableTokenId)) {
            return Status::FAILURE;
        }
        if (!tok2id.emplace(std::move(lit), static_cast<int>(id32)).second) {
            return Status::FAILURE; // 同一字面量双 id
        }
    }
    id2tok.reserve(id2tok_num);
    for (std::uint32_t i = 0; i < id2tok_num; ++i) {
        std::string s;
        if (!ReadStr(in, &s))
            return Status::FAILURE;
        id2tok.push_back(std::move(s));
    }
    if (id2tok.size() != id2tok_num)
        return Status::FAILURE;
    for (std::uint32_t i = 0; i < id2tok_num; ++i) {
        // 同一 id 多字面量：非空字面量冲突拒；空串（gap id）放行——与 JSON
        // 路径 id2tok 空 gap 语义一致（id_to_token 对空串返回 nullopt）
        const std::string& s = id2tok[i];
        if (!s.empty() && !tok2id.count(s))
            return Status::FAILURE;
    }
    merges.reserve(merge_num);
    for (std::uint32_t i = 0; i < merge_num; ++i) {
        std::string a, b;
        if (!ReadStr(in, &a) || !ReadStr(in, &b))
            return Status::FAILURE;
        merges.emplace_back(std::move(a), std::move(b));
    }
    added_tokens.reserve(added_num);
    for (std::uint32_t i = 0; i < added_num; ++i) {
        std::string lit;
        std::uint32_t id32 = 0;
        if (!ReadStr(in, &lit) || !ReadU32(in, &id32))
            return Status::FAILURE;
        if (id32 > static_cast<std::uint32_t>(kMaxReasonableTokenId)) {
            return Status::FAILURE;
        }
        added_tokens.emplace_back(std::move(lit), static_cast<int>(id32));
    }
    special_strip_ids.reserve(special_num);
    for (std::uint32_t i = 0; i < special_num; ++i) {
        std::uint32_t id32 = 0;
        if (!ReadU32(in, &id32))
            return Status::FAILURE;
        if (id32 > static_cast<std::uint32_t>(kMaxReasonableTokenId)) {
            return Status::FAILURE;
        }
        special_strip_ids.push_back(static_cast<int>(id32));
    }
    // 后置排序（缓存写出序未必有序——特殊集源序无序）
    std::sort(special_strip_ids.begin(), special_strip_ids.end());
    // merges 交叉校验（与 JSON 路径同构：引用 token 必在词表）
    for (const auto& [a, b] : merges) {
        if (!tok2id.count(a) || !tok2id.count(b))
            return Status::FAILURE;
    }
    // id 一致性：id2tok 非空槽位的 id 必须回查 tok2id 一致（防 id↔lit 翻转位
    // 漂——双表互证）
    for (std::uint32_t i = 0; i < id2tok_num; ++i) {
        const std::string& s = id2tok[i];
        if (s.empty())
            continue;
        auto it = tok2id.find(s);
        if (it == tok2id.end() || it->second != static_cast<int>(i)) {
            return Status::FAILURE;
        }
    }

    // 尾部元信息
    std::uint8_t byte_fback_u8 = 0;
    std::uint8_t has_replace_u8 = 0;
    if (!ReadStr(in, &model_type))
        return Status::FAILURE;
    if (!in.read(reinterpret_cast<char*>(&byte_fback_u8), 1))
        return Status::FAILURE;
    if (!ReadStr(in, &unk_token))
        return Status::FAILURE;
    if (!in.read(reinterpret_cast<char*>(&has_replace_u8), 1))
        return Status::FAILURE;
    std::string pat, rep;
    if (!ReadStr(in, &pat) || !ReadStr(in, &rep))
        return Status::FAILURE;
    // 尾字节之后必须恰为 EOF（防拼接到别的缓存的截断头+段组合通过校验）
    if (in.get() != std::istream::traits_type::eof())
        return Status::FAILURE;

    byte_fallback = byte_fback_u8 != 0;
    has_replace_normalizer = has_replace_u8 != 0;
    if (has_replace_normalizer && pat.empty())
        return Status::FAILURE;
    if (has_replace_normalizer)
        replace_normalizer = {std::move(pat), std::move(rep)};

    // 全部校验通过：提交成员
    tok2id_ = std::move(tok2id);
    id2tok_ = std::move(id2tok);
    merges_ = std::move(merges);
    added_tokens_ = std::move(added_tokens);
    special_strip_ids_ = std::move(special_strip_ids);
    model_type_ = std::move(model_type);
    byte_fallback_ = byte_fallback;
    unk_token_ = std::move(unk_token);
    has_replace_normalizer_ = has_replace_normalizer;
    replace_normalizer_ = std::move(replace_normalizer);
    return Status::SUCCESS;
}

// ── VocabStore ──────────────────────────────────────────────────────────────

// L4：JSON 全量解析（原 load 本体，SAX → 九成员局部构建 → 校验 → 提交）。
Status VocabStore::load_from_json_(const VocabSource& src)
{
    std::unordered_map<std::string, int> tok2id;
    std::vector<std::string> id2tok;
    std::vector<std::pair<std::string, std::string>> merges;
    std::vector<std::pair<std::string, int>> added_tokens;
    std::vector<int> special_strip_ids;
    std::string model_type, unk_token;
    bool byte_fallback = false;
    bool has_replace_normalizer = false;
    std::pair<std::string, std::string> replace_normalizer;

    // id 合理性上界（外审 F-01：畸形产物的超大 id 曾致 id2tok 巨额分配 bad_alloc
    // 崩溃 / int 静默截断污染词表——现统一按结构非法拒绝）
    constexpr std::int64_t kMaxReasonableTokenId = 10'000'000;

    // 读取产物（路径或内存内容 → 统一流解析，避免整文件双缓冲——外审 F-06）
    std::ifstream file;
    std::istringstream memory;
    std::istream* in = nullptr;
    if (const auto* path = std::get_if<std::filesystem::path>(&src.artifact)) {
        std::error_code ec;
        if (!std::filesystem::exists(*path, ec))
            return Status::FAILURE;
        file.open(*path, std::ios::binary);
        if (!file)
            return Status::FAILURE;
        in = &file;
    } else if (const auto* bytes = std::get_if<std::vector<std::byte>>(&src.artifact)) {
        memory.str(std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size()));
        in = &memory;
    } else {
        return Status::FAILURE;
    }

    // SAX 解析（流式，无中间 string 拷贝）
    TokenizerJsonSax sax;
    if (!nlohmann::json::sax_parse(*in, &sax) || !sax.parse_ok()) {
        return Status::FAILURE;
    }

    // base vocab
    if (sax.vocab_pairs().empty())
        return Status::FAILURE; // 空词表：结构非法
    std::int64_t max_id = -1;
    for (const auto& [lit, id] : sax.vocab_pairs()) {
        if (id < 0 || id > kMaxReasonableTokenId)
            return Status::FAILURE;
        if (!tok2id.emplace(lit, static_cast<int>(id)).second) {
            return Status::FAILURE; // 同一字面量双 id
        }
        max_id = std::max(max_id, id);
    }
    id2tok.assign(static_cast<size_t>(max_id) + 1, std::string());
    for (const auto& [lit, id] : sax.vocab_pairs()) {
        if (!id2tok[static_cast<size_t>(id)].empty()) {
            return Status::FAILURE; // 同一 id 多字面量（vocab dict 内冲突）
        }
        id2tok[static_cast<size_t>(id)] = lit;
    }

    // added_tokens overlay（Qwen 追加式 / gemma 内联式统一：added 覆写双向映射，
    // 与 HF AddedVocabulary 语义一致——外审 F-03 注释订正）
    for (const auto& [lit, id64] : sax.added_tokens()) {
        if (id64 < 0 || id64 > kMaxReasonableTokenId)
            return Status::FAILURE;
        const int id = static_cast<int>(id64);
        const size_t uid = static_cast<size_t>(id);
        added_tokens.emplace_back(lit, id);
        if (uid >= id2tok.size())
            id2tok.resize(uid + 1);
        id2tok[uid] = lit;
        tok2id[lit] = id;
    }
    for (const auto id64 : sax.special_ids()) {
        special_strip_ids.push_back(static_cast<int>(id64));
    }
    std::sort(special_strip_ids.begin(), special_strip_ids.end());

    // merges：字符串格式（"A B" 按首空格二分，token 不得含空格）+ 数组格式
    // （["A","B"] 直出 pair，token 可含空格——外审 F-04）
    for (const auto& m : sax.merges_raw()) {
        const auto sep = m.find(' ');
        if (sep == std::string::npos || m.find(' ', sep + 1) != std::string::npos) {
            return Status::FAILURE; // merges 条目非 "A B" 二元组：结构非法
        }
        merges.emplace_back(m.substr(0, sep), m.substr(sep + 1));
    }
    merges.insert(merges.end(), sax.merge_pairs().begin(), sax.merge_pairs().end());
    // 交叉校验：merges 引用的 token 必须在词表中（TC_102）
    for (const auto& [a, b] : merges) {
        if (!tok2id.count(a) || !tok2id.count(b))
            return Status::FAILURE;
    }

    // 元信息
    model_type = sax.model_type();
    byte_fallback = sax.byte_fallback();
    unk_token = sax.unk_token();
    if (sax.normalizer_type() == "Replace" && !sax.normalizer_pattern().empty()) {
        has_replace_normalizer = true;
        replace_normalizer = {sax.normalizer_pattern(), sax.normalizer_content()};
    }

    // 全部校验通过：提交成员（加载后只读）
    tok2id_ = std::move(tok2id);
    id2tok_ = std::move(id2tok);
    merges_ = std::move(merges);
    added_tokens_ = std::move(added_tokens);
    special_strip_ids_ = std::move(special_strip_ids);
    model_type_ = std::move(model_type);
    byte_fallback_ = byte_fallback;
    unk_token_ = std::move(unk_token);
    has_replace_normalizer_ = has_replace_normalizer;
    replace_normalizer_ = std::move(replace_normalizer);
    return Status::SUCCESS;
}

// load 入口（L4 分流）：path 态且 BincacheEnabled（env CANN_LM_TOKENIZER_
// BINCACHE，默认未设 = 关）→ 先试 .bincache → 失败回退 JSON 全量；成功且
// JSON 路径生效时静默落缓存。默认关与 bytes 态（内存）无缓存面，直走 JSON
// （= L4 合入前基线行为：sidecar 不读不写）。
Status VocabStore::load(const VocabSource& src)
{
    if (const auto* path = std::get_if<std::filesystem::path>(&src.artifact)) {
        if (!BincacheEnabled()) {
            CANN_LLM_LOGD("vocab.bincache: cache off (env CANN_LM_TOKENIZER_BINCACHE unset/\"0\"), "
                          "json direct, path=%s",
                          path->c_str());
            return load_from_json_(src);
        }
        // 先试缓存（键校验在 TryLoadFromCache 内部：源不存在时 SourceKey 失败
        // → FAILURE，与现状 exists() 拒绝语义一致）
        std::filesystem::path cache = *path;
        cache += ".bincache";
        if (TryLoadFromCache(*path, cache).ok()) {
            CANN_LLM_LOGD("vocab.bincache: cache hit path=%s", path->c_str());
            return Status::SUCCESS;
        }
        const Status st = load_from_json_(src);
        if (st.ok()) {
            // 静默落缓存（失败无声——下次启动照走 JSON）：先取源键再写数据，
            // 后键位回填（两个窗口：truncate 首字节仍非 CLMVBIN1 即整包拒，
            // 键位零值对任何真实源恒失配——半写缓存不可用，安全）。
            std::uint64_t src_size = 0;
            std::int64_t src_mtime = 0;
            if (SourceKey(*path, &src_size, &src_mtime) && Serialize(cache).ok()) {
                PlugKey(cache, src_size, src_mtime);
            }
        }
        return st;
    }
    return load_from_json_(src);
}

std::optional<int> VocabStore::token_to_id(std::string_view tok) const
{
    // C++17 unordered_map 无 string_view 异构查找（透明哈希为 C++20）——构造
    // string 查找；encode 热点开销登记 U-1 优化项（string_view 键/透明哈希）
    auto it = tok2id_.find(std::string(tok));
    if (it == tok2id_.end())
        return std::nullopt;
    return it->second;
}

std::optional<std::string_view> VocabStore::id_to_token(int id) const
{
    if (id < 0 || static_cast<size_t>(id) >= id2tok_.size())
        return std::nullopt;
    const std::string& s = id2tok_[static_cast<size_t>(id)];
    if (s.empty())
        return std::nullopt;
    return std::string_view(s);
}

size_t VocabStore::vocab_size() const { return id2tok_.size(); }

const std::vector<std::pair<std::string, std::string>>& VocabStore::merges() const { return merges_; }

const std::vector<std::pair<std::string, int>>& VocabStore::added_tokens() const { return added_tokens_; }

const std::vector<int>& VocabStore::special_strip_ids() const { return special_strip_ids_; }

} // namespace cann_local_lm::engine
