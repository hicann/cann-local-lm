#include "core/component/tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace cann_lm::engine {

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
    bool null() {
        if (state_ == State::kSkip) { ExitSkipOnScalar(); return true; }
        // 已识别键的 null 值（"normalizer": null、"unk_token": null 等）→ 统一回退父态
        switch (state_) {
            case State::kRootKeyAdded:
            case State::kRootKeyModel:
            case State::kRootKeyNormalizer: state_ = State::kRoot; break;
            case State::kAddedId:
            case State::kAddedContent:
            case State::kAddedSpecial: state_ = State::kAddedElem; break;
            case State::kModelType:
            case State::kModelByteFallback:
            case State::kModelUnk: state_ = State::kModel; break;
            case State::kVocabValue: state_ = State::kVocab; break;
            case State::kNormalizerType:
            case State::kNormalizerContent: state_ = State::kNormalizer; break;
            case State::kNormalizerPattern: state_ = State::kNormalizerPatternObj; break;
            default: break;
        }
        return true;
    }
    bool boolean(bool v) {
        if (state_ == State::kSkip) { ExitSkipOnScalar(); return true; }
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
    bool number_unsigned(std::uint64_t v) {
        return HandleNumber(static_cast<std::int64_t>(v));
    }
    bool number_float(double, const std::string&) { return true; }
    bool string(std::string& v) {
        if (state_ == State::kSkip) { ExitSkipOnScalar(); return true; }
        switch (state_) {
            case State::kModelType: model_type_ = v; state_ = State::kModel; break;
            case State::kModelUnk: unk_token_ = v; state_ = State::kModel; break;
            case State::kAddedContent: added_content_ = v; state_ = State::kAddedElem; break;
            case State::kMergesItem: merges_raw_.push_back(v); break;  // 字符串格式 "A B"
            case State::kMergesPair: merge_pair_.push_back(v); break;  // 数组格式 ["A","B"]
            case State::kNormalizerType: normalizer_type_ = v; state_ = State::kNormalizer; break;
            case State::kNormalizerContent: normalizer_content_ = v; state_ = State::kNormalizer; break;
            case State::kNormalizerPattern:
                normalizer_pattern_ = v;
                state_ = State::kNormalizerPatternObj;
                break;
            default: break;  // 其余字符串（version 等）丢弃
        }
        return true;
    }
    bool binary(nlohmann::json::binary_t&) { return true; }
    bool start_object(std::size_t) {
        if (state_ == State::kSkip) { ++skip_depth_; return true; }
        switch (state_) {
            case State::kRootKeyAdded: state_ = State::kAddedElem; break;
            case State::kRootKeyModel: state_ = State::kModel; break;
            case State::kRootKeyNormalizer: state_ = State::kNormalizer; break;
            case State::kNormalizerPatternObj: break;  // pattern 对象内层，保持
            default: break;
        }
        return true;
    }
    bool key(std::string& k) {
        if (state_ == State::kSkip) return true;
        switch (state_) {
            case State::kRoot:
                if (k == "added_tokens") state_ = State::kRootKeyAdded;
                else if (k == "model") state_ = State::kRootKeyModel;
                else if (k == "normalizer") state_ = State::kRootKeyNormalizer;
                else if (!k.empty()) EnterSkip();  // version/truncation/padding/pre_tokenizer/... 跳过
                break;
            case State::kAddedElem:
                if (k == "id") state_ = State::kAddedId;
                else if (k == "content") state_ = State::kAddedContent;
                else if (k == "special") state_ = State::kAddedSpecial;
                else EnterSkip();  // lstrip/rstrip/single_word/normalized 等跳过
                break;
            case State::kModel:
                if (k == "type") state_ = State::kModelType;
                else if (k == "byte_fallback") state_ = State::kModelByteFallback;
                else if (k == "unk_token") state_ = State::kModelUnk;
                else if (k == "vocab") state_ = State::kVocab;
                else if (k == "merges") state_ = State::kMerges;
                else EnterSkip();  // ignore_merges/dropout/continuing_subword 等跳过
                break;
            case State::kVocab:
                vocab_key_ = k;
                state_ = State::kVocabValue;
                break;
            case State::kNormalizer:
                if (k == "type") state_ = State::kNormalizerType;
                else if (k == "content") state_ = State::kNormalizerContent;
                else if (k == "pattern") state_ = State::kNormalizerPatternObj;
                else EnterSkip();
                break;
            case State::kNormalizerPatternObj:
                if (k == "String") state_ = State::kNormalizerPattern;
                else EnterSkip();  // Regex 类型 pattern（本模块不实现）跳过
                break;
            default: break;
        }
        return true;
    }
    bool end_object() {
        if (state_ == State::kSkip) { ExitSkipOnEnd(); return true; }
        switch (state_) {
            case State::kAddedElem:
                if (added_id_ >= 0) {
                    added_tokens_.emplace_back(added_content_, added_id_);
                    if (added_special_) special_ids_.push_back(added_id_);
                }
                added_id_ = -1;
                added_content_.clear();
                added_special_ = false;
                state_ = State::kRootKeyAdded;  // 数组内继续或由 end_array 收尾
                break;
            case State::kVocab: state_ = State::kModel; break;
            case State::kModel: state_ = State::kRoot; break;
            case State::kNormalizer: state_ = State::kRoot; break;
            case State::kNormalizerPatternObj: state_ = State::kNormalizer; break;
            default: break;
        }
        return true;
    }
    bool start_array(std::size_t) {
        if (state_ == State::kSkip) { ++skip_depth_; return true; }
        if (state_ == State::kMerges) state_ = State::kMergesItem;
        else if (state_ == State::kMergesItem) {
            // 数组格式 merges：元素为 ["A","B"] 二元组（tokenizers≥0.20 保存形态）
            merge_pair_.clear();
            state_ = State::kMergesPair;
        }
        // kRootKeyAdded（added_tokens 数组）保持，元素对象由 start_object 处理
        return true;
    }
    bool end_array() {
        if (state_ == State::kSkip) { ExitSkipOnEnd(); return true; }
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
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) {
        parse_ok_ = false;
        return false;
    }

    // 结果访问
    bool parse_ok() const { return parse_ok_; }
    const std::vector<std::pair<std::string, std::int64_t>>& added_tokens() const { return added_tokens_; }
    const std::vector<std::int64_t>& special_ids() const { return special_ids_; }
    const std::vector<std::string>& merges_raw() const { return merges_raw_; }   // 字符串格式
    const std::vector<std::pair<std::string, std::string>>& merge_pairs() const {
        return merge_pairs_;                                                      // 数组格式
    }
    const std::vector<std::pair<std::string, std::int64_t>>& vocab_pairs() const {
        return vocab_pairs_;
    }
    const std::string& model_type() const { return model_type_; }
    bool byte_fallback() const { return byte_fallback_; }
    const std::string& unk_token() const { return unk_token_; }
    const std::string& normalizer_type() const { return normalizer_type_; }
    const std::string& normalizer_pattern() const { return normalizer_pattern_; }
    const std::string& normalizer_content() const { return normalizer_content_; }

private:
    bool HandleNumber(std::int64_t v) {
        if (state_ == State::kSkip) { ExitSkipOnScalar(); return true; }
        if (state_ == State::kAddedId) {
            added_id_ = v;  // int64 原样保存，范围校验在 load 期（F-01）
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
    void EnterSkip() {
        skip_return_ = state_;
        skip_depth_ = 0;
        state_ = State::kSkip;
    }
    void ExitSkipOnScalar() {           // 未识别键的值为标量：消费即退出
        if (skip_depth_ == 0) state_ = skip_return_;
    }
    void ExitSkipOnEnd() {              // 容器结束：递减深度，归零退出
        if (skip_depth_ > 0) --skip_depth_;
        if (skip_depth_ == 0) state_ = skip_return_;
    }

    enum class State {
        kRoot, kRootKeyAdded, kRootKeyModel, kRootKeyNormalizer,
        kAddedElem, kAddedId, kAddedContent, kAddedSpecial,
        kModel, kModelType, kModelByteFallback, kModelUnk, kVocab, kVocabValue,
        kMerges, kMergesItem, kMergesPair,
        kNormalizer, kNormalizerType, kNormalizerContent,
        kNormalizerPatternObj, kNormalizerPattern,
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
    std::vector<std::pair<std::string, std::string>> merge_pairs_;  // 数组格式（免拼接）
    std::vector<std::string> merge_pair_;  // 数组格式 merges 的当前二元组

    std::string model_type_;
    bool byte_fallback_ = false;
    std::string unk_token_;
    std::string normalizer_type_, normalizer_pattern_, normalizer_content_;
};

}  // namespace

// ── VocabStore ──────────────────────────────────────────────────────────────
Status VocabStore::load(const VocabSource& src) {
    // 局部构建、成功才提交成员（外审 F-02：任何失败路径不留"半加载可用"实例——
    // 失败实例 vocab_size()==0，encode/decode 恒 FAILURE）
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
        if (!std::filesystem::exists(*path, ec)) return Status::FAILURE;
        file.open(*path, std::ios::binary);
        if (!file) return Status::FAILURE;
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
    if (sax.vocab_pairs().empty()) return Status::FAILURE;  // 空词表：结构非法
    std::int64_t max_id = -1;
    for (const auto& [lit, id] : sax.vocab_pairs()) {
        if (id < 0 || id > kMaxReasonableTokenId) return Status::FAILURE;
        if (!tok2id.emplace(lit, static_cast<int>(id)).second) {
            return Status::FAILURE;  // 同一字面量双 id
        }
        max_id = std::max(max_id, id);
    }
    id2tok.assign(static_cast<size_t>(max_id) + 1, std::string());
    for (const auto& [lit, id] : sax.vocab_pairs()) {
        if (!id2tok[static_cast<size_t>(id)].empty()) {
            return Status::FAILURE;  // 同一 id 多字面量（vocab dict 内冲突）
        }
        id2tok[static_cast<size_t>(id)] = lit;
    }

    // added_tokens overlay（Qwen 追加式 / gemma 内联式统一：added 覆写双向映射，
    // 与 HF AddedVocabulary 语义一致——外审 F-03 注释订正）
    for (const auto& [lit, id64] : sax.added_tokens()) {
        if (id64 < 0 || id64 > kMaxReasonableTokenId) return Status::FAILURE;
        const int id = static_cast<int>(id64);
        const size_t uid = static_cast<size_t>(id);
        added_tokens.emplace_back(lit, id);
        if (uid >= id2tok.size()) id2tok.resize(uid + 1);
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
            return Status::FAILURE;  // merges 条目非 "A B" 二元组：结构非法
        }
        merges.emplace_back(m.substr(0, sep), m.substr(sep + 1));
    }
    merges.insert(merges.end(), sax.merge_pairs().begin(), sax.merge_pairs().end());
    // 交叉校验：merges 引用的 token 必须在词表中（TC_102）
    for (const auto& [a, b] : merges) {
        if (!tok2id.count(a) || !tok2id.count(b)) return Status::FAILURE;
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

std::optional<int> VocabStore::token_to_id(std::string_view tok) const {
    // C++17 unordered_map 无 string_view 异构查找（透明哈希为 C++20）——构造
    // string 查找；encode 热点开销登记 U-1 优化项（string_view 键/透明哈希）
    auto it = tok2id_.find(std::string(tok));
    if (it == tok2id_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string_view> VocabStore::id_to_token(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id2tok_.size()) return std::nullopt;
    const std::string& s = id2tok_[static_cast<size_t>(id)];
    if (s.empty()) return std::nullopt;
    return std::string_view(s);
}

size_t VocabStore::vocab_size() const { return id2tok_.size(); }

const std::vector<std::pair<std::string, std::string>>& VocabStore::merges() const {
    return merges_;
}

const std::vector<std::pair<std::string, int>>& VocabStore::added_tokens() const {
    return added_tokens_;
}

const std::vector<int>& VocabStore::special_strip_ids() const {
    return special_strip_ids_;
}

}  // namespace cann_lm::engine
