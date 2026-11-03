#include "core/component/tokenizer/tokenizer.h"

#include "infra/log.h"

#include <algorithm>

namespace cann_local_lm::engine {

// ── M-06 SpecialTokenHandler（v0.7：零注入缺省 + 截断预算 + 剥离集 +
//    meta special_tokens 映射/解映射——D-3 meta 优先 + WARN）────────────────

void SpecialTokenHandler::configure(const TokenizerConfig& cfg, std::vector<int> strip_ids)
{
    cfg_ = cfg;

    // meta special_tokens 消费（D-3/D-4）：id→literal 查找表（decode 解映射）；
    // id<0 条目 WARN 跳过（与具名字段 -1 哨兵语义对齐，不静默）；同 id 双字面量
    // WARN 保首条（id→literal 映射天然单值，歧义显式化）
    id2lit_.clear();
    std::vector<int> meta_ids;
    meta_ids.reserve(cfg.special_tokens.size());
    for (const auto& [lit, tid] : cfg.special_tokens) {
        if (tid < 0) {
            CANN_LLM_LOGW("tokenizer.special.meta: negative id ignored (literal=%s, id=%d)", lit.c_str(), tid);
            continue;
        }
        if (auto [it, inserted] = id2lit_.emplace(tid, lit); !inserted) {
            CANN_LLM_LOGW("tokenizer.special.meta: duplicate id (id=%d, literal=%s kept, %s dropped)", tid,
                          it->second.c_str(), lit.c_str());
            continue;
        }
        meta_ids.push_back(tid);
    }

    // meta special id 并入剥离集（视为 special，skip_special_tokens 一并剥离）；
    // 与产物 special_strip_ids 归并去重
    strip_ids.insert(strip_ids.end(), meta_ids.begin(), meta_ids.end());
    std::sort(strip_ids.begin(), strip_ids.end());
    strip_ids.erase(std::unique(strip_ids.begin(), strip_ids.end()), strip_ids.end());
    strip_ids_ = std::move(strip_ids);
}

Status SpecialTokenHandler::inject(std::vector<int>* ids, const EncodeOptions& opts) const
{
    if (!ids)
        return Status::INVALID_ARGUMENT;

    // 注入语义误用检查（M-06 §6：跳过 + WARN，不静默——外审 F-10；
    // SR20260903001 D-13：-1 哨兵 = 未配置）
    if (opts.add_special_tokens && cfg_.add_bos && cfg_.bos_token_id < 0) {
        CANN_LLM_LOGW("tokenizer.special.misconfigured: add_bos=true but bos_id unset; injection skipped");
    }
    if (opts.add_special_tokens && cfg_.add_eos && cfg_.eos_token_id < 0) {
        CANN_LLM_LOGW("tokenizer.special.misconfigured: add_eos=true but eos_id unset; injection skipped");
    }
    const bool add_bos = opts.add_special_tokens && cfg_.add_bos && cfg_.bos_token_id >= 0;
    const bool add_eos = opts.add_special_tokens && cfg_.add_eos && cfg_.eos_token_id >= 0;

    // ① 截断：special 计入 max_length 预算（先截内容，eos 不被截掉——M-3 裁决）
    if (opts.max_length.has_value()) {
        const size_t budget = *opts.max_length > (add_bos ? 1u : 0u) + (add_eos ? 1u : 0u) ?
                                  *opts.max_length - (add_bos ? 1u : 0u) - (add_eos ? 1u : 0u) :
                                  0u;
        if (ids->size() > budget)
            ids->resize(budget);
    }

    // ② 注入：首 bos、尾 eos（仅显式配置时；缺省零注入——H-3 裁决）
    if (add_bos)
        ids->insert(ids->begin(), cfg_.bos_token_id);
    if (add_eos)
        ids->push_back(cfg_.eos_token_id);

    // ③ padding：Fixed 用 pad_to_length；Longest 在单条 encode 语境为 no-op
    //   （批量对齐由 TokenizerBase::batch_encode 承接）
    if (opts.padding == EncodeOptions::Padding::Fixed && opts.pad_to_length.has_value()) {
        if (cfg_.pad_token_id < 0)
            return Status::INVALID_ARGUMENT; // pad 需求无 pad id
        if (ids->empty())
            return Status::INVALID_ARGUMENT; // 空 padding：无真实 token
        const size_t target = *opts.pad_to_length;
        if (ids->size() > target)
            return Status::INVALID_ARGUMENT; // 目标长度小于现有长度
        ids->resize(target, cfg_.pad_token_id);
    }
    return Status::SUCCESS;
}

Status SpecialTokenHandler::strip(const std::vector<int>& ids, std::vector<int>* kept) const
{
    if (!kept)
        return Status::INVALID_ARGUMENT;
    kept->clear();
    kept->reserve(ids.size());
    for (const int id : ids) {
        if (!std::binary_search(strip_ids_.begin(), strip_ids_.end(), id)) {
            kept->push_back(id);
        }
    }
    return Status::SUCCESS;
}

std::optional<int> SpecialTokenHandler::id(std::string_view name) const
{
    // D-13：-1 哨兵 → nullopt（对外 optional 语义维持）
    if (name == "bos")
        return cfg_.bos_token_id >= 0 ? std::optional<int>(cfg_.bos_token_id) : std::nullopt;
    if (name == "eos")
        return cfg_.eos_token_id >= 0 ? std::optional<int>(cfg_.eos_token_id) : std::nullopt;
    if (name == "pad")
        return cfg_.pad_token_id >= 0 ? std::optional<int>(cfg_.pad_token_id) : std::nullopt;
    if (name == "unk")
        return cfg_.unk_token_id >= 0 ? std::optional<int>(cfg_.unk_token_id) : std::nullopt;
    return std::nullopt;
}

std::vector<std::pair<std::string, int>> SpecialTokenHandler::merged_added_tokens(
    const std::vector<std::pair<std::string, int>>& vocab_added) const
{
    std::vector<std::pair<std::string, int>> merged = vocab_added;
    // D-3：meta 显式配置权威——同字面量冲突 → meta 覆盖 + WARN（不静默，
    // meta 可覆盖词表产物的 id 指配）
    for (const auto& [lit, tid] : cfg_.special_tokens) {
        if (tid < 0)
            continue; // configure 已 WARN，此处静默跳过
        // C++17：结构化绑定不可被 lambda 捕获（clang 拒，g++ 扩展收——CI clang
        // 实证）；具名别名后按引用捕获
        const std::string& literal = lit;
        auto it = std::find_if(merged.begin(), merged.end(), [&literal](const auto& e) { return e.first == literal; });
        if (it != merged.end()) {
            if (it->second != tid) {
                CANN_LLM_LOGW(
                    "tokenizer.special.meta: literal conflict, meta wins (literal=%s, vocab_id=%d, meta_id=%d)",
                    lit.c_str(), it->second, tid);
                it->second = tid;
            }
        } else {
            merged.emplace_back(lit, tid);
        }
    }
    return merged;
}

std::optional<std::string_view> SpecialTokenHandler::literal(int id) const
{
    // meta 解映射：词表外 id 的出字面量通道（decode 查序：meta 先于 vocab）
    auto it = id2lit_.find(id);
    if (it == id2lit_.end())
        return std::nullopt;
    return std::string_view(it->second);
}

} // namespace cann_local_lm::engine
