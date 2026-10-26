// HistoryPromptManager 实现 —— PromptCache 缓存台账（design §3.2 规则表）。
#include "pipeline/history_prompt_manager.h"

#include <algorithm>
#include <utility>

namespace cann_lm::engine {

std::size_t HistoryPromptManager::MatchCachedTokens(const std::vector<TokenId>& input, std::size_t kv_max,
                                                    MatchStats* stats)
{
    std::lock_guard<std::mutex> lk(mtx_);
    auto fill_stats = [&](std::size_t lcp, bool wrapped) {
        if (stats) {
            stats->history_len = tokens_.size();
            stats->lcp = lcp;
            stats->wrapped = wrapped;
        }
    };
    // R3 门控：禁用模型特征不存在——不走比对、台账不动、不产生 miss 副作用
    // （pipeline 侧台账与池同属每 Context 新建空态，行为等同首轮全量重算）。
    if (!enabled_) {
        fill_stats(0, false);
        return 0;
    }
    // 台账空（首轮）：miss 但无账可动（Clear 与 ClearLocked_ 幂等，省一次写）。
    if (tokens_.empty()) {
        fill_stats(0, false);
        return 0;
    }

    // 1. LCP = 最大 k ≤ min(|tokens_|, |input|)：tokens_[i]==input[i] ∀i<k。
    //    逐位比对（单 batch 端侧量级，spec 非目标：无哈希/分块加速）。
    const std::size_t common = std::min(tokens_.size(), input.size());
    std::size_t lcp = 0;
    while (lcp < common && tokens_[lcp] == input[lcp])
        ++lcp;
    const std::size_t ledger_len = tokens_.size();

    // 2. 绕回判定（R6）：wrapped = 台账长 > kv_max（绕回已发生）。
    const bool wrapped = ledger_len > kv_max;
    fill_stats(lcp, wrapped);
    if (wrapped && lcp < ledger_len) {
        // 绕回 + 输入未完全覆盖台账（窗口内发散同样判 miss）⇒ 清账重建。
        ClearLocked_();
        return 0;
    }
    if (!wrapped && lcp == 0) {
        // 无绕回且首 token 失配短路：全量重算。
        ClearLocked_();
        return 0;
    }

    // 3. R7：全命中（LCP == 台账长，含绕回全覆盖）强制重算最后 1 个 token——
    //    末 token 的 logits 只存在于 prefill 输出中，不重推无采样输入；
    //    输入比台账长（普通命中）则直接复用 [0, LCP)。
    //    台账长 1 全命中 ⇒ 复用长 0 = 等价全量重算（REQ-5 特例）。
    const std::size_t reusable = (lcp == ledger_len) ? (lcp > 0 ? lcp - 1 : 0) : lcp;

    // 4. 台账截断到复用边界（REQ-2.3：每轮比对后 resize）。
    tokens_.resize(reusable);
    return reusable;
}

void HistoryPromptManager::AppendDecodedToken(TokenId id)
{
    std::lock_guard<std::mutex> lk(mtx_);
    tokens_.push_back(id);
}

std::size_t HistoryPromptManager::Size() const noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    return tokens_.size();
}

std::vector<TokenId> HistoryPromptManager::SnapshotTokens() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return tokens_;
}

void HistoryPromptManager::Clear() noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    ClearLocked_();
}

void HistoryPromptManager::Restore(std::vector<TokenId> tokens)
{
    std::lock_guard<std::mutex> lk(mtx_);
    tokens_ = std::move(tokens);
}

void HistoryPromptManager::SetInitTokenLen(std::size_t len) noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    init_token_len_ = len;
}

std::size_t HistoryPromptManager::InitTokenLen() const noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    return init_token_len_;
}

void HistoryPromptManager::SetEnabled(bool enabled) noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    enabled_ = enabled;
}

bool HistoryPromptManager::Enabled() const noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    return enabled_;
}

void HistoryPromptManager::ClearLocked_() noexcept { tokens_.clear(); }

} // namespace cann_lm::engine
