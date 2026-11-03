// HistoryPromptManager 实现 —— PromptCache 缓存台账（design §3.2 规则表）。
#include "pipeline/history_prompt_manager.h"

#include <algorithm>
#include <utility>

namespace cann_local_lm::engine {

std::size_t HistoryPromptManager::MatchCachedTokens(const std::vector<TokenId>& input, std::size_t keep_cap,
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

    // 2. 绕回判定（R6 + sliding_kv_seq_dim_rebind D6）：wrapped = 台账长 >
    //    keep_cap。keep_cap = 各层环容量最小值（复用保真容量，调用方传入；
    //    新契 = min(kv_max, C=W+P)，旧契/无滑窗层 = kv_max）。先绕回的
    //    窄环层只保留最近环容量个 token 的新主，台账超门后部分命中复用的
    //    该层窗尾已落被覆写圈——门按 kv_max 判会放走"静默读旧代"复用，
    //    故取各层环最小值。
    const bool wrapped = ledger_len > keep_cap;
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

    // 3. 末 token 强制重算（R7 一般化，DTS202610220945 Fix A）：末 token 的
    //    logits 只存在于 prefill 输出中，不重推无采样输入。触发判据一般化为
    //    "本轮增量段为空"，含两种形态，皆复用长 = 边界 − 1（保证至少 1 个
    //    真实 prefill 块，封闭全部空块链来源）：
    //      a. 全命中：LCP == 台账长（含绕回全覆盖）——本规则原有形态；
    //      b. 前缀重合：LCP == 输入长 < 台账长——输入恰为台账真前缀（两轮
    //         重发完全一致输入即此列，原规则落空致下游空块链越界）。
    //    台账长 1 全命中 ⇒ 复用长 0 = 等价全量重算（REQ-5 特例）；输入长 1
    //    前缀重合同退化为 0（台账 resize(0)，池侧由 GenerateAsync miss 收口
    //    分支 ResetLedgerOnSlot 清账，语义与 miss 一致）。
    const bool increments_empty = (lcp == ledger_len) || (lcp == input.size());
    const std::size_t reusable = increments_empty ? (lcp > 0 ? lcp - 1 : 0) : lcp;

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

void HistoryPromptManager::SetKvSlotId(int slot) noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    kv_slot_id_ = slot;
}

int HistoryPromptManager::KvSlotId() const noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    return kv_slot_id_;
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

} // namespace cann_local_lm::engine
