// HistoryPromptManager —— PromptCache 缓存台账（design §3，2026-10-15 定稿）。
// 语义改造（R1，remove_multi_turn.md 落地后）：
//   旧语义（已删）：引擎多轮记忆——Read 出历史拼进 prompt、Append 写回。
//   新语义：**缓存台账**——记录"已写入该 Context KV 的 token 逻辑序"，
//   仅用于前缀比对与复用记账，不注入 prompt；客户端每轮发全量对话文本。
// 组件名沿用 HistoryPromptManager（用户裁定：不引入 ledger 命名）。
#pragma once

#include "infra/status.h"
#include "api/cpp/types.h"

#include <cstddef>
#include <mutex>
#include <vector>

namespace cann_lm::engine {

/// 缓存台账（GLOSSARY.md「缓存台账」词条）。
/// 不变量（spec REQ-2）：任意时刻 tokens_.size() == 所属 Context KV 池的
/// total_written（pipeline 侧接线保证；本类不含池引用）。
/// 线程模型：GenerateAsync 调用线程比对 vs worker 追加，由外层
/// active_generates_ 同 Context 串行兜底；本类内部自带粗粒度互斥（防御）。
class HistoryPromptManager {
public:
    HistoryPromptManager() = default;

    /// REQ-11 观测出参（比对细节；供埋点日志/UT 断言，不影响判定）。
    struct MatchStats {
        std::size_t history_len = 0; ///< 比对前台账长
        std::size_t lcp = 0;         ///< 最长公共前缀（miss = 0）
        bool wrapped = false;        ///< 绕回已发生（台账长 > kv_max）
    };

    /// 命中判定（每轮 Generate、encode 之后、切块之前，单点单次）：
    /// 输入全量 ids 与台账从 0 逐位比对取 LCP，按复用规则（design §3.2 /
    /// spec REQ-4）返回可复用逻辑长度 [0, ret)。内部完成台账截断/清空：
    ///   - ret > 0：台账已截到 ret（pool 记账回退到 ret 由 pipeline 侧处理）；
    ///   - ret == 0：miss，台账已清空（pipeline 侧再清池记账）。
    /// 绕回判定参数 kv_max / wrapped 由调用方传入（台账不含池引用）：
    ///   wrapped = 台账长 > kv_max（绕回已发生）⇒ 仅"输入完全覆盖台账"
    ///   （LCP == 台账长）才复用（R6），否则 miss 清账。
    /// R7：全命中（LCP == 台账长）复用长 = 台账长 − 1（末 token 强制重算，
    /// logits 只存在于 prefill 输出中）。台账长 1 时复用长 0 = 等价全量重算。
    /// stats 非空时全程填写（含 early return：LCP=0/wrapped 按比对前台账长判）。
    std::size_t MatchCachedTokens(const std::vector<TokenId>& input, std::size_t kv_max, MatchStats* stats = nullptr);

    /// decode 每步追加采样输出 token（进 KV 后立即记，REQ-2）。
    void AppendDecodedToken(TokenId id);

    /// 台账长度（逻辑写入长度；供不变量断言与光标回退快照）。
    std::size_t Size() const noexcept;

    /// 台账全量副本（R10 快照拷贝；GenerateAsync 提交期一次，请求终态
    /// 回退 Restore）。持锁拷贝，成本 = 一次 vector copy。
    std::vector<TokenId> SnapshotTokens() const;

    /// R10 失败回退：台账重灌到 tokens（一次到位；快照由 GenerateAsync 提交
    /// 期拷贝）。仅恢复用——miss/失败后 Clear+重灌等价，无需独立前进语义。
    void Restore(std::vector<TokenId> tokens);

    /// 清空台账（miss / 请求失败回退 / DestroyContext）。
    void Clear() noexcept;

    // ── initTokenLen 锚定段记账（R5/R13；宿主 pipeline 维护，池侧同步）──
    /// 锚定段长记录：首个全量 prefill 完成时由 pipeline 设置
    /// （total_written ≤ kv_max 时 = total_written；超长 prompt 场景 = 0）。
    void SetInitTokenLen(std::size_t len) noexcept;
    std::size_t InitTokenLen() const noexcept;

    /// R3 门控：含 linear_attention 层的模型整个 feature 降级为不存在。
    /// enabled_ 由 pipeline 构造期一次写入（NumLinearLayers > 0 ⇒ false），
    /// 此后只读、不再每请求重复判定。
    void SetEnabled(bool enabled) noexcept;
    bool Enabled() const noexcept;

private:
    /// 持锁清理（mtx_ 已持有时用）
    void ClearLocked_() noexcept;

    std::vector<TokenId> tokens_;    ///< 逻辑序全量台账（已写入 KV 的 token）
    std::size_t init_token_len_ = 0; ///< 锚定段长记账（与池同值同步）
    bool enabled_ = true;            ///< R3 门控（构造期定，只读）
    mutable std::mutex mtx_;         ///< 粗粒度防御互斥（比对 vs 追加）
};

} // namespace cann_lm::engine
