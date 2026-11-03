#pragma once

#include "core/component/text_embedding/text_embedding.h"
#include "core/component/tokenizer/tokenizer.h"
#include "infra/status.h"
#include "api/cpp/types.h"

#include <memory>
#include <vector>

namespace cann_local_lm::engine {

/// 单一职责协调预处理链路（需求 #7）。无状态（每次调用纯函数语义，单轮 prompt）。
/// 经非拥有指针引用 TokenizerBase / TextEmbedding：TokenizerBase 由 M-02 经工厂 unique_ptr
/// 持有（多阶段复用：预处理 tokenize + 后处理 detokenize 复用同一实例，01 §4 注）；
/// TextEmbedding 由 M-02 init 期 Load（SR20260828001，经 InitEmbedding_ 的 U-8 路线 b 通道）。
/// tokenId→向量查表在 om 图前向内部完成（非 Preprocessor 调 TextEmbedding 查表，
/// C8 v1.2 per model_executor §4.1.1）。
class Preprocessor {
public:
    Preprocessor(TokenizerBase* tokenizer, TextEmbedding* text_embedding, int default_chunk_size);

    /// Prefill 路径：Prompt → 多个 chunk 的 token 序列
    ///   流程：tokenize Prompt.text → 增量切块（PromptCache R8/REQ-6）。
    ///   reusable（缓存复用长，MatchCachedTokens 出参；0 = 全量零行为变化）：
    ///   切块仅作用 [reusable, len)——首块 [reusable, min(reusable+chunk_size, len))，
    ///   尾块自然收短；无过算、无对齐、无特判（参考实现块拷贝协议不采纳，
    ///   per-token 序列天然免疫）。
    ///   out_ids（可空）：encode 全量产物回传——pipeline 在比对（Match）与
    ///   本调用间单次 encode 复用（比对需要 ids、追加台账也需要）。非空时
    ///   必须与本 prompt 同帧调用。
    ///   （多轮上下文组织归客户端：Prompt.text 为单轮全量文本，
    ///   引擎不拼历史，remove_multi_turn.md）。
    ///   tokenId→向量查表在 om 图内部。
    ///   载体为普通堆内存：token id 仅 host 侧 FillInputs 消费、不进 NPU，
    ///   不占 ION（方案与影响范围见 preprocessor_mem_use.md）。
    ///   错误：encode 失败上抛；reusable > ids.size() → INVALID_ARGUMENT
    ///   （上游 Match 出参恒 ≤ 台账长 ≤ 输入长，违 = 调用时序 bug）。
    Status PreparePrefill(const Prompt& prompt, std::vector<std::vector<TokenId>>* out_chunks, std::size_t reusable = 0,
                          std::vector<TokenId>* out_ids = nullptr);

    /// Decode fast path：单 token id → 输入序列（Prefill 链路子集，需求 #7）。
    ///   载体同上（普通堆内存）。
    Status PrepareDecodeStep(TokenId id, std::vector<TokenId>* out_input);

    /// 增量切块纯切片通道（design §4.1）：ids 全量产物 + 复用长 → 块链。
    /// 与 PreparePrefill 的切块语义逐值一致（chunk_size 三级回退链同
    /// default_chunk_size_）；独立入口供 pipeline 在 Match（比对需要 ids）
    /// 与切片之间插入台账逻辑，不二次 encode。reusable > ids.size() →
    /// INVALID_ARGUMENT（防御）。
    Status SliceChunks(const std::vector<TokenId>& ids, std::size_t reusable,
                       std::vector<std::vector<TokenId>>* out_chunks);

    /// tokenizer 接线（perf_optimize.md §3.5 L5）：构造期 tokenizer 异步创建
    /// （∥ 执行器 Load），join 后由 pipeline 回填。nullptr = 未就绪（
    /// PreparePrefill 持空防御返回 INVALID_STATE，语义同构造期坏 tokenizer）。
    void SetTokenizer(TokenizerBase* tokenizer) { tokenizer_ = tokenizer; }

private:
    TokenizerBase* tokenizer_;      // 非拥有
    TextEmbedding* text_embedding_; // 非拥有（load+bind 由 M-02 init）
    int default_chunk_size_;
};

} // namespace cann_local_lm::engine
