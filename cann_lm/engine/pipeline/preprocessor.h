#pragma once

#include "core/component/text_embedding/text_embedding.h"
#include "core/component/tokenizer/tokenizer.h"
#include "infra/status.h"
#include "api/cpp/types.h"

#include <memory>
#include <vector>

namespace cann_lm::engine {

/// 单一职责协调预处理链路（需求 #7）。无状态（每次调用纯函数语义，历史由调用方传入）。
/// 经非拥有指针引用 TokenizerBase / TextEmbedding：TokenizerBase 由 M-02 经工厂 unique_ptr
/// 持有（多阶段复用：预处理 tokenize + 后处理 detokenize 复用同一实例，01 §4 注）；
/// TextEmbedding 由 M-02 init 期 Load（SR20260828001，经 InitEmbedding_ 的 U-8 路线 b 通道）。
/// tokenId→向量查表在 om 图前向内部完成（非 Preprocessor 调 TextEmbedding 查表，
/// C8 v1.2 per model_executor §4.1.1）。
class Preprocessor {
public:
    Preprocessor(TokenizerBase*     tokenizer,
                 TextEmbedding*  text_embedding,
                 int             default_chunk_size);

    /// Prefill 路径：Prompt + 历史 token → 多个 chunk 的 token 序列
    ///   流程：tokenize Prompt.text → 拼接历史 → 占位符定位（多模态 embedding 按顺序，
    ///        来自 Prompt）→ 切块（基于 chunk_size）。tokenId→向量查表在 om 图内部。
    ///   载体为普通堆内存：token id 仅 host 侧 FillInputs 消费、不进 NPU，
    ///   不占 ION（方案与影响范围见 preprocessor_mem_use.md）。
    Status PreparePrefill(const Prompt& prompt,
                          const std::vector<TokenId>& history,
                          std::vector<std::vector<TokenId>>* out_chunks);

    /// Decode fast path：单 token id → 输入序列（Prefill 链路子集，需求 #7）。
    ///   载体同上（普通堆内存）。
    Status PrepareDecodeStep(TokenId id, std::vector<TokenId>* out_input);

private:
    TokenizerBase*      tokenizer_;       // 非拥有
    TextEmbedding*   text_embedding_;  // 非拥有（load+bind 由 M-02 init）
    int              default_chunk_size_;
};

}  // namespace cann_lm::engine
