#include "pipeline/preprocessor.h"
#include "infra/log.h"
#include "utils/common_utils.h"
#include "utils/debug_tensor_dump.h" // tokenizer ids dump（golden 对齐，env 开关默认关）

#include <algorithm>
#include <cstddef>

namespace cann_lm::engine {

Preprocessor::Preprocessor(TokenizerBase* tokenizer, TextEmbedding* text_embedding, int default_chunk_size)
    : tokenizer_(tokenizer),
      text_embedding_(text_embedding),
      default_chunk_size_(default_chunk_size > 0 ? default_chunk_size : 1024)
{}

Status Preprocessor::PreparePrefill(const Prompt& prompt, std::vector<std::vector<TokenId>>* out_chunks,
                                    std::size_t reusable, std::vector<TokenId>* out_ids)
{
    if (!out_chunks)
        return Status::INVALID_ARGUMENT;
    out_chunks->clear();
    if (out_ids)
        out_ids->clear();

    if (!tokenizer_) {
        CANN_LLM_LOGW("preprocess.prefill: tokenizer unavailable");
        return Status::INVALID_STATE;
    }

    // 1. tokenize prompt text
    EncodeOptions enc_opts;
    enc_opts.add_special_tokens = true;
    std::vector<int> ids;
    Status st = tokenizer_->encode(prompt.text, enc_opts, &ids);
    if (!st.ok()) {
        CANN_LLM_LOGE("preprocess.prefill: encode failed (st=%d)", st.code);
        return st;
    }

    // 调试：打印 tokenize 后的 TokenId 结果
    CANN_LLM_LOGD("preprocess.prefill: prompt tokenized, count=%zu, ids=[%s]", ids.size(), CvtVectorToStr(ids).c_str());

    // tokenizer encode 结果落盘（与工位 dump 同一总闸 CANN_LM_DEBUG_DUMP_TENSORS，
    // env 未设零行为变化）。产出 prefill_tokenizer_ids.bin（int32 裸数组），与
    // Python 参考 tokenizer.encode 逐值比对；同轮的 prefill_in_* dump 的
    // embeddings 前 count 行即该 id 序列查表结果（含 pad_token 补齐行=0）。
    // 增量轮（PromptCache reusable>0）同 dump 同文件名（覆写为当轮全量 ids）:
    // 对精度比对取"最近一次 encode"即够用。
    DebugDumpTokenizerIds("prefill", ids, "ids");

    // int（tokenizer 面）→ TokenId（engine 面）：历史悠久双型（token id 语义
    // 值域 [0, vocab)，非负恒成立，窄转无损）
    std::vector<TokenId> token_ids;
    token_ids.reserve(ids.size());
    for (int id : ids)
        token_ids.push_back(static_cast<TokenId>(id));

    if (out_ids) {
        *out_ids = token_ids;
    }

    // 防御：reusable > 输入长 = 调用时序 bug（Match 出参恒 ≤ 台账长 ≤ 输入长）
    if (reusable > token_ids.size()) {
        CANN_LLM_LOGE("preprocess.prefill: reusable=%zu > ids=%zu (caller bug)", reusable, token_ids.size());
        return Status::INVALID_ARGUMENT;
    }

    return SliceChunks(token_ids, reusable, out_chunks);
}

Status Preprocessor::SliceChunks(const std::vector<TokenId>& ids, std::size_t reusable,
                                 std::vector<std::vector<TokenId>>* out_chunks)
{
    if (!out_chunks)
        return Status::INVALID_ARGUMENT;
    out_chunks->clear();

    // 防御：reusable > 输入长 = 调用时序 bug（Match 出参恒 ≤ 台账长 ≤ 输入长）
    if (reusable > ids.size()) {
        CANN_LLM_LOGE("preprocess.slice: reusable=%zu > ids=%zu (caller bug)", reusable, ids.size());
        return Status::INVALID_ARGUMENT;
    }

    // 增量切块（R8/REQ-6）：仅作用 [reusable, ids.size())。
    // 契约（DTS202610220945 Fix A）：ids 非空时 Match 出参恒 reusable <
    // ids.size()（末 token 强制重算一般化，全命中/前缀重合皆减 1），不产空块链；
    // reusable == ids.size() 形态在此仍按空链返回（防御语义不改，不承担钳制）。
    // 空块链两处可达：ids 为空（zero-token prompt 历史行为）与上游回归——
    // 下游由 Scheduler::Step 空块守卫兜底终态化，不依赖 Submit 拒空。
    if (ids.empty()) {
        return Status::SUCCESS;
    }

    int chunk_size = default_chunk_size_;
    if (chunk_size <= 0)
        chunk_size = static_cast<int>(ids.size());

    for (size_t offset = reusable; offset < ids.size(); offset += static_cast<size_t>(chunk_size)) {
        size_t end = std::min(offset + static_cast<size_t>(chunk_size), ids.size());
        out_chunks->emplace_back(ids.begin() + static_cast<std::ptrdiff_t>(offset),
                                 ids.begin() + static_cast<std::ptrdiff_t>(end));
    }

    return Status::SUCCESS;
}

Status Preprocessor::PrepareDecodeStep(TokenId id, std::vector<TokenId>* out_input)
{
    if (!out_input)
        return Status::INVALID_ARGUMENT;
    *out_input = {id};
    return Status::SUCCESS;
}

} // namespace cann_lm::engine
