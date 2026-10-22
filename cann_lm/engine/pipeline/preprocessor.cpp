#include "pipeline/preprocessor.h"
#include "infra/log.h"
#include "utils/common_utils.h"

#include <algorithm>
#include <cstddef>

namespace cann_lm::engine {

Preprocessor::Preprocessor(TokenizerBase*    tokenizer,
                           TextEmbedding*  text_embedding,
                           int             default_chunk_size)
    : tokenizer_(tokenizer),
      text_embedding_(text_embedding),
      default_chunk_size_(default_chunk_size > 0 ? default_chunk_size : 1024) {}

Status Preprocessor::PreparePrefill(const Prompt& prompt,
                                    const std::vector<TokenId>& history,
                                    std::vector<std::vector<TokenId>>* out_chunks) {
    if (!out_chunks) return Status::INVALID_ARGUMENT;
    out_chunks->clear();

    if (!tokenizer_) {
        CANN_LLM_LOGW( "preprocess.prefill: tokenizer unavailable");
        return Status::INVALID_STATE;
    }

    // 1. tokenize prompt text
    EncodeOptions enc_opts;
    enc_opts.add_special_tokens = true;
    std::vector<int> ids;
    Status st = tokenizer_->encode(prompt.text, enc_opts, &ids);
    if (!st.ok()) {
        CANN_LLM_LOGE( "preprocess.prefill: encode failed (st=%d)", st.code);
        return st;
    }

    // 调试：打印 tokenize 后的 TokenId 结果
    CANN_LLM_LOGD("preprocess.prefill: prompt tokenized, count=%zu, ids=[%s]",
                  ids.size(), CvtVectorToStr(ids).c_str());

    // 2. 拼接历史（历史在前，新 prompt 在后）
    std::vector<TokenId> all_ids = history;
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());

    if (all_ids.empty()) {
        out_chunks->clear();
        return Status::SUCCESS;
    }

    // 3. 切块（token 级，基于 chunk_size）
    int chunk_size = default_chunk_size_;
    if (chunk_size <= 0) chunk_size = static_cast<int>(all_ids.size());

    for (size_t offset = 0; offset < all_ids.size(); offset += static_cast<size_t>(chunk_size)) {
        size_t end = std::min(offset + static_cast<size_t>(chunk_size), all_ids.size());
        out_chunks->emplace_back(all_ids.begin() + static_cast<std::ptrdiff_t>(offset),
                                 all_ids.begin() + static_cast<std::ptrdiff_t>(end));
    }

    return Status::SUCCESS;
}

Status Preprocessor::PrepareDecodeStep(TokenId id, std::vector<TokenId>* out_input) {
    if (!out_input) return Status::INVALID_ARGUMENT;
    *out_input = {id};
    return Status::SUCCESS;
}

}  // namespace cann_lm::engine
