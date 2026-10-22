#include "pipeline/history_prompt_manager.h"

namespace cann_lm::engine {

Status HistoryPromptManager::Append(const std::vector<TokenId>& ids) {
    history_.insert(history_.end(), ids.begin(), ids.end());
    return Status::SUCCESS;
}

Status HistoryPromptManager::Read(std::vector<TokenId>* out) const {
    if (!out) return Status::INVALID_ARGUMENT;
    *out = history_;
    return Status::SUCCESS;
}

Status HistoryPromptManager::Clear() {
    history_.clear();
    return Status::SUCCESS;
}

size_t HistoryPromptManager::Size() const noexcept { return history_.size(); }

}  // namespace cann_lm::engine
