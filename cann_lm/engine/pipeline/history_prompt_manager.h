#pragma once

#include "infra/status.h"
#include "api/cpp/types.h"

#include <cstddef>
#include <vector>

namespace cann_lm::engine {

/// 按 Context 隔离（需求 #9）。存储 token id 序列（非文本，避免反复 tokenize 与非双射损失）。
class HistoryPromptManager {
public:
    HistoryPromptManager() = default;
    /// 追加本轮生成的 token（每步 Decode 后由 pipeline 写）
    Status Append(const std::vector<TokenId>& ids);
    /// 读取历史 token（下一轮 prefill 拼接用）
    Status Read(std::vector<TokenId>* out) const;
    /// 清空（DestroyContext 时）
    Status Clear();
    size_t Size() const noexcept;

private:
    std::vector<TokenId> history_;   // C3：大对象，按值持；跨轮增长
};

}  // namespace cann_lm::engine
