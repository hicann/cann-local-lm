#pragma once

#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/tensor_manager.h"   // ITensorBuffer + TensorDesc + using 声明
#include "infra/status.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cann_lm::engine {

/// 输入组装器（DESIGN M-03 / v0.13 + gemma4 扩展 docs/mask_fill.md）。
/// 按 name→index 填 position_ids/new_kv_cache_pos；mask 工位双模式（按 name_index
/// 动态分派，图说了算）：
///  - 旧图单 attention_mask（float：0.0=attend / -FLT_MAX=屏蔽，kv 维=本步实际长度）；
///  - gemma4 图 mask_full+mask_sliding（BOOL：0=attend / 1=屏蔽，kv 维=静态
///    kv_cache_max_len，整 buffer 每步全量重写——先全置 1 再清 attend 区间）。
///
/// 可选工位 output_pos（gemma4 图内 logits.index_select 裁剪行号，prefill 填
/// chunk_len-1 / decode 填 0，dtype 按 om 工位读出 INT32/INT64）：图带才填，
/// 缺工位跳过——旧图/output_pos 已常量折叠进图的导出零行为变化。
class InputAssembler {
public:
    /// pool：KV 池（查光标/推进）；max_batch_size：decode padding 目标（prefill 实例可传 1）；
    /// sliding_window：gemma4 滑窗宽（meta_json model_config.sliding_window；0=无滑窗，
    /// mask_sliding 与 mask_full 同值；非 gemma4 图不受影响——按工位名分派）。
    InputAssembler(std::shared_ptr<KvCachePool> pool, int max_batch_size,
                   int sliding_window = 0);

    /// 组装 prefill 输入：按 name_index 查找 position_ids/new_kv_cache_pos 与 mask 工位
    /// （attention_mask 或 mask_full+mask_sliding 二选一），
    /// 从 KV 池获取光标位置填入 tensor，推进 slot 光标 chunk_len。
    /// 错误：name_index 缺 position_ids/new_kv_cache_pos 或 mask 工位组合非法
    ///       （都缺/只有其一/并存）→ INVALID_ARGUMENT；
    ///       slot<0 / chunk_len==0 → INVALID_ARGUMENT；光标越界 → FAILURE。
    Status AssemblePrefill(SlotId slot, std::size_t chunk_len,
                           std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                           const std::unordered_map<std::string, std::size_t>& name_index);

    /// 组装 decode 输入：各 active slot 从 KV 池获取光标，填 position_ids/new_kv_cache_pos
    /// 与 mask 工位（同上双模式），推进各 slot 光标 1。
    /// 错误：name_index 缺 position_ids/new_kv_cache_pos 或 mask 工位组合非法 → INVALID_ARGUMENT；
    ///       slots 空 → INVALID_ARGUMENT；光标越界 → FAILURE。
    Status AssembleDecode(const std::vector<SlotId>& slots,
                          std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                          const std::unordered_map<std::string, std::size_t>& name_index);

    int max_batch_size() const noexcept { return max_batch_size_; }

private:
    std::shared_ptr<KvCachePool> pool_;
    int                           max_batch_size_;
    int                           sliding_window_ = 0;   ///< gemma4 滑窗宽（0=无）
};

}  // namespace cann_lm::engine
