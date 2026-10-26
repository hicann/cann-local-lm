#pragma once

#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/tensor_manager.h" // ITensorBuffer + TensorDesc + using 声明
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
    InputAssembler(std::shared_ptr<KvCachePool> pool, int max_batch_size, int sliding_window = 0);

    /// 组装 prefill 输入：按 name_index 查找 position_ids/new_kv_cache_pos 与 mask 工位
    /// （attention_mask 或 mask_full+mask_sliding 二选一），
    /// 从 KV 池获取光标位置填入 tensor，推进 slot 光标 chunk_len。
    /// gemma4 bool 图（P0 修复，Python 标杆 run_gemma4_gen.py build_prefill）：
    /// position_ids/new_kv_cache_pos 按图行数【全行】确定性填充——真实行
    /// cursor+i，padding 行续 arange 落未来未写区（kv_max 非 rows 整倍数时截到
    /// kv_max-1）；dtype 限 INT32、容量显式校验。om 图逐行按 new_kv_cache_pos[row]
    /// scatter 写 KV（input tensors 跨 Forward 复用），残值尾行 = 错误写入位。
    /// 错误：name_index 缺 position_ids/new_kv_cache_pos 或 mask 工位组合非法
    ///       （都缺/只有其一/并存）、slot<0 / chunk_len==0、rows<chunk_len、
    ///       pos 工位 dtype≠INT32 或容量不足 → INVALID_ARGUMENT；
    ///       光标越界（真实块超 kv 容量）/ 恰满短块（padding 行无安全落位，
    ///       正常部署配置永不触发）→ FAILURE。
    Status AssemblePrefill(SlotId slot, std::size_t chunk_len, std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                           const std::unordered_map<std::string, std::size_t>& name_index);

    /// 组装 decode 输入：各 active slot 从 KV 池获取光标，填 position_ids/new_kv_cache_pos
    /// 与 mask 工位（同上双模式），推进各 slot 光标 1。
    /// gemma4 bool 图（P1 修复）：pos 族按图 batch 维【全 lane】确定性填充——
    /// active lane = 本槽光标，inactive lane 复制 lane 0（pos 写入位须合法，
    /// 防残值被 scatter 消费）；mask 侧 inactive lane 保持全 1（golden 对齐，
    /// padding 行不复制有效行）。
    /// 错误：name_index 缺 position_ids/new_kv_cache_pos 或 mask 工位组合非法、
    ///       rows<batch、pos 工位 dtype/容量不符 → INVALID_ARGUMENT；
    ///       slots 空 → INVALID_ARGUMENT；任一 active 槽光标 ≥ kv_max（无位
    ///       可写，原 Advance 后置失败等价终态）→ FAILURE。
    Status AssembleDecode(const std::vector<SlotId>& slots, std::vector<std::shared_ptr<ITensorBuffer>>& tensors,
                          const std::unordered_map<std::string, std::size_t>& name_index);

    int max_batch_size() const noexcept { return max_batch_size_; }

    /// 工位池换绑（PromptCache R9 配套）：RebindKvPool 切 Context 池时同步
    /// 更新本组件持有的池引用——记账推进（Advance）、物理槽映射（MapSlot）、
    /// 光标读取（TotalWritten/NextPosition）全部以当前活跃 Context 池为权威。
    /// 幂等（同池再绑 = no-op）；生命周期由调用方（pipeline worker 串行）保证。
    void RebindPool(std::shared_ptr<KvCachePool> pool) noexcept { pool_ = std::move(pool); }

private:
    std::shared_ptr<KvCachePool> pool_;
    int max_batch_size_;
    int sliding_window_ = 0; ///< gemma4 滑窗宽（0=无）
    /// P2 告警去重：图带 mask_sliding 工位但 sliding_window=0（meta 缺失）→ 滑窗层
    /// 静默退化全因果；实例生命周期内只 WARN 一次（跨 prefill/decode 共享实例去重）。
    bool warned_window_ = false;
};

} // namespace cann_lm::engine
