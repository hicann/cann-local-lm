// ═══════════════════════════════════════════════════════════════════════════
// LinearStatePool（Qwen3.5，support_qwen3.5.md §4.4）。engine 内部组件，
// 仅对 pipelineExecutor / ModelExecutor 暴露。
// linear_attention 层的 conv_state / rstate 状态池：
//  - 与 KV cache 同族复用：双执行器共享（pipeline 持 shared_ptr、双 Attach）、
//    输入/输出工位绑同一 tensor（图推理期原地更新，无回绕、无需光标）；
//  - 形状/dtype 以 om 图工位 tensorDesc 为权威（meta_json 无状态形状字段，
//    §5 D1）：构造期只有层清单无 tensor；首个执行器 Load 时 EnsureStateTensor
//    惰性创建（创建即整体清零），另一图 Load 校验 dims/dtype 一致后复用，
//    失配 INVALID_ARGUMENT（响亮失败）；
//  - 槽位体系沿用 KvCachePool 的 SlotId（batch lane）；lane 复用前经 ResetLane
//    清零该 lane 全层状态切片（引擎侧负责，§5 D4；batch 维约定 = dims[0]，§8 Q3）。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "core/component/kv_cache_pool/kv_cache_pool.h" // SlotId（槽位体系沿用）
#include "core/tensor_manager.h"                        // ITensorBuffer/TensorDesc/DataType/TensorManager
#include "infra/status.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cann_local_lm::engine {

/// LinearAttention 状态池（Qwen3.5）。conv_state / rstate per linear 层各一份。
/// 不含槽位/光标记账（无回绕无需光标；lane 生命周期由 KvCachePool 槽位体系管理，
/// 本池只在新序列复用 lane 前清零状态残留）。
class LinearStatePool {
public:
    /// 状态种类：conv_state（卷积前状态）/ rstate（递归状态）。
    enum class StateKind { Conv, R };

    /// layer_types 同 KvCachePool 口径（meta_json model_config.layer_types；
    /// 空 = 无 linear 层，池空转——零行为变化）。构造期仅派生层清单，不建 tensor。
    explicit LinearStatePool(const std::vector<std::string>& layer_types);

    /// 层 layer 的 conv_state tensor（未 Ensure / 非 linear 层 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> ConvTensor(int layer) const;
    /// 层 layer 的 rstate tensor（未 Ensure / 非 linear 层 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> RTensor(int layer) const;

    /// Load 期消费：该层该类状态 tensor 就绪——无则按 desc 创建并整体清零；
    /// 有则校验 dims/dtype 一致（prefill/decode 双图状态形状一致性在此把关），
    /// 失配 INVALID_ARGUMENT。layer 非 linear_attention → INVALID_ARGUMENT
    /// （图带状态工位而配置未声明，显式报错不静默兼容）。desc 取自 om 图输入工位。
    Status EnsureStateTensor(int layer, StateKind kind, const TensorDesc& desc);

    /// 清零 slot 对应 batch lane 的全层状态切片（新序列复用 lane 前调用；
    /// Scheduler prefill 分配 KV 槽后接线）。batch 维 = dims[0]（§8 Q3）；
    /// 池空 / tensor 未就绪 = no-op；slot 越界（≥ dims[0]）→ WARN 跳过该 tensor。
    Status ResetLane(SlotId slot);

    /// linear_attention 层数（layer_types 空 = 0）。
    int NumLinearLayers() const noexcept;
    /// linear 层全局层号列表（升序；构造后不可变）。
    const std::vector<int>& LinearLayerIds() const noexcept;
    /// 已创建状态 tensor 数（≤ 2 × NumLinearLayers；UT/dfx 观测）。
    std::size_t TensorCount() const noexcept;

private:
    /// 全局层号 → tensors_ 组下标；非 linear 层 → -1（linear_layer_ids_ 升序）。
    int GroupIndex(int layer) const noexcept;

    std::vector<int> linear_layer_ids_; ///< linear 层全局层号（升序，构造后不可变）
    /// 每已就绪 linear 层一组 [conv, r]（与 linear_layer_ids_ 同序）；
    /// 组内元素未 Ensure 时为空。
    std::vector<std::pair<std::shared_ptr<ITensorBuffer>, std::shared_ptr<ITensorBuffer>>> tensors_;
    mutable std::mutex mtx_; ///< 保护 tensors_（linear_layer_ids_ 不可变免锁）
};

} // namespace cann_local_lm::engine
