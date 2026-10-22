// ═══════════════════════════════════════════════════════════════════════════
// KV Cache 池（DESIGN M-02）。engine 内部组件，仅对 pipelineExecutor 暴露。
// pos/光标/槽位管理在本模块；tensor 内存申请/释放经 TensorManager
// （构造时单次批量 CreateKvCacheTensor，C-04 落地——取代旧 AttachBuffer 外部注入）。
// 排布：每建池层独立 K/V tensor（Qwen3.5 稀疏化，support_qwen3.5.md §4.3：
//       layer_types 声明 linear_attention 的层无 KV 工位、不建 tensor；
//       空 layer_types = 全部层稠密建，现状行为），单 tensor 维度
//       [kv_cache_max_len][num_attention_kv_heads][batch][末维]，末维按层型：
//       full_attention 层 = num_attention_head_dims、sliding_attention 层 =
//       sliding_window（gemma4 滑窗异构 KV），按 kv_layer_ids_ 列表序
//       [k0,v0,k1,v1,...]——满足 om 按层绑定整层 KV 的要求
//       （D-13 演进）；tensor 命名用全局层号（kv_cache_k_<全局层号>）。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "core/tensor_manager.h"   // DataType/ITensorBuffer/TensorManager + using 声明
#include "infra/status.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cann_lm::engine {

/// 槽 id = batch lane 索引（0..max_batch_size-1）；-1 表示无槽。
using SlotId = int;

/// 单槽状态快照（dfx/调试用）。
struct SlotInfo {
    SlotId     id        = -1;
    std::size_t cursor   = 0;   ///< 下一个可写位置（已写长度 = cursor），position 维、跨层同步
    bool       active    = false;
    uint64_t   alloc_seq = 0;   ///< 分配序号（FIFO 驱逐依据：值越小越早分配）
};

/// KV ringBuffer 池（DESIGN M-02 / D-13）。
///
/// 定长池，槽位数 = max_batch_size。per-seq 光标只推进 position 维、对所有层
/// 同步（各层同 token 同步推进）。prefill 按 chunk 追加、decode 每 seq 追加单 token。
/// 池满驱逐最早（FIFO，U-4 可调）。底层为每层独立 K/V tensor（经 TensorManager
/// 构造时批量申请，KTensor/VTensor 按层暴露——满足 om 按层绑定整层 KV）。
class KvCachePool {
public:
    /// 构造：容量参数化于 ModelConfig（由调用方 Load 时传入，U-2）。
    /// dtype = KV tensor 元素类型（meta_json kv_cache_type 解析所得，缺省 FLOAT16）。
    /// layer_types = 层型表（meta_json model_config.layer_types）；
    /// 空 = 未声明层型 → 稠密池（全部层建 KV，现状行为，零行为变化）；
    /// 非空（长度须 == num_hidden_layers，解析层已校验）→ 稀疏池，
    /// linear_attention 层不建 tensor，full_attention / sliding_attention
    /// 层建 tensor；长度不符 → WARN 回退稠密（防御，不静默稀疏）。
    /// sliding_window = gemma4 滑窗宽（meta model_config.sliding_window；
    /// >0 且层型为 sliding_attention → 该层 KV 末维取该值；其余情况末维 =
    /// num_attention_head_dims，现状行为）。
    /// 构造期经 TensorManager 申请全部 2×NumKvLayers() 个 tensor（连续同末维
    /// 层合批；无滑窗异构时保持单批建形不变）；
    /// 申请失败（维度非法/乘积溢出/底层分配失败）→ tensor 无产出 + ERROR 日志，
    /// 槽位记账不受影响（all-or-nothing，host malloc 回退下几乎不触发）。
    KvCachePool(int max_batch_size, int kv_cache_max_len, int num_hidden_layers,
                int num_attention_kv_heads, int num_attention_head_dims,
                DataType dtype,
                const std::vector<std::string>& layer_types = {},
                int sliding_window = 0);

    /// 为新 sequence 分配一条 batch lane（槽）。
    /// 池满则驱逐最早 active 槽（FIFO，U-4）；满且无可驱逐 → Status::FAILURE。
    Status AllocKvSlot(SlotId* out);

    /// 释放槽（标记复用）。幂等：重复释放已释放槽返回 Success。
    Status FreeKvSlot(SlotId id);

    /// 推进槽 id 的光标 n 个 token（position 维，跨层同步）。
    /// 越界（cursor + n > kv_cache_max_len）→ Status::FAILURE（KV 越界，质量红线 🔴）。
    Status Advance(SlotId id, std::size_t n);

    /// 下一步写入位置 = cursor（position 维，跨层一致）。
    std::size_t NextPosition(SlotId id) const;

    /// dfx 快照：各槽占用、光标、active。
    Status Snapshot(std::vector<SlotInfo>* out) const;

    // ── KV tensor 只读暴露（零拷贝；供 om 按层绑定）──
    /// 已申请 tensor 数（= 2×NumKvLayers()；构造申请失败为 0）。
    std::size_t TensorCount() const noexcept;
    /// 层 layer（全局层号）的 K tensor（非建池层/越界/未申请 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> KTensor(int layer) const noexcept;
    /// 层 layer（全局层号）的 V tensor（非建池层/越界/未申请 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> VTensor(int layer) const noexcept;
    /// 建池层数（full+sliding；空 layer_types = num_hidden_layers；稀疏口径）。
    int NumKvLayers() const noexcept;
    /// 建池层全局层号列表（升序；与 tensors_ 排布同序；构造后不可变）。
    const std::vector<int>& KvLayerIds() const noexcept;

    /// KV tensor 形状对图对齐（HF 命名族适配，kv_slot_naming_hf.md §8）——按层。
    /// layer = KV 工位全局层号：以该层池 tensor 为比对基准（gemma4 滑窗异构
    /// KV 下各层 dims 可不同，不再以 tensors_.front() 单形代表全池）。
    /// 图权威口径不变：池默认轴序 [len, heads, batch, dim] vs HF 图
    /// [batch, heads, len, dim]，元素总数同。该层一致 → no-op；轴序不同但
    /// 元素总数相同且 dtype 相同 → 仅重建该层 K/V 两个 tensor（kv_layer_ids_/
    /// 槽位记账不变，老 tensor RAII 释放；此时未发生任何工位绑定，重建安全
    /// ——首个 Load 在任何绑定前调用）。
    /// 该层首个调用即锁定该层布局：该层后续再遇不同 dims → INVALID_ARGUMENT
    /// （prefill/decode 双图同名层布局须一致，与 LinearStatePool::
    /// EnsureStateTensor 双图把关同口径）；dtype 不同 / 元素总数不同 /
    /// dims 非法 → INVALID_ARGUMENT（真失配显式报错）。
    /// 构造期申请失败（tensors_ 空）→ SUCCESS（无从对齐；绑定侧 pool tensor
    /// missing 收口）。pool 为空指针时不应调用（调用方保证）。
    Status EnsureKvLayout(int layer, const std::vector<int32_t>& dims, DataType dtype);

    int max_batch_size() const noexcept { return max_batch_size_; }
    int kv_cache_max_len() const noexcept { return kv_cache_max_len_; }
    int num_hidden_layers() const noexcept { return num_hidden_layers_; }

private:
    /// 统一取层 tensor（kv：0=K，1=V）：全局层号经 kv_layer_ids_ 映射到池内下标
    ///（序 [k0,v0,k1,v1,...] 按 kv_layer_ids_ 列表序）。
    std::shared_ptr<ITensorBuffer> TensorAt(int layer, int kv) const noexcept;

    int           max_batch_size_;
    int           kv_cache_max_len_;
    int           num_hidden_layers_;
    int           num_kv_heads_;
    int           head_dim_;
    DataType      dtype_ = DataType::RESERVED;
    /// 建池层全局层号（升序；空 layer_types = 全部层 → 现状稠密）。
    std::vector<int> kv_layer_ids_;
    std::vector<std::shared_ptr<ITensorBuffer>> tensors_;   ///< [k0,v0,k1,v1,...]（按 kv_layer_ids_ 序）
    std::vector<bool> layout_locked_;   ///< 每建池层一把布局锁（kv_layer_ids_ 序；EnsureKvLayout 首调用后逐层置位——双图布局一致性把关）
    std::vector<SlotInfo> slots_;
    uint64_t      next_seq_ = 0;          ///< 全局分配序号（递增，用于 FIFO 驱逐排序）
    mutable std::mutex   mtx_;
};

}  // namespace cann_lm::engine
