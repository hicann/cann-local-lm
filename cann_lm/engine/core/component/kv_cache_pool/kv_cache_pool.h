// ═══════════════════════════════════════════════════════════════════════════
// KV Cache 池（DESIGN M-02 / Cache_pool_simplify.md）。
// 槽位/光标记账 + HF 期望值预计算在构造期；tensor 显存**不在构造期申请**——
// 由首个执行器 Load 时经 EnsureKvTensor 按 om 图工位 desc 惰性创建
// （R1/T1，与 LinearStatePool::EnsureStateTensor 同构 verify-or-create）。
// 排布：每建池层独立 K/V tensor（Qwen3.5 稀疏化，support_qwen3.5.md §4.3：
//       layer_types 声明 linear_attention 的层无 KV 工位、不建 tensor；
//       空 layer_types = 全部层稠密建，现状行为），tensor 实形 = om 图工位
//       desc（R1，om 权威——轴序/dtype 一切细节以图为准），按 kv_layer_ids_
//       列表序 [k0,v0,k1,v1,...] 落位；tensor 命名用全局层号
//       （kv_cache_k_<全局层号>，约定不变）。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "core/tensor_manager.h" // DataType/ITensorBuffer/TensorDesc/TensorManager
#include "infra/status.h"

#include <atomic>
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
    SlotId id = -1;
    std::size_t total_written = 0; ///< 逻辑写入长度（绝对位，单调不减；R4 分离记账——position_ids 语义）
    std::size_t write_pos = 0; ///< 物理写指针 = Map(total_written)（环形槽位；R4——new_kv_cache_pos 语义）
    bool active = false;
    uint64_t alloc_seq = 0; ///< 分配序号（FIFO 驱逐依据：值越小越早分配）
};

/// KV 环形缓冲池（DESIGN M-02 / D-13 / Cache_pool_simplify.md R1-R3+T1；
/// PromptCache 环形化改造 R4/R5——design §4.2，ADR-0001）。
///
/// 定长池，槽位数 = max_batch_size。per-seq cursor 分离记账（R4）：
///   - **total_written**：单调逻辑写入长度（绝对位），语义源 = position_ids；
///   - **write_pos**：物理写指针 = MapSlot(total_written)（环形），语义源 =
///     new_kv_cache_pos 工位值。
/// 物理缓冲 [0, kv_max) 两段（R5 锚定式环形）：[0, init_token_len) 锚定段永不
/// 覆写（首轮 prompt 段）；[init_token_len, kv_max) 尾圆环绕圈覆写。
/// 单 batch 端侧场景下每 Context 一池（S5 池复制），本池单槽即全池状态。
/// 底层为每层独立 K/V tensor，Load 期经 EnsureKvTensor 创建（创建即整只
/// 清零），KTensor/VTensor 按层暴露——满足 om 按层绑定整层 KV。
class KvCachePool {
public:
    /// 构造：容量与期望值参数化于 ModelConfig（调用方 Load 前传入，U-2）。
    /// **构造期不申请 tensor**（T1：显存创建后移到首个执行器 Load，经
    /// EnsureKvTensor 按 om 图工位 desc 直建）——仅做槽位初始化、kv_layer_ids_
    /// 层表推导与每层 HF 期望 dims 预计算（R3 比对值）。
    /// num_attention_kv_heads / num_attention_head_dims / sliding_window：
    /// **期望值**（meta_json model_config 段），不再用于建 tensor——仅组成
    /// HF 期望序 tuple 供 EnsureKvTensor 与图 dims 比对告警/把关。
    /// layer_types = 层型表（meta_json model_config.layer_types）；
    /// 空 = 未声明层型 → 稠密池（全部层建 KV，现状行为，零行为变化）；
    /// 非空（长度须 == num_hidden_layers，解析层已校验）→ 稀疏池，
    /// linear_attention 层不建 tensor，full_attention / sliding_attention
    /// 层建 tensor；长度不符 → WARN 回退稠密（防御，不静默稀疏）。
    /// sliding_window 另供期望末维：sliding_attention 层期望末维 =
    /// sliding_window（>0 时），其余层 = num_attention_head_dims
    /// （gemma4 滑窗异构 KV 期望口径）。
    KvCachePool(int max_batch_size, int kv_cache_max_len, int num_hidden_layers, int num_attention_kv_heads,
                int num_attention_head_dims, const std::vector<std::string>& layer_types = {}, int sliding_window = 0);

    /// 为新 sequence 分配一条 batch lane（槽）。cursor 分离记账（R4）：分配即
    /// total_written/write_pos 归零。池满则驱逐最早 active 槽（FIFO，U-4）；
    /// 满且无可驱逐 → Status::FAILURE。
    Status AllocKvSlot(SlotId* out);

    /// 释放槽（标记复用）。幂等：重复释放已释放槽返回 Success。
    Status FreeKvSlot(SlotId id);

    /// 推进槽 id 的写入记账 n 个 token（R4 分离记账：total_written += n 单调
    /// 逻辑长度；write_pos = Map(total_written) 环形物理指针）。
    /// 环形化（R5/ADR-0001）：越 kv_max 绕回**不报错**——尾圆环
    /// [init_token_len, kv_max) 绕圈覆写，锚定段 [0, init_token_len) 不动；
    /// 单次输入自身超缓冲（ids.size() > kv_max）由 Submit 前置校验拦截
    /// （spec REQ-9），池侧恒 SUCCESS。
    Status Advance(SlotId id, std::size_t n);

    /// 下一步物理写入槽位 = write_pos（= Map(total_written)；R4：new_kv_cache_pos
    /// 工位值语义）。无绕回时与旧 cursor 同值（行为兼容）。
    std::size_t NextPosition(SlotId id) const;

    /// 槽逻辑写入长度（绝对位单调值；R4：position_ids 起填语义）。
    std::size_t TotalWritten(SlotId id) const;

    /// 回退写入记账到指定 total_written（R10 失败一致性：snap 在 GenerateAsync
    /// 提交时记录，请求 Failed/Cancel 终态出队时光标回退）。仅允许回退
    /// （target ≤ 当前值），前进请求按 Advance 走。
    Status RollbackTo(SlotId id, std::size_t target_total_written);

    /// 槽位映射（R5 环形）：物理槽 = pos < kv_max ? pos
    ///   : init_token_len + (pos - init_token_len) % (kv_max - init_token_len)。
    /// init_token_len = 0（全缓冲圆环）或 init_token_len = kv_max（退化保护）时
    /// 图结构不成立，退化为 pos % kv_max；纯函数、无 slot 归属。
    std::size_t MapSlot(std::size_t pos) const noexcept;

    /// 锚定段长（R5/R13：[0, init_token_len) 永不覆写；miss 重建时归零，其余
    /// 链上随链延续）。台账侧 SetInitTokenLen 同步维护（design §3.3）。
    std::size_t init_token_len() const noexcept { return init_token_len_; }
    void SetInitTokenLen(std::size_t len) noexcept;

    // ── PromptCache 台账联动（design §3；miss 时调用方同步清池）──
    /// miss 重建：记账归零 + init_token_len 归零（R13"一切重建"）。
    /// 注意：不清 KV tensor 内容——复用链上的 lane 语义由 ResetLane 承担
    /// （scheduler 对应槽的 state 清理），池只管记账。
    void ResetLedgerOnSlot(SlotId id) noexcept;

    /// 命中复用（REQ-4 第 2 行）：lane 继续本 Context 原有身份——total_written
    /// 回退到复用边界 reusable（池光标 == 台账截断后长度，REQ-2 不变量），
    /// active 置位、init_token_len 保持（R13 随链延续）。与 AllocKvSlot 的
    /// 归零分配（miss 重建链）语义区分：调用方按 MatchCachedTokens 结果选路。
    /// target ≤ 当前 total_written 时回退；前进请求按 Advance 走（防御拒绝）。
    Status ResumeSlot(SlotId id, std::size_t target_total_written) noexcept;

    /// dfx 快照：各槽占用、光标、active。
    Status Snapshot(std::vector<SlotInfo>* out) const;

    // ── KV tensor 只读暴露（零拷贝；供 om 按层绑定）──
    /// 已创建 tensor 数（构造后 = 0；任一图 Load 成功后 = 2×NumKvLayers()）。
    std::size_t TensorCount() const noexcept;
    /// 层 layer（全局层号）的 K tensor（非建池层/越界/未 Ensure 层 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> KTensor(int layer) const noexcept;
    /// 层 layer（全局层号）的 V tensor（非建池层/越界/未 Ensure 层 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> VTensor(int layer) const noexcept;
    /// 建池层数（full+sliding；空 layer_types = num_hidden_layers；稀疏口径）。
    int NumKvLayers() const noexcept;
    /// 建池层全局层号列表（升序；与 tensors_ 排布同序；构造后不可变）。
    const std::vector<int>& KvLayerIds() const noexcept;

    /// Load 期消费：层 layer 的 K（is_value=false）/V（is_value=true）tensor
    /// 就绪——无则按 om 图工位 desc 直接创建（R1），有则校验 dims+dtype
    /// 严格一致（R2 双图把关点，失配 INVALID_ARGUMENT）。
    /// desc 合法性：dims 非空逐维 >0、dtype != RESERVED → 否则 INVALID_ARGUMENT；
    /// layer 不在建池层表内 → FAILURE（资源类防御；Load 侧层型校验在前）。
    ///
    /// 附加 HF 期望校验（R3 分档；**只认 HF 序 [batch,heads,len,dim] 逐位比对，
    /// 不兼容旧约定序 [len,heads,batch,dim]**——2026-10-15 裁定）：
    ///   - **kv_len 维（dims[2]）/ batch 维（dims[0]）失配 → LOGE +
    ///     INVALID_ARGUMENT**（致命终止，Load 失败 ⇒ pipeline 终止初始化）。
    ///     kv_len：光标容量/mask/Advance 越界全线以 meta kv_cache_max_len 为准，
    ///     图短于 meta 将导致光标推进超图缓冲；batch：池槽位数（lane 并发
    ///     上限）来自 meta decode_batch_size，图 batch < meta 槽位数时 decode
    ///     满并发会把超界 lane 的 KV 写入压错位/越 tensor。两张力均 host 记账
    ///     与图强关联，不能放行带病运行（2026-10-15 裁定致命）。
    ///   - **其余维（dims[1]=heads / dims[3]=末维）失配，及 dims 数 ≠ 4 →
    ///     LOGW 告警（每池同类去重一次，warned_），不影响返回值**；dims 数 ≠ 4
    ///     时跳过逐位比对、按图 desc 直建（om 图自洽，host 不直接寻址 KV
    ///     tensor；2026-10-15 裁定）。
    ///
    /// 新建路径创建即整只清零（memset 全 buffer，与 EnsureStateTensor 同口径；
    /// 2026-10-15 裁定）；复用路径不重复清。om 是权威，移植 K 与 V 形状由图
    /// 自己保证，同层 K/V 两工位各自独立 Ensure。pool 为空指针时不应调用
    /// （调用方保证）。
    Status EnsureKvTensor(int layer, bool is_value, const TensorDesc& desc);

    int max_batch_size() const noexcept { return max_batch_size_; }
    int kv_cache_max_len() const noexcept { return kv_cache_max_len_; }
    int num_hidden_layers() const noexcept { return num_hidden_layers_; }

private:
    /// 统一取层 tensor（kv：0=K，1=V）：全局层号经 kv_layer_ids_ 映射到池内下标
    /// （序 [k0,v0,k1,v1,...] 按 kv_layer_ids_ 列表序）。
    std::shared_ptr<ITensorBuffer> TensorAt(int layer, int kv) const noexcept;

    /// 全局 layer → kv_layer_ids_ 池内下标；不在池内返回 -1（线程安全假设：
    /// 仅构造后层表不可变，读无锁；Ensure 独占写）。
    int LayerPos(int layer) const noexcept;

    int max_batch_size_;
    int kv_cache_max_len_;
    int num_hidden_layers_;
    std::size_t init_token_len_ = 0; ///< 锚定段长（R5；随复用链延续，miss 归零）
    int num_kv_heads_;               ///< 期望值（R3 比对用；不参与建 tensor）
    int head_dim_;                   ///< 期望值（R3 比对用；不参与建 tensor）
    int sliding_window_;             ///< 期望末维口径（sliding_attention 层）
    /// 建池层全局层号（升序；空 layer_types = 全部层 → 现状稠密）。
    std::vector<int> kv_layer_ids_;
    /// 每建池层 HF 期望 dims（kv_layer_ids_ 序；[batch,heads,kv_len,末维] HF 序，
    /// 末维 = sliding_attention 层取 sliding_window（>0），否则 head_dim）。
    std::vector<std::vector<int32_t>> expect_dims_;
    std::vector<std::shared_ptr<ITensorBuffer>> tensors_; ///< [k0,v0,k1,v1,...]（按 kv_layer_ids_ 序；Ensure 逐层落位）
    /// R3 告警去重（池生命周期内同类只告一次）：bit0 = dims 数 ≠ 4、
    /// bit1 = heads 维失配、bit2 = 末维失配（2026-10-15 裁定：其余维告警不终止）。
    mutable std::atomic<uint32_t> warned_{0};
    std::vector<SlotInfo> slots_;
    uint64_t next_seq_ = 0; ///< 全局分配序号（递增，用于 FIFO 驱逐排序）
    mutable std::mutex mtx_;
};

} // namespace cann_lm::engine
