// ═══════════════════════════════════════════════════════════════════════════
// KV Cache 池（DESIGN M-02 / Cache_pool_simplify.md + ADR-0002
// Session/SetContext：模板池 desc 账本 + 薄壳转发）。
// 建点双轨（ADR-0002）：模板池 Load 期仅经 EnsureKvDesc 记录 om 图 KV 工位
// desc 账本（不建 tensor、不开显存）；KV tensor 实建延后到 CreateContext——
// per-Context 池按模板池已复验 desc 逐层 EnsureKvTensor 直建（verify-or-create，同
// LinearStatePool::EnsureStateTensor 口径；创建即整只清零、all-or-nothing）。
// SetContext 使模板池经 BindFrom 把 tensors_ 转发指向活动 Context 池列表
// （Q10 乙薄壳）；DestroyContext(活动) 经 ClearTensors 回"未指向"态。
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
#include <optional>
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
/// 覆写（meta_json 静态供给；0 = 全缓冲圆环）；[init_token_len,
/// kv_max) 尾圆环绕圈覆写。
/// 单 batch 端侧场景下 KV tensor 按 Lane 具名独占（Named Lane Ownership，
/// GLOSSARY 规范词；S5），每 Context 一池，本池单槽即全池状态。
/// 底层为每层独立 K/V tensor：模板池 Load 期只经 EnsureKvDesc 记 desc 账本，
/// KV tensor 实建延后到 CreateContext（per-Context 按账本经 EnsureKvTensor
/// 直建，创建即整只清零——ADR-0002）；模板池经 BindFrom 薄壳转发呈现活动
/// Context 的 tensor，KTensor/VTensor 按层暴露——满足 om 按层绑定整层 KV。
class KvCachePool {
public:
    /// 构造：容量与期望值参数化于 ModelConfig（调用方 Load 前传入，U-2）。
    /// **构造期不申请 tensor**（ADR-0002：模板池 Load 期仅 EnsureKvDesc 记
    /// desc 账本；KV tensor 实建延后到 CreateContext，按模板池账本经
    /// EnsureKvTensor 直建）——仅做槽位初始化、kv_layer_ids_
    /// 层表推导与每层 HF 期望 dims 预计算（R3 比对值）。
    /// num_attention_kv_heads / num_attention_head_dims / sliding_window：
    /// **期望值**（meta_json model_config 段）：仅组成 HF 期望序 tuple 供
    /// EnsureKvTensor 与图 dims 比对告警/把关（tensor 实形以 om 图工位
    /// desc 为唯一权威，R1）。
    /// layer_types = 层型表（meta_json model_config.layer_types）；
    /// 空 = 未声明层型 → 稠密池（全部层建 KV，现状行为，零行为变化）；
    /// 非空（长度须 == num_hidden_layers，解析层已校验）→ 稀疏池，
    /// linear_attention 层不建 tensor，full_attention / sliding_attention
    /// 层建 tensor；长度不符 → WARN 回退稠密（防御，不静默稀疏）。
    /// prefill_max / sliding_window_head_dim（sliding_kv_seq_dim_rebind 新图
    /// 契约）＝ PrefillMaxLen(prefill_len) 与 meta sw_head_dim：sw_head_dim>0
    /// （新契）时 sliding_attention 层期望 = [batch, heads, W+P, sw_head_dim]，
    /// 其余（旧契滑窗层 / full 层 / 线性无 KV）期望 = [batch, heads, kv_max,
    /// head_dim]（与 full 层同形）。仅作 R3 比对期望，不用于建 tensor（R1：
    /// om 图工位 desc 为 tensor 实形唯一权威）；新契 meta 配旧图 → kv_len
    /// 失配 fatal（fail-loud 有意设计）。
    /// init_token_len（meta_json model_config.init_token_len）：
    /// 0/缺省 = 无锚定段（全缓冲圆环，现状行为零变化）；0 < v < kv_cache_max_len
    /// = 锚定段 [0,v) 永不覆写，miss 重建（ResetLedgerOnSlot）时恢复本配置值
    /// （configured_init_len_ 基线）而非归零。越界值构造期 WARN 降 0（防御深度；
    /// 解析层已段降级拒绝，正常不可达）。
    KvCachePool(int max_batch_size, int kv_cache_max_len, int num_hidden_layers, int num_attention_kv_heads,
                int num_attention_head_dims, const std::vector<std::string>& layer_types = {}, int sliding_window = 0,
                int init_token_len = 0, int prefill_max = 0, int sliding_window_head_dim = 0);

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

    /// 锚定段长（R5/R13：[0, init_token_len) 永不覆写）。值由 meta_json
    /// model_config.init_token_len 构造期供给（configured_init_len_ 基线），
    /// miss 重建（ResetLedgerOnSlot）恢复基线而非归零，其余链上随链延续。
    /// 台账侧 SetInitTokenLen 同步维护（design §3.3）。
    std::size_t init_token_len() const noexcept { return init_token_len_; }
    void SetInitTokenLen(std::size_t len) noexcept;

    // ── PromptCache 台账联动（design §3；miss 时调用方同步清池）──
    /// miss 重建：记账归零 + init_token_len 恢复配置基线 configured_init_len_
    /// （R13：锚定段为元数据级常量，"一切重建"= 回配置基线，配置 0 时与
    /// 归零等价）。
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
    /// 已登记/已建 tensor 数（构造后 = 0；模板池 = 活动池 BindFrom 后 = 活动池
    /// 非空 tensor 数；Context 池 = EnsureKvTensor 实建数，host/stub 零 desc → 0 合法）。
    std::size_t TensorCount() const noexcept;
    /// 层 layer（全局层号）的 K tensor（非建池层/越界/未 Ensure 层 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> KTensor(int layer) const noexcept;
    /// 层 layer（全局层号）的 V tensor（非建池层/越界/未 Ensure 层 → 空 shared_ptr）。
    std::shared_ptr<ITensorBuffer> VTensor(int layer) const noexcept;
    /// 建池层数（full+sliding；空 layer_types = num_hidden_layers；稀疏口径）。
    int NumKvLayers() const noexcept;
    /// 建池层全局层号列表（升序；与 tensors_ 排布同序；构造后不可变）。
    const std::vector<int>& KvLayerIds() const noexcept;

    // ── ADR-0002：desc 账本（模板池 Load 期记账面；不建显存）──
    /// Load 期模板池消费：记录层 layer 的 K（is_value=false）/V（is_value=true）
    /// 工位 desc——首次记账即录，重复调用复验 dims+dtype 严格一致（R2 双图
    /// 把关点，失配 INVALID_ARGUMENT）。**不建 tensor、不开显存**（真机形态
    /// KV tensor 建点在 CreateContext）。
    /// desc 合法性：dims 非空逐维 >0、dtype != RESERVED → 否则 INVALID_ARGUMENT；
    /// layer 不在建池层表内 → FAILURE（资源类防御；Load 侧层型校验在前）。
    /// HF 期望校验（R3 分档）与 EnsureKvTensor 共用同一比对（kv_len/batch
    /// 失配致命、其余维 WARN 池内去重）；校验时点整体留在 Load（Q7 裁定：
    /// 双图一致性只有 Load 摸得到图 desc）。
    Status EnsureKvDesc(int layer, bool is_value, const TensorDesc& desc);

    /// 层 layer 的 K（is_value=false）/V（is_value=true）已记账 desc（Load 期
    /// EnsureKvDesc 首记后恒有）；未记账（host/stub 无 Load）→ 空 optional。
    /// CreateContext 按账本逐层 EnsureKvTensor 实建 KV tensor 的权威形状源。
    std::optional<TensorDesc> KvDesc(int layer, bool is_value) const;

    /// desc 账本是否完整（每建池层 K/V 全记）：模板池 Load 完成 = true；
    /// host/stub（未跑 Load）= false——SetContext/CreateContext 零 tensor
    /// 形态合法性判据（Q12 裁定 a）。
    bool DescLedgerComplete() const noexcept;

    /// CreateContext 期 Context 池消费（ADR-0002 KV tensor 建点）：层 layer
    /// 的 K（is_value=false）/V（is_value=true）tensor 就绪——无则按 desc
    /// 直接创建（R1；desc 通常取自模板池 KvDesc 账本，om 图工位 desc 为
    /// tensor 实形唯一权威），有则校验 dims+dtype 严格一致（R2 复验口径，
    /// 失配 INVALID_ARGUMENT）。
    /// desc 合法性：dims 非空逐维 >0、dtype != RESERVED → 否则 INVALID_ARGUMENT；
    /// layer 不在建池层表内 → FAILURE（资源类防御；CreateContext 建建池层全表）。
    /// 失败该层不落位（未建可重试，调用方 all-or-nothing 回滚天然成立，Q11）。
    ///
    /// 附加 HF 期望校验（R3 分档；**只认 HF 序 [batch,heads,len,dim] 逐位比对，
    /// 不兼容旧约定序 [len,heads,batch,dim]**）：
    ///   - **kv_len 维（dims[2]）/ batch 维（dims[0]）失配 → LOGE +
    ///     INVALID_ARGUMENT**（致命终止，Load 失败 ⇒ pipeline 终止初始化）。
    ///     kv_len：光标容量/mask/Advance 越界全线以 meta kv_cache_max_len 为准，
    ///     图短于 meta 将导致光标推进超图缓冲；batch：池槽位数（lane 并发
    ///     上限）来自 meta decode_batch_size，图 batch < meta 槽位数时 decode
    ///     满并发会把超界 lane 的 KV 写入压错位/越 tensor。两张力均 host 记账
    ///     与图强关联，不能放行带病运行（致命终止）。
    ///   - **其余维（dims[1]=heads / dims[3]=末维）失配，及 dims 数 ≠ 4 →
    ///     LOGW 告警（每池同类去重一次，warned_），不影响返回值**；dims 数 ≠ 4
    ///     时跳过逐位比对、按图 desc 直建（om 图自洽，host 不直接寻址 KV
    ///     tensor）。
    ///
    /// 新建路径创建即整只清零（memset 全 buffer，与 EnsureStateTensor 同口径；
    /// 同口径）；复用路径不重复清。om 是权威，K 与 V 形状由图
    /// 自己保证，同层 K/V 两工位各自独立 Ensure。pool 为空指针时不应调用
    /// （调用方保证）。
    Status EnsureKvTensor(int layer, bool is_value, const TensorDesc& desc);

    // ── ADR-0002：薄壳转发登记（SetContext/DestroyContext 消费）──
    /// 模板池消费（Q10 乙）：把 tensors_ 各槽位指向活动池 src 对应槽位
    /// （"模板池 KV tensor 列表指向 Context 的列表"字面语义——shared_ptr
    /// 共享引用，真显存属 src 池）。src 半建（任一槽空）→ FAILURE（半建池
    /// 不允许被激活，二次防御）；src 为零 tensor 形态（host/stub 无 Load
    /// → 零 desc → ContextCreate 零 tensor 记账合法）→ 清空回"未指向"态
    /// 仍 SUCCESS（Q12 裁定 a：生命周期 API 恒成功）。两池必须同参构造
    /// （层表一致，调用方保证；此处守 size+同层表防御）。
    Status BindFrom(const KvCachePool& src);

    /// 清空转发登记（DestroyContext 消费）：tensors_ 回全空"未指向"态。
    /// 只清指针不动显存（真显存在 Context 池，随其析构释放）。
    void ClearTensors() noexcept;

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

    /// EnsureKvDesc / EnsureKvTensor 共用前半套校验（锁外、纯只读）：
    /// desc 合法性（dims 非空逐维 >0、dtype != RESERVED）→ INVALID_ARGUMENT；
    /// LayerPos 层表防御 → FAILURE；R3 HF 期望比对（kv_len/batch 失配致命、
    /// 其余维 WARN 池内去重——与既有 EnsureKvTensor 行为逐字一致）。
    /// 通过则 *pos = 池内下标。
    Status CheckKvDescCommon(int layer, bool is_value, const TensorDesc& desc, int* pos) const;

    int max_batch_size_;
    int kv_cache_max_len_;
    int num_hidden_layers_;
    std::size_t init_token_len_ = 0; ///< 锚定段长现值（R5；随复用链延续，miss 恢复基线）
    /// 锚定配置基线：构造期由 meta_json 供给定型、此后
    /// 只读；miss 重建（ResetLedgerOnSlot）恢复本值。0 = 无锚定段配置。
    std::size_t configured_init_len_ = 0;
    int num_kv_heads_;    ///< 期望值（R3 比对用；不参与建 tensor）
    int head_dim_;        ///< 期望值（R3 比对用；不参与建 tensor）
    int sliding_window_;  ///< 滑窗宽 W（attend 窗宽；sliding_attention 层）
    int prefill_max_ = 0; ///< P = max(prefill_len)（新契 C=W+P 的入参；构造期定型）
    /// 新图契约 flag + 滑窗层末维期望（meta sliding_window_head_dim）：
    /// >0 = 新契（滑窗层期望 = [b,h,W+P,本值]）；0 = 旧契（与 full 层同形）。
    int sliding_window_head_dim_ = 0;
    /// 建池层全局层号（升序；空 layer_types = 全部层 → 现状稠密）。
    std::vector<int> kv_layer_ids_;
    /// 每建池层 HF 期望 dims（kv_layer_ids_ 序；[batch,heads,kv_len,末维] HF 序；
    /// 末维 = 新契滑窗层取 sw_head_dim，其余（旧契滑窗层 / full 层）取 head_dim；
    /// 序列轴 = 新契滑窗层取 C=W+P，其余取 kv_max）。
    std::vector<std::vector<int32_t>> expect_dims_;
    /// desc 账本（ADR-0002）：每建池层 K/V 工位 TensorDesc（序 [k0,v0,k1,v1,...]
    /// 按 kv_layer_ids_ 列表序，对齐 tensors_ 排布）。模板池 Load 期 EnsureKvDesc
    /// 首记；Context 池建 tensor 的形状权威源。维度应 tensors_ 同维（同一下标语义
    /// 对应同一 K/V 工位）。
    std::vector<std::optional<TensorDesc>> kv_descs_;
    std::vector<std::shared_ptr<ITensorBuffer>> tensors_; ///< [k0,v0,k1,v1,...]（按 kv_layer_ids_ 序；模板池=BindFrom
                                                          ///< 转发活动池 / Context 池=EnsureKvTensor 实建落位）
    /// R3 告警去重（池生命周期内同类只告一次）：bit0 = dims 数 ≠ 4、
    /// bit1 = heads 维失配、bit2 = 末维失配（其余维告警不终止）。
    mutable std::atomic<uint32_t> warned_{0};
    std::vector<SlotInfo> slots_;
    uint64_t next_seq_ = 0; ///< 全局分配序号（递增，用于 FIFO 驱逐排序）
    mutable std::mutex mtx_;
};

} // namespace cann_lm::engine
