// ═══════════════════════════════════════════════════════════════════════════
// ModelExecutor（v0.13：Session/SetContext——Load 期 KV 工位 desc 记账 + 空占位，
// RebindKvPool 保留为 SetContext 内部工位刷新原语，ADR-0002）。
// 前向壳 + 控制面：Load(CreateBuiltModel+Restore + Init + CreateTensor) / Forward(model_manager->Run) /
// Unload(model_manager->DeInit) 输入/输出 tensor 用私有 input_tensors_/output_tensors_（ITensorBuffer，经 TensorManager
// 代理申请） single-flight；prefill/decode 两实例经模板池薄壳转发面共享活动 Context KV （Load 期不建 KV 显存，创建点在
// CreateContext；F1：vector 直持 shared_ptr，
//  SetContext 必须显式 RebindKvPool 刷新工位）。LoRA：搁置。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "api/cpp/types.h" // ArtifactSource/ModelArtifact（M-01）
#include "core/component/dfx_collector/dfx_collector.h"
#include "core/component/input_assembler/input_assembler.h"
#include "core/component/kv_cache_pool/kv_cache_pool.h"
#include "core/component/linear_state_pool/linear_state_pool.h" // Qwen3.5 状态池
#include "core/tensor_manager.h"                                // backend 类型 + TensorManager + map_backend_status
#include "infra/status.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cann_lm::engine {

/// 模型执行器配置承载结构体（SR20260903001：自 tensor_manager.h 迁入，D-06）。
/// 字段集承接 meta_json model_config 段 schema（原 docs/model_executor/
/// model_config.md 15 字段 + init_token_len + sliding_window_head_dim，
/// 共 17 字段——2026-10-27 增 sw_hd，sliding_kv_seq_dim_rebind S1）。
struct ModelConfig {
    int hidden_size = 0;
    /// 单请求总 token 预算（REQ-9 预算闸基准）：输入 + 生成共享。meta 缺省
    /// （键缺失/null）⇒ 默认 4096；显式配置 ≤0 ⇒ 段降级、已部署形态 Init 以
    /// meta_json invalid 终止（解析层把关，TC_204 先例同口径）。
    int max_io_tokens = 4096;
    std::string model_type;       // 承载不校验（本期不驱动分支）
    int num_hidden_layers = 0;    // KV 缓存层数
    std::vector<int> prefill_len; // prefill 长度档位（本期取 [0]）
    std::vector<int> decode_len;  // decode 长度档位（承载）
    int pad_token_id = 0;         // 跨段：tokenizer_config 消费
    size_t vocab_size = 0;        // 跨段：embedding_config 消费
    size_t vocab_align_size = 0;
    int kv_cache_max_len = 0;    // KV 池序列容量权威来源
    std::string kv_cache_format; // 承载不校验
    // ── KV cache tensor 形状/类型（KvCachePool 定容 + 显存申请）──
    int num_attention_kv_heads = 0;  // 缺省由解析层补桩默认 4
    int num_attention_head_dims = 0; // 缺省由解析层补桩默认 8
    int batch_num = 0;               // decode_batch_size 最高优先级来源
    std::string kv_cache_type;       // KV tensor dtype 名（缺省/未知回退 FLOAT16）
    // ── Qwen3.5 层型映射表（support_qwen3.5.md §4.1）──
    /// meta_json model_config.layer_types。空 = 非 Qwen3.5 / 未声明 → 全部层视为
    /// full_attention（现状行为）；非空时元素个数 == num_hidden_layers，取值
    /// "full_attention" | "linear_attention"。KV 池稀疏化与 LinearStatePool 消费。
    std::vector<std::string> layer_types;
    // ── Gemma4 滑动窗口（docs/mask_fill.md D7）──
    /// meta_json model_config.sliding_window。0/缺省 = 无滑窗（mask_sliding 与
    /// mask_full 同值，非 gemma4 图零影响）；>0 = sliding_attention 层窗口宽
    /// （gemma4-e2b = 512）。InputAssembler 消费（mask_sliding 填充）。
    int sliding_window = 0;
    // ── Gemma4 滑窗层 KV 形状（sliding_kv_seq_dim_rebind 新图契约 flag）──
    /// meta_json model_config.sliding_window_head_dim。0/缺省 = 旧契：滑窗层
    /// 序列轴期望 kv_cache_max_len、末维期望 head_dim（与 full 层同形，
    /// mask_sliding 走 kv_max 大环）；>0 = 新契：滑窗层序列轴期望 =
    /// sliding_window + max(prefill_len)、末维期望 = 本值；mask_sliding 列宽
    /// 随之 = W+P。KvCachePool（R3 期望）/CorrectTensorDescs（修形）/
    /// InputAssembler（填充）/pipeline keep_cap 消费。
    int sliding_window_head_dim = 0;
    /// meta_json model_config.init_token_len。0/缺省 = 无锚定段（全缓冲圆环，
    /// MapSlot 纯模绕回——R5 退化分支，零行为变化）；0 < v < kv_cache_max_len =
    /// 锚定段 [0,v) 永不覆写（MapSlot 环形枝点）、miss 重建时恢复本配置值。
    /// KvCachePool 构造消费（模板池 + 每 Context 池同参）。
    int init_token_len = 0;
    // ── Embedding dequant scale 维度修正（跨段：embedding_config 消费）──
    bool embedding_per_group = false; // PerGroup=true; PerToken=false
    int embedding_group_size = 0;     // PerGroup 必填；PerToken 忽略
};

/// Forward 控制参数。
struct ForwardOptions {
    int batch = 1;             ///< prefill=1；decode ≤ max_batch_size
    std::size_t chunk_len = 1; ///< 本步 token 数（prefill=chunk；decode=1）
    std::vector<SlotId> slots; ///< 本次 batch 对应的 KV 槽
    bool dump = false;         ///< dfx 开关
};

// ── 原地更新族（KV/线性状态）输入工位命名解析 ──
// 引擎旧约定族：past_key_in<层> / past_value_in<层>；
// HF 命名族：past_key_values_<层>_<0|1>（omc 导出名原样，末位 0=key / 1=value，
// 与 HF past_key_values[i][0|1] 同构——10101015_debug.log input[3]/[4] 实名）。
// 两族同语义，Load 工位绑定、CorrectTensorDescs 修正/保护统一在此解析；
// 置于头文件与 kEmbedSlotName 同口径（tests/ut 可直接锁语义——model_executor_ut
// 为 hiai WIP 预损坏目标，独立 UT 仅能经头文件缝隙触达本命名族）。
inline constexpr char kPastKeyPrefix[] = "past_key_in";
inline constexpr char kPastValuePrefix[] = "past_value_in";
inline constexpr char kPastKeyValuesPrefix[] = "past_key_values_";
inline constexpr char kConvStateInPrefix[] = "conv_state_in_layer_";
inline constexpr char kRStateInPrefix[] = "rstate_in_layer_";

/// KV 输入工位名解析 → 全局层号。
/// 覆盖旧约定族 past_key_in<层>/past_value_in<层> 与 HF 命名族
/// past_key_values_<层>_<0|1>（0=key / 1=value）。
/// 返回：layer（≥0 命中）；-1 非该族（常规工位）；-2 命中但后缀非法。
/// *is_value：true = value 工位，false = key 工位。
inline int ParseKvInputLayer(const std::string& name, bool* is_value)
{
    *is_value = false;
    const std::array<std::pair<const char*, bool>, 2> prefixes{std::pair{kPastKeyPrefix, false},
                                                               std::pair{kPastValuePrefix, true}};
    for (const auto& [prefix, val] : prefixes) {
        const std::size_t n = std::strlen(prefix);
        if (name.compare(0, n, prefix) != 0)
            continue;
        *is_value = val;
        const std::string suffix = name.substr(n);
        if (suffix.empty() ||
            !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return -2; // 前缀命中但层数后缀非纯数字
        }
        return std::atoi(suffix.c_str());
    }
    const std::size_t n = std::strlen(kPastKeyValuesPrefix);
    if (name.compare(0, n, kPastKeyValuesPrefix) == 0) {
        const std::string suffix = name.substr(n);
        const std::size_t sep = suffix.rfind('_');
        if (sep == std::string::npos || sep == 0) {
            return -2; // 后缀缺 <层>_<0|1> 双段结构
        }
        const std::string layer = suffix.substr(0, sep);
        const std::string kv = suffix.substr(sep + 1);
        if (!std::all_of(layer.begin(), layer.end(), [](unsigned char c) { return std::isdigit(c) != 0; }) ||
            (kv != "0" && kv != "1")) {
            return -2; // 层号非纯数字 / k-v 末位非 0|1
        }
        *is_value = (kv == "1");
        return std::atoi(layer.c_str());
    }
    return -1;
}

/// Qwen3.5 线性状态输入工位名解析（support_qwen3.5.md §4.5-2）。
/// kind：0 = conv_state，1 = rstate。
/// 返回：layer（≥0 命中，全局层号）；-1 非该族前缀；-2 命中但层数后缀非法。
inline int ParseStateInputLayer(const std::string& name, int* kind)
{
    const std::array<std::pair<const char*, int>, 2> prefixes{std::pair{kConvStateInPrefix, 0},
                                                              std::pair{kRStateInPrefix, 1}};
    for (const auto& [prefix, k] : prefixes) {
        const std::size_t n = std::strlen(prefix);
        if (name.compare(0, n, prefix) != 0)
            continue;
        *kind = k;
        const std::string suffix = name.substr(n);
        if (suffix.empty() ||
            !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return -2; // 前缀命中但层数后缀非纯数字
        }
        return std::atoi(suffix.c_str());
    }
    return -1;
}

/// 原地更新族（KV 旧约定族 + HF 命名族 + Qwen3.5 线性状态族）工位名判定。
/// CorrectTensorDescs 的输入侧跳过与末输出保护共用此口径（问题 3 方案 b：
/// HF 命名 past_key_values_* 同判，防末输出被 2D 截断误伤 KV 形输出）。
inline bool IsInPlaceUpdateSlotName(const std::string& name)
{
    return name.find(kPastKeyPrefix) == 0 || name.find(kPastValuePrefix) == 0 || name.find(kPastKeyValuesPrefix) == 0 ||
           name.find(kConvStateInPrefix) == 0 || name.find(kRStateInPrefix) == 0;
}

/// 配置 prefill 档位上界（sliding_kv_seq_dim_rebind D1）：P = max(prefill_len)；
/// 空数组 → 0（新契 D10 校验已挡空组合，此处回退仅防御）。
/// 单源：KvCachePool 期望 / mask 列宽 / keep_cap 三处共用，防口径漂移。
inline int PrefillMaxLen(const std::vector<int>& prefill_len) noexcept
{
    if (prefill_len.empty())
        return 0;
    return *std::max_element(prefill_len.begin(), prefill_len.end());
}

/// 新契越界组合校验（sliding_kv_seq_dim_rebind D10 宁拒勿错）——meta 解析
/// 双路径单源（pipeline_executor ParseModelSection 与 golden_verify
/// ParseModelConfig 共用，防两处手工同步漂移）：字段自检（sw>0、sw_hd≥0）
/// 之后调用；sw_hd>0（新契）须 sw>0 且 prefill 档非空全正，否则 C=W+P
/// 算不出——静默兜底会让配置者拿到错误形状期望。合法 → true，越界 → false
/// （调用方按段降级处理）。
inline bool SlidingHeadDimComboValid(int sliding_window, int sw_head_dim, const std::vector<int>& prefill_len) noexcept
{
    if (sw_head_dim < 0 || sliding_window < 0)
        return false;
    if (sw_head_dim == 0)
        return true; // 旧契：无组合约束
    if (sliding_window <= 0)
        return false;
    if (prefill_len.empty())
        return false;
    for (const int p : prefill_len) {
        if (p <= 0)
            return false;
    }
    return true;
}

/// 滑窗层 KV 序列容量 + mask 列宽（sliding_kv_seq_dim_rebind 唯一口径 D4/D5）：
/// 仅新契（sliding_window_head_dim>0，由调用方判定）调用；sw>0 且 P>0 →
/// C = W + P（attend 窗宽仍 W；C 恒 > W，纯模环 j%C，attend 槽恒存本代内容
/// ——方案 §2.2 三段论证：块内互异 + 本块覆写不到窗口旧位 + 跨块同理）；
/// 任一非正 → 0（防御；新契 D10 已挡空组合）。旧契（sw_head_dim=0）无滑窗
/// 小环概念——滑窗层与 full 层同槽域 kv_max，消费点直用 kv_max，不经本助手。
inline std::size_t SlidingLayerWidth(int sliding_window, int prefill_max) noexcept
{
    if (sliding_window > 0 && prefill_max > 0) {
        return static_cast<std::size_t>(sliding_window) + static_cast<std::size_t>(prefill_max);
    }
    return 0;
}

/// 具体类。PrefillExecutor 与 DecodeExecutor 是该类的两个实例，非子类。
class ModelExecutor {
public:
    ModelExecutor(std::shared_ptr<KvCachePool> pool, std::shared_ptr<InputAssembler> assembler,
                  std::shared_ptr<DfxCollector> dfx);

    Status AttachPool(std::shared_ptr<KvCachePool> pool);
    Status AttachAssembler(std::shared_ptr<InputAssembler> assembler);
    /// Qwen3.5 线性状态池接线（support_qwen3.5.md §4.5-1）。可空 = 非 Qwen3.5
    /// （不作为就绪条件；图带状态工位而池缺失时 Load 显式报错）。生命周期锁内赋值。
    Status AttachStatePool(std::shared_ptr<LinearStatePool> pool);

    /// 工位重绑（Session/SetContext 内部原语，ADR-0002 + F1 事实修正）：
    /// 把 input_tensors_/output_tensors_ 中 KV 族条目（Load 期记 kv_input_slots_/
    /// kv_output_slots_ 下标）逐条替换到 new_pool 的同层 K/V tensor——执行器
    /// vector 直持 shared_ptr（F1："模板池薄壳指向"不会自动刷新 vector），
    /// SetContext 必须**显式调用本原语**刷新工位。O(KV 工位数) 指针替换，不拷数据。
    /// 前提：new_pool 与当前池层数/层表一致（池同源构造保证）；Load 期 KV 工位为
    /// 空占位（ADR-0002）——本原语在 SetContext（无在飞窗口）填实。LinearStatePool
    /// 为 pipeline 级共享，不在重绑面。
    Status RebindKvPool(std::shared_ptr<KvCachePool> new_pool);

    /// KV 工位清空占位（DestroyContext(活动) 消费，ADR-0002）：把 input_tensors_/
    /// output_tensors_ 中 KV 族条目逐条置空占位、pool_ 引用置空——回"未指向"态。
    /// S2（无活动 Context Generate 拒绝）保证空占位工位不可能被 Run 消费。
    Status DetachKvPool();

    // ── 控制面 ──
    /// Load：单图位双态入口（prefill/decode 执行器各传各自 omc 图位，D-10）。
    /// 流程（参考 framework model_executor_for_generation.cpp）：
    /// CreateModelManager → CreateBuiltModel → Restore（双态，D-15：buffer 优先，
    /// 经 CreateLocalBuffer 包装后 RestoreFromBuffer；否则 RestoreFromFile）
    /// → SetName → Init → GetInput/OutputTensorDescs。
    /// weight 路径非空则 InitWeights（图与权重分离）。
    /// lifecycle_mutex 串行。单活跃：已加载先 DeInit。
    /// 错误：kv_cache_max_len<=0 或 num_hidden_layers<=0 → INVALID_ARGUMENT（TC_106）；
    ///       CreateModelManager/CreateBuiltModel 返空→FAILURE；Restore/Init 失败→FAILURE
    /// seq_len：序列长度，兼做 prefill/decode 模式判别（>1=prefill, ==1=decode），
    ///          用于修正 om 图 tensorDesc 中的占位维度。
    /// share_weights_model_id：权重共享绑定（decode 执行器消费）。>0 = 源模型
    ///    （prefill）ModelId() 返回值，Load 内在 model_manager Init 之前填入
    ///    ModelInitOptions::shareWeightsModelId。绑定后本模型照常按自身 weight
    ///    实参 InitWeights；哪些权值复用源模型已加载份由底层同源规则决定
    ///    （部分共享），上层无需感知（docs/separate_pd_weights.md §4.2）；
    ///    0 = 不绑定（默认，行为不变）。绑定失败（老 ROM 缺符号）快速失败：Load 返 FAILURE。
    Status Load(const ArtifactSource& omc, const std::string& weight, const ModelConfig& config, int seq_len,
                uint32_t share_weights_model_id = 0);
    Status Unload();
    Status GetConfig(ModelConfig* out) const;

    // ── 前向面 ──
    Status SetInputData(const std::shared_ptr<ITensorBuffer>& data);
    Status Forward(const ForwardOptions& opts);
    // 精度验证通道：跳过 InputAssembler，直接用已填充的 input_tensors_ 调
    // model_manager->Run()。golden 二进制文件直接灌入输入工位后前向，不经过 assembler
    // 填 position_ids/attention_mask/KV 光标等。调用方须自行填充全部输入工位。
    Status ForwardRaw();
    std::shared_ptr<ITensorBuffer> Logits() const;

    // ── 输入工位只读暴露（C-02，SR20260828001：TextEmbedding::FillInputs 消费）──
    /// 输入工位 bundle 只读访问（Load 后有效；Unload 后为空）。
    const std::vector<std::shared_ptr<ITensorBuffer>>& InputTensors() const noexcept;
    /// name→index 映射只读访问（Load 后有效；Unload 后为空）。
    const std::unordered_map<std::string, std::size_t>& InputNameIndex() const noexcept;
    /// 输出工位 bundle 只读访问（Load 后有效；含 KV 形输出绑定的池 tensor）。
    const std::vector<std::shared_ptr<ITensorBuffer>>& OutputTensors() const noexcept;

    std::shared_ptr<KvCachePool> Pool() const noexcept;
    /// 线性状态池只读暴露（未 Attach → 空；Scheduler ResetLane 接线消费）。
    std::shared_ptr<LinearStatePool> StatePool() const noexcept;
    /// 本模型权重共享源 id 暴露（供 decode 执行器绑定为 share_weights_model_id）。
    /// Load 成功后有效（model_manager Init 后经 GetModelID 取值）；Unload/重载清零。
    /// 0 = 未加载 / 取值失败 = 不绑定。
    uint32_t ModelId() const noexcept { return model_id_; }
    explicit operator bool() const noexcept;
    std::size_t InputTensorCount() const noexcept;
    std::size_t OutputTensorCount() const noexcept;

    ~ModelExecutor();
    ModelExecutor(const ModelExecutor&) = delete;
    ModelExecutor& operator=(const ModelExecutor&) = delete;
    ModelExecutor(ModelExecutor&&) noexcept;
    ModelExecutor& operator=(ModelExecutor&&) noexcept;

private:
    /// dump 轮次分配：dump 开启时自增并返回新轮次号（forward_mutex_ 已被
    /// Forward/ForwardRaw 持有，此处无需再取锁）；未开启返回当前值不自增。
    std::size_t NextDumpRound();

    std::shared_ptr<IModelManager> model_manager_; ///< Load 时经 hiai::CreateModelManager() 创建并持有
    std::shared_ptr<KvCachePool> pool_;
    std::shared_ptr<LinearStatePool> state_pool_; ///< Qwen3.5 线性状态池（可空 = 非 Qwen3.5）
    std::shared_ptr<InputAssembler> assembler_;
    std::shared_ptr<DfxCollector> dfx_;
    ModelConfig config_{};

    std::vector<std::shared_ptr<ITensorBuffer>> input_tensors_;
    std::vector<std::shared_ptr<ITensorBuffer>> output_tensors_;
    std::unordered_map<std::string, std::size_t> input_name_index_; ///< name→index 映射（Load 后建）
    /// KV 工位在 input/output_tensors_ 中的下标（Load 期记录，RebindKvPool/
    /// DetachKvPool 消费；ADR-0002：Load 期对应槽位为空占位，SetContext 换绑填实）。
    std::vector<std::array<std::size_t, 3>> kv_input_slots_;  ///< [tensor 下标, 全局层号, is_value]
    std::vector<std::array<std::size_t, 3>> kv_output_slots_; ///< [tensor 下标, 全局层号, is_value]
    /// Load 期已记账 KV desc 账本镜像（seq = kv_input_slots_ 同序；SetContext
    /// 换绑时复验新池 tensor desc 与账本一致——半建池防御的执行器侧二次把关）。
    std::vector<TensorDesc> kv_slot_descs_;
    std::size_t logits_index_ = 0; ///< logits 工位下标（Q13 裁定：定 0 号位 output[0]）
    uint32_t model_id_ = 0; ///< model_manager Init 后 GetModelID 取值（权重共享源 id；Load 成功后有效）
    /// dump 轮次计数（debug_tensor_dump 文件名用）：每次前向自增，同一轮
    /// in/out dump 同号。仅 DebugDumpEnabled() 时递增——未开开关零行为。
    std::size_t dump_round_ = 0;

    std::mutex forward_mutex_;   ///< single-flight
    std::mutex lifecycle_mutex_; ///< load/unload 串行
};

} // namespace cann_lm::engine
