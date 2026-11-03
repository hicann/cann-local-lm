// ModelExecutor 实现（v0.13：builtModel 改用 backend/api built_model.h——
// 参考 framework model_executor_for_generation.cpp 的 modelManager 流程）。
// Load: CreateModelManager → CreateBuiltModel → RestoreFromBuffer/RestoreFromFile
//       → SetName → Init → GetInput/OutputTensorDescs → GetModelID
//       → CreateTensor（经 TensorManager 代理）
// Forward: assembler 填 input_tensors_ + model_manager->Run(input_tensors_, output_tensors_)
// Unload: model_manager->DeInit()
// Status: engine Status 不改——边界 map_backend_status() 映射。
#include "core/component/model_executor/model_executor.h"
#include "core/component/text_embedding/text_embedding.h" // 问题 1 方案 b：embed 工位名/别名解析（kEmbedSlotName 等）
#include "core/tensor_manager.h"
#include "infra/log.h"
#include "utils/debug_tensor_dump.h" // 推理前后 dump 工位（env 开关默认关）
#include "utils/perf_timer.h"        // NowUs/UsToMs：加载/推理耗时观测（perf.* 协议见其头注释）

#include "model/built_model.h" // hiai::IBuiltModel + CreateBuiltModel

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <utility>

namespace cann_local_lm::engine {

namespace {
void ReleaseLocked(std::shared_ptr<IModelManager>& model_manager)
{
    if (model_manager) {
        (void)model_manager->DeinitWeights();
        model_manager->DeInit();
    }
    model_manager.reset();
}

// 原地更新族工位命名常量与解析（kPastKeyPrefix/ParseKvInputLayer/
// IsInPlaceUpdateSlotName 等）已升入 model_executor.h（问题 3 方案 b），
// 供 CorrectTensorDescs 修正/保护与 UT 直接共用。

const char* DataTypeName(DataType t)
{
    switch (t) {
        case DataType::UINT8:
            return "UINT8";
        case DataType::FLOAT32:
            return "FLOAT32";
        case DataType::FLOAT16:
            return "FLOAT16";
        case DataType::INT32:
            return "INT32";
        case DataType::INT8:
            return "INT8";
        case DataType::INT16:
            return "INT16";
        case DataType::BOOL:
            return "BOOL";
        case DataType::INT64:
            return "INT64";
        case DataType::UINT32:
            return "UINT32";
        case DataType::DOUBLE:
            return "DOUBLE";
        case DataType::RESERVED:
            return "RESERVED";
    }
    return "unknown";
}

const char* FormatName(Format f)
{
    switch (f) {
        case Format::NCHW:
            return "NCHW";
        case Format::NHWC:
            return "NHWC";
        case Format::ND:
            return "ND";
        case Format::RESERVED:
            return "RESERVED";
    }
    return "unknown";
}

/// 打印 NDTensorDesc 列表（每个 desc 一行，格式统一）
void LogTensorDescs(const char* label, const std::vector<hiai::NDTensorDesc>& descs)
{
    auto dimsStr = [](const std::vector<int32_t>& dims) -> std::string {
        std::string s = "[";
        for (std::size_t d = 0; d < dims.size(); ++d) {
            if (d)
                s += ",";
            s += std::to_string(dims[d]);
        }
        s += "]";
        return s;
    };
    CANN_LLM_LOGI("loader.load.%s_descs: count=%zu", label, descs.size());
    for (std::size_t i = 0; i < descs.size(); ++i) {
        const auto& d = descs[i];
        CANN_LLM_LOGD("    [%zu] name=%-24s shape=%-20s dtype=%-8s format=%s", i, d.tensorName.c_str(),
                      dimsStr(d.dims).c_str(), DataTypeName(d.dataType), FormatName(d.format));
    }
}
// ── TensorDesc 维度修正参数（仅 ModelExecutor 内部使用）──
struct ShapeCorrectionParams {
    int seq_len = 0;
    int batch_size = 0;
    int kv_cache_max_len = 0;
    int hidden_size = 0;
    int vocab_align_size = 0;
    bool embedding_per_group = false;
    int embedding_group_size = 0;
    int sliding_window = 0;
    int prefill_max = 0; ///< P = max(prefill_len)（新契 C=W+P 入参）
    /// 新图契约 flag（meta sliding_window_head_dim）：>0 = 新契——
    /// mask_sliding 列宽 = C = W+P（滑窗层 KV 槽空间）；0 = 旧契——列宽 kv_max。
    int sliding_window_head_dim = 0;
};

ShapeCorrectionParams DeriveCorrectionParams(const ModelConfig& config, int seq_len)
{
    ShapeCorrectionParams correction;
    correction.seq_len = seq_len;
    correction.batch_size = (seq_len > 1) ? 1 : config.batch_num;
    correction.kv_cache_max_len = config.kv_cache_max_len;
    correction.hidden_size = config.hidden_size;
    correction.vocab_align_size = static_cast<int>(config.vocab_align_size);
    correction.embedding_per_group = config.embedding_per_group;
    correction.embedding_group_size = config.embedding_group_size;
    correction.sliding_window = config.sliding_window;
    correction.prefill_max = PrefillMaxLen(config.prefill_len);
    correction.sliding_window_head_dim = config.sliding_window_head_dim;
    return correction;
}

/// 按 tensor 名称修正输入/输出 NDTensorDesc 的 dims。
/// 仅修正已知名称的工位；未匹配的工位保持 om 原始 dims 不变。
/// 原地更新族工位（past_key_in*/past_value_in*/past_key_values_*（HF 命名，
/// 问题 3 方案 b）/conv_state_in_layer_*/rstate_in_layer_*）跳过——由池绑定逻辑处理。
/// 输出侧 logits 通过 shape 特征识别（4D 且 dims[3] == vocab_align_size），
/// 修正后恢复为 3D [batch, 1, vocab_align_size]——GatherV2(output_pos) 单位置
/// 输出，第二维恒 1，不随 seq_len 扩张（decode 同形）。
void CorrectTensorDescs(std::vector<hiai::NDTensorDesc>& in_descs, std::vector<hiai::NDTensorDesc>& out_descs,
                        const ShapeCorrectionParams& correction)
{
    auto dimsStr = [](const std::vector<int32_t>& dims) -> std::string {
        std::string s = "[";
        for (std::size_t d = 0; d < dims.size(); ++d) {
            if (d)
                s += ",";
            s += std::to_string(dims[d]);
        }
        s += "]";
        return s;
    };

    // ── 输入侧修正 ──
    for (auto& d : in_descs) {
        // 原地更新族工位跳过（KV 旧约定族 + HF 命名族 + Qwen3.5 线性状态族；
        // support_qwen3.5.md §4.5-6：池绑定逻辑处理，dims 以 om 图为权威——
        // 即便不显式跳过也无模式可命中，此处显式化防未来模式误伤）
        if (IsInPlaceUpdateSlotName(d.tensorName)) {
            continue;
        }

        std::vector<int32_t> old_dims = d.dims;
        if (IsEmbedSlotName(d.tensorName) && correction.hidden_size > 0) {
            // 问题 1 方案 b：omc 导出名 embeddings 与规范名 input_embed 同口径修正
            d.dims = {correction.batch_size, correction.seq_len, correction.hidden_size};
        } else if (d.tensorName == "input_ids") {
            d.dims = {correction.batch_size, correction.seq_len};
        } else if (d.tensorName == "attention_mask" && correction.kv_cache_max_len > 0) {
            d.dims = {correction.batch_size, 1, correction.seq_len, correction.kv_cache_max_len};
        } else if (d.tensorName == "mask_full" && correction.kv_cache_max_len > 0) {
            // Gemma4 全因果层 mask（docs/mask_fill.md D8）：BOOL 值域 0=attend/
            // 1=屏蔽；列轴 = kv_max 锚定环物理槽（MapSlot 语义），恒全 kv_max 宽。
            d.dims = {correction.batch_size, 1, correction.seq_len, correction.kv_cache_max_len};
        } else if (d.tensorName == "mask_sliding" && correction.kv_cache_max_len > 0) {
            // Gemma4 滑窗层 mask（sliding_kv_seq_dim_rebind D5，mask-KV 对偶）：
            // 列数必须恰等于滑窗层 KV 槽空间宽度。新契（sw_head_dim>0）：
            // 列宽 = C = W+P（纯模小环，列物理槽 = 逻辑位 j%C，对应
            // InputAssembler::WriteBoolMask kSliding 环）；旧契：列宽 =
            // kv_max（滑窗层与 full 层同槽域，kAnchored MapSlot 环，
            // pre-b688221 语义）。
            const std::size_t width = (correction.sliding_window_head_dim > 0) ?
                                          SlidingLayerWidth(correction.sliding_window, correction.prefill_max) :
                                          static_cast<std::size_t>(correction.kv_cache_max_len);
            d.dims = {correction.batch_size, 1, correction.seq_len, static_cast<int32_t>(width)};
        } else if (d.tensorName == "position_ids") {
            d.dims = {correction.batch_size, correction.seq_len};
        } else if (d.tensorName == "new_kv_cache_pos") {
            if (correction.batch_size > 1) {
                d.dims = {correction.batch_size, 1};
            } else {
                d.dims = {correction.seq_len};
            }
        } else if (d.tensorName == "sliding_kv_pos") {
            // 新图滑窗层专用 scatter 写位工位（sliding_kv_pos，与 new_kv_cache_pos
            // 成对，1027 首图实证）：修形口径与 new_kv_cache_pos 同构——
            // batch>1 → [batch,1]、单 batch → [seq_len] 一维。旧图不携带该工位
            // 零行为变化；新契（sw_hd>0）由 meta 侧 SlidingHeadDimComboValid 保证
            // C=W+P 可算，修形不依赖 sw_hd（SeqLen 工位容量只随图 seq 维走）。
            if (correction.batch_size > 1) {
                d.dims = {correction.batch_size, 1};
            } else {
                d.dims = {correction.seq_len};
            }
        } else if (IsEmbedScaleSlotName(d.tensorName)) {
            // 工位名匹配：规范名 embed_scales 优先、omc 导出别名 embeddings_scale
            // 兜底（与上方 embed 工位同口径）。
            // 参考 SetEmbeddingDequantScalesTensorDims：
            // PerGroup: {batch, seq, hidden/group_size, 1}
            // PerToken: {batch, seq, 1}（int4 未支持，U-3）
            if (correction.embedding_per_group && correction.embedding_group_size > 0 && correction.hidden_size > 0) {
                d.dims = {correction.batch_size, correction.seq_len,
                          correction.hidden_size / correction.embedding_group_size, 1};
            } else {
                d.dims = {correction.batch_size, correction.seq_len, 1};
            }
        } else if (d.tensorName == "enable_llm_config" || d.tensorName == "indices" || d.tensorName == "topk") {
            // 标量/配置类输入：修正为 1D，值取原始 dims[0]
            if (!d.dims.empty()) {
                d.dims = {d.dims[0]};
            }
        } else if (d.tensorName == "output_pos") {
            // gemma4 logits 裁剪行号（可选工位，InputAssembler 每步填 prefill
            // chunk_len-1 / decode 0；dtype 以 om 读出为准 INT32/INT64）：恒 1 维
            // 标量索引，归一为 [dims[0]]（om 导出 [1]，防 omc 输出 [1,1] 等广播形）
            if (!d.dims.empty()) {
                d.dims = {d.dims[0]};
            }
        } else {
            continue;
        }
        if (old_dims != d.dims) {
            CANN_LLM_LOGD("loader.load.correct_input: '%s' shape %s → %s", d.tensorName.c_str(),
                          dimsStr(old_dims).c_str(), dimsStr(d.dims).c_str());
        }
    }

    // ── 输出侧修正 ──
    // 输出 tensor 没有名称，logits 通过 shape 特征识别：
    // om 图错误地将所有 tensor 描述为 4D，但 vocab 维度的值保留在 dims[3] 中。
    // 第二维恒 1（不随 seq_len 扩张）：om 图 logits 工位为 GatherV2(output_pos)
    // ——图内已按 output_pos 挑出单位置 logits，实际写出量 = batch×1×vocab_align
    // （om 占位 [1,1,1,vocab_align] 的 dims[2]=1 即位置数）。若按 seq_len 建形
    // 会超额申请输出 tensor（seq=128 时 64MB vs 实际 512KB）且与图写出量不符。
    for (auto& d : out_descs) {
        if (correction.vocab_align_size > 0 && d.dims.size() == 4 && d.dims[3] == correction.vocab_align_size) {
            std::vector<int32_t> old_dims = d.dims;
            d.dims = {correction.batch_size, 1, correction.vocab_align_size};
            CANN_LLM_LOGD("loader.load.correct_output: logits shape %s → %s", dimsStr(old_dims).c_str(),
                          dimsStr(d.dims).c_str());
        }
    }
    // 最后一个输出 tensor：修正为 2D，保留原始 dims[0] 和 dims[1]。
    // Qwen3.5 保护（support_qwen3.5.md §4.5-6）：末输出若与任一 KV/线性状态
    // 输入工位 dims+dtype 全等（即原地更新族的状态形输出，≥3 维），截断会
    // 破坏输出侧按形识别 → 跳过。工位族口径 = IsInPlaceUpdateSlotName
    // （问题 3 方案 b：HF 命名 past_key_values_<层>_<0|1> 同判，否则该类图
    // 末输出被误截 [1,1,2048,256] → [1,1]，hcl CheckOutputs 长度失配 st=1）。
    // 参考 = 输入工位 desc（自包含：修正先于池 tensor 绑定执行，而 KV/状态族
    // 输入工位不参与修正、dims 保持 om 原值，与池 tensor/图输出 dims 全等
    // ——Load 期一致性校验兜底）。
    if (!out_descs.empty()) {
        auto& last = out_descs.back();
        bool stateful_shaped = false;
        for (const auto& d : in_descs) {
            if (!IsInPlaceUpdateSlotName(d.tensorName)) {
                continue; // 非原地更新族输入
            }
            if (last.dims == d.dims && last.dataType == d.dataType) {
                stateful_shaped = true;
                break;
            }
        }
        if (last.dims.size() > 2 && !stateful_shaped) {
            std::vector<int32_t> old_dims = last.dims;
            last.dims = {last.dims[0], last.dims[1]};
            CANN_LLM_LOGD("loader.load.correct_output: last output shape %s → %s", dimsStr(old_dims).c_str(),
                          dimsStr(last.dims).c_str());
        }
    }
}
} // namespace

ModelExecutor::ModelExecutor(std::shared_ptr<KvCachePool> pool, std::shared_ptr<InputAssembler> assembler,
                             std::shared_ptr<DfxCollector> dfx)
    : pool_(std::move(pool)), assembler_(std::move(assembler)), dfx_(std::move(dfx))
{}

Status ModelExecutor::AttachPool(std::shared_ptr<KvCachePool> pool)
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    pool_ = std::move(pool);
    return Status::SUCCESS;
}

Status ModelExecutor::AttachAssembler(std::shared_ptr<InputAssembler> assembler)
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    assembler_ = std::move(assembler);
    return Status::SUCCESS;
}

Status ModelExecutor::AttachStatePool(std::shared_ptr<LinearStatePool> pool)
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    state_pool_ = std::move(pool);
    return Status::SUCCESS;
}

Status ModelExecutor::DetachKvPool()
{
    // DestroyContext(活动) 消费（ADR-0002）：KV 工位回空占位、池引用置空
    // （S2 门禁保证空工位不可达 Run；模板池侧 ClearTensors 由调用方同步）。
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    for (const auto& slot_entry : kv_input_slots_) {
        input_tensors_[slot_entry[0]] = nullptr;
    }
    for (const auto& slot_entry : kv_output_slots_) {
        output_tensors_[slot_entry[0]] = nullptr;
    }
    pool_.reset();
    return Status::SUCCESS;
}

Status ModelExecutor::RebindKvPool(std::shared_ptr<KvCachePool> new_pool)
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    if (!new_pool)
        return Status::INVALID_ARGUMENT;
    // 未 Load / 非 KV 型图：无 KV 工位可换绑——Attach 口径即可（无 tensor 面）。
    if (kv_input_slots_.empty() && kv_output_slots_.empty()) {
        pool_ = std::move(new_pool);
        return Status::SUCCESS;
    }
    // 层数/层表一致性防御：池复制下各 Context 池同源构造（同 meta），
    // 失配 = 构造期 bug，宁拒不混绑。比对锚 = 本执行器 Load 记账（kv_input_slots_
    // 数/2），不用旧 pool_——DestroyContext(活动) 后 pool_ 已置空（回无活动态），
    // 后续 SetContext 到新 Context 必须照常可绑（S6→新 Set 无中间态）。
    const int loaded_kv_layers = static_cast<int>(kv_input_slots_.size() / 2);
    if (new_pool->NumKvLayers() != loaded_kv_layers) {
        CANN_LLM_LOGE("executor.rebind: kv layer count mismatch (loaded=%d, new=%d)", loaded_kv_layers,
                      new_pool->NumKvLayers());
        return Status::INVALID_ARGUMENT;
    }
    // KV 族工位条目逐个替换（SetContext 工位刷新原语，ADR-0002：Load 期记的
    // kv_*_slots_ 下标 + 空占位，O(工位数) 指针替换）。新池 tensor 必须已
    // Ensure（CreateContext 按模板池 desc 账本实建，同源等形；空 = 半建池 =
    // all-or-nothing 已否决，此为二次防御）：取 tensor 时顺带复验其 desc 与
    // Load 期账本镜像 kv_slot_descs_ 一致（execution 侧第二道把关联）。
    for (std::size_t si = 0; si < kv_input_slots_.size(); ++si) {
        const auto& slot_entry = kv_input_slots_[si];
        auto t = slot_entry[2] ? new_pool->VTensor(static_cast<int>(slot_entry[1])) :
                                 new_pool->KTensor(static_cast<int>(slot_entry[1]));
        if (!t) {
            CANN_LLM_LOGE("executor.rebind: new pool %s tensor missing (layer=%zu)", slot_entry[2] ? "V" : "K",
                          slot_entry[1]);
            return Status::FAILURE;
        }
        if (si < kv_slot_descs_.size()) {
            const auto& cur = t->GetTensorDesc();
            const auto& ledger_desc = kv_slot_descs_[si];
            if (cur.dims != ledger_desc.dims || cur.dataType != ledger_desc.dataType) {
                CANN_LLM_LOGE("executor.rebind: new pool %s tensor desc mismatch "
                              "(layer=%zu, dims/dtype differs from Load ledger)",
                              slot_entry[2] ? "V" : "K", slot_entry[1]);
                return Status::INVALID_ARGUMENT;
            }
        }
        input_tensors_[slot_entry[0]] = std::move(t);
    }
    for (std::size_t si = 0; si < kv_output_slots_.size(); ++si) {
        const auto& slot_entry = kv_output_slots_[si];
        auto t = slot_entry[2] ? new_pool->VTensor(static_cast<int>(slot_entry[1])) :
                                 new_pool->KTensor(static_cast<int>(slot_entry[1]));
        if (!t) {
            CANN_LLM_LOGE("executor.rebind: new pool %s tensor missing (layer=%zu)", slot_entry[2] ? "V" : "K",
                          slot_entry[1]);
            return Status::FAILURE;
        }
        // 输出与输入同对象（同一 Context 池同层 tensor），复验共用同序镜像
        // （kv_output_slots_ 与 kv_input_slots_ 同层同 is_value 序，
        // 下标 si 与账本镜像一一对应）。
        if (si < kv_slot_descs_.size()) {
            const auto& cur = t->GetTensorDesc();
            const auto& ledger_desc = kv_slot_descs_[si];
            if (cur.dims != ledger_desc.dims || cur.dataType != ledger_desc.dataType) {
                CANN_LLM_LOGE("executor.rebind: new pool %s output tensor desc mismatch "
                              "(layer=%zu, dims/dtype differs from Load ledger)",
                              slot_entry[2] ? "V" : "K", slot_entry[1]);
                return Status::INVALID_ARGUMENT;
            }
        }
        output_tensors_[slot_entry[0]] = std::move(t);
    }
    pool_ = std::move(new_pool);
    return Status::SUCCESS;
}

Status ModelExecutor::Load(const ArtifactSource& omc, const std::string& weight, const ModelConfig& config, int seq_len,
                           uint32_t share_weights_model_id)
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    ReleaseLocked(model_manager_);
    input_tensors_.clear();
    output_tensors_.clear();
    input_name_index_.clear();
    kv_input_slots_.clear();
    kv_output_slots_.clear();
    kv_slot_descs_.clear();
    logits_index_ = 0;
    model_id_ = 0;

    // 容量校验（加载层，TC_106；解析层降级后回退的桩默认值须通过本域）
    if (config.kv_cache_max_len <= 0 || config.num_hidden_layers <= 0) {
        CANN_LLM_LOGW("loader.load: invalid config (kv_cache_max_len=%d, num_hidden_layers=%d)",
                      config.kv_cache_max_len, config.num_hidden_layers);
        return Status::INVALID_ARGUMENT;
    }

    // 加载耗时观测：graph 段 = 自此起至工位建立完毕（含
    // restore/init/descs/修正/tensor 申请/池绑定）；weights 段 = InitWeights。
    // 仅成功路径输出（失败路径的信息不完整、无观测价值，log.h 无级联抑制）。
    const std::int64_t graph_start = NowUs();

    // 1) CreateModelManager（backend 工厂，NDK 通路；无参——设备路由在 backend
    //    内部经 NDK compilation 构建时处理）。
    model_manager_ = hiai::CreateModelManager();
    if (!model_manager_) {
        CANN_LLM_LOGE("loader.load.model_manager: CreateModelManager returned null");
        return Status::FAILURE;
    }

    // 2) CreateBuiltModel + Restore（图位双态：buffer 优先（D-15））。
    //    参考 model_executor_for_generation.cpp：buffer 态经 CreateLocalBuffer
    //    包装后 RestoreFromBuffer；path 态 RestoreFromFile。
    //    失败路径须 ReleaseLocked(model_manager_) 维持"model_manager_ 非空 ⇒ 已加载"不变量。
    std::shared_ptr<IBuiltModel> built = hiai::CreateBuiltModel();
    if (!built) {
        CANN_LLM_LOGE("loader.load.built_model: CreateBuiltModel returned null");
        ReleaseLocked(model_manager_);
        return Status::FAILURE;
    }
    if (omc.data.data != nullptr && omc.data.size > 0) {
        if (!omc.path.empty()) {
            CANN_LLM_LOGW("artifact.dual_state: omc buffer 优先，path 忽略（D-15）");
        }
        // CreateLocalBuffer 形参 void*（非拥有包装，shouldOwn=false）；
        // Restore 只读解析，const 视图侧 const_cast 桥接（对外 API Buffer.data 保持 const）。
        std::shared_ptr<IBuffer> buffer = hiai::CreateLocalBuffer(const_cast<void*>(omc.data.data), omc.data.size);
        if (!buffer) {
            CANN_LLM_LOGE("loader.load.restore: CreateLocalBuffer returned null");
            ReleaseLocked(model_manager_);
            return Status::FAILURE;
        }
        if (built->RestoreFromBuffer(buffer) != hiai::SUCCESS) {
            CANN_LLM_LOGE("loader.load.restore: RestoreFromBuffer failed (invalid omc buffer)");
            ReleaseLocked(model_manager_);
            return Status::FAILURE;
        }
    } else if (!omc.path.empty()) {
        if (built->RestoreFromFile(omc.path.c_str()) != hiai::SUCCESS) {
            CANN_LLM_LOGE("loader.load.restore: RestoreFromFile failed (path=%s)", omc.path.c_str());
            ReleaseLocked(model_manager_);
            return Status::FAILURE;
        }
    } else {
        CANN_LLM_LOGE("loader.load.restore: omc 图位双态均空（无图可加载）");
        ReleaseLocked(model_manager_);
        return Status::FAILURE;
    }

    // SetName（须在 Restore 后：NDK 实现在 compilation 未构造时拒绝）。
    // 名字仅元数据（日志）：path 态取文件名，buffer 态（无 path）用默认名。
    std::string model_name = "llm_model";
    if (!omc.path.empty()) {
        const std::size_t slash = omc.path.find_last_of('/');
        model_name = (slash == std::string::npos) ? omc.path : omc.path.substr(slash + 1);
        if (model_name.empty())
            model_name = "llm_model";
    }
    built->SetName(model_name);

    // 3) Init（backend：绑定 model+device；perfMode/bandMode 显式设置
    //    （EXTREME_HIGH/HIGH——perf 优先拉满；bandMode 仅支持 UNSET/HIGH），
    //    cache 策略仍留默认）。
    //    权重共享绑定：>0 在 Init 之前填入（SetInitOption 于 compilation 构造
    //    executor 前生效）；0 = 不绑定，默认加载路径不受老 ROM 缺符号影响。
    ModelInitOptions init_opts;
    init_opts.perfMode = hiai::PerfMode::EXTREME_HIGH;
    init_opts.bandMode = hiai::PerfMode::HIGH;
    init_opts.shareWeightsModelId = share_weights_model_id;
    if (share_weights_model_id != 0) {
        // 附记本模型 weightDir：分离部署（prefill/decode 各传各目录）与共用
        // 部署的加载归因排查（docs/separate_pd_weights.md §4.2）
        CANN_LLM_LOGI("loader.load.share_weights: binding to source model %u (weight=%s)", share_weights_model_id,
                      weight.c_str());
    }
    auto be_st = model_manager_->Init(init_opts, built, nullptr);
    if (be_st != hiai::SUCCESS) {
        CANN_LLM_LOGE("loader.load.init: backend Init failed (st=%d)", static_cast<int>(be_st));
        ReleaseLocked(model_manager_);
        return Status::FAILURE;
    }

    // 4) GetInput/OutputTensorDescs（Init 后；空 → FAILURE——参考实现同）。
    auto in_descs = built->GetInputTensorDescs();
    auto out_descs = built->GetOutputTensorDescs();
    LogTensorDescs("input", in_descs);
    LogTensorDescs("output", out_descs);
    if (in_descs.empty() || out_descs.empty()) {
        CANN_LLM_LOGE("loader.load.descs: input/output tensor descs empty (invalid omc)");
        ReleaseLocked(model_manager_);
        return Status::FAILURE;
    }

    // TensorDesc 维度修正：om 图导出的 dims 为占位值，根据 config + seq_len 修正。
    // seq_len <= 0 时跳过修正（保留 om 原始 dims）。
    if (seq_len > 0) {
        auto correction = DeriveCorrectionParams(config, seq_len);
        CorrectTensorDescs(in_descs, out_descs, correction);
        LogTensorDescs("input_corrected", in_descs);
        LogTensorDescs("output_corrected", out_descs);
    } else {
        CANN_LLM_LOGW("loader.load: seq_len=%d, skipping tensorDesc correction", seq_len);
    }

    // 5) GetModelID（model_manager Init 之后；存成员供权重共享绑定消费——decode 执行器
    //    Load 时绑定为 shareWeightsModelId）+ 经 TensorManager 代理申请 input/output tensor。
    //    KV 工位（past_key_in<层>/past_value_in<层>，HF 命名族
    //    past_key_values_<层>_<0|1> 同口径——问题 3 方案 b）例外（ADR-0002）：
    //    不申请 tensor、不建池显存——经 EnsureKvDesc 只记 desc 账本（模板池），
    //    input_tensors_ 对应槽位 push 空占位 + kv_input_slots_ 记账 +
    //    kv_slot_descs_ 账本镜像；KV tensor 实建延后到 CreateContext，SetContext
    //    经 RebindKvPool 把空占位填实（om 原地更新 KV；形状/dtype 权威 = om 图
    //    工位 desc，meta 仅作 HF 期望比对——R1-R3 校验整体留在 Load，Q7 裁定）。
    model_id_ = model_manager_->GetModelID();
    const uint32_t model_id = model_id_;
    auto& tm = TensorManager::Instance();
    int kv_input_slots = 0;
    int conv_input_slots = 0; // Qwen3.5 状态工位族内计数（support_qwen3.5.md §4.5）
    int r_input_slots = 0;
    for (const auto& d : in_descs) {
        bool is_value = false;
        const int layer = ParseKvInputLayer(d.tensorName, &is_value);
        if (layer == -2) {
            CANN_LLM_LOGE("loader.load.kv_input: malformed KV slot name '%s' "
                          "(expect past_key_in<layer> / past_key_values_<layer>_<0|1>)",
                          d.tensorName.c_str());
            ReleaseLocked(model_manager_);
            return Status::INVALID_ARGUMENT;
        }
        if (layer >= 0) {
            // 层型↔图一致（Qwen3.5，§4.5-4）：KV 工位层号须为建池层
            // （linear_attention 层无 KV 工位；layer_types 空 = 未声明层型，
            // 全层建池，现状行为）
            if (!config.layer_types.empty() &&
                (layer >= static_cast<int>(config.layer_types.size()) ||
                 config.layer_types[static_cast<std::size_t>(layer)] == "linear_attention")) {
                CANN_LLM_LOGE("loader.load.kv_input: layer %d is linear_attention per "
                              "layer_types - no KV slot (slot '%s')",
                              layer, d.tensorName.c_str());
                ReleaseLocked(model_manager_);
                return Status::INVALID_ARGUMENT;
            }
            // KV desc 按图记账/复验（ADR-0002 + Cache_pool_simplify.md R1/R2/R3）——
            // EnsureKvDesc：首图 Load 记 desc 账本（不建 tensor 不开显存；om 图
            // 工位 desc 为形状唯一权威），次图复验 dims+dtype 严格一致（失配
            // INVALID_ARGUMENT）；附加 HF 期望分档校验在池内（kv_len/batch 维
            // 失配致命终止，heads/末维/dims≠4 告警一次后继续）。
            // 失败 → Load 失败 ⇒ pipeline 终止初始化。
            if (pool_) {
                Status ensured = pool_->EnsureKvDesc(layer, is_value, d);
                if (!ensured.ok()) {
                    CANN_LLM_LOGE("loader.load.kv_input: EnsureKvDesc failed for '%s' "
                                  "(layer=%d, st=%d)",
                                  d.tensorName.c_str(), layer, ensured.code);
                    ReleaseLocked(model_manager_);
                    return ensured;
                }
            }
            // 空占位 + 记账（ADR-0002）：tensor 建点在 CreateContext；SetContext
            // 经 RebindKvPool 按本下标填实。无池（host/stub 无 KV 工位面）→
            // 同样只记空占位。
            input_tensors_.push_back(nullptr);
            kv_input_slots_.push_back({input_tensors_.size() - 1, static_cast<std::size_t>(layer),
                                       is_value ? 1u : 0u}); // SetContext 换绑记账（R9 沿用）
            kv_slot_descs_.push_back(d);                     // Load 期 desc 账本镜像（换绑时复验）
            ++kv_input_slots;
            continue;
        }
        // ── Qwen3.5 线性状态工位（§4.5-3）：conv_state_in_layer_<x> /
        //    rstate_in_layer_<x> → EnsureStateTensor（惰性创建/跨图校验）后绑池 tensor
        int state_kind = -1;
        const int state_layer = ParseStateInputLayer(d.tensorName, &state_kind);
        if (state_layer == -2) {
            CANN_LLM_LOGE("loader.load.state_input: malformed state slot name '%s' "
                          "(expect conv_state_in_layer_<layer> / rstate_in_layer_<layer>)",
                          d.tensorName.c_str());
            ReleaseLocked(model_manager_);
            return Status::INVALID_ARGUMENT;
        }
        if (state_layer >= 0) {
            // 图带状态工位而配置未声明 Qwen3.5（layer_types 空 → 未建状态池）
            // → 显式报错，不静默兼容（§4.5-4）
            if (!state_pool_) {
                if (config.layer_types.empty()) {
                    CANN_LLM_LOGE("loader.load.state_input: state slot '%s' but layer_types "
                                  "absent (Qwen3.5 config not declared)",
                                  d.tensorName.c_str());
                    ReleaseLocked(model_manager_);
                    return Status::INVALID_ARGUMENT;
                }
                CANN_LLM_LOGE("loader.load.state_input: state pool missing (slot '%s')", d.tensorName.c_str());
                ReleaseLocked(model_manager_);
                return Status::FAILURE; // 接线缺失（同 KV 池缺失口径）
            }
            const auto kind = state_kind == 0 ? LinearStatePool::StateKind::Conv : LinearStatePool::StateKind::R;
            // Ensure：无则按图 desc 惰性创建（创建即整体清零），有则校验双图
            // dims/dtype 一致（失配 INVALID_ARGUMENT）
            Status es = state_pool_->EnsureStateTensor(state_layer, kind, d);
            if (!es.ok()) {
                ReleaseLocked(model_manager_);
                return es;
            }
            auto t = state_kind == 0 ? state_pool_->ConvTensor(state_layer) : state_pool_->RTensor(state_layer);
            if (!t) { // 防御（Ensure 成功 ⇒ tensor 就绪）
                ReleaseLocked(model_manager_);
                return Status::FAILURE;
            }
            input_tensors_.push_back(t);
            (state_kind == 0) ? ++conv_input_slots : ++r_input_slots;
            continue;
        }
        std::shared_ptr<ITensorBuffer> t;
        Status t_st = tm.CreateTensor(d, static_cast<int>(model_id), &t);
        if (!t_st.ok()) {
            CANN_LLM_LOGE("loader.load.create_tensor: input '%s' create failed (st=%d)", d.tensorName.c_str(),
                          t_st.code);
            ReleaseLocked(model_manager_);
            return Status::FAILURE;
        }
        input_tensors_.push_back(t);
    }
    // KV 型 om：输入工位数须与池建池层数一致（meta_json ↔ om 图；
    // Qwen3.5 稀疏口径：NumKvLayers = 建池层数（full+sliding；linear 不建），
    // 空 layer_types 时
    // == num_hidden_layers，与旧稠密校验逐值一致，§4.5-4）
    if (kv_input_slots > 0 && (!pool_ || kv_input_slots != 2 * pool_->NumKvLayers())) {
        CANN_LLM_LOGE("loader.load.kv_input: KV slot count mismatch (om=%d, expect=2x%d)", kv_input_slots,
                      pool_ ? pool_->NumKvLayers() : -1);
        ReleaseLocked(model_manager_);
        return Status::INVALID_ARGUMENT;
    }
    // Qwen3.5 状态工位数校验（§4.5-4）：状态池在场 → conv/r 输入工位数须与
    // linear 层数一致（prefill/decode 双图均须声明全部 linear 层状态工位）。
    // 池不在场但有状态工位已在输入循环显式 INVALID_ARGUMENT。
    if (state_pool_ && 2 * state_pool_->NumLinearLayers() != conv_input_slots + r_input_slots) {
        CANN_LLM_LOGE("loader.load.state_input: state slot count mismatch "
                      "(om conv=%d r=%d, expect=2x%d)",
                      conv_input_slots, r_input_slots, state_pool_->NumLinearLayers());
        ReleaseLocked(model_manager_);
        return Status::INVALID_ARGUMENT;
    }
    // 建立 name→index 映射（InputAssembler 按名查工位）
    for (std::size_t i = 0; i < in_descs.size(); ++i) {
        input_name_index_[in_descs[i].tensorName] = i;
    }
    // 输出侧 KV 工位：名称未知，按 dims+dtype 识别（仅 KV 型 om）。ADR-0002
    // 改造（F4/Q13 落地难关）：模板池 Load 期无活体 tensor——原"按池 K tensor
    // 形状识别"恒不命中且计数守恒必挂；比对源改为 **Load 期已记账 desc 账本**
    // （kv_slot_descs_ 镜像 = 输入侧 EnsureKvDesc 首记族，输出循环在输入循环
    // 之后，账本已就绪；dims+dataType 全等语义不变）。命中后按层序
    // [k0,v0,k1,v1,...] push 空占位 + kv_output_slots_ 记账 + 账本镜像校验
    // （第 nth 位输出 desc 与账本第 nth 位全等——防撞形错绑，语义照旧严格；
    // 真绑定同 tensor 在 SetContext 的 RebindKvPool 完成——输出与输入同
    // Context 池同层 K/V，om 原地更新 KV 语义不变）。
    // 稀疏池（§4.5-5）：层序经 kv_input_slots_ 记账序隐含
    // （[k(l0),v(l0),k(l1),v(l1),...]，li = 建池层全局层号升序）。
    // gemma4 滑窗异构 KV：各层 dims 可不同（sliding_kv_seq_dim_rebind 新契
    // sliding 层 = [b,h,W+P,sw_head_dim]、旧契与 full 层同形 [b,h,kv_max,
    // head_dim]）——族形 = 命中账本内任一 K desc 形状。
    const bool kv_om = kv_input_slots > 0;
    auto is_kv_family_shape = [&](const hiai::NDTensorDesc& d) -> bool {
        for (std::size_t i = 0; i < kv_slot_descs_.size(); i += 2) {
            // 只比 K 位（is_value=0.desc 镜像序 = kv_input_slots_ 序 [k0,v0,...]）
            const auto& ledger_desc = kv_slot_descs_[i];
            if (d.dims == ledger_desc.dims && d.dataType == ledger_desc.dataType) {
                return true;
            }
        }
        return false;
    };
    // Qwen3.5 状态形输出识别参考（§4.5-5）：状态池首个已就绪层的 Conv/RTensor
    // desc（输出绑定在输入循环之后——本图声明的层已 Ensure 就绪）。conv 与 r
    // 撞形（dims+dtype 相同）时族间不可按形区分，按层内 conv 先 r 交替消歧
    // （与 KV k/v 同形消歧同构；层内顺序假设 [conv_x, r_x]，族间顺序无关）。
    const bool state_om = (conv_input_slots + r_input_slots) > 0;
    const TensorDesc* conv_ref = nullptr;
    const TensorDesc* r_ref = nullptr;
    if (state_om && state_pool_) {
        for (int layer : state_pool_->LinearLayerIds()) {
            if (!conv_ref) {
                if (auto t = state_pool_->ConvTensor(layer)) {
                    conv_ref = &t->GetTensorDesc();
                }
            }
            if (!r_ref) {
                if (auto t = state_pool_->RTensor(layer)) {
                    r_ref = &t->GetTensorDesc();
                }
            }
            if (conv_ref && r_ref)
                break;
        }
    }
    const bool state_same_shape = conv_ref && r_ref && conv_ref->dims == r_ref->dims &&
                                  conv_ref->dataType == r_ref->dataType;
    int kv_output_slots = 0;
    int conv_output_slots = 0;
    int r_output_slots = 0;
    for (const auto& d : out_descs) {
        if (kv_om && pool_ && is_kv_family_shape(d)) {
            const int nth = kv_output_slots / 2;
            if (!pool_ || nth >= pool_->NumKvLayers() || kv_output_slots >= static_cast<int>(kv_slot_descs_.size())) {
                CANN_LLM_LOGE("loader.load.kv_output: layer index out of range "
                              "(nth=%d, kv_layers=%d, ledger=%zu)",
                              nth, pool_ ? pool_->NumKvLayers() : -1, kv_slot_descs_.size());
                ReleaseLocked(model_manager_);
                return Status::INVALID_ARGUMENT;
            }
            // 账本镜像第 kv_output_slots 位 = 输出序位应绑的 K/V desc
            // （kv_input_slots_ 记账序 [k0,v0,k1,v1,...]，输出按同序消歧——
            // 本校验兼防撞形：族形命中但序位 desc 不全等 → 显式报错）
            const auto& ledger_desc = kv_slot_descs_[static_cast<std::size_t>(kv_output_slots)];
            const bool is_v = kv_output_slots % 2 == 1;
            const int layer = static_cast<int>(kv_input_slots_[static_cast<std::size_t>(kv_output_slots)][1]);
            if (d.dims != ledger_desc.dims || d.dataType != ledger_desc.dataType) {
                CANN_LLM_LOGE("loader.load.kv_output: KV output shape mismatch at nth=%d "
                              "(layer=%d, slot '%s') - expect ledger dims on this slot",
                              nth, layer, d.tensorName.c_str());
                ReleaseLocked(model_manager_);
                return Status::INVALID_ARGUMENT;
            }
            // 空占位 + 记账（ADR-0002）：SetContext 经 RebindKvPool 填实同层
            // K/V（与输入同 tensor 对象 = om 原地更新语义不变）
            output_tensors_.push_back(nullptr);
            kv_output_slots_.push_back({output_tensors_.size() - 1, static_cast<std::size_t>(layer),
                                        is_v ? 1u : 0u}); // SetContext 换绑记账（R9 沿用）
            ++kv_output_slots;
            continue;
        }
        // Qwen3.5 状态形输出：dims+dtype 与 conv/r 参考全等 → 绑池 tensor
        // （与输入同 buffer，om 原地更新状态；无光标无回绕）
        if (state_om && (conv_ref || r_ref)) {
            int idx = -1;
            LinearStatePool::StateKind kind = LinearStatePool::StateKind::Conv;
            bool hit = false;
            if (conv_ref && d.dims == conv_ref->dims && d.dataType == conv_ref->dataType) {
                if (state_same_shape) {
                    const int seq = conv_output_slots + r_output_slots;
                    idx = seq / 2;
                    kind = (seq % 2 == 0) ? LinearStatePool::StateKind::Conv : LinearStatePool::StateKind::R;
                } else {
                    idx = conv_output_slots;
                    kind = LinearStatePool::StateKind::Conv;
                }
                hit = true;
            } else if (r_ref && d.dims == r_ref->dims && d.dataType == r_ref->dataType) {
                idx = r_output_slots;
                kind = LinearStatePool::StateKind::R;
                hit = true;
            }
            if (hit) {
                const auto& ids = state_pool_->LinearLayerIds();
                if (idx < 0 || idx >= static_cast<int>(ids.size())) {
                    CANN_LLM_LOGE("loader.load.state_output: layer index out of range "
                                  "(idx=%d, linear_layers=%zu)",
                                  idx, ids.size());
                    ReleaseLocked(model_manager_);
                    return Status::INVALID_ARGUMENT;
                }
                const int layer = ids[static_cast<std::size_t>(idx)];
                std::shared_ptr<ITensorBuffer> t = kind == LinearStatePool::StateKind::Conv ?
                                                       state_pool_->ConvTensor(layer) :
                                                       state_pool_->RTensor(layer);
                if (!t) {
                    CANN_LLM_LOGE("loader.load.state_output: pool tensor missing (layer=%d)", layer);
                    ReleaseLocked(model_manager_);
                    return Status::FAILURE;
                }
                output_tensors_.push_back(t);
                (kind == LinearStatePool::StateKind::Conv) ? ++conv_output_slots : ++r_output_slots;
                continue;
            }
        }
        std::shared_ptr<ITensorBuffer> t;
        Status t_st = tm.CreateTensor(d, static_cast<int>(model_id), &t);
        if (!t_st.ok()) {
            CANN_LLM_LOGE("loader.load.create_tensor: output '%s' create failed (st=%d)", d.tensorName.c_str(),
                          t_st.code);
            ReleaseLocked(model_manager_);
            return Status::FAILURE;
        }
        output_tensors_.push_back(t);
    }
    // 输出侧 KV 形工位数须与输入侧一致
    if (kv_om && kv_output_slots != kv_input_slots) {
        CANN_LLM_LOGE("loader.load.kv_output: KV-shaped output count mismatch (out=%d, in=%d)", kv_output_slots,
                      kv_input_slots);
        ReleaseLocked(model_manager_);
        return Status::INVALID_ARGUMENT;
    }
    // Qwen3.5 状态形输出族内计数校验（§4.5-5）：conv/r 各自 输出 == 输入
    if (state_om && (conv_output_slots != conv_input_slots || r_output_slots != r_input_slots)) {
        CANN_LLM_LOGE("loader.load.state_output: state-shaped output count mismatch "
                      "(conv out=%d/in=%d, r out=%d/in=%d)",
                      conv_output_slots, conv_input_slots, r_output_slots, r_input_slots);
        ReleaseLocked(model_manager_);
        return Status::INVALID_ARGUMENT;
    }
    // 5b) logits 工位定位（Q13 裁定：恒钉 output[0] 为 logits 工位。
    //    真机 gemma4-quant 已以 debug dump 对账确认 0 号位即 logits；若实测
    //    非 logits（图布局变化），回退 shape 扫描并向用户回报（design §8.2 开放项）。
    logits_index_ = 0;

    // 6) 加载权重（图与权重分离模型；Init 只加载图结构，权重经 InitWeights 单独加载）
    if (!weight.empty()) {
        const std::int64_t weights_start = NowUs();
        CANN_LLM_LOGI("perf.load.graph: phase=%s elapsed=%.3f ms", seq_len > 1 ? "prefill" : "decode",
                      UsToMs(weights_start - graph_start));
        auto w_st = model_manager_->InitWeights(weight);
        if (w_st != hiai::SUCCESS) {
            CANN_LLM_LOGE("loader.load.init_weights: backend InitWeights failed (st=%d, weight=%s)",
                          static_cast<int>(w_st), weight.c_str());
            ReleaseLocked(model_manager_);
            return Status::FAILURE;
        }
        CANN_LLM_LOGI("perf.load.weights: phase=%s elapsed=%.3f ms", seq_len > 1 ? "prefill" : "decode",
                      UsToMs(NowUs() - weights_start));
    } else {
        // 无权重路径（图权重合一/测试形态）：graph 段耗时在此收口
        CANN_LLM_LOGI("perf.load.graph: phase=%s elapsed=%.3f ms", seq_len > 1 ? "prefill" : "decode",
                      UsToMs(NowUs() - graph_start));
    }

    config_ = config;
    return Status::SUCCESS;
}

Status ModelExecutor::Unload()
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    ReleaseLocked(model_manager_);
    input_tensors_.clear();
    output_tensors_.clear();
    input_name_index_.clear();
    kv_input_slots_.clear();
    kv_output_slots_.clear();
    kv_slot_descs_.clear();
    logits_index_ = 0;
    model_id_ = 0;
    return Status::SUCCESS;
}

Status ModelExecutor::GetConfig(ModelConfig* out) const
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (!model_manager_) {
        CANN_LLM_LOGW("executor.config: not loaded");
        return Status::INVALID_STATE;
    }
    *out = config_;
    return Status::SUCCESS;
}

Status ModelExecutor::SetInputData(const std::shared_ptr<ITensorBuffer>& data)
{
    if (!data)
        return Status::INVALID_ARGUMENT;
    // v0.13：按 name→index 查 input_embed 工位（不硬编码下标）；
    // 问题 1 方案 b：omc 导出别名 embeddings 兜底（FindEmbedSlot）
    auto it = FindEmbedSlot(input_name_index_);
    if (it == input_name_index_.end() || it->second >= input_tensors_.size() || !input_tensors_[it->second]) {
        CANN_LLM_LOGW("executor.set_input: input slot 'input_embed'/'embeddings' missing "
                      "(model not loaded or no such input)");
        return Status::INVALID_STATE;
    }
    auto& buf = input_tensors_[it->second];
    void* dst = buf->GetData();
    const void* src = data->GetData();
    std::size_t n = std::min(data->GetSize(), buf->GetSize());
    if (n && dst && src)
        std::memcpy(dst, src, n);
    return Status::SUCCESS;
}

Status ModelExecutor::Forward(const ForwardOptions& opts)
{
    if (!static_cast<bool>(*this)) {
        CANN_LLM_LOGW("executor.forward: executor not ready (model not loaded or pool/assembler missing)");
        return Status::FAILURE; // A2:未加载返 FAILURE(非 INVALID_STATE)— bbit 契约只认 SUCCESS/FAILURE
    }
    if (opts.slots.empty()) {
        CANN_LLM_LOGW("executor.forward: empty slots");
        return Status::INVALID_ARGUMENT;
    }
    if (output_tensors_.empty()) {
        CANN_LLM_LOGW("executor.forward: no output tensors (not loaded?)");
        return Status::INVALID_STATE;
    }

    std::lock_guard<std::mutex> lk(forward_mutex_);

    Status st;
    if (opts.chunk_len > 1) {
        if (opts.slots.size() != 1) {
            CANN_LLM_LOGW("executor.forward: prefill chunk requires exactly 1 slot");
            return Status::INVALID_ARGUMENT;
        }
        st = assembler_->AssemblePrefill(opts.slots[0], opts.chunk_len, input_tensors_, input_name_index_);
    } else {
        st = assembler_->AssembleDecode(opts.slots, input_tensors_, input_name_index_);
    }
    // assemble 失败细节由 InputAssembler 记录（缺工位名/KV 光标越界），此处不重复记。
    if (!st.ok())
        return st;

    if (dfx_)
        (void)dfx_->OnForwardStart(input_tensors_, opts.slots, opts.dump);
    // model_manager->Run(input_tensors_, output_tensors_) — model manager 填 output buffer 内容
    // 打印输入 tensor 名字 + shape + dtype（每个 tensor 一行）
    {
        auto dimsStr = [](const std::vector<int32_t>& dims) -> std::string {
            std::string s = "[";
            for (std::size_t d = 0; d < dims.size(); ++d) {
                if (d)
                    s += ",";
                s += std::to_string(dims[d]);
            }
            s += "]";
            return s;
        };
        CANN_LLM_LOGD("executor.forward.inputs:");
        for (const auto& t : input_tensors_) {
            const auto& desc = t->GetTensorDesc();
            CANN_LLM_LOGD("    input[%zu] name=%-24s shape=%-20s dtype=%s", &t - input_tensors_.data(),
                          desc.tensorName.c_str(), dimsStr(desc.dims).c_str(), DataTypeName(desc.dataType));
        }
    }
    // 打印输出 tensor 名字 + shape + dtype（每个 tensor 一行）
    {
        auto dimsStr = [](const std::vector<int32_t>& dims) -> std::string {
            std::string s = "[";
            for (std::size_t d = 0; d < dims.size(); ++d) {
                if (d)
                    s += ",";
                s += std::to_string(dims[d]);
            }
            s += "]";
            return s;
        };
        CANN_LLM_LOGD("executor.forward.outputs:");
        for (const auto& t : output_tensors_) {
            const auto& desc = t->GetTensorDesc();
            CANN_LLM_LOGD("    output[%zu] name=%-24s shape=%-20s dtype=%s", &t - output_tensors_.data(),
                          desc.tensorName.c_str(), dimsStr(desc.dims).c_str(), DataTypeName(desc.dataType));
        }
    }
    // 推理前后 dump 输入/输出工位（CANN_LM_DEBUG_DUMP_TENSORS 开关，env 未设
    // 零行为变化；KV 工位需额外开 CANN_LM_DEBUG_DUMP_KV）。放在 Run 前/后：
    // 前 = 送进模型的输入（assembler 已填完），后 = 模型吐出的结果。轮次：每次
    // Forward 自增（未开 dump 不递增）；同一轮 in/out dump 同号，便于逐轮精度
    // 对比成套比对 dump_prefill_in_{r}_x.bin ↔ dump_prefill_out_{r}_y.bin。
    const char* dump_mode = (opts.chunk_len > 1) ? "prefill" : "decode";
    const std::size_t dump_round = NextDumpRound();
    DebugDumpTensors(input_tensors_, "in", dump_mode, dump_round, pool_);

    auto be_st = model_manager_->Run(input_tensors_, output_tensors_);

    DebugDumpTensors(output_tensors_, "out", dump_mode, dump_round, pool_);
    if (dfx_)
        (void)dfx_->OnForwardEnd(output_tensors_.empty() ? nullptr : output_tensors_[0], 0.0);
    Status mapped = map_backend_status(be_st);
    if (!mapped.ok()) {
        CANN_LLM_LOGE("executor.forward: backend Run failed (st=%d)", static_cast<int>(be_st));
    }
    return mapped;
}

// ── ForwardRaw：精度验证通道 ─────────────────────────────────────────────────
// 跳过 InputAssembler，直接用 input_tensors_ 调 model_manager->Run()。golden 二进制文件
// 直接灌入输入工位后前向（调用方须自行填充全部输入工位），供 golden_verify 用。
Status ModelExecutor::ForwardRaw()
{
    if (!static_cast<bool>(*this)) {
        CANN_LLM_LOGW("executor.forward_raw: executor not ready (model not loaded or pool/assembler missing)");
        return Status::FAILURE;
    }
    if (output_tensors_.empty()) {
        CANN_LLM_LOGW("executor.forward_raw: no output tensors (not loaded?)");
        return Status::INVALID_STATE;
    }

    std::lock_guard<std::mutex> lk(forward_mutex_);

    // dump 输入工位（与 Forward 同口径，便于比对）
    const char* dump_mode = "raw";
    const std::size_t dump_round = NextDumpRound();
    DebugDumpTensors(input_tensors_, "in", dump_mode, dump_round, pool_);

    auto be_st = model_manager_->Run(input_tensors_, output_tensors_);

    DebugDumpTensors(output_tensors_, "out", dump_mode, dump_round, pool_);
    Status mapped = map_backend_status(be_st);
    if (!mapped.ok()) {
        CANN_LLM_LOGE("executor.forward_raw: backend Run failed (st=%d)", static_cast<int>(be_st));
    }
    return mapped;
}

std::size_t ModelExecutor::NextDumpRound()
{
    // 仅 dump 开启时递增（未开开关零行为变化）；调用点在 forward_mutex_
    // 持有区内，无并发。
    if (!DebugDumpEnabled())
        return dump_round_;
    ++dump_round_;
    return dump_round_ - 1;
}

std::shared_ptr<ITensorBuffer> ModelExecutor::Logits() const
{
    if (output_tensors_.empty())
        return nullptr;
    return logits_index_ < output_tensors_.size() ? output_tensors_[logits_index_] : output_tensors_[0];
}

std::shared_ptr<KvCachePool> ModelExecutor::Pool() const noexcept { return pool_; }

std::shared_ptr<LinearStatePool> ModelExecutor::StatePool() const noexcept { return state_pool_; }

const std::vector<std::shared_ptr<ITensorBuffer>>& ModelExecutor::InputTensors() const noexcept
{
    return input_tensors_;
}

const std::vector<std::shared_ptr<ITensorBuffer>>& ModelExecutor::OutputTensors() const noexcept
{
    return output_tensors_;
}

const std::unordered_map<std::string, std::size_t>& ModelExecutor::InputNameIndex() const noexcept
{
    return input_name_index_;
}

std::size_t ModelExecutor::InputTensorCount() const noexcept { return input_tensors_.size(); }
std::size_t ModelExecutor::OutputTensorCount() const noexcept { return output_tensors_.size(); }

ModelExecutor::operator bool() const noexcept
{
    return static_cast<bool>(model_manager_) && static_cast<bool>(pool_) && static_cast<bool>(assembler_);
}
// 注：pool_ 检查保留——host/stub 未 Load（model_manager_ 空）报 FAILURE 照旧；SetContext
// 后 pool_ 非空；DestroyContext(活动) 后 DetachKvPool 置空 pool_ → Forward
// 走"not ready"FAILURE（恰为 S6 后无活动态的正确门禁，bbit 契约兼容：bbit
// 形态 host 未 Load，model_manager_ 空先短路，不受影响）。

ModelExecutor::~ModelExecutor()
{
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    ReleaseLocked(model_manager_);
}

ModelExecutor::ModelExecutor(ModelExecutor&& other) noexcept
{
    std::lock_guard<std::mutex> lk(other.lifecycle_mutex_);
    model_manager_ = std::move(other.model_manager_);
    pool_ = std::move(other.pool_);
    state_pool_ = std::move(other.state_pool_);
    assembler_ = std::move(other.assembler_);
    dfx_ = std::move(other.dfx_);
    config_ = other.config_;
    input_tensors_ = std::move(other.input_tensors_);
    output_tensors_ = std::move(other.output_tensors_);
    input_name_index_ = std::move(other.input_name_index_);
    kv_input_slots_ = std::move(other.kv_input_slots_);
    kv_output_slots_ = std::move(other.kv_output_slots_);
    kv_slot_descs_ = std::move(other.kv_slot_descs_);
    logits_index_ = other.logits_index_;
    model_id_ = other.model_id_;
    dump_round_ = other.dump_round_;
    other.logits_index_ = 0;
    other.model_id_ = 0;
    other.dump_round_ = 0;
}

ModelExecutor& ModelExecutor::operator=(ModelExecutor&& other) noexcept
{
    if (this != &other) {
        std::lock_guard<std::mutex> lk1(lifecycle_mutex_);
        std::lock_guard<std::mutex> lk2(other.lifecycle_mutex_);
        ReleaseLocked(model_manager_);
        model_manager_ = std::move(other.model_manager_);
        pool_ = std::move(other.pool_);
        state_pool_ = std::move(other.state_pool_);
        assembler_ = std::move(other.assembler_);
        dfx_ = std::move(other.dfx_);
        config_ = other.config_;
        input_tensors_ = std::move(other.input_tensors_);
        output_tensors_ = std::move(other.output_tensors_);
        input_name_index_ = std::move(other.input_name_index_);
        kv_input_slots_ = std::move(other.kv_input_slots_);
        kv_output_slots_ = std::move(other.kv_output_slots_);
        kv_slot_descs_ = std::move(other.kv_slot_descs_);
        logits_index_ = other.logits_index_;
        model_id_ = other.model_id_;
        dump_round_ = other.dump_round_;
        other.logits_index_ = 0;
        other.model_id_ = 0;
        other.dump_round_ = 0;
    }
    return *this;
}

} // namespace cann_local_lm::engine
