// LinearStatePool 实现（support_qwen3.5.md §4.4）。
// 惰性创建 + 跨图一致性校验 + lane 切片清零；tensor 申请经 TensorManager
// 通用通道（CreateTensor），TensorManager 零改动。
#include "core/component/linear_state_pool/linear_state_pool.h"
#include "infra/log.h"

#include <cstring>

namespace cann_local_lm::engine {

LinearStatePool::LinearStatePool(const std::vector<std::string>& layer_types)
{
    for (int i = 0; i < static_cast<int>(layer_types.size()); ++i) {
        // 解析层已校验取值 ∈ {"full_attention","linear_attention"}（§4.2）；
        // 此处只关心 linear，其余（含空表）一律视为 full。
        if (layer_types[static_cast<std::size_t>(i)] == "linear_attention") {
            linear_layer_ids_.push_back(i);
        }
    }
    tensors_.resize(linear_layer_ids_.size());
}

int LinearStatePool::NumLinearLayers() const noexcept { return static_cast<int>(linear_layer_ids_.size()); }

const std::vector<int>& LinearStatePool::LinearLayerIds() const noexcept { return linear_layer_ids_; }

int LinearStatePool::GroupIndex(int layer) const noexcept
{
    for (std::size_t i = 0; i < linear_layer_ids_.size(); ++i) {
        if (linear_layer_ids_[i] == layer)
            return static_cast<int>(i);
    }
    return -1;
}

std::shared_ptr<ITensorBuffer> LinearStatePool::ConvTensor(int layer) const
{
    std::lock_guard<std::mutex> lk(mtx_);
    const int group = GroupIndex(layer);
    return group < 0 ? nullptr : tensors_[static_cast<std::size_t>(group)].first;
}

std::shared_ptr<ITensorBuffer> LinearStatePool::RTensor(int layer) const
{
    std::lock_guard<std::mutex> lk(mtx_);
    const int group = GroupIndex(layer);
    return group < 0 ? nullptr : tensors_[static_cast<std::size_t>(group)].second;
}

std::size_t LinearStatePool::TensorCount() const noexcept
{
    std::lock_guard<std::mutex> lk(mtx_);
    std::size_t n = 0;
    for (const auto& pr : tensors_) {
        n += (pr.first ? 1u : 0u) + (pr.second ? 1u : 0u);
    }
    return n;
}

Status LinearStatePool::EnsureStateTensor(int layer, StateKind kind, const TensorDesc& desc)
{
    std::lock_guard<std::mutex> lk(mtx_);
    const int group = GroupIndex(layer);
    if (group < 0) {
        CANN_LLM_LOGE("state_pool.ensure: layer %d is not linear_attention "
                      "(pool linear layers = %d) —— 图带状态工位而配置未声明",
                      layer, NumLinearLayers());
        return Status::INVALID_ARGUMENT;
    }
    auto& slot = (kind == StateKind::Conv) ? tensors_[static_cast<std::size_t>(group)].first :
                                             tensors_[static_cast<std::size_t>(group)].second;
    const char* kind_name = (kind == StateKind::Conv) ? "conv" : "r";

    // 已就绪 → 跨图一致性校验后复用（prefill/decode 双图状态形状把关点）
    if (slot) {
        const auto& cur = slot->GetTensorDesc();
        if (cur.dims != desc.dims || cur.dataType != desc.dataType) {
            CANN_LLM_LOGE("state_pool.ensure: layer %d kind %s dims/dtype mismatch "
                          "(existing vs graph desc) —— 双图状态形状不一致",
                          layer, kind_name);
            return Status::INVALID_ARGUMENT;
        }
        return Status::SUCCESS; // 一致 → 复用
    }

    // 惰性创建：desc 合法性（dims 非空、全 >0、dtype 非 RESERVED）
    if (desc.dims.empty() || desc.dataType == DataType::RESERVED) {
        CANN_LLM_LOGE("state_pool.ensure: layer %d kind %s invalid desc "
                      "(dims empty or dtype RESERVED)",
                      layer, kind_name);
        return Status::INVALID_ARGUMENT;
    }
    for (const int32_t d : desc.dims) {
        if (d <= 0) {
            CANN_LLM_LOGE("state_pool.ensure: layer %d kind %s non-positive dim", layer, kind_name);
            return Status::INVALID_ARGUMENT;
        }
    }
    TensorDesc d = desc; // 形状/dtype 以图工位 desc 为权威（§5 D1）
    d.tensorName = std::string(kind == StateKind::Conv ? "conv_state_layer_" : "rstate_layer_") + std::to_string(layer);
    std::shared_ptr<ITensorBuffer> t;
    Status create_status = TensorManager::Instance().CreateTensor(d, /*modelId=*/0, &t);
    if (!create_status.ok()) {
        CANN_LLM_LOGE("state_pool.ensure: layer %d kind %s CreateTensor failed "
                      "(st=%d)",
                      layer, kind_name, create_status.code);
        return create_status;
    }
    // 创建即整体清零（图首次读须是干净状态）
    std::memset(t->GetData(), 0, t->GetSize());
    slot = std::move(t);
    CANN_LLM_LOGI("state_pool.ensure: layer %d kind %s tensor created (dims[0]=%d, "
                  "bytes=%zu)",
                  layer, kind_name, desc.dims[0], slot->GetSize());
    return Status::SUCCESS;
}

Status LinearStatePool::ResetLane(SlotId slot)
{
    std::lock_guard<std::mutex> lk(mtx_);
    for (std::size_t group = 0; group < tensors_.size(); ++group) {
        for (auto& t : {tensors_[group].first, tensors_[group].second}) {
            if (!t)
                continue;
            const auto& dims = t->GetTensorDesc().dims;
            if (dims.empty() || dims[0] <= 0) {
                CANN_LLM_LOGW("state_pool.reset_lane: invalid batch dim (layer "
                              "group %zu)",
                              group);
                continue;
            }
            const auto batch = static_cast<SlotId>(dims[0]);
            if (slot < 0 || slot >= batch) {
                CANN_LLM_LOGW("state_pool.reset_lane: slot %d out of range [0,%d) — skip", slot, batch);
                continue;
            }
            // lane 步长 = 实际 buffer 大小 / batch（dims[0]；§8 Q3）
            const std::size_t lane_bytes = t->GetSize() / static_cast<std::size_t>(batch);
            auto* base = static_cast<uint8_t*>(t->GetData());
            if (!base || lane_bytes == 0)
                continue;
            std::memset(base + static_cast<std::size_t>(slot) * lane_bytes, 0, lane_bytes);
        }
    }
    return Status::SUCCESS;
}

} // namespace cann_local_lm::engine
