// TensorManager 实现（v0.13：backend 代理）。
// 内部委托 hiai::CreateNDTensorBuffer(NDTensorDesc)。
#include "core/tensor_manager.h"

#include "infra/log.h"

#include <limits>
#include <string>
#include <utility>

namespace cann_lm::engine {

namespace {
/// 调试用 dims 格式化（临时）——与下方两条申请日志同属 DEBUG-TODO-REMOVE
/// 打包删除（grep "[DEBUG-TODO-REMOVE]" 一并清除）。[a, b, c] 方括号包裹，
/// 与 kv_cache_pool 的 FormatDimsForLog 日志风格一致。
std::string DebugFormatDims(const std::vector<int32_t>& dims)
{
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}
} // namespace

TensorManager::TensorManager() = default;

TensorManager& TensorManager::Instance()
{
    static TensorManager inst;
    return inst;
}

Status TensorManager::CreateTensor(const TensorDesc& desc, int modelId, std::shared_ptr<ITensorBuffer>* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    // [DEBUG-TODO-REMOVE] 调试日志：tensor 申请入口（name/dims/dtype/modelId）。
    // spec 联调期排查用，正式版删除（与本文件内同名标记一并清除）。
    CANN_LLM_LOGD("tensor_mgr.create: name='%s', dims=%s, dtype=%d modelId=%d "
                  "[DEBUG-TODO-REMOVE]",
                  desc.tensorName.c_str(), DebugFormatDims(desc.dims).c_str(), static_cast<int>(desc.dataType),
                  modelId);
    // 委托 backend CreateNDTensorBuffer 工厂（NDK 通路；host 无 so 时返回 nullptr）
    auto tb = hiai::CreateNDTensorBuffer(desc);
    if (!tb)
        return Status::FAILURE;
    // 检查底层 buffer 分配是否成功（分配失败时 GetData() 返 nullptr）
    if (!tb->GetData())
        return Status::FAILURE;
    *out = tb;
    (void)modelId; // 桩：modelId 暂忽略（真实后端按 modelId 绑定 NPU 上下文）
    return Status::SUCCESS;
}

Status TensorManager::CreateKvCacheTensor(const std::vector<int>& layer_ids, int kv_cache_max_len,
                                          int num_attention_kv_heads, int batch, int num_attention_head_dims,
                                          DataType dtype, std::vector<std::shared_ptr<ITensorBuffer>>* out)
{
    // 默认轴序 [len, heads, batch, dim]（引擎旧口径）——委托显式 dims 版，行为逐值不变化。
    const std::vector<int32_t> dims = {static_cast<int32_t>(kv_cache_max_len),
                                       static_cast<int32_t>(num_attention_kv_heads), static_cast<int32_t>(batch),
                                       static_cast<int32_t>(num_attention_head_dims)};
    return CreateKvCacheTensor(layer_ids, dims, dtype, out);
}

Status TensorManager::CreateKvCacheTensor(const std::vector<int>& layer_ids, const std::vector<int32_t>& dims,
                                          DataType dtype, std::vector<std::shared_ptr<ITensorBuffer>>* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (layer_ids.empty() || dims.empty() || dtype == DataType::RESERVED) {
        return Status::INVALID_ARGUMENT;
    }
    for (const int layer : layer_ids) {
        if (layer < 0)
            return Status::INVALID_ARGUMENT;
    }
    for (const int32_t dim : dims) {
        if (dim <= 0)
            return Status::INVALID_ARGUMENT;
    }
    // 字节数 = 逐维乘积 × dtype 字节数（uint64 累乘 + 溢出防护）
    const auto mul = [](uint64_t a, uint64_t b, uint64_t* r) -> bool {
        if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
            return false;
        *r = a * b;
        return true;
    };
    uint64_t bytes = 1;
    for (const int32_t dim : dims) {
        if (!mul(bytes, static_cast<uint64_t>(dim), &bytes)) {
            return Status::INVALID_ARGUMENT; // 维度乘积溢出
        }
    }
    if (!mul(bytes, static_cast<uint64_t>(DataTypeSize(dtype)), &bytes) ||
        bytes > std::numeric_limits<std::size_t>::max()) {
        return Status::INVALID_ARGUMENT;
    }

    // 2×layer_ids.size() 个 tensor：按列表序 [k(l0),v(l0),k(l1),v(l1),...]，
    // tensorName 用列表中的全局层号（与图工位层号对齐），all-or-nothing
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    ts.reserve(layer_ids.size() * 2);
    for (const int layer : layer_ids) {
        for (const char* kv : {"kv_cache_k_", "kv_cache_v_"}) {
            TensorDesc d;
            d.tensorName = std::string(kv) + std::to_string(layer);
            d.dataType = dtype;
            d.format = Format::ND;
            d.dims = dims;
            auto tb = hiai::CreateNDTensorBuffer(d);
            if (!tb || !tb->GetData())
                return Status::FAILURE; // 失败不产出
            ts.push_back(std::move(tb));
        }
    }
    *out = std::move(ts);
    return Status::SUCCESS;
}

Status TensorManager::CreateKvCacheTensorWithDescs(const std::vector<TensorDesc>& descs,
                                                   std::vector<std::shared_ptr<ITensorBuffer>>* out)
{
    if (!out)
        return Status::INVALID_ARGUMENT;
    if (descs.empty())
        return Status::INVALID_ARGUMENT;
    // [DEBUG-TODO-REMOVE] 调试日志：KV tensor 批量申请入口（逐 desc
    // name/dims/dtype）。spec 联调期排查用，正式版删除（同上打包清除）。
    for (const auto& d : descs) {
        CANN_LLM_LOGD("tensor_mgr.create_kv: name='%s', dims=%s, dtype=%d "
                      "[DEBUG-TODO-REMOVE]",
                      d.tensorName.c_str(), DebugFormatDims(d.dims).c_str(), static_cast<int>(d.dataType));
    }
    // 逐个 desc 校验 + 溢出防护
    const auto mul = [](uint64_t a, uint64_t b, uint64_t* r) -> bool {
        if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
            return false;
        *r = a * b;
        return true;
    };
    for (const auto& d : descs) {
        if (d.dims.empty() || d.dataType == DataType::RESERVED) {
            return Status::INVALID_ARGUMENT;
        }
        for (const int32_t dim : d.dims) {
            if (dim <= 0)
                return Status::INVALID_ARGUMENT;
        }
        uint64_t bytes = 1;
        for (const int32_t dim : d.dims) {
            if (!mul(bytes, static_cast<uint64_t>(dim), &bytes)) {
                return Status::INVALID_ARGUMENT;
            }
        }
        if (!mul(bytes, static_cast<uint64_t>(DataTypeSize(d.dataType)), &bytes) ||
            bytes > std::numeric_limits<std::size_t>::max()) {
            return Status::INVALID_ARGUMENT;
        }
    }
    // 逐个创建（all-or-nothing）
    std::vector<std::shared_ptr<ITensorBuffer>> ts;
    ts.reserve(descs.size());
    for (const auto& d : descs) {
        auto tb = hiai::CreateNDTensorBuffer(d);
        if (!tb || !tb->GetData())
            return Status::FAILURE;
        ts.push_back(std::move(tb));
    }
    *out = std::move(ts);
    return Status::SUCCESS;
}

Status TensorManager::Release(const std::shared_ptr<ITensorBuffer>& t)
{
    (void)t; // RAII 自动释放（ITensorBuffer shared_ptr 析构时释放底层 buffer）
    return Status::SUCCESS;
}

} // namespace cann_lm::engine
