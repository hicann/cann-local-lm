// ═══════════════════════════════════════════════════════════════════════════
// TensorManager — backend 代理枢纽（v0.13：全链路改 hiai NDK backend 类型）。
//
// 替代旧 core/tensor.h + i_built_model.h + i_model_manager.h 三头。
// 所用 backend 类型（INDTensorBuffer/NDTensorDesc/DataType/Format/IBuiltModel/
// IModelManager 等）经 backend/api 头（built_model.h / model_manager.h /
// nd_tensor_buffer.h）引入 + using 声明，engine 代码透明使用。
//
// TensorManager 保留为单例代理——所有 tensor 申请/管理经此入口，
// 内部委托 hiai::CreateNDTensorBuffer(NDTensorDesc)。
//
// engine Status 不改（infra/status.h 不动）——经 map_backend_status() 映射。
// ModelConfig 框（内部类，后续从 meta_json 反序列化）。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

// ── backend 头文件（backend/api，include 根经 cann_backend PUBLIC 传递）──
#include "types.h"                       // Status/DataType/Format/NDTensorDesc/ModelInitOptions
#include "model/built_model.h"           // hiai::IBuiltModel + CreateBuiltModel
#include "model_manager/model_manager.h" // hiai::IModelManager + CreateModelManager
#include "tensor/nd_tensor_buffer.h"     // hiai::INDTensorBuffer + CreateNDTensorBuffer

#include "infra/status.h" // engine Status（不改）

#include <memory>
#include <string>
#include <vector>

namespace cann_lm::engine {

// ── backend 类型别名（engine 代码透明使用，无需 hiai:: 前缀）──
using ITensorBuffer = hiai::INDTensorBuffer;
using TensorDesc = hiai::NDTensorDesc;
using DataType = hiai::DataType;
using Format = hiai::Format;
using IBuffer = hiai::IBuffer;
using IBuiltModel = hiai::IBuiltModel;
using IModelManager = hiai::IModelManager;
using IModelManagerListener = hiai::IModelManagerListener;
using ModelInitOptions = hiai::ModelInitOptions;
// 注：不用 using hiai::Context（与 engine 自己的 Context 类冲突）
// 注：ModelConfig 已迁至 core/component/model_executor/model_executor.h（SR20260903001 D-06，模块所有权）

// ── Status 映射（backend hiai::Status(uint32_t) → engine Status）──
inline Status map_backend_status(hiai::Status s)
{
    switch (s) {
        case hiai::SUCCESS:
            return Status(Status::SUCCESS); // 0
        case hiai::INVALID_PARAM:
            return Status(Status::INVALID_ARGUMENT); // 3
        default:
            return Status(Status::FAILURE); // 1/2/4…
    }
}

// ── dtype → 字节数（engine 侧通用辅助；RESERVED → 0）──
inline std::size_t DataTypeSize(DataType t)
{
    switch (t) {
        case DataType::BOOL:
        case DataType::INT8:
        case DataType::UINT8:
            return 1;
        case DataType::INT16:
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:
        case DataType::DOUBLE:
            return 8;
        default:
            return 0; // RESERVED
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TensorManager 代理（单例，所有 tensor 申请/管理经此入口）。
// ═══════════════════════════════════════════════════════════════════════════
class TensorManager {
public:
    /// 单例访问。
    static TensorManager& Instance();

    /// 按 desc 申请 tensor。
    /// 内部委托 hiai::CreateNDTensorBuffer(desc)（NDK 通路；host 无 NDK so
    /// 时工厂返回 nullptr → FAILURE，优雅降级）。
    /// modelId 关联已加载模型（真实后端用于绑定 NPU 上下文；桩忽略）。
    Status CreateTensor(const TensorDesc& desc, int modelId, std::shared_ptr<ITensorBuffer>* out);

    /// KV cache 池专用批量通道：一次调用创建全部 2×layer_ids.size() 个 tensor。
    /// 维度 [kv_cache_max_len, num_attention_kv_heads, batch, num_attention_head_dims]，
    /// 命名 kv_cache_k_<layer> / kv_cache_v_<layer>（layer 取 layer_ids 中的全局层号，
    /// Qwen3.5 稀疏池下与图工位层号对齐，support_qwen3.5.md §4.3），输出序按列表序
    /// [k(l0),v(l0),k(l1),v(l1),...]（KvCachePool 按 2*pos+{0=K,1=V} 直接消费，
    /// pos = 层号在列表中的下标）。
    /// 与 CreateTensor 的差异：无 modelId——池在模型 Load 前创建，不绑定具体模型。
    /// 参数校验（layer_ids 空/含负值 / 其余任一 ≤ 0 / dtype RESERVED / out 空）→
    /// INVALID_ARGUMENT；all-or-nothing：任一 tensor 申请失败 → FAILURE，out 不产出。
    Status CreateKvCacheTensor(const std::vector<int>& layer_ids, int kv_cache_max_len, int num_attention_kv_heads,
                               int batch, int num_attention_head_dims, DataType dtype,
                               std::vector<std::shared_ptr<ITensorBuffer>>* out);

    /// 显式 dims 版批量通道（保留供 UT 引用；生产路径走 descs 版）。
    /// dims 须非空且逐维 > 0、dtype 非 RESERVED，维度乘积 × dtype
    /// 字节数 uint64 溢出防护同四参版；命名/输出序同四参版
    /// （kv_cache_k_<layer>/kv_cache_v_<layer>，[k,v] 对，all-or-nothing）。
    Status CreateKvCacheTensor(const std::vector<int>& layer_ids, const std::vector<int32_t>& dims, DataType dtype,
                               std::vector<std::shared_ptr<ITensorBuffer>>* out);

    /// 按完整 TensorDesc 列表批量创建 KV cache tensor（异构 dims 通道；
    /// KvCachePool::EnsureKvTensor 新建路径消费——形状/dtype 权威 = om 图
    /// 工位 desc，Cache_pool_simplify.md R1）。
    /// descs 已含 dims/dataType/format/tensorName，本方法逐个委托
    /// hiai::CreateNDTensorBuffer 创建，输出序与输入序一致。适用于各层 dims
    /// 可不同的场景（含同层 K/V 异形）；同形场景与显式 dims 版等价。
    /// 参数校验（descs 空/任一 dims 空/任一 dim ≤ 0/dtype RESERVED/out 空）→
    /// INVALID_ARGUMENT；all-or-nothing：任一 tensor 申请失败 → FAILURE，out 不产出。
    Status CreateKvCacheTensorWithDescs(const std::vector<TensorDesc>& descs,
                                        std::vector<std::shared_ptr<ITensorBuffer>>* out);

    /// 显式释放（可选；RAII 亦自动释放）。
    Status Release(const std::shared_ptr<ITensorBuffer>& t);

private:
    TensorManager();
    TensorManager(const TensorManager&) = delete;
    TensorManager& operator=(const TensorManager&) = delete;
};

} // namespace cann_lm::engine
