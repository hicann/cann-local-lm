/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef HIAI_API_MODEL_MANAGER_H
#define HIAI_API_MODEL_MANAGER_H

#include "types.h"
#include "model/built_model.h"
#include "tensor/nd_tensor_buffer.h"

namespace hiai {
// 未使用：引擎链路 Init 时传 nullptr（model_executor.cpp），OnRunDone/OnServiceDied 全仓无实现无调用。
// 上游接口保真预留，暂不删除。
class IModelManagerListener {
public:
    virtual ~IModelManagerListener() = default;

    virtual void OnRunDone(
        const Context& context, Status result, std::vector<std::shared_ptr<INDTensorBuffer>>& outputs) = 0;

    virtual void OnServiceDied() = 0;
};

class IModelManager {
public:
    virtual ~IModelManager() = default;

    virtual Status Init(const ModelInitOptions& options, const std::shared_ptr<IBuiltModel>& builtModel,
        const std::shared_ptr<IModelManagerListener>& listener) = 0;

    virtual Status InitWeights(const std::string& weightDir) = 0;

    virtual Status DeinitWeights() = 0;

    // 未使用：引擎链路未调用（权重字节流读通路预留）。
    virtual Status GetWeightBuffer(const std::string& weightName, std::shared_ptr<IBuffer>& weightBuffer) = 0;

    // 未使用：引擎链路未调用（权重字节流刷写通路预留）。
    virtual Status FlushWeight(const std::string& weightName, size_t offset, size_t size) = 0;

    // 未使用：引擎链路未调用（权重字节流更新通路预留）。
    virtual Status UpdateWeight(
        const std::string& weightName, size_t offset, size_t size, const void* weightBuffer) = 0;

    virtual uint32_t GetModelID() = 0;

    virtual Status Run(const std::vector<std::shared_ptr<INDTensorBuffer>>& inputs,
        std::vector<std::shared_ptr<INDTensorBuffer>>& outputs) = 0;

    // 未使用：引擎链路未调用（内存恢复通路预留，NDK impl 内部亦未触发）。
    virtual Status ReleaseMemory(MemoryMode mode) = 0;

    // 未使用：引擎链路未调用（内存恢复通路预留）。
    virtual Status RecoverMemory(MemoryMode mode) = 0;

    virtual void DeInit() = 0;
};

std::shared_ptr<IModelManager> CreateModelManager();
} // namespace hiai
#endif // HIAI_API_MODEL_MANAGER_H
