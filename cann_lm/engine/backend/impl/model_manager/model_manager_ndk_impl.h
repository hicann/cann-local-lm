/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_INC_MODEL_MANAGER_MODEL_MANAGER_NDK_IMPL_H
#define FRAMEWORK_INC_MODEL_MANAGER_MODEL_MANAGER_NDK_IMPL_H

#include "model_manager/model_manager.h"
#include "neural_network_runtime/neural_network_core.h"

#include <mutex>

namespace hiai {

class ModelManagerNDKImpl : public IModelManager {
public:
    ModelManagerNDKImpl() = default;
    ~ModelManagerNDKImpl() override;

private:
    Status Init(const ModelInitOptions& options, const std::shared_ptr<IBuiltModel>& builtModel,
                const std::shared_ptr<IModelManagerListener>& listener) override;

    Status InitWeights(const std::string& weightDir) override;

    Status DeinitWeights() override;

    Status ReleaseMemory(MemoryMode mode) override;

    Status RecoverMemory(MemoryMode mode) override;

    Status GetWeightBuffer(const std::string& weightName, std::shared_ptr<IBuffer>& weightBuffer) override;

    Status FlushWeight(const std::string& weightName, size_t offset, size_t size) override;

    Status UpdateWeight(const std::string& weightName, size_t offset, size_t size, const void* weightBuffer) override;

    uint32_t GetModelID() override;

    Status Run(const std::vector<std::shared_ptr<INDTensorBuffer>>& inputs,
               std::vector<std::shared_ptr<INDTensorBuffer>>& outputs) override;

    void DeInit() override;

private:
    Status PrepareModelManager(const ModelInitOptions& options, const std::shared_ptr<IBuiltModel>& builtModel);

    void UnLoad();

private:
    std::mutex modelManagerMutex_;
    std::shared_ptr<OH_NNExecutor> executor_{nullptr};
    std::shared_ptr<OH_NNCompilation> nnCompilation_{nullptr};

    CustomModelData customModelData_;

    bool isAsyncMode_{false};
};
} // namespace hiai
#endif // FRAMEWORK_INC_MODEL_MANAGER_MODEL_MANAGER_NDK_IMPL_H
