/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_MODEL_BUILT_MODEL_NDK_IMPL_H
#define FRAMEWORK_MODEL_BUILT_MODEL_NDK_IMPL_H

#include <atomic>

// api/framework
#include "model/built_model.h"
#include "neural_network_runtime/neural_network_core.h"
// inc
#include "base_buffer.h"

namespace hiai {
class BuiltModelNDKImpl : public IBuiltModel {
public:
    BuiltModelNDKImpl();
    BuiltModelNDKImpl(std::shared_ptr<OH_NNCompilation> nnCompilation, std::shared_ptr<IBuffer> modelBuffer);
    ~BuiltModelNDKImpl() override;

    uint64_t GetBuiltModelId() { return builtModelId_; }
    std::shared_ptr<OH_NNCompilation> GetNNCompilation();

private:
    Status RestoreFromBuffer(const std::shared_ptr<IBuffer>& buffer) override;
    Status RestoreFromFile(const char* file) override;

    void SetName(const std::string& name) override;

    std::vector<NDTensorDesc> GetInputTensorDescs() const override;
    std::vector<NDTensorDesc> GetOutputTensorDescs() const override;

    void SetCustomData(const CustomModelData& customModelData) override;
    const CustomModelData& GetCustomData() override;

    std::shared_ptr<OH_NNExecutor> GetExecutor() const;

private:
    std::shared_ptr<OH_NNCompilation> nnCompilation_ {nullptr};
    std::shared_ptr<BaseBuffer> buffer_ {nullptr};
    CustomModelData customModelData_;
    std::shared_ptr<IBuffer> modelBuffer_ {nullptr};
    uint64_t builtModelId_ {0};

    std::string modelName_ {""};
    std::string modelPath_ {""};

private:
    static std::atomic<uint64_t> idGenerator_;
};
} // namespace hiai
#endif // FRAMEWORK_MODEL_BUILT_MODEL_NDK_IMPL_H
