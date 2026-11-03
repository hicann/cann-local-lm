/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#include "model_manager_ndk_impl.h"

#include <unistd.h>
#include <algorithm>

#include "infra/assertion.h"
#include "infra/securestl.h"

#include "infra/log.h"
#ifdef AI_SUPPORT_DECODER_OFFLOAD
#include "framework/cl_itf/decode_offload_manager.h"
#endif

#include "built_model/built_model_ndk_impl.h"
#include "tensor/nd_tensor_buffer_ndk_impl.h"

#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_options.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_nncore.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_tensor.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_executor.h"
#include "types.h"
#include "model_manager/executor_container.h"

namespace hiai {
ModelManagerNDKImpl::~ModelManagerNDKImpl()
{
    if (executor_ != nullptr) {
        UnLoad();
    }
}

namespace {
Status SetInitOption(OH_NNCompilation* compilation, const ModelInitOptions& options, bool isAsyncMode)
{
    static const int MIDDLE_BAND_MIDDLE_COMPUTE_UNIT = 202;
    uint32_t bandMode = static_cast<uint32_t>(options.bandMode);
    uint32_t perfMode = static_cast<uint32_t>(options.perfMode);
    uint32_t modelCacheStrategy = static_cast<uint32_t>(options.modelCacheStrategy);
    if (perfMode == HIAI_PERFMODE_HIGH_COMPUTE_UNIT || perfMode == HIAI_PERFMODE_MIDDLE_BAND_NORMAL_COMPUTE_UNIT) {
        bandMode = perfMode / 100;
        perfMode = perfMode % 100;
    }

    Status status = HIAI_NDK_HiAIOptions_SetBandMode(compilation, static_cast<HiAI_BandMode>(bandMode));
    HIAI_EXPECT_TRUE(status == SUCCESS);

    status = HIAI_NDK_NNCompilation_SetPerformanceMode(compilation, static_cast<OH_NN_PerformanceMode>(perfMode));
    HIAI_EXPECT_TRUE(status == SUCCESS);

    status = HIAI_NDK_HiAIOptions_SetAsyncModeEnable(compilation, isAsyncMode);
    HIAI_EXPECT_TRUE(status == SUCCESS);

    // 处理DDK和ROM不兼容：SetModelCacheStrategy为后加特性，历史版本hiai_foundtion和HDI可能不支持
    // 为了不影响加载流程，因此忽略其返回值
    (void)HIAI_NDK_HiAIOptions_SetModelCacheStrategy(compilation,
                                                     static_cast<HiAI_ModelCacheStrategy>(modelCacheStrategy));

    // 权重共享绑定：id>0 绑定本 compilation 复用源模型已加载权重，绑定失败快速失败
    // （老 ROM 缺符号 shim 返 FAILURE，不让绑定请求被静默吞掉）；id=0 不调 HMS 接口，
    // 默认加载路径不受老 ROM 缺符号影响。
    if (options.shareWeightsModelId != 0) {
        status = HIAI_NDK_HiAIOptions_SetShareWeightsModelId(compilation, options.shareWeightsModelId);
        HIAI_EXPECT_TRUE(status == SUCCESS);
    }
    return SUCCESS;
}

bool IsOptionsChanged(bool isAsyncMode, const ModelInitOptions& options)
{
    if (isAsyncMode || options.perfMode != PerfMode::MIDDLE || options.bandMode != PerfMode::UNSET ||
        options.shareWeightsModelId != 0) {
        CANN_LLM_LOGI("isAsync or options has changed, need reload");
        return true;
    }
    CANN_LLM_LOGI("don't need reload");
    return false;
}
} // namespace

Status ModelManagerNDKImpl::PrepareModelManager(const ModelInitOptions& options,
                                                const std::shared_ptr<IBuiltModel>& builtModel)
{
    auto builtModelNDKImpl = std::dynamic_pointer_cast<BuiltModelNDKImpl>(builtModel);
    HIAI_EXPECT_NOT_NULL(builtModelNDKImpl);

    auto nnCompilation = builtModelNDKImpl->GetNNCompilation();
    HIAI_EXPECT_NOT_NULL(nnCompilation);

    // 如果加载选项有变化，需要重新加载
    uint64_t builtModelId = builtModelNDKImpl->GetBuiltModelId();
    ExecutorContainer* container = ExecutorContainer::GetInstance();
    HIAI_EXPECT_NOT_NULL(container);
    auto executor = container->GetExecutor(builtModelId);
    if (executor == nullptr || IsOptionsChanged(isAsyncMode_, options)) {
        if (executor != nullptr) { // 卸载以前的executor
            container->RemoveExecutor(executor);
            executor.reset();
        }

        HIAI_EXPECT_EXEC(SetInitOption(nnCompilation.get(), options, isAsyncMode_));
        executor_ = std::shared_ptr<OH_NNExecutor>(HIAI_NDK_NNExecutor_Construct(nnCompilation.get()),
                                                   [](OH_NNExecutor* p) { HIAI_NDK_NNExecutor_Destroy(&p); });
        HIAI_EXPECT_NOT_NULL(executor_);
        container->AddExecutor(builtModelId, executor_);
    } else {
        executor_ = executor;
    }
    nnCompilation_ = nnCompilation;
    return SUCCESS;
}

Status ModelManagerNDKImpl::Init(const ModelInitOptions& options, const std::shared_ptr<IBuiltModel>& builtModel,
                                 const std::shared_ptr<IModelManagerListener>& listener)
{
    HIAI_EXPECT_NOT_NULL(builtModel);
    std::lock_guard<std::mutex> lock(modelManagerMutex_);
    HIAI_EXPECT_TRUE(executor_ == nullptr);

    customModelData_ = builtModel->GetCustomData();

    isAsyncMode_ = (listener != nullptr);

    HIAI_EXPECT_EXEC(PrepareModelManager(options, builtModel));

    return SUCCESS;
}

Status ModelManagerNDKImpl::InitWeights(const std::string& weightDir)
{
    HIAI_EXPECT_NOT_NULL(executor_);
    return HIAI_NDK_HiAIExecutor_InitWeights(executor_.get(), weightDir.c_str());
}

Status ModelManagerNDKImpl::DeinitWeights()
{
    HIAI_EXPECT_NOT_NULL(executor_);
    return HIAI_NDK_HiAIExecutor_DeinitWeights(executor_.get());
}

Status ModelManagerNDKImpl::ReleaseMemory(MemoryMode mode)
{
    HIAI_EXPECT_NOT_NULL(executor_);
    HIAI_EXPECT_TRUE_R(mode >= MemoryMode::ALL && mode <= MemoryMode::FEATURE_MAP, INVALID_PARAM);
    return HIAI_NDK_HiAIExecutor_ReleaseMemory(executor_.get(), static_cast<HiAI_MemoryMode>(mode));
}

Status ModelManagerNDKImpl::RecoverMemory(MemoryMode mode)
{
    HIAI_EXPECT_NOT_NULL(executor_);
    HIAI_EXPECT_TRUE_R(mode >= MemoryMode::ALL && mode <= MemoryMode::FEATURE_MAP, INVALID_PARAM);
    return HIAI_NDK_HiAIExecutor_RecoverMemory(executor_.get(), static_cast<HiAI_MemoryMode>(mode));
}

Status ModelManagerNDKImpl::GetWeightBuffer(const std::string& weightName, std::shared_ptr<IBuffer>& weightBuffer)
{
    HIAI_EXPECT_NOT_NULL(executor_);
    void* data = nullptr;
    size_t size = 0;
    Status ret = HIAI_NDK_HiAIExecutor_GetWeightBuffer(executor_.get(), weightName.c_str(), &data, &size);
    if (ret != SUCCESS) {
        CANN_LLM_LOGE("GetWeightBuffer failed.");
        return ret;
    }
    HIAI_EXPECT_NOT_NULL(data);
    HIAI_EXPECT_TRUE(size > 0);
    weightBuffer = CreateLocalBuffer(data, size, false);

    return SUCCESS;
}

Status ModelManagerNDKImpl::FlushWeight(const std::string& weightName, size_t offset, size_t size)
{
    HIAI_EXPECT_NOT_NULL(executor_);
    return HIAI_NDK_HiAIExecutor_FlushWeight(executor_.get(), weightName.c_str(), offset, size);
}

Status ModelManagerNDKImpl::UpdateWeight(const std::string& weightName, size_t offset, size_t size,
                                         const void* weightBuffer)
{
    HIAI_EXPECT_NOT_NULL(executor_);
    return HIAI_NDK_HiAIExecutor_UpdateWeight(executor_.get(), weightName.c_str(), offset, size, weightBuffer);
}

uint32_t ModelManagerNDKImpl::GetModelID()
{
    std::lock_guard<std::mutex> lock(modelManagerMutex_);
    uint32_t modelId = 0;
    (void)HIAI_NDK_HiAIExecutor_GetModelID(executor_.get(), &modelId);
    return modelId;
}

namespace {
NN_Tensor* GetRawBufferFromNDTensorBufferNDK(const std::shared_ptr<INDTensorBuffer>& buffer)
{
    std::shared_ptr<NDTensorBufferNDKImpl> bufferNDKImpl = std::dynamic_pointer_cast<NDTensorBufferNDKImpl>(buffer);
    if (bufferNDKImpl == nullptr) {
        CANN_LLM_LOGE("invalid buffer");
        return nullptr;
    }
    return bufferNDKImpl->GetNNTensor();
}

std::vector<NN_Tensor*> Convert2NNTensorBuffers(const std::vector<std::shared_ptr<INDTensorBuffer>>& buffers)
{
    std::vector<NN_Tensor*> nnBuffers;
    for (size_t i = 0; i < buffers.size(); i++) {
        HIAI_EXPECT_NOT_NULL_R(buffers[i], nnBuffers);
        NN_Tensor* nnTensor = GetRawBufferFromNDTensorBufferNDK(buffers[i]);
        HIAI_EXPECT_NOT_NULL_R(nnTensor, nnBuffers);
        nnBuffers.push_back(nnTensor);
    }
    return nnBuffers;
}
} // namespace

Status ModelManagerNDKImpl::Run(const std::vector<std::shared_ptr<INDTensorBuffer>>& inputs,
                                std::vector<std::shared_ptr<INDTensorBuffer>>& outputs)
{
    std::vector<NN_Tensor*> nnInputs = Convert2NNTensorBuffers(inputs);
    HIAI_EXPECT_TRUE_R(nnInputs.size() == inputs.size(), INVALID_PARAM);

    std::vector<NN_Tensor*> nnOutputs = Convert2NNTensorBuffers(outputs);
    HIAI_EXPECT_TRUE_R(nnOutputs.size() == outputs.size(), INVALID_PARAM);

    std::lock_guard<std::mutex> lock(modelManagerMutex_);
    return HIAI_NDK_NNExecutor_RunSync(executor_.get(), nnInputs, nnOutputs);
}

void ModelManagerNDKImpl::UnLoad()
{
    std::lock_guard<std::mutex> lock(modelManagerMutex_);
    ExecutorContainer* container = ExecutorContainer::GetInstance();
    HIAI_EXPECT_NOT_NULL_VOID(container);
    container->RemoveExecutor(executor_);
    executor_.reset();
    executor_ = nullptr;
}

void ModelManagerNDKImpl::DeInit() { UnLoad(); }

std::shared_ptr<IModelManager> CreateModelManagerFromNDK() { return make_shared_nothrow<ModelManagerNDKImpl>(); }
} // namespace hiai
