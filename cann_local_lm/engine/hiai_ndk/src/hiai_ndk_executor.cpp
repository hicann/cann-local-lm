/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_executor.h"

#include "securec.h"
#include "infra/assertion.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/ndk_proxy.h"

namespace hiai {
Status HIAI_NDK_HiAIExecutor_InitWeights(OH_NNExecutor* executor, const char* weightDir)
{
    HIAI_EXPECT_NOT_NULL(executor);
    auto initWeightsFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_InitWeights)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_InitWeights"));
    HIAI_EXPECT_NOT_NULL(initWeightsFunc);
    OH_NN_ReturnCode retCode = initWeightsFunc(executor, weightDir);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIExecutor_DeinitWeights(OH_NNExecutor* executor)
{
    HIAI_EXPECT_NOT_NULL(executor);
    auto deinitWeightsFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_DeinitWeights)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_DeinitWeights"));
    HIAI_EXPECT_NOT_NULL(deinitWeightsFunc);
    OH_NN_ReturnCode retCode = deinitWeightsFunc(executor);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIExecutor_ReleaseMemory(OH_NNExecutor* executor, HiAI_MemoryMode mode)
{
    HIAI_EXPECT_NOT_NULL(executor);
    auto releaseMemoryFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_ReleaseMemory)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_ReleaseMemory"));
    HIAI_EXPECT_NOT_NULL(releaseMemoryFunc);
    OH_NN_ReturnCode retCode = releaseMemoryFunc(executor, mode);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIExecutor_RecoverMemory(OH_NNExecutor* executor, HiAI_MemoryMode mode)
{
    HIAI_EXPECT_NOT_NULL(executor);
    auto recoverMemoryFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_RecoverMemory)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_RecoverMemory"));
    HIAI_EXPECT_NOT_NULL(recoverMemoryFunc);
    OH_NN_ReturnCode retCode = recoverMemoryFunc(executor, mode);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIExecutor_GetWeightBuffer(OH_NNExecutor* executor, const char* weightName, void** data, size_t* size)
{
    HIAI_EXPECT_NOT_NULL(executor);
    HIAI_EXPECT_NOT_NULL(weightName);
    auto getWeightBufferFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_GetWeightBuffer)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_GetWeightBuffer"));
    HIAI_EXPECT_NOT_NULL(getWeightBufferFunc);
    OH_NN_ReturnCode retCode = getWeightBufferFunc(executor, weightName, data, size);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIExecutor_FlushWeight(OH_NNExecutor* executor, const char* weightName, size_t offset, size_t size)
{
    HIAI_EXPECT_NOT_NULL(executor);
    HIAI_EXPECT_NOT_NULL(weightName);
    auto flushWeightFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_FlushWeight)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_FlushWeight"));
    HIAI_EXPECT_NOT_NULL(flushWeightFunc);
    OH_NN_ReturnCode retCode = flushWeightFunc(executor, weightName, offset, size);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIExecutor_UpdateWeight(OH_NNExecutor* executor, const char* weightName, size_t offset, size_t size,
                                          const void* weightBuffer)
{
    HIAI_EXPECT_NOT_NULL(executor);
    HIAI_EXPECT_NOT_NULL(weightName);
    auto updateWeightFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_UpdateWeight)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_UpdateWeight"));
    if (updateWeightFunc != nullptr) {
        OH_NN_ReturnCode retCode = updateWeightFunc(executor, weightName, offset, size, weightBuffer);
        return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
    } else {
        void* weightData = nullptr;
        size_t weightSize = 0;
        HIAI_EXPECT_TRUE(HIAI_NDK_HiAIExecutor_GetWeightBuffer(executor, weightName, &weightData, &weightSize) ==
                         hiai::SUCCESS);
        HIAI_EXPECT_NOT_NULL(weightData);
        HIAI_EXPECT_TRUE(weightSize > 0);
        HIAI_EXPECT_TRUE(weightSize >= offset + size);

        HIAI_EXPECT_TRUE(memcpy_s(reinterpret_cast<uint8_t*>(weightData) + offset, size, weightBuffer, size) == 0);

        return HIAI_NDK_HiAIExecutor_FlushWeight(executor, weightName, offset, size);
    }
}

Status HIAI_NDK_HiAIExecutor_GetModelID(OH_NNExecutor* executor, uint32_t* modelId)
{
    HIAI_EXPECT_NOT_NULL(executor);
    HIAI_EXPECT_NOT_NULL(modelId);
    auto getModelIDFunc = reinterpret_cast<decltype(HMS_HiAIExecutor_GetModelID)*>(
        NDKProxy::GetSymbol("HMS_HiAIExecutor_GetModelID"));
    HIAI_EXPECT_NOT_NULL(getModelIDFunc);
    OH_NN_ReturnCode retCode = getModelIDFunc(executor, modelId);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}
} // namespace hiai
