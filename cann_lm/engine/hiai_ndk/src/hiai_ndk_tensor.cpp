/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_tensor.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_nncore.h"

#include "infra/assertion.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/ndk_proxy.h"

namespace hiai {
Status HIAI_NDK_HiAITensor_SetCacheStatus(NN_Tensor* tensor, uint8_t cacheStatus)
{
    HIAI_EXPECT_NOT_NULL(tensor);
    auto setCacheStatusFunc = reinterpret_cast<decltype(HMS_HiAITensor_SetCacheStatus)*>(
        NDKProxy::GetSymbol("HMS_HiAITensor_SetCacheStatus"));
    HIAI_EXPECT_NOT_NULL(setCacheStatusFunc);
    HIAI_EXPECT_TRUE(setCacheStatusFunc(tensor, cacheStatus) == OH_NN_SUCCESS);
    return SUCCESS;
}

Status HIAI_NDK_HiAITensor_GetCacheStatus(NN_Tensor* tensor, uint8_t* cacheStatus)
{
    HIAI_EXPECT_NOT_NULL(tensor);
    HIAI_EXPECT_NOT_NULL(cacheStatus);
    auto getCacheStatusFunc = reinterpret_cast<decltype(HMS_HiAITensor_GetCacheStatus)*>(
        NDKProxy::GetSymbol("HMS_HiAITensor_GetCacheStatus"));
    HIAI_EXPECT_NOT_NULL(getCacheStatusFunc);
    HIAI_EXPECT_TRUE(getCacheStatusFunc(tensor, cacheStatus) == OH_NN_SUCCESS);
    return SUCCESS;
}

NN_Tensor* HIAI_NDK_HiAITensor_CreateKVCache(NN_TensorDesc* tensorDesc, size_t kvCacheNum, size_t tileSize)
{
    HIAI_EXPECT_NOT_NULL_R(tensorDesc, nullptr);
    auto createFunc = reinterpret_cast<decltype(HMS_HiAITensor_CreateKVCache)*>(
        NDKProxy::GetSymbol("HMS_HiAITensor_CreateKVCache"));
    HIAI_EXPECT_NOT_NULL_R(createFunc, nullptr);
    size_t id = HIAI_NDK_GetDeviceID();
    HIAI_EXPECT_TRUE_R(id != 0, nullptr);
    return createFunc(tensorDesc, kvCacheNum, tileSize, id);
}

Status HIAI_NDK_HiAITensor_ResizeKVCache(NN_Tensor* tensor, size_t size)
{
    HIAI_EXPECT_NOT_NULL(tensor);
    auto resizeFunc = reinterpret_cast<decltype(HMS_HiAITensor_ResizeKVCache)*>(
        NDKProxy::GetSymbol("HMS_HiAITensor_ResizeKVCache"));
    HIAI_EXPECT_NOT_NULL(resizeFunc);
    if (resizeFunc(tensor, size) != OH_NN_SUCCESS) {
        return FAILURE;
    }
    return SUCCESS;
}

} // namespace hiai
