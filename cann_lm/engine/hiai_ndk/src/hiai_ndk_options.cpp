/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_options.h"

#include "infra/assertion.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/ndk_proxy.h"

namespace hiai {
Status HIAI_NDK_HiAIOptions_SetBandMode(OH_NNCompilation* nnCompilation, HiAI_BandMode bandMode)
{
    auto setBandModeFunc = reinterpret_cast<decltype(HMS_HiAIOptions_SetBandMode)*>(
        NDKProxy::GetSymbol("HMS_HiAIOptions_SetBandMode"));
    HIAI_EXPECT_NOT_NULL(setBandModeFunc);

    OH_NN_ReturnCode retCode = setBandModeFunc(nnCompilation, bandMode);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIOptions_SetModelCacheStrategy(OH_NNCompilation* nnCompilation,
    HiAI_ModelCacheStrategy modelCacheStrategy)
{
    auto setModelCacheStrategyFunc = reinterpret_cast<decltype(HMS_HiAIOptions_SetModelCacheStrategy)*>(
        NDKProxy::GetSymbol("HMS_HiAIOptions_SetModelCacheStrategy"));
    // 这里处理DDK和ROM版本不兼容的问题
    HIAI_EXPECT_NOT_NULL_R(setModelCacheStrategyFunc, SUCCESS);

    OH_NN_ReturnCode retCode = setModelCacheStrategyFunc(nnCompilation, modelCacheStrategy);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIOptions_SetAsyncModeEnable(OH_NNCompilation* nnCompilation, bool isEnable)
{
    auto setAsyncModeEnableFunc = reinterpret_cast<decltype(HMS_HiAIOptions_SetAsyncModeEnable)*>(
        NDKProxy::GetSymbol("HMS_HiAIOptions_SetAsyncModeEnable"));
    HIAI_EXPECT_NOT_NULL(setAsyncModeEnableFunc);

    OH_NN_ReturnCode retCode = setAsyncModeEnableFunc(nnCompilation, isEnable);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_HiAIOptions_SetShareWeightsModelId(OH_NNCompilation* nnCompilation, uint32_t srcModelId)
{
    auto setShareWeightsModelIdFunc = reinterpret_cast<decltype(HMS_HiAIOptions_SetShareWeightsModelId)*>(
        NDKProxy::GetSymbol("HMS_HiAIOptions_SetShareWeightsModelId"));
    HIAI_EXPECT_NOT_NULL(setShareWeightsModelIdFunc);

    OH_NN_ReturnCode retCode = setShareWeightsModelIdFunc(nnCompilation, srcModelId);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}



}
