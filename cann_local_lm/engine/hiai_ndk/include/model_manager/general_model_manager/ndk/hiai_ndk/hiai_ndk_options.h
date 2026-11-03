/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#ifndef HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_OPTIONS_H
#define HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_OPTIONS_H

#include "neural_network_runtime/neural_network_core.h"

#include "types.h"
#include "hiai_foundation/hiai_options.h"
#include "hiai_foundation/inner_api/hiai_options_inner.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_align.h"

namespace hiai {

Status HIAI_NDK_HiAIOptions_SetBandMode(OH_NNCompilation* nnCompilation, HiAI_BandMode bandMode);

Status HIAI_NDK_HiAIOptions_SetModelCacheStrategy(OH_NNCompilation* nnCompilation,
                                                  HiAI_ModelCacheStrategy modelCacheStrategy);

Status HIAI_NDK_HiAIOptions_SetAsyncModeEnable(OH_NNCompilation* nnCompilation, bool isEnable);

Status HIAI_NDK_HiAIOptions_SetShareWeightsModelId(OH_NNCompilation* nnCompilation, uint32_t srcModelId);

} // namespace hiai

#endif
