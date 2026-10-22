/**
 * Copyright (c) 2025. All rights reserved.
 * @file hiai_acl/src/ndk/hiai_ndk_options.h
 * @brief Build options wrapper — translates ACL build options to NDK HMS_HiAIOptions_*.
 */
#ifndef HIAI_ACL_NDK_OPTIONS_H
#define HIAI_ACL_NDK_OPTIONS_H
#include "neural_network_runtime/neural_network_core.h"
#include "acl/acl_base.h"
#include "hiai_options.h"  // C2 follow-on: HiAI_*Enum + HMS_HiAIOptions_* decltype
#include <string>
#include <vector>
namespace hiai {
aclError HIAI_NDK_HiAIOptions_SetInputTensorShapes(OH_NNCompilation* compilation, NN_TensorDesc* inputTensorDescs[], size_t shapeCount);
aclError HIAI_NDK_HiAIOptions_SetFormatMode(OH_NNCompilation* compilation, HiAI_FormatMode formatMode);
aclError HIAI_NDK_HiAIOptions_SetDynamicShapeStatus(OH_NNCompilation* compilation, HiAI_DynamicShapeStatus status);
aclError HIAI_NDK_HiAIOptions_SetDynamicShapeMaxCache(OH_NNCompilation* compilation, size_t maxCacheCount);
aclError HIAI_NDK_HiAIOptions_SetDynamicShapeCacheMode(OH_NNCompilation* compilation, HiAI_DynamicShapeCacheMode cacheMode);
aclError HIAI_NDK_HiAIOptions_SetTuningStrategy(OH_NNCompilation* compilation, HiAI_TuningStrategy tuningStrategy);
aclError HIAI_NDK_HiAIOptions_SetTuningMode(OH_NNCompilation* compilation, HiAI_TuningMode tuningMode);
aclError HIAI_NDK_HiAIOptions_SetTuningCacheDir(OH_NNCompilation* compilation, const char* cacheDir);
aclError HIAI_NDK_HiAIOptions_SetQuantConfig(OH_NNCompilation* compilation, void* data, size_t size);
aclError HIAI_NDK_HiAIOptions_SetBandMode(OH_NNCompilation* compilation, HiAI_BandMode bandMode);
aclError HIAI_NDK_HiAIOptions_SetFallbackMode(OH_NNCompilation* compilation, HiAI_FallbackMode fallbackMode);
aclError HIAI_NDK_HiAIOptions_SetDeviceMemoryReusePlan(OH_NNCompilation* compilation, HiAI_DeviceMemoryReusePlan plan);
// C2 follow-on: SetOperatorDeviceOrder/SetModelDeviceOrder removed (HiAI_ExecuteDevice*; not a ModelOptions field).
aclError HIAI_NDK_HiAIOptions_SetOmOptions(OH_NNCompilation* compilation, HiAI_OmType type, const char* outputDir);
}
#endif
