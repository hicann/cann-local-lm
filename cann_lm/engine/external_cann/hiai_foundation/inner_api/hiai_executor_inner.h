/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

/**
 * @addtogroup HiAIFoundation
 * @{
 *
 * @brief Provides APIs for HiAI Foundation model inference.
 *
 * @syscap SystemCapability.AI.HiAIFoundation
 * @since 4.1.0(11)
 */

/**
 * @file hiai_executor_inner.h
 * @kit HiAIFoundationKit
 * @library libhiai_foundation.so
 * @syscap SystemCapability.AI.HiAIFoundation
 *
 * @brief Auxiliary APIs
 *
 *
 * @since 5.0.0(12)
 */

#ifndef HIAI_FOUNDATION_EXECUTOR_INNER_H
#define HIAI_FOUNDATION_EXECUTOR_INNER_H
#include "neural_network_runtime/neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HIAI_MEMORY_MODE_ALL = 0,
    HIAI_MEMORY_MODE_WEIGHT_ALL = 1,
    HIAI_MEMORY_MODE_WEIGHT_RES = 2,
    HIAI_MEMORY_MODE_FEATURE_MAP = 3,
} HiAI_MemoryMode;

OH_NN_ReturnCode HMS_HiAIExecutor_InitWeights(OH_NNExecutor *executor, const char* weightDir);

OH_NN_ReturnCode HMS_HiAIExecutor_DeinitWeights(OH_NNExecutor *executor);

OH_NN_ReturnCode HMS_HiAIExecutor_ReleaseMemory(OH_NNExecutor *executor, HiAI_MemoryMode mode);

OH_NN_ReturnCode HMS_HiAIExecutor_RecoverMemory(OH_NNExecutor *executor, HiAI_MemoryMode mode);

OH_NN_ReturnCode HMS_HiAIExecutor_GetWeightBuffer(OH_NNExecutor *executor, const char* weightName, void** data, size_t* size);

OH_NN_ReturnCode HMS_HiAIExecutor_FlushWeight(OH_NNExecutor *executor, const char* weightName, size_t offset, size_t size);

OH_NN_ReturnCode HMS_HiAIExecutor_GetModelID(OH_NNExecutor *executor, uint32_t* modelId);

OH_NN_ReturnCode HMS_HiAIExecutor_UpdateWeight(
    OH_NNExecutor* executor, const char* weightName, size_t offset, size_t size, const void* weightBuffer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // HIAI_FOUNDATION_EXECUTOR_INNER_H
