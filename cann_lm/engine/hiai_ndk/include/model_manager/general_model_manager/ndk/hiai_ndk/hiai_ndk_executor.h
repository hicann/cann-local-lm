/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#ifndef HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_EXECUTOR_H
#define HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_EXECUTOR_H

#include "hiai_foundation/inner_api/hiai_executor_inner.h"
#include "types.h"

namespace hiai {
Status HIAI_NDK_HiAIExecutor_InitWeights(OH_NNExecutor* executor, const char* weightDir);

Status HIAI_NDK_HiAIExecutor_DeinitWeights(OH_NNExecutor* executor);

Status HIAI_NDK_HiAIExecutor_ReleaseMemory(OH_NNExecutor* executor, HiAI_MemoryMode mode);

Status HIAI_NDK_HiAIExecutor_RecoverMemory(OH_NNExecutor* executor, HiAI_MemoryMode mode);

Status HIAI_NDK_HiAIExecutor_GetWeightBuffer(OH_NNExecutor* executor, const char* weightName, void** data, size_t* size);

Status  HIAI_NDK_HiAIExecutor_FlushWeight(OH_NNExecutor* executor, const char* weightName, size_t offset, size_t size);

Status  HIAI_NDK_HiAIExecutor_UpdateWeight(
    OH_NNExecutor* executor, const char* weightName, size_t offset, size_t size, const void* weightBuffer);

Status  HIAI_NDK_HiAIExecutor_GetModelID(OH_NNExecutor* executor, uint32_t* modelId);
}

#endif