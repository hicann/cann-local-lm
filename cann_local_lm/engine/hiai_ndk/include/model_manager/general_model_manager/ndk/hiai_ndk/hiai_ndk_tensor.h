/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#ifndef HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_TENSOR_H
#define HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_TENSOR_H

#include <vector>

#include "hiai_foundation/inner_api/hiai_tensor_inner.h"
#include "types.h"

namespace hiai {
Status HIAI_NDK_HiAITensor_SetCacheStatus(NN_Tensor* tensor, uint8_t cacheStatus);

Status HIAI_NDK_HiAITensor_GetCacheStatus(NN_Tensor* tensor, uint8_t* cacheStatus);

NN_Tensor* HIAI_NDK_HiAITensor_CreateKVCache(NN_TensorDesc* tensorDesc, size_t kvCacheNum, size_t tileSize);

Status HIAI_NDK_HiAITensor_ResizeKVCache(NN_Tensor* tensor, size_t size);
} // namespace hiai

#endif
