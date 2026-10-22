/**
 * Copyright (c) 2025. All rights reserved.
 * @file hiai_acl/src/ndk/hiai_ndk_tensor.h
 * @brief Tensor + cache status wrapper.
 */
#ifndef HIAI_ACL_NDK_TENSOR_H
#define HIAI_ACL_NDK_TENSOR_H
#include <vector>
#include "neural_network_runtime/neural_network_core.h"
#include "acl/acl_base.h"
namespace hiai {
aclError HIAI_NDK_HiAITensor_SetCacheStatus(NN_Tensor* tensor, uint8_t cacheStatus);
aclError HIAI_NDK_HiAITensor_GetCacheStatus(NN_Tensor* tensor, uint8_t* cacheStatus);
}
#endif
