/**
 * Copyright (c) 2025. All rights reserved.
 * @file hiai_acl/src/ndk/hiai_ndk_executor.h
 * @brief Executor helper — model ID extraction from OH_NNExecutor.
 */
#ifndef HIAI_ACL_NDK_EXECUTOR_H
#define HIAI_ACL_NDK_EXECUTOR_H
#include "neural_network_runtime/neural_network_core.h"
#include "acl/acl_base.h"
namespace hiai {
aclError HIAI_NDK_HiAIExecutor_GetModelID(OH_NNExecutor* executor, uint32_t* modelId);
}
#endif
