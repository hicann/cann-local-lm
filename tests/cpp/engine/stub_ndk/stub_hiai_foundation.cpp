/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file tests/stub_ndk/stub_hiai_foundation.cpp
 * @brief Host stub of libhiai_foundation.so — minimal:just loadable
 *        (NDKProxy 要 nncore+hiai 两个 .so 都 Init 成功才 initialized_=true)。
 *        导出 HMS_HiAIExecutor_GetModelID(hiai_ndk_executor.cpp 用到)。
 *        其余 HMS_* 未在模型生命周期路径调用,无需桩;后续若 build/test 暴露缺口再补。
 */
#include "neural_network_runtime/neural_network_runtime_type.h" // OH_NN_ReturnCode / OH_NNExecutor

// HMS_* 签名来自 hiai_ndk_executor.cpp 的 extern "C" 声明
extern "C" {

OH_NN_ReturnCode HMS_HiAIExecutor_GetModelID(OH_NNExecutor* executor, uint32_t* modelId)
{
    (void)executor;
    if (modelId)
        *modelId = 1;
    return OH_NN_SUCCESS;
}

} // extern "C"
