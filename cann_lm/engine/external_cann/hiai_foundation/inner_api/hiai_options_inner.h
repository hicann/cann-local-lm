/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
 * @file hiai_options_inner.h
 * @kit HiAIFoundationKit
 * @library libhiai_foundation.so
 * @syscap SystemCapability.AI.HiAIFoundation
 *
 * @brief Defines the API for build options.
 *
 * Allows you to update model shapes, and set dynamic shapes, data layout formats, operator fusion strategies,
 * quantization configurations, operator-level tuning, model-level tuning, assisted tuning, and bandwidth modes.
 *
 * @since 4.1.0(11)
 */
#ifndef HIAI_FOUNDATION_OPTIONS_INNER_H
#define HIAI_FOUNDATION_OPTIONS_INNER_H

#include "neural_network_runtime/neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

OH_NN_ReturnCode HMS_HiAIOptions_SetAsyncModeEnable(OH_NNCompilation* compilation, bool isEnable);

bool HMS_HiAIOptions_GetAsyncModeEnable(const OH_NNCompilation* compilation);

typedef struct HiAI_NativeHandle HiAI_NativeHandle;

typedef void  (*onAllocate) (void* userData, uint32_t requiredSize, HiAI_NativeHandle* handles[], size_t* handlesSize);

typedef void (*onFree) (void* userData, HiAI_NativeHandle* handles[], size_t handlesSize);

/**
 * @brief Creates HiAI_NativeHandle with the specified file descriptor, size, and offset.
 *
 * @param [in] fd The file descriptor.
 * @param [in] size The size of the memory resource in bytes.
 * @param [in] offset The offset within the memory resource from which to start the handle.
 * @return A pointer to the created HiAI_NativeHandle. Returns NULL if the creation fails.
 * @since 5.0.0(12)
 */
HiAI_NativeHandle* HMS_HiAINativeHandle_Create(int fd, int size, int offset);

/**
 * @brief Destroys HiAI_NativeHandle  and releases its associated resources.
 *
 * @param [in] handle A pointer to a pointer to the HiAI_NativeHandle to be destroyed.
 * @since 5.0.0(12)
 */
void HMS_HiAINativeHandle_Destroy(HiAI_NativeHandle** handle);

/**
 * @brief Set the allocate function in compilation options.
 *
 * @param [in] compilation Pointer to {@link OH_NNCompilation}. The value cannot be null. Otherwise, a null pointer is
 * returned.
 * @param [in] allocateFunc A function pointer of type onAllocate.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 5.0.0(12)
 */
OH_NN_ReturnCode HMS_HiAIOptions_SetAllocate(OH_NNCompilation* compilation, onAllocate allocateFunc);

/**
 * @brief Set the free function in compilation options.
 *
 * @param [in] compilation Pointer to {@link OH_NNCompilation}. The value cannot be null. Otherwise, a null pointer is
 * returned.
 * @param [in] freeFunc  A function pointer of type onFree.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 5.0.0(12)
 */
OH_NN_ReturnCode HMS_HiAIOptions_SetFree(OH_NNCompilation* compilation, onFree freeFunc);

/**
 * @brief Set the userData in compilation options.
 *
 * @param [in] compilation Pointer to {@link OH_NNCompilation}. The value cannot be null. Otherwise, a null pointer is
 * returned.
 * @param [in] userData Pointer to userData.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 5.0.0(12)
 */
OH_NN_ReturnCode HMS_HiAIOptions_SetUserData(OH_NNCompilation* compilation, void* userData);

/**
 * @brief Sets the custom operator path.
 *
 * This method takes effect only in the model building phase.
 *
 * @param [in] compilation Pointer to {@link OH_NNCompilation}. The value cannot be null. Otherwise, an error code is
 * returned.
 * @param [in] customPath The custom operator path.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 4.1.0(11)
 */
OH_NN_ReturnCode HMS_HiAIOptions_SetCustomOP(OH_NNCompilation* compilation, const char* customPath);

/**
* @brief Define the Model Cache Strategy
 * @since 4.1.0(11)
 */
typedef enum {
    /** Turn off the model cahce strategy. */
    HIAI_MODEL_CACHE_STRATEGY_OFF = 0,
    /** Turn on the model cache strategy. */
    HIAI_MODEL_CACHE_STRATEGY_ON = 1,
} HiAI_ModelCacheStrategy;

/**
 * @brief Sets the Model Cache Strategy.
 *
 * Sets the Model Cache Strategy.
 *
 * @param [in] compilation Pointer to {@link OH_NNCompilation}. The value cannot be null. Otherwise, an error code is
 * returned.
 * @param [in] modelCacheStrategy model cache strategy {@link HiAI_ModelCacheStrategy}.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 4.1.0(11)
 */

OH_NN_ReturnCode HMS_HiAIOptions_SetModelCacheStrategy(OH_NNCompilation* compilation, HiAI_ModelCacheStrategy modelCacheStrategy);

/**
 * @brief Get the Model Cache Strategy.
 *
 * @param [in] compilation Pointer to {@link OH_NNCompilation}. The value cannot be null. Otherwise, the default value
 * is returned.
 * @return Returns the bandwidth mode {@link HiAI_BandMode} if the operation is successful; returns the default value
 * otherwise.
 * @since 4.1.0(11)
 */
HiAI_ModelCacheStrategy HMS_HiAIOptions_GetModelCacheStrategy(const OH_NNCompilation* compilation);

OH_NN_ReturnCode HMS_HiAIOptions_SetShareWeightsModelId(OH_NNCompilation* compilation, uint32_t srcModelId);
#ifdef __cplusplus
}
#endif

/** @} */
#endif // HIAI_FOUNDATION_OPTIONS_INNER_H
