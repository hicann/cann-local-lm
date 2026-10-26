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
 * @file hiai_tensor_inner.h
 * @kit HiAIFoundationKit
 * @library libhiai_foundation.so
 * @syscap SystemCapability.AI.HiAIFoundation
 *
 * @brief Auxiliary APIs related to input and output memory during HiAI Foundation model inference.
 *
 * You can call the following APIs to associate aippParam with a tensor or calculate the tensor memory size required for
 * the image format.
 *
 * @since 4.1.0(11)
 */

#ifndef HIAI_FOUNDATION_TENSOR_INNER_H
#define HIAI_FOUNDATION_TENSOR_INNER_H
#include "neural_network_runtime/neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Get the cache status of the NN_Tensor.
 *
 * @param [in] tensor Pointer to {@link NN_TensorDesc}. The value cannot be null. Otherwise, 0 is returned.
 * @param [in] cacheStatus Pointer to cacheStatus.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 5.0.0(12)
 */
OH_NN_ReturnCode HMS_HiAITensor_GetCacheStatus(NN_Tensor* tensor, uint8_t* cacheStatus);

/**
 * @brief Sets cache status for NN_Tensor.
 *
 * @param [in] tensor Pointer to {@link NN_Tensor}. The value cannot be null. Otherwise, 0 is returned.
 * @param [in] cacheStatus cache status to be set.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @see OH_NNTensor_Destroy
 * @since 5.0.0(12)
 */
OH_NN_ReturnCode HMS_HiAITensor_SetCacheStatus(NN_Tensor* tensor, uint8_t cacheStatus);

/**
 * @brief Create NN_Tensor which can only be used by KVCache. The seqlen dimension of KVCache can be dynamically
 * resized.
 *
 * @param [in] tensorDesc Pointer to {@link NN_TensorDesc}. The value cannot be null. Otherwise, null is returned.
 * @param [in] tileSize Used when concat multiple kvcache into a single tensor.
 * @param [in] tileSize Specify the tile size on the headDim dimension.
 * For the ND format, the tile size is equal to headDim; for the NZ format, the tile size is the actual block size.
 * @param [in] deviceID deviceID can be obtained by {@link HIAI_NDK_GetDeviceID}
 * @return Pointer to {@link NN_Tensor}
 * @see HIAI_NDK_GetDeviceID
 */
NN_Tensor* HMS_HiAITensor_CreateKVCache(NN_TensorDesc* tensorDesc, size_t kvCacheNum, size_t tileSize, size_t deviceID);

/**
 * @brief Resize the seqlen dimension of KVCache. New size must be larger than current size.
 *
 * @param [in] tensor Pointer to {@link NN_Tensor}. The value cannot be null. The value must be create by {@link
 * HMS_HiAITensor_CreateKVCache}.
 * @param [in] size New size for the seqlen dimension.
 * @return Function execution result. Returns OH_NN_SUCCESS if the operation is successful; returns an error code
 * otherwise. For details about the error codes, see {@link OH_NN_ReturnCode}.
 */
OH_NN_ReturnCode HMS_HiAITensor_ResizeKVCache(NN_Tensor* tensor, size_t size);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // HIAI_FOUNDATION_TENSOR_INNER_H
