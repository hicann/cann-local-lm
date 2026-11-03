/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#ifndef HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_NNCORE_H
#define HIAI_FOUNDATION_HIAI_NDK_HIAI_NDK_NNCORE_H

#include "neural_network_runtime/neural_network_core.h"

#include <vector>

#include "types.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_align.h"

namespace hiai {
OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructWithOfflineModelBuffer(const void* modelBuffer, size_t modelSize);

// Build 的内部前置（设备选择），虽不在 LLM 接口表，但为 NNCompilation_Build 的调用闭包
Status HIAI_NDK_NNCompilation_SetDevice(OH_NNCompilation* nnCompilation);

Status HIAI_NDK_NNCompilation_SetPerformanceMode(OH_NNCompilation* nnCompilation,
                                                 OH_NN_PerformanceMode performanceMode);

OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructForCache();

Status HIAI_NDK_NNCompilation_SetCache(OH_NNCompilation* nnCompilation, const char* cachePath, uint32_t version);

Status HIAI_NDK_NNCompilation_Build(OH_NNCompilation* compilation);

void HIAI_NDK_NNCompilation_Destroy(OH_NNCompilation** compilation);

NN_TensorDesc* HIAI_NDK_NNTensorDesc_Create();

void HIAI_NDK_NNTensorDesc_Destroy(NN_TensorDesc** tensorDesc);

Status HIAI_NDK_NNTensorDesc_SetDataType(NN_TensorDesc* tensorDesc, OH_NN_DataType dataType);

Status HIAI_NDK_NNTensorDesc_GetDataType(const NN_TensorDesc* tensorDesc, OH_NN_DataType* dataType);

Status HIAI_NDK_NNTensorDesc_SetShape(NN_TensorDesc* tensorDesc, const int32_t* shape, size_t shapeLength);

Status HIAI_NDK_NNTensorDesc_GetShape(const NN_TensorDesc* tensorDesc, int32_t** shape, size_t* shapeLength);

Status HIAI_NDK_NNTensorDesc_SetFormat(NN_TensorDesc* tensorDesc, OH_NN_Format format);

Status HIAI_NDK_NNTensorDesc_GetFormat(const NN_TensorDesc* tensorDesc, OH_NN_Format* format);

Status HIAI_NDK_NNTensorDesc_GetName(const NN_TensorDesc* tensorDesc, const char** name);

NN_Tensor* HIAI_NDK_NNTensor_Create(NN_TensorDesc* tensorDesc);

NN_Tensor* HIAI_NDK_NNTensor_CreateWithFd(NN_TensorDesc* tensorDesc, int fd, size_t size, size_t offset);

void HIAI_NDK_NNTensor_Destroy(NN_Tensor** nnTensor);

void* HIAI_NDK_NNTensor_GetDataBuffer(NN_Tensor* nnTensor);

Status HIAI_NDK_NNTensor_GetSize(NN_Tensor* nnTensor, size_t* size);

OH_NNExecutor* HIAI_NDK_NNExecutor_Construct(OH_NNCompilation* nnCompilation);

void HIAI_NDK_NNExecutor_Destroy(OH_NNExecutor** executor);

Status HIAI_NDK_NNExecutor_GetInputCount(const OH_NNExecutor* executor, size_t* inputCount);

Status HIAI_NDK_NNExecutor_GetOutputCount(const OH_NNExecutor* executor, size_t* outputCount);

NN_TensorDesc* HIAI_NDK_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor* executor, size_t index);

NN_TensorDesc* HIAI_NDK_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor* executor, size_t index);

Status HIAI_NDK_NNExecutor_RunSync(OH_NNExecutor* executor, std::vector<NN_Tensor*> inputs,
                                   std::vector<NN_Tensor*> outputs);

// 内部调用闭包：NNTensor_Create/CreateWithFd 与 HiAITensor_CreateKVCache 的设备 ID 前置
size_t HIAI_NDK_GetDeviceID();
} // namespace hiai
#endif
