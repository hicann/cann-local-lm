/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/ndk/hiai_ndk_nncore.h
 * @brief HIAI_NDK_* — C++ wrapper functions for NDK API (via NDKProxy).
 *
 * Each function resolves the corresponding NDK symbol via NDKProxy::GetSymbol
 * and calls through the function pointer. This avoids direct linking to
 * NDK .so files.
 *
 * Evolved from harmony_acl — added GetInputDimRange, GetOutputShape,
 * RunAsync, SetOnRunDone, SetOnServiceDied declarations.
 */

#ifndef HIAI_ACL_NDK_NNCORE_H
#define HIAI_ACL_NDK_NNCORE_H

#include "neural_network_runtime/neural_network_core.h"
#include <vector>
#include "acl/acl_base.h"

namespace hiai {

// ===== Compilation =====
OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructWithOfflineModelFile(const char* modelPath);
OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructWithOfflineModelBuffer(const void* modelBuffer, size_t modelSize);
OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructForCache();
aclError HIAI_NDK_NNCompilation_SetDevice(OH_NNCompilation* nnCompilation);
aclError HIAI_NDK_NNCompilation_SetPerformanceMode(OH_NNCompilation* nnCompilation,
                                                   OH_NN_PerformanceMode performanceMode);
aclError HIAI_NDK_NNCompilation_SetPriority(OH_NNCompilation* nnCompilation, OH_NN_Priority priority);
aclError HIAI_NDK_NNCompilation_SetCache(OH_NNCompilation* nnCompilation, const char* cachePath, uint32_t version);
aclError HIAI_NDK_NNCompilation_EnableFloat16(OH_NNCompilation* compilation, bool enableFloat16);
aclError HIAI_NDK_NNCompilation_AddExtensionConfig(OH_NNCompilation* compilation, const char* configName,
                                                   const void* configValue, size_t configValueSize);
aclError HIAI_NDK_NNCompilation_ExportCacheToBuffer(OH_NNCompilation* nnCompilation, const void* buffer, size_t length,
                                                    size_t* modelSize);
aclError HIAI_NDK_NNCompilation_ImportCacheFromBuffer(OH_NNCompilation* nnCompilation, const void* buffer,
                                                      size_t modelSize);
aclError HIAI_NDK_NNCompilation_Build(OH_NNCompilation* compilation);
void HIAI_NDK_NNCompilation_Destroy(OH_NNCompilation** compilation);

// ===== TensorDesc =====
NN_TensorDesc* HIAI_NDK_NNTensorDesc_Create();
void HIAI_NDK_NNTensorDesc_Destroy(NN_TensorDesc** tensorDesc);
aclError HIAI_NDK_NNTensorDesc_SetDataType(NN_TensorDesc* tensorDesc, OH_NN_DataType dataType);
aclError HIAI_NDK_NNTensorDesc_GetDataType(const NN_TensorDesc* tensorDesc, OH_NN_DataType* dataType);
aclError HIAI_NDK_NNTensorDesc_SetShape(NN_TensorDesc* tensorDesc, const int32_t* shape, size_t shapeLength);
aclError HIAI_NDK_NNTensorDesc_GetShape(const NN_TensorDesc* tensorDesc, int32_t** shape, size_t* shapeLength);
aclError HIAI_NDK_NNTensorDesc_SetFormat(NN_TensorDesc* tensorDesc, OH_NN_Format format);
aclError HIAI_NDK_NNTensorDesc_GetFormat(const NN_TensorDesc* tensorDesc, OH_NN_Format* format);
aclError HIAI_NDK_NNTensorDesc_GetName(const NN_TensorDesc* tensorDesc, const char** name);
aclError HIAI_NDK_NNTensorDesc_GetByteSize(const NN_TensorDesc* tensorDesc, size_t* byteSize);

// ===== Tensor =====
NN_Tensor* HIAI_NDK_NNTensor_Create(NN_TensorDesc* tensorDesc);
NN_Tensor* HIAI_NDK_NNTensor_CreateWithSize(NN_TensorDesc* tensorDesc, size_t size);
NN_Tensor* HIAI_NDK_NNTensor_CreateWithFd(NN_TensorDesc* tensorDesc, int fd, size_t size, size_t offset);
void HIAI_NDK_NNTensor_Destroy(NN_Tensor** nnTensor);
void* HIAI_NDK_NNTensor_GetDataBuffer(NN_Tensor* nnTensor);
aclError HIAI_NDK_NNTensor_GetSize(NN_Tensor* nnTensor, size_t* size);
aclError HIAI_NDK_NNTensor_GetFd(const NN_Tensor* nnTensor, int* fd);
aclError HIAI_NDK_NNTensor_GetOffset(const NN_Tensor* nnTensor, size_t* offset);

// ===== Executor =====
OH_NNExecutor* HIAI_NDK_NNExecutor_Construct(OH_NNCompilation* nnCompilation);
void HIAI_NDK_NNExecutor_Destroy(OH_NNExecutor** executor);
aclError HIAI_NDK_NNExecutor_GetInputCount(const OH_NNExecutor* executor, size_t* inputCount);
aclError HIAI_NDK_NNExecutor_GetOutputCount(const OH_NNExecutor* executor, size_t* outputCount);
NN_TensorDesc* HIAI_NDK_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor* executor, size_t index);
NN_TensorDesc* HIAI_NDK_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor* executor, size_t index);
aclError HIAI_NDK_NNExecutor_GetInputDimRange(const OH_NNExecutor* executor, size_t index, size_t** minInputDims,
                                              size_t** maxInputDims, size_t* shapeLength);
aclError HIAI_NDK_NNExecutor_GetOutputShape(
    OH_NNExecutor* executor, uint32_t outputIndex, int32_t** shape,
    uint32_t* shapeLength); // C3: non-const (matches CANNKit OH_NNExecutor_GetOutputShape)
aclError HIAI_NDK_NNExecutor_SetOnRunDone(OH_NNExecutor* executor, NN_OnRunDone onRunDone);
aclError HIAI_NDK_NNExecutor_SetOnServiceDied(OH_NNExecutor* executor, NN_OnServiceDied onServiceDied);
aclError HIAI_NDK_NNExecutor_RunSync(OH_NNExecutor* executor, std::vector<NN_Tensor*> inputs,
                                     std::vector<NN_Tensor*> outputs);
aclError HIAI_NDK_NNExecutor_RunAsync(OH_NNExecutor* executor, std::vector<NN_Tensor*> inputs,
                                      std::vector<NN_Tensor*> outputs, int32_t timeout, void* userData);

// ===== Device =====
aclError HIAI_NDK_NNDevice_GetAllDevicesID(const size_t** allDevicesID, uint32_t* deviceCount);
aclError HIAI_NDK_NNDevice_GetName(size_t deviceID, const char** name);
aclError HIAI_NDK_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType* deviceType);
size_t HIAI_NDK_GetDeviceID();

} // namespace hiai

#endif // HIAI_ACL_NDK_NNCORE_H
