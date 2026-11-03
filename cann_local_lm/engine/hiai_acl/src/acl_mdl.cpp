/* Copyright (C) 2020-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cstring>
#include <vector>

#include "acl_manager.h"
#include "acl_data.h"
#include "utils/log.h"
#include "utils/assert.h"
#include "utils/securestl.h"
#include "ndk/hiai_ndk_nncore.h"

using namespace hiai;

std::map<OH_NN_DataType, aclDataType> formatMap = {
    {OH_NN_FLOAT16, ACL_FLOAT16}, {OH_NN_INT8, ACL_INT8},   {OH_NN_UINT8, ACL_UINT8},   {OH_NN_INT16, ACL_INT16},
    {OH_NN_UINT16, ACL_UINT16},   {OH_NN_INT32, ACL_INT32}, {OH_NN_UINT32, ACL_UINT32}, {OH_NN_INT64, ACL_INT64},
    {OH_NN_UINT64, ACL_UINT64},   {OH_NN_BOOL, ACL_BOOL}};

constexpr size_t DATA_MEMORY_ALIGN_SIZE = 32UL;
constexpr int32_t PRIORITY_MIN = 1;
constexpr int32_t PRIORITY_MAX = 3;
// gACLManager and aclInitRefCount are defined in acl_manager.cpp (hiai namespace)
// They are accessed here via `using namespace hiai;` above.

aclError aclInit(const char* configPath)
{
    static std::mutex initMutex;
    const std::lock_guard<std::mutex> locker(initMutex);
    ACL_EXPECT_TRUE_R_WITH_LOG(aclInitRefCount == 0, ACL_ERROR_REPEAT_INITIALIZE, "repeatedly initialized");
    if (configPath == nullptr || strlen(configPath) == 0) { // 目前端侧只支持configpath为空
        gACLManager = hiai::make_shared_nothrow<ACLManager>();
        aclInitRefCount = 1;
        ACL_LOGI("create ACLManager success");
    } else {
        ACL_LOGE("only support nullptr config");
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclFinalize()
{
    static std::mutex initMutex;
    const std::lock_guard<std::mutex> locker(initMutex);
    ACL_EXPECT_TRUE_R_WITH_LOG(aclInitRefCount == 1, ACL_ERROR_INVALID_PARAM, "repeatedly initialized");
    gACLManager.reset();

    aclInitRefCount = 0;
    return ACL_SUCCESS;
}

// C3: `namespace ge { aclgrphBuildModelFromFile/FromData/SaveModel/SaveModelToBuffer/
// DestroyModel }` removed — these graph-build APIs reference ACL graph/ge types
// (AscendString, graphStatus, ModelBufferData, char_t, ge::GRAPH_*) not vendored in
// external_acl/ (acl_mdl.h is empty) nor external_cann/ (CANNKit/NDK only). The NDK
// backend uses the offline-model path (aclmdlLoadFromFile/Execute), not graph-build
// — no callers (verified via workspace search). TODO: re-implement with real ACL
// graph headers when available.

aclmdlConfigHandle* aclmdlCreateConfigHandle() { return new (std::nothrow) aclmdlConfigHandle(); }

aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle* handle)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(handle != nullptr, ACL_ERROR_INVALID_PARAM, "the config to be destroyed is nullptr");
    delete handle;
    handle = nullptr;
    return ACL_SUCCESS;
}

// C3: aclmdlSetConfigOpt removed — signature uses `aclmdlConfigAttr` (ACL graph enum
// not vendored in external_acl/) and body uses ACL_MDL_LOAD_FROM_FILE/MEM. No
static aclError ReadFileToBuffer(const char* file, void** buffer, size_t* size)
{
    if (file == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    FILE* fp = fopen(file, "rb");
    if (fp == nullptr) {
        ACL_LOGE("open file failed: %s", file);
        return ACL_ERROR_FAILURE;
    }
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fileSize <= 0) {
        fclose(fp);
        ACL_LOGE("file size invalid: %ld", fileSize);
        return ACL_ERROR_FAILURE;
    }
    void* data = malloc(static_cast<size_t>(fileSize));
    if (data == nullptr) {
        fclose(fp);
        ACL_LOGE("malloc failed for cache buffer");
        return ACL_ERROR_FAILURE;
    }
    size_t readSize = fread(data, 1, static_cast<size_t>(fileSize), fp);
    fclose(fp);
    if (readSize != static_cast<size_t>(fileSize)) {
        free(data);
        ACL_LOGE("read size mismatch: %zu vs %ld", readSize, fileSize);
        return ACL_ERROR_FAILURE;
    }
    *buffer = data;
    *size = static_cast<size_t>(fileSize);
    return ACL_SUCCESS;
}

aclError aclmdlLoadFromCacheBuffer(const void* buffer, size_t size, uint32_t* modelId)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    ACL_EXPECT_TRUE_R_WITH_LOG(buffer != nullptr && size > 0, ACL_ERROR_INVALID_PARAM, "the cache buffer is invalid");
    OH_NNCompilation* compilation = HIAI_NDK_NNCompilation_ConstructForCache();
    if (compilation == nullptr) {
        ACL_LOGE("ConstructForCache failed");
        return ACL_ERROR_FAILURE;
    }
    auto ret = HIAI_NDK_NNCompilation_ImportCacheFromBuffer(compilation, buffer, size);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("ImportCacheFromBuffer failed");
        HIAI_NDK_NNCompilation_Destroy(&compilation);
        return ACL_ERROR_FAILURE;
    }
    ret = HIAI_NDK_NNCompilation_Build(compilation);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("Build failed (cache restore)");
        HIAI_NDK_NNCompilation_Destroy(&compilation);
        return ACL_ERROR_FAILURE;
    }
    auto aclRet = gACLManager->LoadModel(compilation, modelId);
    HIAI_NDK_NNCompilation_Destroy(&compilation);
    return aclRet;
}

aclError aclmdlLoadFromCacheFile(const char* cacheFilePath, uint32_t* modelId)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    ACL_EXPECT_TRUE_R_WITH_LOG(cacheFilePath != nullptr, ACL_ERROR_INVALID_PARAM, "the cache file path is nullptr");
    void* buffer = nullptr;
    size_t size = 0;
    auto ret = ReadFileToBuffer(cacheFilePath, &buffer, &size);
    if (ret != ACL_SUCCESS)
        return ret;
    auto aclRet = aclmdlLoadFromCacheBuffer(buffer, size, modelId);
    free(buffer);
    return aclRet;
}

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    if (devPtr == nullptr) {
        ACL_LOGE("the pointer to be malloced is nullptr");
        return ACL_ERROR_INVALID_PARAM;
    }
    if (size == 0) {
        ACL_LOGE("the size to be malloced is below zero");
        return ACL_ERROR_INVALID_PARAM;
    }
    if (policy != ACL_MEM_MALLOC_NORMAL_ONLY) {
        ACL_LOGE("the malloc policy is not supported");
        return ACL_ERROR_INVALID_PARAM;
    }
    if ((size + (DATA_MEMORY_ALIGN_SIZE * 2)) < size) {
        ACL_LOGE("the size to be malloced is too large: %zu", size);
        return ACL_ERROR_INVALID_PARAM;
    }

    const size_t appendSize = DATA_MEMORY_ALIGN_SIZE * 2;
    size_t alignedSize = (size + appendSize - 1) / DATA_MEMORY_ALIGN_SIZE * DATA_MEMORY_ALIGN_SIZE;
    *devPtr = malloc(alignedSize);
    if (*devPtr == nullptr) {
        ACL_LOGE("Malloc failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtFree(void* devPtr)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(devPtr != nullptr, ACL_ERROR_INVALID_PARAM, "the free data is nullptr");
    free(devPtr);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(devPtr != nullptr, ACL_ERROR_INVALID_PARAM, "the memset data is nullptr");
    ACL_EXPECT_TRUE_R_WITH_LOG(maxCount > count, ACL_ERROR_INVALID_PARAM, "the memset is larger than max count");
    if (memset(devPtr, static_cast<size_t>(value), count) != devPtr) {
        ACL_LOGE("buffer memset failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

// C3: 5 dataset functions (aclmdlCreateDataset / aclmdlAddDatasetBuffer /
// aclmdlDestroyDataset / aclmdlGetDatasetNumBuffers / aclmdlGetDatasetBuffer) removed
// — DUPLICATE definitions. acl_data.cpp is the canonical home (its comment: "moved
// here for cohesion from reference acl_mdl.cpp") but the move was incomplete — these
// stale copies were never deleted, causing "multiple definition" link errors when an
// exe pulls both .obj files (cann_backend_test is the first consumer). The acl_data.cpp
// definitions (C-linkage via the acl_mdl.h include) are authoritative.

aclmdlDesc* aclmdlCreateDesc() { return new (std::nothrow) aclmdlDesc(); }

aclError aclmdlDestroyDesc(aclmdlDesc* modelDesc)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, ACL_ERROR_INVALID_PARAM, "the modeldesc to destroy is nullptr");
    if (modelDesc->inputDescs.size() != 0) {
        for (size_t i = 0U; i < modelDesc->inputDescs.size(); ++i) {
            if (modelDesc->inputDescs[i] != nullptr) {
                HIAI_NDK_NNTensorDesc_Destroy(&modelDesc->inputDescs[i]);
                modelDesc->inputDescs[i] = nullptr;
            }
        }
    }
    if (modelDesc->outputDescs.size() != 0) {
        for (size_t i = 0U; i < modelDesc->outputDescs.size(); i++) {
            if (modelDesc->outputDescs[i] != nullptr) {
                HIAI_NDK_NNTensorDesc_Destroy(&modelDesc->outputDescs[i]);
                modelDesc->outputDescs[i] = nullptr;
            }
        }
    }
    delete modelDesc;
    modelDesc = nullptr;
    return ACL_SUCCESS;
}

aclError aclmdlGetDesc(aclmdlDesc* modelDesc, uint32_t modelId)
{
    return gACLManager->GetTensorDesc(modelDesc, modelId);
}

size_t aclmdlGetNumInputs(aclmdlDesc* modelDesc)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, 0, "the desc to get input number is nullptr");
    return modelDesc->inputDescs.size();
}

size_t aclmdlGetNumOutputs(aclmdlDesc* modelDesc)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, 0, "the desc to destroy is nullptr");
    return modelDesc->outputDescs.size();
}

size_t aclmdlGetInputSizeByIndex(aclmdlDesc* modelDesc, size_t index)
{
    size_t inputSize = 0;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, 0, "the desc to get input number is nullptr");
    if (index >= modelDesc->inputDescs.size()) {
        ACL_LOGE("index is out of range of inputDescs");
        return 0;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetByteSize(modelDesc->inputDescs[index], &inputSize);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("Getinputsize failed");
    }
    return inputSize;
}

size_t aclmdlGetOutputSizeByIndex(aclmdlDesc* modelDesc, size_t index)
{
    size_t outputSize = 0;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, 0, "the desc to get output number is nullptr");
    if (index >= modelDesc->outputDescs.size()) {
        ACL_LOGE("index is out of range of outputDescs");
        return 0;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetByteSize(modelDesc->outputDescs[index], &outputSize);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("Getoutputsize failed");
    }
    return outputSize;
}

const char* aclmdlGetInputNameByIndex(const aclmdlDesc* modelDesc, size_t index)
{
    const char* name = nullptr;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, nullptr, "the desc to get input name is nullptr");
    if (index >= modelDesc->inputDescs.size()) {
        ACL_LOGE("index is out of the range of inputDescs");
        return name;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetName(modelDesc->inputDescs[index], &name);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetInputName failed");
    }
    return name;
}

const char* aclmdlGetOutputNameByIndex(const aclmdlDesc* modelDesc, size_t index)
{
    const char* name = nullptr;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, nullptr, "the desc to get output name is nullptr");
    if (index >= modelDesc->outputDescs.size()) {
        ACL_LOGE("index is out of the range of outputDescs");
        return name;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetName(modelDesc->outputDescs[index], &name);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetOutputName failed");
    }
    return name;
}

aclFormat aclmdlGetInputFormat(const aclmdlDesc* modelDesc, size_t index)
{
    OH_NN_Format inputFormat = OH_NN_FORMAT_NONE;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, ACL_FORMAT_UNDEFINED, "the desc to get input format is nullptr");
    if (index >= modelDesc->inputDescs.size()) {
        ACL_LOGE("index is out of the range of inputDescs");
        return ACL_FORMAT_UNDEFINED;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetFormat(modelDesc->inputDescs[index], &inputFormat);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetInputName failed");
        return ACL_FORMAT_UNDEFINED;
    }
    if (inputFormat == OH_NN_FORMAT_NCHW) {
        return ACL_FORMAT_NCHW;
    } else if (inputFormat == OH_NN_FORMAT_NHWC) {
        return ACL_FORMAT_NHWC;
    } else if (inputFormat == OH_NN_FORMAT_ND) {
        return ACL_FORMAT_ND;
    }
    return ACL_FORMAT_UNDEFINED;
}

aclFormat aclmdlGetOutputFormat(const aclmdlDesc* modelDesc, size_t index)
{
    OH_NN_Format outputFormat = OH_NN_FORMAT_NONE;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, ACL_FORMAT_UNDEFINED, "the desc to get output format is nullptr");
    if (index >= modelDesc->outputDescs.size()) {
        ACL_LOGE("index is out of the range of outputDescs");
        return ACL_FORMAT_UNDEFINED;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetFormat(modelDesc->outputDescs[index], &outputFormat);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetOutputName failed");
        return ACL_FORMAT_UNDEFINED;
    }
    if (outputFormat == OH_NN_FORMAT_NCHW) {
        return ACL_FORMAT_NCHW;
    } else if (outputFormat == OH_NN_FORMAT_NHWC) {
        return ACL_FORMAT_NHWC;
    } else if (outputFormat == OH_NN_FORMAT_ND) {
        return ACL_FORMAT_ND;
    }
    return ACL_FORMAT_UNDEFINED;
}

aclDataType aclmdlGetInputDataType(const aclmdlDesc* modelDesc, size_t index)
{
    OH_NN_DataType inputType = OH_NN_UNKNOWN;
    aclDataType dataType = ACL_DT_UNDEFINED;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, ACL_DT_UNDEFINED, "the desc to get input datatype is nullptr");
    if (index >= modelDesc->inputDescs.size()) {
        ACL_LOGE("index is out of the range of inputDescs");
        return dataType;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetDataType(modelDesc->inputDescs[index], &inputType);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("getinputName failed");
        return dataType;
    }
    auto it = formatMap.find(inputType);
    if (it != formatMap.end()) {
        dataType = it->second;
    } else {
        ACL_LOGE("don't support this inputformat");
    }
    return dataType;
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc* modelDesc, size_t index)
{
    OH_NN_DataType outputType = OH_NN_UNKNOWN;
    aclDataType dataType = ACL_DT_UNDEFINED;
    ACL_EXPECT_TRUE_R_WITH_LOG(modelDesc != nullptr, ACL_DT_UNDEFINED, "the desc to get output datatype is nullptr");
    if (index >= modelDesc->outputDescs.size()) {
        ACL_LOGE("indexis out of the range of outputDescs");
        return dataType;
    }
    auto ret = HIAI_NDK_NNTensorDesc_GetDataType(modelDesc->outputDescs[index], &outputType);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("getoutputName failed");
        return dataType;
    }
    auto it = formatMap.find(outputType);
    if (it != formatMap.end()) {
        dataType = it->second;
    } else {
        ACL_LOGE("don't support this outputformat");
    }
    return dataType;
}

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    return gACLManager->RunModel(modelId, input, output);
}

aclError aclmdlUnload(uint32_t modelId)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    return gACLManager->UnloadModel(modelId);
}

aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset* dataset, size_t index, const aclmdlIODims* dims)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    return gACLManager->SetInputDynamic(modelId, dataset, index, dims);
}

// ===== Gap APIs (not in harmony_acl reference, needed for LLM backend) =====

aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output, aclrtStream stream)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    // H1: Phase 1 — call RunModel (SYNC) instead of RunModelAsync. RunModelAsync leaks
    // input/output tensors (the NDK callback that should DestroyTensors isn't wired —
    // SetOnRunDone is a stub). RunModel manages tensors inline (create→RunSync→destroy),
    // so no leak. ExecuteAsync is effectively sync (acceptable per SDD §14 #7 Phase-1
    // placeholder). Real async (RunAsync + SetOnRunDone callback forwarding + virtual
    // stream queue) is §15.5.
    (void)stream;
    return gACLManager->RunModel(modelId, input, output);
}

aclError aclmdlGetCurOutputDims(const aclmdlDesc* modelDesc, size_t index, aclmdlIODims* dims)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    ACL_EXPECT_NOT_NULL_R(modelDesc, ACL_ERROR_INVALID_PARAM);
    ACL_EXPECT_NOT_NULL_R(dims, ACL_ERROR_INVALID_PARAM);
    // Look up executor via modelId stored in desc
    auto executorPtr = gACLManager->GetExecutorByModelId(modelDesc->modelId);
    ACL_EXPECT_NOT_NULL_R(executorPtr, ACL_ERROR_INVALID_PARAM);
    auto* executor = executorPtr.get(); // H2(a): raw ptr, executorPtr keeps executor alive
    int32_t* shape = nullptr;
    uint32_t shapeLength = 0;
    auto ret = HIAI_NDK_NNExecutor_GetOutputShape(executor, static_cast<uint32_t>(index), &shape, &shapeLength);
    if (ret != ACL_SUCCESS || shape == nullptr) {
        return ret;
    }
    dims->dimCount = shapeLength;
    for (uint32_t i = 0; i < shapeLength && i < ACL_MAX_DIM_CNT; ++i) {
        dims->dims[i] = static_cast<int64_t>(shape[i]);
    }
    return ACL_SUCCESS;
}

aclError aclmdlGetInputDimsRange(const aclmdlDesc* modelDesc, size_t index, aclmdlIODimsRange* dimsRange)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(gACLManager != nullptr, ACL_ERROR_INVALID_PARAM, "not init aclmanager");
    ACL_EXPECT_NOT_NULL_R(modelDesc, ACL_ERROR_INVALID_PARAM);
    ACL_EXPECT_NOT_NULL_R(dimsRange, ACL_ERROR_INVALID_PARAM);
    auto executorPtr = gACLManager->GetExecutorByModelId(modelDesc->modelId);
    ACL_EXPECT_NOT_NULL_R(executorPtr, ACL_ERROR_INVALID_PARAM);
    auto* executor = executorPtr.get(); // H2(a): raw ptr, executorPtr keeps executor alive
    size_t* minShape = nullptr;
    size_t* maxShape = nullptr;
    size_t shapeLength = 0;
    auto ret = HIAI_NDK_NNExecutor_GetInputDimRange(executor, index, &minShape, &maxShape, &shapeLength);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    // Fill aclmdlIODimsRange: range[i][0] = min, range[i][1] = max
    dimsRange->rangeCount = shapeLength;
    for (size_t i = 0; i < shapeLength && i < ACL_MAX_DIM_CNT; ++i) {
        dimsRange->range[i][0] = static_cast<int64_t>(minShape[i]);
        dimsRange->range[i][1] = static_cast<int64_t>(maxShape[i]);
    }
    return ACL_SUCCESS;
}

// C3: aclmdlGetInputDims / aclmdlGetOutputDims — were MISSING from the shim (the
// backend's ndk_device_model.h GetInputDesc/GetOutputDesc call them; the shim only
// had aclmdlGetCurOutputDims + aclmdlGetInputDimsRange). Implemented by querying the
// NN_TensorDesc shape stored in aclmdlDesc (inputDescs/outputDescs, filled by
// aclmdlGetDesc → ACLManager::GetTensorDesc).
aclError aclmdlGetInputDims(const aclmdlDesc* modelDesc, size_t index, aclmdlIODims* dims)
{
    ACL_EXPECT_NOT_NULL_R(modelDesc, ACL_ERROR_INVALID_PARAM);
    ACL_EXPECT_NOT_NULL_R(dims, ACL_ERROR_INVALID_PARAM);
    if (index >= modelDesc->inputDescs.size())
        return ACL_ERROR_INVALID_PARAM;
    int32_t* shape = nullptr;
    size_t shapeLength = 0;
    auto ret = HIAI_NDK_NNTensorDesc_GetShape(modelDesc->inputDescs[index], &shape, &shapeLength);
    if (ret != ACL_SUCCESS)
        return ret;
    dims->dimCount = (shapeLength < ACL_MAX_DIM_CNT) ? shapeLength : ACL_MAX_DIM_CNT;
    for (size_t i = 0; i < dims->dimCount; ++i) {
        dims->dims[i] = static_cast<int64_t>(shape[i]);
    }
    return ACL_SUCCESS;
}

aclError aclmdlGetOutputDims(const aclmdlDesc* modelDesc, size_t index, aclmdlIODims* dims)
{
    ACL_EXPECT_NOT_NULL_R(modelDesc, ACL_ERROR_INVALID_PARAM);
    ACL_EXPECT_NOT_NULL_R(dims, ACL_ERROR_INVALID_PARAM);
    if (index >= modelDesc->outputDescs.size())
        return ACL_ERROR_INVALID_PARAM;
    int32_t* shape = nullptr;
    size_t shapeLength = 0;
    auto ret = HIAI_NDK_NNTensorDesc_GetShape(modelDesc->outputDescs[index], &shape, &shapeLength);
    if (ret != ACL_SUCCESS)
        return ret;
    dims->dimCount = (shapeLength < ACL_MAX_DIM_CNT) ? shapeLength : ACL_MAX_DIM_CNT;
    for (size_t i = 0; i < dims->dimCount; ++i) {
        dims->dims[i] = static_cast<int64_t>(shape[i]);
    }
    return ACL_SUCCESS;
}
