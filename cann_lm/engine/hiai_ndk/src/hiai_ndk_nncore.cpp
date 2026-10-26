/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
 */

#include "model_manager/general_model_manager/ndk/hiai_ndk/hiai_ndk_nncore.h"

#include "infra/assertion.h"
#include "infra/log.h"
#include "model_manager/general_model_manager/ndk/hiai_ndk/ndk_proxy.h"

namespace hiai {
Status ConvertOHStatusToHIAI(OH_NN_ReturnCode ohStatus)
{
    static std::map<OH_NN_ReturnCode, Status> NNtohiaiStatus = {
        {OH_NN_SUCCESS, SUCCESS},
        {OH_NN_FAILED, FAILURE},
        {OH_NN_OPERATION_FORBIDDEN, UNINITIALIZED},
        {OH_NN_INVALID_PARAMETER, INVALID_PARAM},
        {OH_NN_TIMEOUT, TIMEOUT},
        {OH_NN_UNSUPPORTED, UNSUPPORTED},
        {OH_NN_MEMORY_ERROR, MEMORY_EXCEPTION},
        {OH_NN_NULL_PTR, INVALID_POINTER},
        {OH_NN_INVALID_PATH, FILE_NOT_EXIST},
        {OH_NN_MEMORY_ERROR, DATA_OVERFLOW},
        {OH_NN_CONNECTION_EXCEPTION, CONNECT_EXCEPTION},
    };

    auto iter = NNtohiaiStatus.find(ohStatus);
    if (iter == NNtohiaiStatus.end()) {
        return FAILURE;
    }
    return iter->second;
}

static size_t g_deviceID = 0;

Status HIAI_NDK_NNCompilation_SetPerformanceMode(OH_NNCompilation* nnCompilation, OH_NN_PerformanceMode performanceMode)
{
    auto setPerformanceModeFunc = reinterpret_cast<decltype(OH_NNCompilation_SetPerformanceMode)*>(
        NDKProxy::GetSymbol("OH_NNCompilation_SetPerformanceMode"));
    HIAI_EXPECT_NOT_NULL(setPerformanceModeFunc);

    OH_NN_ReturnCode retCode = setPerformanceModeFunc(nnCompilation, performanceMode);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructForCache()
{
    auto constructForCacheFunc = reinterpret_cast<decltype(OH_NNCompilation_ConstructForCache)*>(
        NDKProxy::GetSymbol("OH_NNCompilation_ConstructForCache"));
    return constructForCacheFunc == nullptr ? nullptr : constructForCacheFunc();
}

Status HIAI_NDK_NNCompilation_SetCache(OH_NNCompilation* nnCompilation, const char* cachePath, uint32_t version)
{
    auto setCacheFunc = reinterpret_cast<decltype(OH_NNCompilation_SetCache)*>(
        NDKProxy::GetSymbol("OH_NNCompilation_SetCache"));
    HIAI_EXPECT_NOT_NULL(setCacheFunc);

    OH_NN_ReturnCode retCode = setCacheFunc(nnCompilation, cachePath, version);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

OH_NNExecutor* HIAI_NDK_NNExecutor_Construct(OH_NNCompilation* nnCompilation)
{
    auto constructFunc = reinterpret_cast<decltype(OH_NNExecutor_Construct)*>(
        NDKProxy::GetSymbol("OH_NNExecutor_Construct"));
    return constructFunc == nullptr ? nullptr : constructFunc(nnCompilation);
}

Status HIAI_NDK_NNExecutor_RunSync(OH_NNExecutor* executor, std::vector<NN_Tensor*> inputs,
                                   std::vector<NN_Tensor*> outputs)
{
    auto runSyncFunc = reinterpret_cast<decltype(OH_NNExecutor_RunSync)*>(NDKProxy::GetSymbol("OH_NNExecutor_RunSync"));
    HIAI_EXPECT_NOT_NULL(runSyncFunc);

    OH_NN_ReturnCode retCode = runSyncFunc(executor, inputs.data(), inputs.size(), outputs.data(), outputs.size());
    return ConvertOHStatusToHIAI(retCode);
}

Status HIAI_NDK_NNCompilation_SetDevice(OH_NNCompilation* nnCompilation)
{
    size_t deviceID = HIAI_NDK_GetDeviceID();
    HIAI_EXPECT_TRUE(deviceID != 0);

    auto setDeviceFunc = reinterpret_cast<decltype(OH_NNCompilation_SetDevice)*>(
        NDKProxy::GetSymbol("OH_NNCompilation_SetDevice"));
    HIAI_EXPECT_NOT_NULL(setDeviceFunc);

    OH_NN_ReturnCode ret = setDeviceFunc(nnCompilation, deviceID);
    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

void HIAI_NDK_NNCompilation_Destroy(OH_NNCompilation** compilation)
{
    auto destroyCompilationFunc = reinterpret_cast<decltype(OH_NNCompilation_Destroy)*>(
        NDKProxy::GetSymbol("OH_NNCompilation_Destroy"));
    if (destroyCompilationFunc != nullptr) {
        destroyCompilationFunc(compilation);
        compilation = nullptr;
    }
}

size_t HIAI_NDK_GetDeviceID()
{
    if (g_deviceID != 0) {
        return g_deviceID;
    }

    size_t deviceID = 0;
    auto getAllDeviceIdFunc = reinterpret_cast<decltype(OH_NNDevice_GetAllDevicesID)*>(
        NDKProxy::GetSymbol("OH_NNDevice_GetAllDevicesID"));
    auto getNameFunc = reinterpret_cast<decltype(OH_NNDevice_GetName)*>(NDKProxy::GetSymbol("OH_NNDevice_GetName"));
    if (getAllDeviceIdFunc == nullptr || getNameFunc == nullptr) {
        CANN_LLM_LOGE("getAllDeviceIdFunc or getNameFunc is nullptr");
        return deviceID;
    }

    const size_t* allDevicesID = nullptr;
    uint32_t deviceCount = 0;
    OH_NN_ReturnCode ret = getAllDeviceIdFunc(&allDevicesID, &deviceCount);
    if (ret != OH_NN_SUCCESS) {
        CANN_LLM_LOGE("OH_NNDevice_GetAllDevicesID failed");
        return deviceID;
    }
    if (deviceCount == 0) {
        return deviceID;
    }

    for (uint32_t i = 0; i < deviceCount; i++) {
        const char* name = nullptr;
        ret = getNameFunc(allDevicesID[i], &name);
        if (ret != OH_NN_SUCCESS) {
            CANN_LLM_LOGE("OH_NNDevice_GetName failed");
            return deviceID;
        }
        if (name != nullptr && std::string(name) == "HIAI_F") {
            deviceID = allDevicesID[i];
            break;
        }
    }
    g_deviceID = deviceID;
    return deviceID;
}

OH_NNCompilation* HIAI_NDK_NNCompilation_ConstructWithOfflineModelBuffer(const void* modelBuffer, size_t modelSize)
{
    auto constructCompilationFunc = reinterpret_cast<decltype(OH_NNCompilation_ConstructWithOfflineModelBuffer)*>(
        NDKProxy::GetSymbol("OH_NNCompilation_ConstructWithOfflineModelBuffer"));
    return constructCompilationFunc == nullptr ? nullptr : constructCompilationFunc(modelBuffer, modelSize);
}

Status HIAI_NDK_NNCompilation_Build(OH_NNCompilation* compilation)
{
    HIAI_EXPECT_EXEC(HIAI_NDK_NNCompilation_SetDevice(compilation));

    auto buildFunc = reinterpret_cast<decltype(OH_NNCompilation_Build)*>(NDKProxy::GetSymbol("OH_NNCompilation_Build"));
    HIAI_EXPECT_NOT_NULL(buildFunc);

    OH_NN_ReturnCode ret = buildFunc(compilation);
    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNTensorDesc_GetShape(const NN_TensorDesc* tensorDesc, int32_t** shape, size_t* shapeLength)
{
    auto getShapeFunc = reinterpret_cast<decltype(OH_NNTensorDesc_GetShape)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_GetShape"));
    HIAI_EXPECT_NOT_NULL(getShapeFunc);
    OH_NN_ReturnCode ret = getShapeFunc(tensorDesc, shape, shapeLength);

    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNTensorDesc_GetDataType(const NN_TensorDesc* tensorDesc, OH_NN_DataType* dataType)
{
    auto getDataTypeFunc = reinterpret_cast<decltype(OH_NNTensorDesc_GetDataType)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_GetDataType"));
    HIAI_EXPECT_NOT_NULL(getDataTypeFunc);
    OH_NN_ReturnCode ret = getDataTypeFunc(tensorDesc, dataType);

    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNTensorDesc_GetFormat(const NN_TensorDesc* tensorDesc, OH_NN_Format* format)
{
    auto getFormatFunc = reinterpret_cast<decltype(OH_NNTensorDesc_GetFormat)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_GetFormat"));
    HIAI_EXPECT_NOT_NULL(getFormatFunc);
    OH_NN_ReturnCode ret = getFormatFunc(tensorDesc, format);

    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNTensorDesc_GetName(const NN_TensorDesc* tensorDesc, const char** name)
{
    auto getNameFunc = reinterpret_cast<decltype(OH_NNTensorDesc_GetName)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_GetName"));
    HIAI_EXPECT_NOT_NULL(getNameFunc);

    OH_NN_ReturnCode ret = getNameFunc(tensorDesc, name);
    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNExecutor_GetInputCount(const OH_NNExecutor* executor, size_t* inputCount)
{
    auto getInputCountFunc = reinterpret_cast<decltype(OH_NNExecutor_GetInputCount)*>(
        NDKProxy::GetSymbol("OH_NNExecutor_GetInputCount"));
    HIAI_EXPECT_NOT_NULL(getInputCountFunc);
    OH_NN_ReturnCode ret = getInputCountFunc(executor, inputCount);

    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNExecutor_GetOutputCount(const OH_NNExecutor* executor, size_t* outputCount)
{
    auto getOutputCountFunc = reinterpret_cast<decltype(OH_NNExecutor_GetOutputCount)*>(
        NDKProxy::GetSymbol("OH_NNExecutor_GetOutputCount"));
    HIAI_EXPECT_NOT_NULL(getOutputCountFunc);
    OH_NN_ReturnCode ret = getOutputCountFunc(executor, outputCount);

    return ret == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

void HIAI_NDK_NNExecutor_Destroy(OH_NNExecutor** executor)
{
    if (executor != nullptr) {
        auto executorDestroyFunc = reinterpret_cast<decltype(OH_NNExecutor_Destroy)*>(
            NDKProxy::GetSymbol("OH_NNExecutor_Destroy"));
        if (executorDestroyFunc != nullptr) {
            executorDestroyFunc(executor);
            executor = nullptr;
        }
    }
}

NN_TensorDesc* HIAI_NDK_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor* executor, size_t index)
{
    auto createInputTensorDescFunc = reinterpret_cast<decltype(OH_NNExecutor_CreateInputTensorDesc)*>(
        NDKProxy::GetSymbol("OH_NNExecutor_CreateInputTensorDesc"));

    return createInputTensorDescFunc != nullptr ? createInputTensorDescFunc(executor, index) : nullptr;
}

NN_TensorDesc* HIAI_NDK_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor* executor, size_t index)
{
    auto createOutputTensorDescFunc = reinterpret_cast<decltype(OH_NNExecutor_CreateOutputTensorDesc)*>(
        NDKProxy::GetSymbol("OH_NNExecutor_CreateOutputTensorDesc"));

    return createOutputTensorDescFunc != nullptr ? createOutputTensorDescFunc(executor, index) : nullptr;
}

void HIAI_NDK_NNTensorDesc_Destroy(NN_TensorDesc** tensorDesc)
{
    auto destroyTensorDescFunc = reinterpret_cast<decltype(OH_NNTensorDesc_Destroy)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_Destroy"));
    if (destroyTensorDescFunc != nullptr) {
        destroyTensorDescFunc(tensorDesc);
        tensorDesc = nullptr;
    }
}

NN_TensorDesc* HIAI_NDK_NNTensorDesc_Create()
{
    auto createTensorDescFunc = reinterpret_cast<decltype(OH_NNTensorDesc_Create)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_Create"));
    HIAI_EXPECT_NOT_NULL_R(createTensorDescFunc, nullptr);
    NN_TensorDesc* tensorDesc = createTensorDescFunc();
    return tensorDesc;
}

Status HIAI_NDK_NNTensorDesc_SetShape(NN_TensorDesc* tensorDesc, const int32_t* shape, size_t shapeLength)
{
    auto setShapeFunc = reinterpret_cast<decltype(OH_NNTensorDesc_SetShape)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_SetShape"));
    HIAI_EXPECT_NOT_NULL(setShapeFunc);
    OH_NN_ReturnCode retCode = setShapeFunc(tensorDesc, shape, shapeLength);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNTensorDesc_SetDataType(NN_TensorDesc* tensorDesc, OH_NN_DataType dataType)
{
    auto setDataTypeFunc = reinterpret_cast<decltype(OH_NNTensorDesc_SetDataType)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_SetDataType"));
    HIAI_EXPECT_NOT_NULL(setDataTypeFunc);
    OH_NN_ReturnCode retCode = setDataTypeFunc(tensorDesc, dataType);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

Status HIAI_NDK_NNTensorDesc_SetFormat(NN_TensorDesc* tensorDesc, OH_NN_Format format)
{
    auto setFormatFunc = reinterpret_cast<decltype(OH_NNTensorDesc_SetFormat)*>(
        NDKProxy::GetSymbol("OH_NNTensorDesc_SetFormat"));
    HIAI_EXPECT_NOT_NULL(setFormatFunc);
    OH_NN_ReturnCode retCode = setFormatFunc(tensorDesc, format);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

NN_Tensor* HIAI_NDK_NNTensor_Create(NN_TensorDesc* tensorDesc)
{
    size_t deviceID = HIAI_NDK_GetDeviceID();
    HIAI_EXPECT_TRUE_R(deviceID != 0, nullptr);

    auto createNNTensorFunc = reinterpret_cast<decltype(OH_NNTensor_Create)*>(
        NDKProxy::GetSymbol("OH_NNTensor_Create"));
    return createNNTensorFunc == nullptr ? nullptr : createNNTensorFunc(deviceID, tensorDesc);
}

NN_Tensor* HIAI_NDK_NNTensor_CreateWithFd(NN_TensorDesc* tensorDesc, int fd, size_t size, size_t offset)
{
    size_t deviceID = HIAI_NDK_GetDeviceID();
    HIAI_EXPECT_TRUE_R(deviceID != 0, nullptr);

    auto createNNTensorWithFdFunc = reinterpret_cast<decltype(OH_NNTensor_CreateWithFd)*>(
        NDKProxy::GetSymbol("OH_NNTensor_CreateWithFd"));
    return createNNTensorWithFdFunc == nullptr ? nullptr :
                                                 createNNTensorWithFdFunc(deviceID, tensorDesc, fd, size, offset);
}

void HIAI_NDK_NNTensor_Destroy(NN_Tensor** nnTensor)
{
    auto destroyNNTensorFunc = reinterpret_cast<decltype(OH_NNTensor_Destroy)*>(
        NDKProxy::GetSymbol("OH_NNTensor_Destroy"));
    if (destroyNNTensorFunc != nullptr) {
        destroyNNTensorFunc(nnTensor);
        nnTensor = nullptr;
    }
}

void* HIAI_NDK_NNTensor_GetDataBuffer(NN_Tensor* nnTensor)
{
    auto getDataBufferFunc = reinterpret_cast<decltype(OH_NNTensor_GetDataBuffer)*>(
        NDKProxy::GetSymbol("OH_NNTensor_GetDataBuffer"));

    return getDataBufferFunc == nullptr ? nullptr : getDataBufferFunc(nnTensor);
}

Status HIAI_NDK_NNTensor_GetSize(NN_Tensor* nnTensor, size_t* size)
{
    auto getSizeFunc = reinterpret_cast<decltype(OH_NNTensor_GetSize)*>(NDKProxy::GetSymbol("OH_NNTensor_GetSize"));
    HIAI_EXPECT_NOT_NULL(getSizeFunc);

    OH_NN_ReturnCode retCode = getSizeFunc(nnTensor, size);
    return retCode == OH_NN_SUCCESS ? SUCCESS : FAILURE;
}

} // namespace hiai
