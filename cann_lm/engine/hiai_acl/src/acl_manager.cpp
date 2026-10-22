/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/acl_manager.cpp
 * @brief ACLManager singleton — model lifecycle, execution, virtual stream/event/context.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/acl_manager.cpp
 * with additions for: virtual stream/event/context, RunModelAsync, device management.
 *
 * JSON device config parsing (parseJsonFile etc.) is stubbed — not needed
 * for the LLM use case.  Can be re-enabled if device config is required.
 */
#include "acl_manager.h"

#include <type_traits>
#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <cstring>

#include "utils/log.h"
#include "utils/assert.h"
#include "utils/string_utils.h"
#include "utils/ir_option.h"

#include "acl_data.h"
#include "ndk/hiai_ndk_tensor.h"

using namespace hiai;

// ===== Global singleton (defined here, declared extern in acl_manager.h) =====
std::shared_ptr<ACLManager> hiai::gACLManager = nullptr;
uint64_t hiai::aclInitRefCount = 0;

// ===== Constants =====
constexpr size_t DATA_MEMORY_ALIGN_SIZE = 32UL;
constexpr int32_t PRIORITY_MIN = 1;
constexpr int32_t PRIORITY_MAX = 3;
// C3: UNDEFINED_CONFIG moved to acl_manager.h (shared with acl_mdl.cpp's aclmdlLoadFromModel).

// ===== Helper functions =====

static void DestroyTensors(std::vector<NN_Tensor*>& tensors)
{
    for (auto tensor : tensors) {
        HIAI_NDK_NNTensor_Destroy(&tensor);
    }
    tensors.clear();
}

static aclError SetInputTensorData(std::vector<NN_Tensor*> inputTensors, const aclmdlDataset* input)
{
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        void* data = HIAI_NDK_NNTensor_GetDataBuffer(inputTensors[i]);
        size_t dataSize = 0;
        HIAI_NDK_NNTensor_GetSize(inputTensors[i], &dataSize);
        if (data == nullptr || dataSize != input->buffers[i]->size) {
            ACL_LOGE("invalid data or dataSize");
            return ACL_ERROR_FAILURE;
        }
        if (memcpy(data, input->buffers[i]->data, input->buffers[i]->size) != data) {
            ACL_LOGE("memcpy failed for input tensor %zu", i);
            return ACL_ERROR_FAILURE;
        }
    }
    return ACL_SUCCESS;
}

static aclError SetOutputTensorData(std::vector<NN_Tensor*> outputTensors, const aclmdlDataset* output)
{
    for (size_t i = 0; i < outputTensors.size(); ++i) {
        void* data = HIAI_NDK_NNTensor_GetDataBuffer(outputTensors[i]);
        size_t dataSize = 0;
        HIAI_NDK_NNTensor_GetSize(outputTensors[i], &dataSize);
        if (data == nullptr || dataSize != output->buffers[i]->size) {
            ACL_LOGE("invalid data or dataSize for output %zu", i);
            return ACL_ERROR_FAILURE;
        }
        if (output->buffers[i]->size < dataSize) {
            ACL_LOGE("output buffer too small");
            return ACL_ERROR_FAILURE;
        }
        if (memcpy(output->buffers[i]->data, data, dataSize) != output->buffers[i]->data) {
            ACL_LOGE("memcpy failed for output tensor %zu", i);
            return ACL_ERROR_FAILURE;
        }
    }
    return ACL_SUCCESS;
}

static aclError PrepareNormalModelInputs(OH_NNExecutor* executor, const aclmdlDataset* input,
    std::vector<NN_Tensor*>& inputTensors)
{
    size_t inputCount = 0;
    auto ret = HIAI_NDK_NNExecutor_GetInputCount(executor, &inputCount);
    if (ret != ACL_SUCCESS || inputCount != input->buffers.size()) {
        ACL_LOGE("OH_NNExecutor_GetInputCount failed or size mismatch: %zu vs %zu",
            inputCount, input->buffers.size());
        return ACL_ERROR_INVALID_PARAM;
    }

    for (size_t i = 0; i < inputCount; ++i) {
        NN_TensorDesc* tensorDesc = HIAI_NDK_NNExecutor_CreateInputTensorDesc(executor, i);
        NN_Tensor* tensor = HIAI_NDK_NNTensor_Create(tensorDesc);
        if (tensor != nullptr) {
            inputTensors.push_back(tensor);
        }
        HIAI_NDK_NNTensorDesc_Destroy(&tensorDesc);
    }
    ret = SetInputTensorData(inputTensors, input);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("SetInputTensorData failed");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

// C3: 5 build-option helpers (ParseInputShape / CreateTensorWithShape /
// SetCompilationShapeConfig / SetCompilationReuseConfig / SetCompilationPrecisionConfig)
// removed — dead (caller BuildModel removed earlier in C3) AND referenced CANNKit
// HiAI_* enums / HIAI_NDK_HiAIOptions_* from the now-excluded hiai_ndk_options.cpp.
// Re-implement in C2 when wiring ModelOptions via the config-handle path (with
// hiai_options.h included + int32_t-vs-HiAI_*Enum signature reconciliation).

// ===== File write helper =====

static aclError WriteBufferToFile(const void* data, uint32_t size, const char* file)
{
    if (file == nullptr) return ACL_ERROR_INVALID_PARAM;
    FILE* fp = fopen(file, "wb");
    if (fp == nullptr) {
        ACL_LOGE("open file failed: %s", file);
        return ACL_ERROR_FAILURE;
    }
    uint32_t writeSize = static_cast<uint32_t>(fwrite(data, 1, size, fp));
    if (writeSize != size) {
        fclose(fp);
        ACL_LOGE("write size mismatch: %u vs %u", writeSize, size);
        return ACL_ERROR_FAILURE;
    }
    fclose(fp);
    return ACL_SUCCESS;
}

// ===== ACLManager methods =====

// C3: ACLManager::BuildModel removed — referenced ACL graph/ge types (AscendString,
// ge::ir_option::INPUT_SHAPE/EXEC_DISABLE_REUSED_MEMORY/PRECISION_MODE_V2) not
// vendored in external_acl/ nor external_cann/. Only caller was the removed
// aclgrphBuildModelFromFile/FromData chain. The 5 build-option helpers below
// (ParseInputShape/CreateTensorWithShape/SetCompilationShapeConfig/
// SetCompilationReuseConfig/SetCompilationPrecisionConfig) are now dead but compile
// cleanly (use CANNKit + utils only) — left in place; remove in a cleanup pass.
// Re-implement BuildModel in C2 with real ACL graph headers when wiring ModelOptions.

aclError ACLManager::LoadModel(OH_NNCompilation* compilation, uint32_t* modelId)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(compilation != nullptr, ACL_ERROR_INVALID_PARAM,
        "the compilation to load is nullptr");
    auto executor = std::shared_ptr<OH_NNExecutor>(
        HIAI_NDK_NNExecutor_Construct(compilation),
        [](OH_NNExecutor* p) { HIAI_NDK_NNExecutor_Destroy(&p); });
    ACL_EXPECT_TRUE_R(executor != nullptr, ACL_ERROR_INVALID_PARAM);
    auto ptr = executor.get();
    auto ret = HIAI_NDK_HiAIExecutor_GetModelID(ptr, modelId);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("get modelid failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    executorMap_.emplace(std::make_pair(*modelId, executor));
    return ACL_SUCCESS;
}

// Anonymous namespace for config-based load helpers
namespace {
aclError SetBuildConfig(OH_NNCompilation* compilation, const aclmdlConfigHandle* handle)
{
    aclError ret = ACL_ERROR_FAILURE;
    if (handle->priority != UNDEFINED_CONFIG) {
        ACL_EXPECT_TRUE(handle->priority <= static_cast<int32_t>(OH_NN_PRIORITY_HIGH) &&
            handle->priority >= 0);
        OH_NN_Priority priority = static_cast<OH_NN_Priority>(handle->priority);
        ret = HIAI_NDK_NNCompilation_SetPriority(compilation, priority);
        if (ret != ACL_SUCCESS) {
            ACL_LOGE("set priority failed");
            return ret;
        }
    }
    if (handle->prefMode != UNDEFINED_CONFIG) {
        OH_NN_PerformanceMode performanceMode = static_cast<OH_NN_PerformanceMode>(handle->prefMode);
        ret = HIAI_NDK_NNCompilation_SetPerformanceMode(compilation, performanceMode);
        if (ret != ACL_SUCCESS) {
            ACL_LOGE("set perfmode failed");
            return ret;
        }
    }
    return ACL_SUCCESS;
}

// C3: SetLoadConfig removed — referenced ACL_MDL_LOAD_FROM_FILE/MEM (ACL graph
// constants not vendored). Was only called by the removed LoadModel(handle) overload.
// Re-implement in C2 with the config-handle path.
}  // anonymous namespace

// C3: ACLManager::LoadModel(const aclmdlConfigHandle*, uint32_t*) removed — called
// the removed SetLoadConfig (ACL_MDL_LOAD_FROM_FILE/MEM). Only caller was the stubbed
// aclmdlLoadWithConfig. Re-implement in C2 when wiring ModelOptions via config handle.
// The LoadModel(OH_NNCompilation*, uint32_t*) overload above is RETAINED (offline-model
// path used by aclmdlLoadFromFile/aclmdlLoadFromMem).

aclError ACLManager::RunModel(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(input != nullptr && output != nullptr, ACL_ERROR_INVALID_PARAM,
        "input or output is nullptr");

    auto executorPtr = GetExecutorByModelId(modelId);
    ACL_EXPECT_NOT_NULL_R(executorPtr, ACL_ERROR_INVALID_PARAM);
    OH_NNExecutor* executor = executorPtr.get();  // H2(a): raw ptr, executorPtr keeps executor alive

    // Create output tensors
    std::vector<NN_Tensor*> outputTensors;
    size_t outputCount = 0;
    auto ret = HIAI_NDK_NNExecutor_GetOutputCount(executor, &outputCount);
    ACL_EXPECT_TRUE_R_WITH_LOG(ret == ACL_SUCCESS, ACL_ERROR_INVALID_PARAM, "GetOutputCount failed");

    for (size_t i = 0; i < outputCount; i++) {
        NN_TensorDesc* tensorDesc = HIAI_NDK_NNExecutor_CreateOutputTensorDesc(executor, i);
        NN_Tensor* tensor = HIAI_NDK_NNTensor_CreateWithSize(tensorDesc, output->buffers[i]->size);
        if (tensor != nullptr) {
            outputTensors.push_back(tensor);
        }
        HIAI_NDK_NNTensorDesc_Destroy(&tensorDesc);
    }

    // H2(b): Check for dynamic input — locked (move tensors out + erase under lock;
    // RunSync + Destroy outside to not hold mutex during NDK execution).
    std::vector<NN_Tensor*> dynamicTensors;
    bool hasDynamic = false;
    {
        std::lock_guard<std::mutex> dynaLock(mutex_);
        auto dyna = dynamicInput_.find(modelId);
        if (dyna != dynamicInput_.end()) {
            dynamicTensors = std::move(dyna->second);
            dynamicInput_.erase(dyna);
            hasDynamic = true;
        }
    }
    if (hasDynamic) {
        ret = HIAI_NDK_NNExecutor_RunSync(executor, dynamicTensors, outputTensors);
        DestroyTensors(dynamicTensors);
        if (ret != ACL_SUCCESS) {  // S4: dynamic path — RunSync failure must not be swallowed
            ACL_LOGE("RunSync failed (dynamic path)");
            DestroyTensors(outputTensors);
            return ACL_ERROR_INVALID_PARAM;
        }
    } else {
        std::vector<NN_Tensor*> inputTensors;
        ret = PrepareNormalModelInputs(executor, input, inputTensors);
        if (ret != ACL_SUCCESS) {
            DestroyTensors(inputTensors);
            DestroyTensors(outputTensors);
            return ACL_ERROR_INVALID_PARAM;
        }
        ret = HIAI_NDK_NNExecutor_RunSync(executor, inputTensors, outputTensors);
        DestroyTensors(inputTensors);
        if (ret != ACL_SUCCESS) {  // S3: non-dynamic path — destroy outputTensors on RunSync failure
            ACL_LOGE("RunSync failed");
            DestroyTensors(outputTensors);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    ret = SetOutputTensorData(outputTensors, output);
    DestroyTensors(outputTensors);
    return ret;
}

// NOTE (review S8/+1A): currently DEAD — aclmdlExecuteAsync now routes to RunModel
// (sync) per H1. The success path leaks input/output tensors because SetOnRunDone is
// a stub (the NDK callback that should DestroyTensors isn't wired). Do NOT call until
// §15.5 real async (RunAsync + callback forwarding + virtual stream queue) is implemented.
aclError ACLManager::RunModelAsync(uint32_t modelId, const aclmdlDataset* input,
    aclmdlDataset* output, int32_t timeoutMs, void* userData)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(input != nullptr && output != nullptr, ACL_ERROR_INVALID_PARAM,
        "input or output is nullptr");

    auto executorPtr = GetExecutorByModelId(modelId);
    ACL_EXPECT_NOT_NULL_R(executorPtr, ACL_ERROR_INVALID_PARAM);
    OH_NNExecutor* executor = executorPtr.get();  // H2(a): raw ptr, executorPtr keeps executor alive

    // Prepare input tensors (same as RunModel)
    std::vector<NN_Tensor*> inputTensors;
    auto ret = PrepareNormalModelInputs(executor, input, inputTensors);
    if (ret != ACL_SUCCESS) {
        DestroyTensors(inputTensors);
        return ACL_ERROR_INVALID_PARAM;
    }

    // Prepare output tensors
    std::vector<NN_Tensor*> outputTensors;
    size_t outputCount = 0;
    ret = HIAI_NDK_NNExecutor_GetOutputCount(executor, &outputCount);
    ACL_EXPECT_TRUE_R_WITH_LOG(ret == ACL_SUCCESS, ACL_ERROR_INVALID_PARAM, "GetOutputCount failed");

    for (size_t i = 0; i < outputCount; i++) {
        NN_TensorDesc* tensorDesc = HIAI_NDK_NNExecutor_CreateOutputTensorDesc(executor, i);
        NN_Tensor* tensor = HIAI_NDK_NNTensor_CreateWithSize(tensorDesc, output->buffers[i]->size);
        if (tensor != nullptr) {
            outputTensors.push_back(tensor);
        }
        HIAI_NDK_NNTensorDesc_Destroy(&tensorDesc);
    }

    // Call async — NDK callback will fire when done, copying data back
    // Note: inputTensors and outputTensors are owned by the NDK callback path.
    // The callback handler must DestroyTensors after copying output data.
    ret = HIAI_NDK_NNExecutor_RunAsync(executor, inputTensors, outputTensors, timeoutMs, userData);
    if (ret != ACL_SUCCESS) {
        DestroyTensors(inputTensors);
        DestroyTensors(outputTensors);
        ACL_LOGE("RunAsync failed");
    }
    return ret;
}

std::shared_ptr<OH_NNExecutor> ACLManager::GetExecutorByModelId(uint32_t modelId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = executorMap_.find(modelId);
    ACL_EXPECT_TRUE_R_WITH_LOG(iter != executorMap_.end(), nullptr, "modelId not found");
    return iter->second;  // H2(a): shared_ptr copy — keeps executor alive beyond the lock
}

aclError ACLManager::GetTensorDesc(aclmdlDesc* modelDesc, uint32_t modelId)
{
    auto executorPtr = GetExecutorByModelId(modelId);
    ACL_EXPECT_NOT_NULL_R(executorPtr, ACL_ERROR_INVALID_PARAM);
    OH_NNExecutor* executor = executorPtr.get();  // H2(a): raw ptr, executorPtr keeps executor alive

    modelDesc->modelId = modelId;  // store for GetCurOutputDims lookup

    size_t inputCount = 0;
    size_t outCount = 0;
    auto ret = HIAI_NDK_NNExecutor_GetInputCount(executor, &inputCount);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetInputCount failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < inputCount; i++) {
        NN_TensorDesc* tensorDesc = HIAI_NDK_NNExecutor_CreateInputTensorDesc(executor, i);
        if (tensorDesc != nullptr) {
            modelDesc->inputDescs.push_back(tensorDesc);
        }
    }
    ret = HIAI_NDK_NNExecutor_GetOutputCount(executor, &outCount);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetOutputCount failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < outCount; i++) {
        NN_TensorDesc* tensorDesc = HIAI_NDK_NNExecutor_CreateOutputTensorDesc(executor, i);
        if (tensorDesc != nullptr) {
            modelDesc->outputDescs.push_back(tensorDesc);
        }
    }
    return ACL_SUCCESS;
}

aclError ACLManager::SetInputDynamic(uint32_t modelId, aclmdlDataset* dataset,
    size_t index, const aclmdlIODims* dims)
{
    (void)index;
    ACL_EXPECT_TRUE_R_WITH_LOG(dataset != nullptr, ACL_ERROR_INVALID_PARAM,
        "dataset is nullptr");

    auto executorPtr = GetExecutorByModelId(modelId);
    ACL_EXPECT_NOT_NULL_R(executorPtr, ACL_ERROR_INVALID_PARAM);
    OH_NNExecutor* executor = executorPtr.get();  // H2(a): raw ptr, executorPtr keeps executor alive

    std::vector<NN_Tensor*> inputTensors;
    size_t inputCount = 0;
    auto ret = HIAI_NDK_NNExecutor_GetInputCount(executor, &inputCount);
    ACL_EXPECT_TRUE_R_WITH_LOG(ret == ACL_SUCCESS && inputCount == dataset->buffers.size(),
        ACL_ERROR_INVALID_PARAM, "GetInputCount failed or size mismatch");

    size_t descCount = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        NN_TensorDesc* tensorDesc = HIAI_NDK_NNExecutor_CreateInputTensorDesc(executor, i);
        int32_t* shape = nullptr;
        size_t shapeLength = 0;
        ret = HIAI_NDK_NNTensorDesc_GetShape(tensorDesc, &shape, &shapeLength);
        if (ret != ACL_SUCCESS) {  // S5: clean up tensorDesc + accumulated inputTensors
            HIAI_NDK_NNTensorDesc_Destroy(&tensorDesc);
            DestroyTensors(inputTensors);
            return ACL_ERROR_INVALID_PARAM;
        }
        if (descCount + shapeLength > dims->dimCount) {
            ACL_LOGE("dynamic dims too small");
            HIAI_NDK_NNTensorDesc_Destroy(&tensorDesc);
            DestroyTensors(inputTensors);  // S5: free prior-iteration tensors
            return ACL_ERROR_INVALID_PARAM;
        }
        std::vector<int32_t> tmpShapes;
        for (size_t j = 0; j < shapeLength; j++) {
            tmpShapes.push_back(static_cast<int32_t>(*(dims->dims + descCount + j)));
        }
        ret = HIAI_NDK_NNTensorDesc_SetShape(tensorDesc, tmpShapes.data(), shapeLength);
        descCount = descCount + shapeLength;
        NN_Tensor* tensor = HIAI_NDK_NNTensor_Create(tensorDesc);
        if (tensor != nullptr) {
            inputTensors.push_back(tensor);
        }
        HIAI_NDK_NNTensorDesc_Destroy(&tensorDesc);
    }
    ret = SetInputTensorData(inputTensors, dataset);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("SetInputTensorData failed");
        DestroyTensors(inputTensors);
        return ACL_ERROR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    dynamicInput_.emplace(std::make_pair(modelId, inputTensors));
    return ACL_SUCCESS;
}

aclError ACLManager::UnloadModel(uint32_t modelId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = executorMap_.find(modelId);
    if (iter == executorMap_.end()) {
        ACL_LOGE("modelId not found: %u", modelId);
        return ACL_ERROR_INVALID_PARAM;
    }
    executorMap_.erase(iter);
    // Also clean up any dynamic input for this model
    auto dynaIter = dynamicInput_.find(modelId);
    if (dynaIter != dynamicInput_.end()) {
        DestroyTensors(dynaIter->second);
        dynamicInput_.erase(dynaIter);
    }
    return ACL_SUCCESS;
}

ACLManager::~ACLManager()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : dynamicInput_) {
        DestroyTensors(pair.second);
    }
    dynamicInput_.clear();
    executorMap_.clear();
    // Clean up virtual streams and events
    streamMap_.clear();
    eventMap_.clear();
}

// ===== Device management (new — gap API support) =====

aclError ACLManager::GetDeviceCount(uint32_t* count)
{
    if (count == nullptr) return ACL_ERROR_INVALID_PARAM;
    const size_t* allDevicesID = nullptr;
    uint32_t deviceCount = 0;
    auto ret = HIAI_NDK_NNDevice_GetAllDevicesID(&allDevicesID, &deviceCount);
    if (ret != ACL_SUCCESS) {
        ACL_LOGE("GetAllDevicesID failed");
        return ret;
    }
    deviceCount_ = deviceCount;
    *count = deviceCount;
    return ACL_SUCCESS;
}

aclError ACLManager::SetDevice(int32_t deviceId)
{
    currentDeviceId_ = deviceId;
    return ACL_SUCCESS;
}

aclError ACLManager::GetDevice(int32_t* deviceId)
{
    if (deviceId == nullptr) return ACL_ERROR_INVALID_PARAM;
    *deviceId = currentDeviceId_;
    return ACL_SUCCESS;
}

aclError ACLManager::ResetDevice(int32_t deviceId)
{
    (void)deviceId;
    // No-op for NDK — device resources managed by executor lifetime
    return ACL_SUCCESS;
}

// ===== Virtual stream (phase 1: no-op, returns SUCCESS) =====

aclError ACLManager::CreateStream(aclrtStream* stream)
{
    if (stream == nullptr) return ACL_ERROR_INVALID_PARAM;
    std::lock_guard<std::mutex> lock(mutex_);
    auto vs = std::make_unique<VirtualStream>();
    uint64_t id = nextStreamId_++;
    streamMap_[id] = std::move(vs);
    *stream = reinterpret_cast<aclrtStream>(id);
    return ACL_SUCCESS;
}

aclError ACLManager::DestroyStream(aclrtStream stream)
{
    uint64_t id = reinterpret_cast<uint64_t>(stream);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = streamMap_.find(id);
    if (iter == streamMap_.end()) {
        ACL_LOGE("stream not found: %llu", static_cast<unsigned long long>(id));
        return ACL_ERROR_INVALID_PARAM;
    }
    // Phase 1: no worker thread to stop. Just erase.
    streamMap_.erase(iter);
    return ACL_SUCCESS;
}

aclError ACLManager::SynchronizeStream(aclrtStream stream)
{
    uint64_t id = reinterpret_cast<uint64_t>(stream);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = streamMap_.find(id);
    if (iter == streamMap_.end()) {
        // Stream may have been destroyed or never created — treat as sync'd
        return ACL_SUCCESS;
    }
    // Phase 1: no queued tasks. Synchronize is a no-op.
    // Phase 2: wait for taskQueue to drain (cv.wait until pendingCount == 0)
    return ACL_SUCCESS;
}

// ===== Virtual event (std::promise/future) =====

aclError ACLManager::CreateEvent(aclrtEvent* event)
{
    if (event == nullptr) return ACL_ERROR_INVALID_PARAM;
    std::lock_guard<std::mutex> lock(mutex_);
    auto ve = std::make_unique<VirtualEvent>();
    uint64_t id = nextEventId_++;
    eventMap_[id] = std::move(ve);
    *event = reinterpret_cast<aclrtEvent>(id);
    return ACL_SUCCESS;
}

aclError ACLManager::DestroyEvent(aclrtEvent event)
{
    uint64_t id = reinterpret_cast<uint64_t>(event);
    std::lock_guard<std::mutex> lock(mutex_);
    eventMap_.erase(id);
    return ACL_SUCCESS;
}

aclError ACLManager::RecordEvent(aclrtEvent event, aclrtStream stream)
{
    uint64_t id = reinterpret_cast<uint64_t>(event);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = eventMap_.find(id);
    if (iter == eventMap_.end() || iter->second == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // Phase 1: immediately set the promise (no real stream queue)
    if (!iter->second->recorded) {
        iter->second->promise.set_value();
        iter->second->recorded = true;
    }
    (void)stream;  // stream not used in phase 1
    return ACL_SUCCESS;
}

aclError ACLManager::SynchronizeEvent(aclrtEvent event)
{
    uint64_t id = reinterpret_cast<uint64_t>(event);
    VirtualEvent* ve = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = eventMap_.find(id);
        if (iter == eventMap_.end() || iter->second == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        ve = iter->second.get();
    }
    // Wait outside lock to avoid deadlock
    if (ve->future.valid()) {
        ve->future.wait();
    }
    return ACL_SUCCESS;
}

aclError ACLManager::QueryEventStatus(aclrtEvent event)
{
    uint64_t id = reinterpret_cast<uint64_t>(event);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = eventMap_.find(id);
    if (iter == eventMap_.end() || iter->second == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (!iter->second->future.valid()) {
        return ACL_ERROR_INVALID_PARAM;  // not recorded yet
    }
    auto status = iter->second->future.wait_for(std::chrono::milliseconds(0));
    return (status == std::future_status::ready) ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

// ===== Device synchronization =====

aclError ACLManager::SynchronizeDevice()
{
    // Phase 1: no pending async ops tracked. No-op.
    // Phase 2: wait for all virtual streams to drain.
    return ACL_SUCCESS;
}

// ===== Private helpers =====

VirtualStream* ACLManager::GetStream(aclrtStream stream)
{
    uint64_t id = reinterpret_cast<uint64_t>(stream);
    auto iter = streamMap_.find(id);
    return (iter != streamMap_.end()) ? iter->second.get() : nullptr;
}

// C2: two-step Load+Compile — compilationMap_ holds pre-build OH_NNCompilation*
// (from aclmdlConstructFromFile/FromMem) until aclmdlFinishLoad (which Build+LoadModel
// + EraseCompilation). EraseCompilation destroys the compilation (matches the existing
// aclmdlLoadFromFile's destroy-after-LoadModel pattern; the executor doesn't hold it).

