/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file tests/stub_ndk/stub_neural_network_core.cpp
 * @brief Host stub of libneural_network_core.so — exports OH_NN* (external_ndk 接口)
 *        with canned + controllable (env-var) returns, so NDKProxy dlopens it on host
 *        and the REAL shim (acl*→OH_NN* translation) + REAL backend (NdkDevice*+factory
 *        +ModelManagerImpl+CannRuntime+model_executor) run, exposing backend bugs.
 *
 * Round 1 (A1/A2/B1/B2+B3): canned 4-in/1-out model, RunSync argmax=8, RunAsync sync-fire.
 * Round 2 (this file): controllable test fixture — env vars inject failure / async delay /
 *        service-died, so bug-sweep probes can stress untested backend paths (C+B+A).
 *
 * shim 与 backend 零改。MinGW SHARED,OUTPUT_NAME neural_network_core + SUFFIX .so →
 * 文件名 libneural_network_core.so,NDKProxy bare-name LoadLibraryA 在 exe 目录命中。
 *
 * Env vars (read at each call; unset = canned default = Round 1 behavior preserved):
 *   STUB_COMPILE_FAIL=1     → ConstructWithOfflineModelFile returns nullptr (Load failure)
 *   STUB_RUN_FAIL=1         → RunSync returns OH_NN_FAILED (Run failure)
 *   STUB_RUNASYNC_FAIL=1    → RunAsync fires callback w/ FAILED + returns FAILED
 *   STUB_ASYNC_DELAY_MS=N   → RunAsync fires callback from detached thread after N ms (real async)
 *   STUB_TENSOR_ALLOC_FAIL=1→ OH_NNTensor_Create* return nullptr
 *   STUB_FIRE_SERVICE_DIED=1→ next RunSync/RunAsync fires g_onServiceDied instead of running
 */
#include "neural_network_runtime/neural_network_core.h"

#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <thread>
#include <chrono>

// ── env-flag helpers (controllable test fixture modes)──
static bool envFlag(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}
static long envLong(const char* name, long def)
{
    const char* v = std::getenv(name);
    return v ? std::atol(v) : def;
}

// ── 完成头文件里的不透明类型(header 只有 typedef struct X X;)──
// byte size 不存字段:由 shape×dtype 派生(见 ByteSizeFromDesc,真 NNRt 协议)。
struct NN_TensorDesc {
    const char* name;
    OH_NN_DataType dataType;
    OH_NN_Format format;
    int32_t shape[8];
    size_t shapeLen;
};
struct NN_Tensor {
    void* data;
    size_t size;
};

// ── byte size 推导(真 NNRt 协议:元素数 × 元素字节;无 SetByteSize API)──
// Round 3 fix:首版 GetByteSize/OH_NNTensor_Create 直接读 desc 存储 byteSize 字段,
// 而经引擎路径(Convert2NNTensorDesc)只设 shape/dtype/format → 字段恒 0 →
// TensorManager::CreateTensor 成功但 GetSize()==0(host UT Ensure/ResetLane 字节级
// 断言全塌)。按头文件协议从 shape×dtype 派生(与 canned descs 内嵌值一致:
// input_embed 64×i32=256、attention_mask 64×64×f32=16384、logits 349×f32=1396)。
static size_t ElemSizeOf(OH_NN_DataType dt)
{
    switch (dt) {
        case OH_NN_BOOL:
            return 1;
        case OH_NN_INT8:
            return 1;
        case OH_NN_UINT8:
            return 1;
        case OH_NN_INT16:
            return 2;
        case OH_NN_UINT16:
            return 2;
        case OH_NN_FLOAT16:
            return 2;
        case OH_NN_INT32:
            return 4;
        case OH_NN_UINT32:
            return 4;
        case OH_NN_FLOAT32:
            return 4;
        case OH_NN_INT64:
            return 8;
        case OH_NN_UINT64:
            return 8;
        case OH_NN_FLOAT64:
            return 8;
        default:
            return 0; // OH_NN_UNKNOWN 等
    }
}
static size_t ByteSizeFromDesc(const NN_TensorDesc* t)
{
    if (t == nullptr)
        return 0;
    size_t c = 1;
    for (size_t i = 0; i < t->shapeLen; ++i) {
        c *= static_cast<size_t>(t->shape[i]);
    }
    return c * ElemSizeOf(t->dataType);
}

// ── 静态 canned descs(单例;Destroy no-op)──
static NN_TensorDesc g_in_descs[4] = {
    {"input_embed", OH_NN_INT32, OH_NN_FORMAT_ND, {64, 0}, 1},
    {"position_ids", OH_NN_INT32, OH_NN_FORMAT_ND, {64, 0}, 1},
    {"new_kv_cache_pos", OH_NN_INT32, OH_NN_FORMAT_ND, {64, 0}, 1},
    {"attention_mask", OH_NN_FLOAT32, OH_NN_FORMAT_ND, {64, 64}, 2},
};
static NN_TensorDesc g_out_descs[1] = {
    {"logits", OH_NN_FLOAT32, OH_NN_FORMAT_ND, {1, 349}, 2},
};

// ── Device(Init 设备枚举:HIAI_NDK_GetDeviceID 要 name=="HIAI_F")──
OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t** allDevicesID, uint32_t* deviceCount)
{
    static size_t id = 1;
    if (allDevicesID)
        *allDevicesID = &id;
    if (deviceCount)
        *deviceCount = 1;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNDevice_GetName(size_t deviceID, const char** name)
{
    (void)deviceID;
    if (name)
        *name = "HIAI_F";
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType* deviceType)
{
    (void)deviceID;
    if (deviceType)
        *deviceType = OH_NN_ACCELERATOR;
    return OH_NN_SUCCESS;
}

// ── Compilation(Load+Compile:返假句柄,SUCCESS)──
OH_NNCompilation* OH_NNCompilation_ConstructWithOfflineModelFile(const char* modelPath)
{
    (void)modelPath;
    if (envFlag("STUB_COMPILE_FAIL"))
        return nullptr;
    return reinterpret_cast<OH_NNCompilation*>(uintptr_t(0x1));
}
OH_NNCompilation* OH_NNCompilation_ConstructWithOfflineModelBuffer(const void* modelBuffer, size_t modelSize)
{
    (void)modelBuffer;
    (void)modelSize;
    return reinterpret_cast<OH_NNCompilation*>(uintptr_t(0x1));
}
OH_NNCompilation* OH_NNCompilation_ConstructForCache() { return reinterpret_cast<OH_NNCompilation*>(uintptr_t(0x1)); }
OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation* c, size_t deviceID)
{
    (void)c;
    (void)deviceID;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation* c, OH_NN_PerformanceMode m)
{
    (void)c;
    (void)m;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNCompilation_SetPriority(OH_NNCompilation* c, OH_NN_Priority p)
{
    (void)c;
    (void)p;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNCompilation_SetCache(OH_NNCompilation* c, const char* cachePath, uint32_t version)
{
    (void)c;
    (void)cachePath;
    (void)version;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNCompilation_EnableFloat16(OH_NNCompilation* c, bool enableFloat16)
{
    (void)c;
    (void)enableFloat16;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNCompilation_AddExtensionConfig(OH_NNCompilation* c, const char* configName,
                                                     const void* configValue, const size_t configValueSize)
{
    (void)c;
    (void)configName;
    (void)configValue;
    (void)configValueSize;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNCompilation_ExportCacheToBuffer(OH_NNCompilation* c, const void* buffer, size_t length,
                                                      size_t* modelSize)
{
    (void)c;
    (void)buffer;
    (void)length;
    if (modelSize)
        *modelSize = 0;
    return OH_NN_UNSUPPORTED;
}
OH_NN_ReturnCode OH_NNCompilation_ImportCacheFromBuffer(OH_NNCompilation* c, const void* buffer, size_t modelSize)
{
    (void)c;
    (void)buffer;
    (void)modelSize;
    return OH_NN_UNSUPPORTED;
}
OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation* c)
{
    (void)c;
    return OH_NN_SUCCESS;
}
void OH_NNCompilation_Destroy(OH_NNCompilation** compilation)
{
    if (compilation)
        *compilation = nullptr;
}

// ── TensorDesc(IO 查询:返 canned 字段)──
NN_TensorDesc* OH_NNTensorDesc_Create()
{
    auto* t = static_cast<NN_TensorDesc*>(std::calloc(1, sizeof(NN_TensorDesc)));
    return t;
}
OH_NN_ReturnCode OH_NNTensorDesc_Destroy(NN_TensorDesc** tensorDesc)
{
    if (tensorDesc) {
        std::free(*tensorDesc);
        *tensorDesc = nullptr;
    }
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_SetName(NN_TensorDesc* t, const char* name)
{
    if (t)
        t->name = name;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetName(const NN_TensorDesc* t, const char** name)
{
    if (!t || !name)
        return OH_NN_INVALID_PARAMETER;
    *name = t->name;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_SetDataType(NN_TensorDesc* t, OH_NN_DataType dataType)
{
    if (t)
        t->dataType = dataType;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetDataType(const NN_TensorDesc* t, OH_NN_DataType* dataType)
{
    if (!t || !dataType)
        return OH_NN_INVALID_PARAMETER;
    *dataType = t->dataType;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_SetShape(NN_TensorDesc* t, const int32_t* shape, size_t shapeLength)
{
    if (!t || !shape || shapeLength == 0 || shapeLength > 8)
        return OH_NN_INVALID_PARAMETER;
    std::memcpy(t->shape, shape, shapeLength * sizeof(int32_t));
    t->shapeLen = shapeLength;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetShape(const NN_TensorDesc* t, int32_t** shape, size_t* shapeLength)
{
    if (!t || !shape || !shapeLength)
        return OH_NN_INVALID_PARAMETER;
    *shape = const_cast<int32_t*>(t->shape); // 调用方不释放(头契约)
    *shapeLength = t->shapeLen;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_SetFormat(NN_TensorDesc* t, OH_NN_Format format)
{
    if (t)
        t->format = format;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetFormat(const NN_TensorDesc* t, OH_NN_Format* format)
{
    if (!t || !format)
        return OH_NN_INVALID_PARAMETER;
    *format = t->format;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetElementCount(const NN_TensorDesc* t, size_t* elementCount)
{
    if (!t || !elementCount)
        return OH_NN_INVALID_PARAMETER;
    size_t c = 1;
    for (size_t i = 0; i < t->shapeLen; ++i)
        c *= static_cast<size_t>(t->shape[i]);
    *elementCount = c;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensorDesc_GetByteSize(const NN_TensorDesc* t, size_t* byteSize)
{
    if (!t || !byteSize)
        return OH_NN_INVALID_PARAMETER;
    *byteSize = ByteSizeFromDesc(t); // 从 shape×dtype 派生(见 ElemSizeOf 注释)
    return OH_NN_SUCCESS;
}

// ── Tensor(Execute:malloc-backed)──
NN_Tensor* OH_NNTensor_Create(size_t deviceID, NN_TensorDesc* tensorDesc)
{
    (void)deviceID;
    if (!tensorDesc)
        return nullptr;
    if (envFlag("STUB_TENSOR_ALLOC_FAIL"))
        return nullptr;
    auto* t = static_cast<NN_Tensor*>(std::calloc(1, sizeof(NN_Tensor)));
    t->size = ByteSizeFromDesc(tensorDesc);
    t->data = std::calloc(1, t->size ? t->size : 1);
    return t;
}
NN_Tensor* OH_NNTensor_CreateWithSize(size_t deviceID, NN_TensorDesc* tensorDesc, size_t size)
{
    (void)deviceID;
    if (!tensorDesc)
        return nullptr;
    if (envFlag("STUB_TENSOR_ALLOC_FAIL"))
        return nullptr;
    auto* t = static_cast<NN_Tensor*>(std::calloc(1, sizeof(NN_Tensor)));
    t->size = size;
    t->data = std::calloc(1, size ? size : 1);
    return t;
}
NN_Tensor* OH_NNTensor_CreateWithFd(size_t deviceID, NN_TensorDesc* tensorDesc, int fd, size_t size, size_t offset)
{
    (void)deviceID;
    (void)fd;
    (void)offset;
    if (!tensorDesc)
        return nullptr;
    if (envFlag("STUB_TENSOR_ALLOC_FAIL"))
        return nullptr;
    auto* t = static_cast<NN_Tensor*>(std::calloc(1, sizeof(NN_Tensor)));
    t->size = size;
    t->data = std::calloc(1, size ? size : 1);
    return t;
}
OH_NN_ReturnCode OH_NNTensor_Destroy(NN_Tensor** tensor)
{
    if (!tensor || !*tensor)
        return OH_NN_INVALID_PARAMETER;
    std::free((*tensor)->data);
    std::free(*tensor);
    *tensor = nullptr;
    return OH_NN_SUCCESS;
}
NN_TensorDesc* OH_NNTensor_GetTensorDesc(const NN_Tensor* tensor)
{
    (void)tensor;
    return nullptr;
}
void* OH_NNTensor_GetDataBuffer(const NN_Tensor* tensor) { return tensor ? tensor->data : nullptr; }
OH_NN_ReturnCode OH_NNTensor_GetSize(const NN_Tensor* tensor, size_t* size)
{
    if (!tensor || !size)
        return OH_NN_INVALID_PARAMETER;
    *size = tensor->size;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNTensor_GetFd(const NN_Tensor* tensor, int* fd)
{
    (void)tensor;
    (void)fd;
    return OH_NN_UNSUPPORTED;
}
OH_NN_ReturnCode OH_NNTensor_GetOffset(const NN_Tensor* tensor, size_t* offset)
{
    (void)tensor;
    if (offset)
        *offset = 0;
    return OH_NN_SUCCESS;
}

// ── Executor(Construct/IO/Run)──
static NN_OnRunDone g_onRunDone = nullptr;
static NN_OnServiceDied g_onServiceDied = nullptr;

OH_NNExecutor* OH_NNExecutor_Construct(OH_NNCompilation* compilation)
{
    (void)compilation;
    return reinterpret_cast<OH_NNExecutor*>(uintptr_t(0x2));
}
void OH_NNExecutor_Destroy(OH_NNExecutor** executor)
{
    if (executor)
        *executor = nullptr;
}
OH_NN_ReturnCode OH_NNExecutor_GetInputCount(const OH_NNExecutor* executor, size_t* inputCount)
{
    (void)executor;
    if (inputCount)
        *inputCount = 4;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_GetOutputCount(const OH_NNExecutor* executor, size_t* outputCount)
{
    (void)executor;
    if (outputCount)
        *outputCount = 1;
    return OH_NN_SUCCESS;
}
NN_TensorDesc* OH_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor* executor, size_t index)
{
    (void)executor;
    if (index >= 4)
        return nullptr;
    // 堆拷贝(shim 会 OH_NNTensorDesc_Destroy 之;静态指针 free → 堆损坏)
    auto* t = static_cast<NN_TensorDesc*>(std::malloc(sizeof(NN_TensorDesc)));
    if (t)
        *t = g_in_descs[index];
    return t;
}
NN_TensorDesc* OH_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor* executor, size_t index)
{
    (void)executor;
    if (index >= 1)
        return nullptr;
    auto* t = static_cast<NN_TensorDesc*>(std::malloc(sizeof(NN_TensorDesc)));
    if (t)
        *t = g_out_descs[0];
    return t;
}
OH_NN_ReturnCode OH_NNExecutor_GetInputDimRange(const OH_NNExecutor* executor, size_t index, size_t** minInputDims,
                                                size_t** maxInputDims, size_t* shapeLength)
{
    (void)executor;
    (void)index;
    static size_t lo = 1, hi = 64;
    if (minInputDims)
        *minInputDims = &lo;
    if (maxInputDims)
        *maxInputDims = &hi;
    if (shapeLength)
        *shapeLength = 1;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_GetOutputShape(OH_NNExecutor* executor, uint32_t outputIndex, int32_t** shape,
                                              uint32_t* shapeLength)
{
    (void)executor;
    (void)outputIndex;
    static int32_t sh[2] = {1, 349};
    if (shape)
        *shape = sh;
    if (shapeLength)
        *shapeLength = 2;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_SetOnRunDone(OH_NNExecutor* executor, NN_OnRunDone onRunDone)
{
    (void)executor;
    g_onRunDone = onRunDone;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_SetOnServiceDied(OH_NNExecutor* executor, NN_OnServiceDied onServiceDied)
{
    (void)executor;
    g_onServiceDied = onServiceDied;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor* executor, NN_Tensor* inputs[], size_t inputCount,
                                       NN_Tensor* outputs[], size_t outputCount)
{
    (void)executor;
    (void)inputs;
    (void)inputCount;
    if (envFlag("STUB_FIRE_SERVICE_DIED")) {
        if (g_onServiceDied)
            g_onServiceDied(nullptr);
        return OH_NN_FAILED;
    }
    if (envFlag("STUB_RUN_FAIL"))
        return OH_NN_FAILED;
    if (outputCount == 0 || !outputs || !outputs[0] || !outputs[0]->data)
        return OH_NN_INVALID_PARAMETER;
    float* lg = static_cast<float*>(outputs[0]->data);
    const size_t V = 349;
    if (outputs[0]->size < V * sizeof(float))
        return OH_NN_INVALID_PARAMETER;
    for (size_t i = 0; i < V; ++i)
        lg[i] = -FLT_MAX;
    lg[8] = 0.0f; // argmax=8 → token "e"
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode OH_NNExecutor_RunAsync(OH_NNExecutor* executor, NN_Tensor* inputs[], size_t inputCount,
                                        NN_Tensor* outputs[], size_t outputCount, int32_t timeout, void* userData)
{
    (void)timeout;
    if (envFlag("STUB_FIRE_SERVICE_DIED")) {
        if (g_onServiceDied)
            g_onServiceDied(nullptr);
        return OH_NN_FAILED;
    }
    if (envFlag("STUB_RUNASYNC_FAIL")) {
        if (g_onRunDone)
            g_onRunDone(userData, OH_NN_FAILED, nullptr, 0);
        return OH_NN_FAILED;
    }
    long delay = envLong("STUB_ASYNC_DELAY_MS", 0);
    if (delay > 0) {
        // real async: fire from a detached thread after delay (exposes cross-thread races
        // the Phase-1 sync path hid; probe_real_async checks for double-fire / UAF)
        std::thread([executor, inputs, inputCount, outputs, outputCount, userData, delay]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            OH_NN_ReturnCode r = OH_NNExecutor_RunSync(executor, inputs, inputCount, outputs, outputCount);
            if (g_onRunDone)
                g_onRunDone(userData, r, reinterpret_cast<void**>(outputs), static_cast<int32_t>(outputCount));
        }).detach();
        return OH_NN_SUCCESS;
    }
    // Phase-1 sync-immediate path (Round 1 behavior)
    OH_NN_ReturnCode r = OH_NNExecutor_RunSync(executor, inputs, inputCount, outputs, outputCount);
    if (g_onRunDone)
        g_onRunDone(userData, r, reinterpret_cast<void**>(outputs), static_cast<int32_t>(outputCount));
    return OH_NN_SUCCESS;
}
