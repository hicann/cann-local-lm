/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/acl_manager.h
 * @brief ACLManager singleton — manages modelId→executor mapping and execution.
 *
 * Evolved from harmony_acl. Key additions:
 *   - Virtual stream/event/context management (for aclrtCreateStream etc.)
 *   - RunAsync support (for aclmdlExecuteAsync)
 *   - Device management (for aclrtSetDevice/GetDeviceCount)
 *
 * The ACLManager is the core of the shim: it translates ACL API calls
 * into NDK API calls via NDKProxy.
 */

#ifndef HIAI_ACL_MANAGER_H
#define HIAI_ACL_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <future> // std::promise/std::shared_future (VirtualEvent) — C3: was missing

#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"

#include "acl_data.h"
#include "ndk/hiai_ndk_nncore.h"
#include "ndk/hiai_ndk_options.h"
#include "ndk/hiai_ndk_executor.h"

// C3: ACL model types (aclmdlConfigHandle / aclmdlDesc / aclmdlModel) +
// UNDEFINED_CONFIG moved to GLOBAL scope (out of namespace hiai) to match the
// vendored external_acl/acl/acl_mdl.h's global opaque typedefs. Real ACL's aclmdl*
// types are C-linkage/global; the shim previously put them in namespace hiai,
// diverging from real ACL and breaking the backend's global aclmdlDesc* usage.
// These structs use only CANNKit (NN_TensorDesc*) + std members (all global), so
// the move is safe; VirtualStream/Event/Context remain in namespace hiai (shim-internal).
// C2: aclmdlConfigHandle moved to external_acl/acl/acl_mdl.h (PUBLIC) so the backend
// can access members directly (shim's non-opaque configHandle; real ACL keeps it opaque
// + uses aclmdlSetConfigOpt — a future migration if real ACL is installed). The struct
// mirrors cann_llm_engine::ModelOptions; the backend fills it, aclmdlFinishLoad reads
// + applies.

// C3: UNDEFINED_CONFIG sentinel for unset config-handle fields. Was a file-scope
// constexpr in acl_manager.cpp (invisible to acl_mdl.cpp's aclmdlLoadFromModel).
inline constexpr int32_t UNDEFINED_CONFIG = -1;

struct aclmdlDesc {
    uint32_t modelId = 0; // set by GetTensorDesc, used by GetCurOutputDims
    std::vector<NN_TensorDesc*> inputDescs;
    std::vector<NN_TensorDesc*> outputDescs;
};

// ===== ACL Model (opaque, wraps OH_NNCompilation*) =====
typedef void* aclmdlModel;

namespace hiai {

// ===== Virtual Stream (software queue + worker thread) =====
struct VirtualStream {
    struct AsyncTask {
        uint32_t modelId;
        const aclmdlDataset* input;
        aclmdlDataset* output;
        int32_t timeoutMs;
        void* userData;
    };
    std::queue<AsyncTask> taskQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    std::thread workerThread;
    bool running = true;
    size_t pendingCount = 0;
};

// ===== Virtual Event (std::promise/future) =====
struct VirtualEvent {
    std::promise<void> promise;
    std::shared_future<void> future;
    bool recorded = false;
    VirtualEvent() : future(promise.get_future().share()) {}
};

// ===== Virtual Context (thread_local device binding) =====
struct VirtualContext {
    int32_t deviceId = 0;
};

class ACLManager {
public:
    ACLManager() = default;
    ACLManager(const ACLManager&) = delete;
    ~ACLManager();

    // ===== Model lifecycle =====
    aclError LoadModel(OH_NNCompilation* compilation, uint32_t* modelId);
    // C3: BuildModel(...) and LoadModel(const aclmdlConfigHandle*, uint32_t*) removed —
    // they referenced ACL graph/ge types (AscendString, ge::ir_option, ACL_MDL_*) not
    // vendored in external_acl/. Re-add in C2 when wiring ModelOptions via the
    // config-handle path, with real ACL graph headers or a local compat enum.
    aclError RunModel(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output);
    aclError RunModelAsync(uint32_t modelId, const aclmdlDataset* input, aclmdlDataset* output, int32_t timeoutMs,
                           void* userData);
    aclError GetTensorDesc(aclmdlDesc* modelDesc, uint32_t modelId);
    aclError UnloadModel(uint32_t modelId);
    aclError SetInputDynamic(uint32_t modelId, aclmdlDataset* dataset, size_t index, const aclmdlIODims* dims);

    // ===== Device management (new: for aclrtSetDevice etc.) =====
    aclError GetDeviceCount(uint32_t* count);
    aclError SetDevice(int32_t deviceId);
    aclError GetDevice(int32_t* deviceId);
    aclError ResetDevice(int32_t deviceId);

    // ===== Virtual stream/event/context (new) =====
    aclError CreateStream(aclrtStream* stream);
    aclError DestroyStream(aclrtStream stream);
    aclError SynchronizeStream(aclrtStream stream);

    aclError CreateEvent(aclrtEvent* event);
    aclError DestroyEvent(aclrtEvent event);
    aclError RecordEvent(aclrtEvent event, aclrtStream stream);
    aclError SynchronizeEvent(aclrtEvent event);
    aclError QueryEventStatus(aclrtEvent event);

    aclError SynchronizeDevice();

    // C3: GetExecutorByModelId made public — aclmdlGetCurOutputDims /
    // aclmdlGetInputDimsRange (acl_mdl.cpp, non-member) look up the executor by
    // modelId. Was private; the accessor was always called from outside the class
    // (masked by earlier build failures). Returns shared_ptr<OH_NNExecutor> — the
    // copy keeps the executor alive beyond the lock (H2(a): prevents UAF if
    // UnloadModel runs concurrently). Callers extract the raw ptr via .get().
    std::shared_ptr<OH_NNExecutor> GetExecutorByModelId(uint32_t modelId);

private:
    VirtualStream* GetStream(aclrtStream stream);

private:
    // Model management
    std::unordered_map<uint32_t, std::shared_ptr<OH_NNExecutor>> executorMap_;
    std::unordered_map<uint32_t, std::vector<NN_Tensor*>> dynamicInput_;
    std::mutex mutex_;

    // Virtual stream management
    std::unordered_map<uint64_t, std::unique_ptr<VirtualStream>> streamMap_;
    uint64_t nextStreamId_ = 1;

    // Virtual event management
    std::unordered_map<uint64_t, std::unique_ptr<VirtualEvent>> eventMap_;
    uint64_t nextEventId_ = 1;

    // Device state
    int32_t currentDeviceId_ = 0;
    size_t deviceCount_ = 0;
};

// ===== Global ACLManager accessor (defined in acl_mdl.cpp) =====
// aclInit() creates the singleton; aclFinalize() destroys it.
// acl_mdl.cpp and acl_rt.cpp call through this global.
extern std::shared_ptr<ACLManager> gACLManager;
extern uint64_t aclInitRefCount;

/// Get the ACLManager singleton. Returns nullptr if not initialized.
inline ACLManager* GetACLManager() { return gACLManager.get(); }

} // namespace hiai

#endif // HIAI_ACL_MANAGER_H
