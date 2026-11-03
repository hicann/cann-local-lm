/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/acl_rt.cpp
 * @brief aclrt* runtime API — gap implementations for the shim.
 *
 * This file implements runtime APIs that are NOT in the harmony_acl reference
 * (acl_mdl.cpp already implements aclrtMalloc/Free/Memset).  All functions
 * delegate to the ACLManager singleton (virtual stream/event/context,
 * device management, host memory, memcpy).
 *
 * See proposal.md section 4.6.2 for the gap API list.
 */
#include "acl/acl_rt.h"
#include "acl_manager.h"
#include "utils/log.h"
#include "utils/assert.h"

#include <cstring>

using namespace hiai;

// ===== Device management =====

aclError aclrtGetDeviceCount(uint32_t* count)
{
    if (count == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->GetDeviceCount(count);
}

aclError aclrtSetDevice(int32_t deviceId)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->SetDevice(deviceId);
}

aclError aclrtGetDevice(int32_t* deviceId)
{
    if (deviceId == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    if (gACLManager == nullptr) {
        *deviceId = 0;
        return ACL_ERROR_INVALID_PARAM;
    }
    return gACLManager->GetDevice(deviceId);
}

aclError aclrtResetDevice(int32_t deviceId)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->ResetDevice(deviceId);
}

// ===== Context (virtual — thread_local device binding) =====

aclError aclrtCreateContext(aclrtContext* context, int32_t deviceId)
{
    if (context == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    // Phase 1: no-op, return a dummy handle. Real context management
    // requires thread_local state tracking.
    (void)deviceId;
    *context = reinterpret_cast<aclrtContext>(1); // non-null dummy
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext context)
{
    (void)context;
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    (void)context;
    return ACL_SUCCESS;
}

// ===== Stream (virtual — queue + worker thread, phase 1: no-op) =====

aclError aclrtCreateStream(aclrtStream* stream)
{
    if (stream == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->CreateStream(stream);
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->DestroyStream(stream);
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->SynchronizeStream(stream);
}

// ===== Event (virtual — std::promise/future) =====

aclError aclrtCreateEvent(aclrtEvent* event)
{
    if (event == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->CreateEvent(event);
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->DestroyEvent(event);
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->RecordEvent(event, stream);
}

aclError aclrtSynchronizeEvent(aclrtEvent event)
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->SynchronizeEvent(event);
}

// ===== Device synchronization =====

aclError aclrtSynchronizeDevice()
{
    if (gACLManager == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    return gACLManager->SynchronizeDevice();
}

// ===== Host memory =====

aclError aclrtMallocHost(void** hostPtr, size_t size)
{
    if (hostPtr == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    if (size == 0) {
        *hostPtr = nullptr;
        return ACL_ERROR_INVALID_PARAM;
    }
    *hostPtr = malloc(size);
    return (*hostPtr != nullptr) ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

aclError aclrtFreeHost(void* hostPtr)
{
    if (hostPtr == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    free(hostPtr);
    return ACL_SUCCESS;
}

// ===== Memcpy =====

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    if (dst == nullptr || src == nullptr)
        return ACL_ERROR_INVALID_PARAM;
    if (destMax < count)
        return ACL_ERROR_INVALID_PARAM;
    // NDK shared memory: all pointers are host-accessible.
    // H2H, H2D, D2H → memcpy (same pointer space)
    // D2D → no-op (same device pointer, nothing to copy)
    (void)kind; // kind doesn't matter for shared memory model
    if (count == 0)
        return ACL_SUCCESS;
    if (memcpy(dst, src, count) != dst) {
        ACL_LOGE("memcpy failed");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}
