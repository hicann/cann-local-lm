/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/acl_data.cpp
 * @brief aclDataBuffer creation/destruction/query — ACL-compatible data carriers.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/acl_data.cpp.
 * No platform-specific code — pure C++ data structure management.
 */
#include "acl_data.h"

#include "acl/acl_base.h"
#include "acl/acl_mdl.h" // C3: aclmdl*Dataset* extern "C" decls — so the definitions below inherit C linkage (matches the backend's C-linkage calls via acl_mdl.h)
#include "utils/assert.h"

#include <new>

// ===== aclDataBuffer =====

aclDataBuffer* aclCreateDataBuffer(void* data, size_t size)
{
    aclDataBuffer* dataBuffer = new (std::nothrow) aclDataBuffer(data, size);
    ACL_EXPECT_NOT_NULL_R(dataBuffer, nullptr);
    return dataBuffer;
}

aclError aclDestroyDataBuffer(const aclDataBuffer* dataBuffer)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(dataBuffer != nullptr, ACL_ERROR_INVALID_PARAM, "the dataBuffer to destroy is nullptr");
    delete dataBuffer;
    return ACL_SUCCESS;
}

void* aclGetDataBufferAddr(const aclDataBuffer* dataBuffer)
{
    ACL_EXPECT_NOT_NULL_R(dataBuffer, nullptr);
    return dataBuffer->data;
}

size_t aclGetDataBufferSizeV2(const aclDataBuffer* dataBuffer)
{
    ACL_EXPECT_NOT_NULL_R(dataBuffer, 0);
    return static_cast<size_t>(dataBuffer->size);
}

// C3: return type fixed to uint32_t to match the vendored declaration in
// external_acl/acl_base_rt.h:324 (ACL_DEPRECATED uint32_t aclGetDataBufferSize).
// The shim previously declared uint64_t — ambiguating the vendored uint32_t
// declaration (redefinition with different return type). dataBuffer->size is
// uint64_t; this narrows, matching the deprecated ACL behavior (use V2 for 64-bit).
uint32_t aclGetDataBufferSize(const aclDataBuffer* dataBuffer)
{
    ACL_EXPECT_NOT_NULL_R(dataBuffer, 0);
    return static_cast<uint32_t>(dataBuffer->size);
}

// ===== aclmdlDataset (from reference acl_mdl.cpp — moved here for cohesion) =====

aclmdlDataset* aclmdlCreateDataset()
{
    aclmdlDataset* dataset = new (std::nothrow) aclmdlDataset();
    ACL_EXPECT_NOT_NULL_R(dataset, nullptr);
    return dataset;
}

aclError aclmdlAddDatasetBuffer(aclmdlDataset* dataset, aclDataBuffer* dataBuffer)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(dataBuffer != nullptr && dataset != nullptr, ACL_ERROR_INVALID_PARAM,
                               "the dataset or dataBuffer is nullptr");
    dataset->buffers.push_back(dataBuffer); // C1: raw, non-owning (matches real ACL)
    return ACL_SUCCESS;
}

aclError aclmdlDestroyDataset(const aclmdlDataset* dataset)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(dataset != nullptr, ACL_ERROR_INVALID_PARAM, "the dataset to destroy is nullptr");
    delete dataset;
    return ACL_SUCCESS;
}

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset* dataset)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(dataset != nullptr, 0, "the dataset to get num is nullptr");
    return dataset->buffers.size();
}

aclDataBuffer* aclmdlGetDatasetBuffer(const aclmdlDataset* dataset, size_t index)
{
    ACL_EXPECT_TRUE_R_WITH_LOG(dataset != nullptr, nullptr, "the dataset to get buffer is nullptr");
    return (index < dataset->buffers.size()) ? dataset->buffers[index] : nullptr; // C1: raw ptr (no shared_ptr wrap)
}
