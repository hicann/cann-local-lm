/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/acl_data.h
 * @brief aclDataBuffer and aclmdlDataset struct definitions.
 *
 * These are the ACL-compatible data carrier types used by the shim.
 * Both ACL and NDK backends use the same types via the ACL API surface.
 */

#ifndef HIAI_ACL_DATA_H
#define HIAI_ACL_DATA_H

#include <cstdint>
#include <cstddef> // size_t (aclmdlIODims) — C3
#include <vector>

struct aclDataBuffer {
    void* data;
    uint64_t size;
    aclDataBuffer(void* dataIn, const uint64_t len) : data(dataIn), size(len) {}
};

// C1: buffers is NON-OWNING (raw aclDataBuffer*) — matches real ACL semantics
// (caller owns + destroys via aclDestroyDataBuffer; aclmdlDestroyDataset frees only
// the container, not the buffers). Pre-fix this was vector<shared_ptr<aclDataBuffer>>
// (taking ownership) → aclmdlDestroyDataset freed the struct via shared_ptr dtor →
// the backend's DestroyTensor (aclDestroyDataBuffer) double-freed (heap corruption).
// AIPP fields removed — not in LLM interface.
struct aclmdlDataset {
    std::vector<aclDataBuffer*> buffers;
};

// C3: aclmdlIODims / aclmdlIODimsRange / ACL_MAX_DIM_CNT moved to vendored
// external_acl/acl/acl_mdl.h (PUBLIC) so the backend sees them too.

#endif // HIAI_ACL_DATA_H
