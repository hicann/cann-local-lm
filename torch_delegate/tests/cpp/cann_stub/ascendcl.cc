/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <acl/acl_rt.h>

#include "cann_stub_control.h"

// ---------------------------------------------------------------------------
// Global stub state
// ---------------------------------------------------------------------------

namespace {

struct CannStubState {
  bool initialized_ = false;
  int32_t current_device_ = -1;
  std::unordered_set<void *> live_streams_;
  std::unordered_map<void *, size_t> live_allocs_;
  std::mutex state_mutex_;
  aclError next_error_ = ACL_SUCCESS;
};

CannStubState g_state;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

/// Consume and return the pending one-shot error (caller must hold state_mutex_).
aclError ConsumeNextError() {
  aclError err = g_state.next_error_;
  g_state.next_error_ = ACL_SUCCESS;
  return err;
}

/// Sentinel type used as the backing allocation for aclrtStream handles.
struct StreamSentinel {};

}  // namespace

// ---------------------------------------------------------------------------
// Test control interface (cann_stub namespace)
// ---------------------------------------------------------------------------

namespace cann_stub {

void Reset() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  g_state.initialized_ = false;
  g_state.current_device_ = -1;
  g_state.live_streams_.clear();
  g_state.live_allocs_.clear();
  g_state.next_error_ = ACL_SUCCESS;
}

bool IsInitialized() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  return g_state.initialized_;
}

int32_t CurrentDevice() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  return g_state.current_device_;
}

size_t LiveStreamCount() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  return g_state.live_streams_.size();
}

size_t LiveAllocCount() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  return g_state.live_allocs_.size();
}

void SetNextError(aclError err) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  g_state.next_error_ = err;
}

std::vector<size_t> LiveAllocSizes() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  std::vector<size_t> sizes;
  sizes.reserve(g_state.live_allocs_.size());
  for (const auto &kv : g_state.live_allocs_) {
    sizes.push_back(kv.second);
  }
  return sizes;
}

}  // namespace cann_stub

// ---------------------------------------------------------------------------
// ACL runtime API stubs
// ---------------------------------------------------------------------------

aclError aclInit(const char * /*configPath*/) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  g_state.initialized_ = true;
  return ACL_SUCCESS;
}

aclError aclFinalize() {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  g_state.initialized_ = false;
  return ACL_SUCCESS;
}

aclError aclrtSetDevice(int32_t deviceId) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  g_state.current_device_ = deviceId;
  return ACL_SUCCESS;
}

aclError aclrtResetDevice(int32_t /*deviceId*/) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  g_state.current_device_ = -1;
  return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  auto *sentinel = new StreamSentinel{};
  *stream = static_cast<aclrtStream>(sentinel);
  g_state.live_streams_.insert(*stream);
  return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  delete static_cast<StreamSentinel *>(stream);
  g_state.live_streams_.erase(stream);
  return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream /*stream*/) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  return ACL_SUCCESS;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy /*policy*/) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  void *p = std::malloc(size);
  *devPtr = p;
  g_state.live_allocs_[p] = size;
  return ACL_SUCCESS;
}

aclError aclrtFree(void *devPtr) {
  std::lock_guard<std::mutex> lk(g_state.state_mutex_);
  if (aclError err = ConsumeNextError(); err != ACL_SUCCESS) {
    return err;
  }
  std::free(devPtr);
  g_state.live_allocs_.erase(devPtr);
  return ACL_SUCCESS;
}
