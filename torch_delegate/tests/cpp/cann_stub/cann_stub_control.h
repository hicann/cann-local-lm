/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <vector>

#include <acl/acl_rt.h>

/// Test-only control interface for the CANN stub.
/// Include this header only from test files, never from production code.
namespace cann_stub {

/// Reset all stub state. Call from GTest SetUp().
void Reset();

/// Returns true if aclInit has been called without a subsequent aclFinalize.
bool IsInitialized();

/// Returns the device ID passed to the most recent aclrtSetDevice call,
/// or -1 if aclrtSetDevice has not been called (or aclrtResetDevice was called).
int32_t CurrentDevice();

/// Returns the number of streams currently alive (created but not yet destroyed).
size_t LiveStreamCount();

/// Returns the number of device memory allocations currently alive.
size_t LiveAllocCount();

/// Injects a one-shot error: the next ACL API call returns err instead of
/// ACL_SUCCESS. The error is consumed after one use. Pass ACL_SUCCESS to
/// cancel a previously set injection.
void SetNextError(aclError err);

/// Returns a snapshot of the sizes of all currently live device allocations.
std::vector<size_t> LiveAllocSizes();

}  // namespace cann_stub
