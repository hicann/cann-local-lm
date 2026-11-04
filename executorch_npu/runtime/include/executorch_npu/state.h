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

#include <cstddef>
#include <cstdint>

namespace executorch_npu {

enum class StateStatus {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kInvalidState,
  kResourceError,
};

struct StateTensorData {
  const void *data = nullptr;
  size_t nbytes = 0;
  const char *dtype = nullptr;
  const int64_t *shape = nullptr;
  size_t rank = 0;
};

// Installs an initializer before a stateful method is first executed. The
// bytes and metadata are copied; the caller retains ownership of the view.
StateStatus SetStateInitializer(int device_id, const char *state_namespace, const char *state_id,
                                const StateTensorData &initializer);

// Replaces an existing state whose manifest reset_policy is "explicit".
StateStatus ResetState(int device_id, const char *state_namespace, const char *state_id, const StateTensorData &source);

// Waits for in-flight executions and releases all state in the namespace.
StateStatus ReleaseStateNamespace(int device_id, const char *state_namespace);

}  // namespace executorch_npu
