/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/epair_model_loader_c.h"

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#include <graph/graph.h>

#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "common/memory_utils.h"
#include "torch_delegate/epair_model_loader.h"

namespace {

// 线程局部错误串：多实例并发下各线程互不干扰，仅在返回非 0 前刷新。
thread_local std::string g_last_error;

void SetLastError(std::string message, td::Status status, const char *path = nullptr) {
  message += ": status=";
  message += td::StatusName(status);
  if (path != nullptr) {
    message += ", path=";
    message += path;
  }
  g_last_error = std::move(message);
}

// C ABI 异常边界：C++ 异常不得穿过 ctypes 边界，否则会触发 std::terminate。
template <typename Callable>
int64_t RunGuarded(const char *stage, const char *path, Callable &&callable) {
  try {
    return callable();
  } catch (const std::exception &e) {
    TDLOGE("TdEpairModelLoader%s exception: %s", stage, e.what());
    try {
      SetLastError(std::string(stage) + " failed: " + e.what(), td::INTERNAL_ERROR, path);
    } catch (...) {
      // 错误串构造失败时保持静默，优先保证状态码返回。
    }
    return td::INTERNAL_ERROR;
  } catch (...) {
    TDLOGE("TdEpairModelLoader%s unknown exception", stage);
    try {
      SetLastError(std::string(stage) + " failed: unknown exception", td::INTERNAL_ERROR, path);
    } catch (...) {
    }
    return td::INTERNAL_ERROR;
  }
}

}  // namespace

int64_t TdEpairModelLoaderCreate(const char *archive_path, TdEpairModelLoaderHandle **handle) {
  return RunGuarded("Create", archive_path, [&]() -> int64_t {
    if (handle == nullptr) {
      TDLOGE("TdEpairModelLoaderCreate failed: handle out param is null");
      SetLastError("Create failed: handle out param is null", td::INVALID_PARAM);
      return td::INVALID_PARAM;
    }
    *handle = nullptr;

    if (archive_path == nullptr || archive_path[0] == '\0') {
      TDLOGE("TdEpairModelLoaderCreate failed: archive path is null or empty");
      SetLastError("Create failed: archive path is null or empty", td::INVALID_PARAM);
      return td::INVALID_PARAM;
    }

    auto loader = td::MakeUnique<td::EpairModelLoader>(archive_path);
    if (loader == nullptr) {
      TDLOGE("TdEpairModelLoaderCreate failed: loader allocation failed, path=%s", archive_path);
      SetLastError("Create failed: loader allocation failed", td::INTERNAL_ERROR, archive_path);
      return td::INTERNAL_ERROR;
    }

    const td::Status init_status = loader->GetInitStatus();
    if (init_status != td::SUCCESS) {
      TDLOGE("TdEpairModelLoaderCreate failed: loader initialization failed, status=%s, path=%s",
             td::StatusName(init_status), archive_path);
      SetLastError("Create failed: archive parse or graph build failed", init_status, archive_path);
      return init_status;
    }

    *handle = loader.release();
    return td::SUCCESS;
  });
}

int64_t TdEpairModelLoaderGetGraph(TdEpairModelLoaderHandle *handle, void *graph /* ge::Graph* */) {
  return RunGuarded("GetGraph", nullptr, [&]() -> int64_t {
    if (handle == nullptr) {
      TDLOGE("TdEpairModelLoaderGetGraph failed: handle is null");
      SetLastError("GetGraph failed: handle is null", td::INTERNAL_ERROR);
      return td::INTERNAL_ERROR;
    }
    if (graph == nullptr) {
      TDLOGE("TdEpairModelLoaderGetGraph failed: graph out param is null");
      SetLastError("GetGraph failed: graph out param is null", td::INVALID_PARAM);
      return td::INVALID_PARAM;
    }

    auto *loader = static_cast<td::EpairModelLoader *>(handle);
    auto *ge_graph = static_cast<ge::Graph *>(graph);
    const td::Status status = loader->GetGEGraph(ge_graph);
    if (status != td::SUCCESS) {
      TDLOGE("TdEpairModelLoaderGetGraph failed: GetGEGraph return status=%s", td::StatusName(status));
      SetLastError("GetGraph failed: copy internal graph to caller graph", status);
      return status;
    }
    return td::SUCCESS;
  });
}

int64_t TdEpairModelLoaderDestroy(TdEpairModelLoaderHandle *handle) {
  return RunGuarded("Destroy", nullptr, [&]() -> int64_t {
    if (handle == nullptr) {
      return td::SUCCESS;
    }
    delete static_cast<td::EpairModelLoader *>(handle);
    return td::SUCCESS;
  });
}

const char *TdEpairModelLoaderLastError(void) {
  return g_last_error.c_str();
}
