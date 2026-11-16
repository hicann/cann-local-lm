/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * EpairModelLoader C ABI stub（Python UT 用）。
 *
 * 导出与 epair_model_loader_c.h 同签名的四个符号，不依赖真实 .epair 与设备。
 * 行为经环境变量注入：
 *   TD_STUB_CREATE_STATUS    TdEpairModelLoaderCreate 返回的状态码（默认 0）
 *   TD_STUB_GET_GRAPH_STATUS TdEpairModelLoaderGetGraph 返回的状态码（默认 0）
 * 另导出 TdEpairStubDestroyCount() 供用例断言 Destroy 调用次数。
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 以 C++ 编译但导出 C ABI 符号，供 ctypes 按头文件签名加载。
extern "C" {

typedef void TdEpairModelLoaderHandle;

static thread_local char g_last_error[256];
static uint64_t g_destroy_count = 0;

static int64_t StubStatus(const char *env_name) {
  const char *value = getenv(env_name);
  return (value == NULL || value[0] == '\0') ? 0 : (int64_t)strtoll(value, NULL, 10);
}

static const char *StubStatusName(int64_t status) {
  switch (status) {
    case 0:
      return "SUCCESS";
    case 1:
      return "FAILED";
    case 2:
      return "INVALID_PARAM";
    case 3:
      return "INTERNAL_ERROR";
    default:
      return "UNKNOWN_STATUS";
  }
}

static void StubSetError(const char *stage, int64_t status) {
  (void)snprintf(g_last_error, sizeof(g_last_error), "stub error: %s failed: status=%s", stage, StubStatusName(status));
}

int64_t TdEpairModelLoaderCreate(const char *archive_path, TdEpairModelLoaderHandle **handle) {
  const int64_t status = StubStatus("TD_STUB_CREATE_STATUS");
  if (handle == NULL) {
    StubSetError("Create", 2);
    return 2;
  }
  *handle = NULL;
  if (status != 0) {
    StubSetError("Create", status);
    return status;
  }
  *handle = malloc(1);
  if (*handle == NULL) {
    StubSetError("Create", 3);
    return 3;
  }
  (void)archive_path;
  return 0;
}

int64_t TdEpairModelLoaderGetGraph(TdEpairModelLoaderHandle *handle, void *graph) {
  const int64_t status = StubStatus("TD_STUB_GET_GRAPH_STATUS");
  (void)handle;
  (void)graph;
  if (status != 0) {
    StubSetError("GetGraph", status);
    return status;
  }
  return 0;
}

int64_t TdEpairModelLoaderDestroy(TdEpairModelLoaderHandle *handle) {
  free(handle);
  g_destroy_count++;
  return 0;
}

const char *TdEpairModelLoaderLastError(void) {
  return g_last_error;
}

uint64_t TdEpairStubDestroyCount(void) {
  return g_destroy_count;
}

}  // extern "C"
