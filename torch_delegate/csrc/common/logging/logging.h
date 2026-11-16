/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_DELEGATE_COMMON_LOGGING_LOGGING_H
#define TORCH_DELEGATE_COMMON_LOGGING_LOGGING_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "acl/acl_base_rt.h"

namespace td {
inline const char *TDBaseName(const char *path) {
  const char *base = std::strrchr(path, '/');
  return base == nullptr ? path : base + 1;
}

struct LogSite {
  const char *func;
  const char *file;
  uint32_t line;
};

inline void LogChunked(aclLogLevel level, const LogSite &site, const char *prefix, const std::string &text) {
  constexpr size_t kMaxLogChunkSize = 800;
  const size_t chunks = (text.size() + kMaxLogChunkSize - 1) / kMaxLogChunkSize;
  size_t chunk_idx = 0;
  for (size_t pos = 0; pos < text.size(); pos += kMaxLogChunkSize, ++chunk_idx) {
    aclAppLog(level, site.func, site.file, site.line, "%s [%zu/%zu]: %s", prefix, chunk_idx + 1, chunks,
              text.substr(pos, kMaxLogChunkSize).c_str());
  }
}
}  // namespace td
#define TDLOGI(fmt, ...) aclAppLog(ACL_INFO, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOGE(fmt, ...) aclAppLog(ACL_ERROR, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOGW(fmt, ...) aclAppLog(ACL_WARNING, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOGD(fmt, ...) aclAppLog(ACL_DEBUG, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOG_CHUNKED(level, prefix, text) \
  td::LogChunked(level, td::LogSite{__FUNCTION__, td::TDBaseName(__FILE__), __LINE__}, prefix, text)
#endif
