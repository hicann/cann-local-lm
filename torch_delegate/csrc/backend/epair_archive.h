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

#include <filesystem>
#include <memory>
#include <vector>

#include "common/epair_archive_reader.h"

namespace td {

enum class EpairKind : uint32_t {
  kAnnotated = 0,
  kDeployable = 1,
  kInvalidKind,
};

class EpairArchive {
 public:
  static std::unique_ptr<EpairArchive> TryCreate(const char *archive_path);
  explicit EpairArchive(const char *archive_path);
  ~EpairArchive();

  EpairArchive(const EpairArchive &) = delete;
  EpairArchive &operator=(const EpairArchive &) = delete;

  const char *Path() const noexcept;
  bool HasRecord(const char *record_name) const noexcept;
  RecordData ReadRecord(const char *record_name) const noexcept;

 private:
  std::filesystem::path archive_path_;
  EpairArchiveReader reader_;
  EpairKind kind_ = EpairKind::kAnnotated;
};

}  // namespace td
