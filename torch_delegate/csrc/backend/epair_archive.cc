/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "epair_archive.h"

#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "common/abi_compat_utils.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "common/memory_utils.h"

namespace td {

std::unique_ptr<EpairArchive> EpairArchive::TryCreate(const char *archive_path) {
  if (archive_path == nullptr) {
    TDLOGE("EpairArchive::TryCreate failed: archive_path is null");
    return nullptr;
  }
  try {
    std::unique_ptr<EpairArchive> archive = MakeUnique<EpairArchive>(archive_path);
    if (archive == nullptr) {
      TDLOGE("EpairArchive::TryCreate failed: memory allocation failed for %s", archive_path);
      return nullptr;
    }
    return archive;
  } catch (const std::exception &e) {
    TDLOGE("EpairArchive::TryCreate failed: %s for %s", e.what(), archive_path);
    return nullptr;
  }
}

EpairArchive::EpairArchive(const char *archive_path)
    : archive_path_(internal::CharToPath(archive_path)), reader_(archive_path_.string()) {}

EpairArchive::~EpairArchive() = default;

const char *EpairArchive::Path() const noexcept {
  return internal::PathToChar(archive_path_);
}

bool EpairArchive::HasRecord(const char *record_name) const noexcept {
  if (record_name == nullptr) {
    return false;
  }
  return reader_.HasRecord(record_name);
}

RecordData EpairArchive::ReadRecord(const char *record_name) const noexcept {
  TD_ASSERT_NOTNULL(record_name, "Invalid record_name, record_name is null");

  return reader_.ReadRecord(record_name);
}

}  // namespace td
