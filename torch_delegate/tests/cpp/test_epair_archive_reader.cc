/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/epair_archive_reader.h"

// FIXTURES_DIR is injected by CMake. This fallback is only reached outside CMake.
#ifndef FIXTURES_DIR
#define FIXTURES_DIR "build/tests/cpp/fixtures"
#endif

static const std::string kAllNpuEpair = std::string(FIXTURES_DIR) + "/all_npu.epair";
static const std::string kModelJson = "models/model.json";
namespace td {
class EpairArchiveReaderTest : public ::testing::Test {};

// ---------------------------------------------------------------------------
// Test: ReadRecord returns non-empty string for models/model.json
// ---------------------------------------------------------------------------
TEST_F(EpairArchiveReaderTest, ReadModelJsonNonEmpty) {
  EpairArchiveReader reader(kAllNpuEpair);
  const RecordData record = reader.ReadRecord(kModelJson);
  std::string content = record.AsString();
  EXPECT_FALSE(content.empty());
  // Should be valid JSON starting with '{'
  EXPECT_EQ(content.front(), '{');
}

// ---------------------------------------------------------------------------
// Test: HasRecord returns true for existing key, false for missing key
// ---------------------------------------------------------------------------
TEST_F(EpairArchiveReaderTest, HasRecordExistingAndMissing) {
  EpairArchiveReader reader(kAllNpuEpair);
  EXPECT_TRUE(reader.HasRecord(kModelJson));
  EXPECT_FALSE(reader.HasRecord("nonexistent/path.json"));
}

TEST_F(EpairArchiveReaderTest, ListRecordsContainsModelJson) {
  EpairArchiveReader reader(kAllNpuEpair);
  const auto records = reader.ListRecords();
  EXPECT_NE(std::find(records.begin(), records.end(), kModelJson), records.end());
}

// ---------------------------------------------------------------------------
// Test: Constructor throws for non-existent file path
// ---------------------------------------------------------------------------
TEST_F(EpairArchiveReaderTest, ThrowsOnMissingFile) {
  EXPECT_THROW({ EpairArchiveReader reader(std::string("/tmp/does_not_exist_12345.epair")); }, std::exception);
}
}  // namespace td
