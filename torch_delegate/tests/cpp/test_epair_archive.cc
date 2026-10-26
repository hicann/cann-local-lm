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

#include <stdexcept>
#include <string>

#include "backend/epair_archive.h"
#include "common/epair_archive_constants.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "build/tests/cpp/fixtures"
#endif

namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";

}  // namespace

TEST(EpairArchiveTest, OpensAnnotatedFixture) {
  td::EpairArchive archive(kAnnotated.c_str());
  EXPECT_TRUE(archive.HasRecord(td::epair_spec::kAnnotatedManifestJson));
  EXPECT_TRUE(archive.HasRecord(td::epair_spec::kAnnotatedModelJson));
}

TEST(EpairArchiveTest, MissingArchiveThrows) {
  EXPECT_THROW({ td::EpairArchive archive("/tmp/does_not_exist_98765.epair"); }, c10::Error);
}
