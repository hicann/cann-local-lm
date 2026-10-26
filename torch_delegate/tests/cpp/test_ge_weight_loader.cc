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

#include <string>

#include <nlohmann/json.hpp>

#include "backend/ge_weight_loader.h"
#include "backend/parsed_model.h"
#include "common/abi_compat_utils.h"
#include "common/epair_archive_constants.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "torch_delegate/td_types.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "build/tests/cpp/fixtures"
#endif

namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";
const std::string kMissingWeightRecord = std::string(FIXTURES_DIR) + "/annotated_missing_weight_record.epair";
const std::string kBadWeightSize = std::string(FIXTURES_DIR) + "/annotated_bad_weight_size.epair";
const std::string kGraphSource = td::epair_spec::kAnnotatedModelJson;

}  // namespace

namespace td {
class GeWeightLoaderTest : public ::testing::Test {};

TEST_F(GeWeightLoaderTest, ResolvesAndLoadsTensorHolderByPlaceholderName) {
  EpairArchive epair_archive(kAnnotated.c_str());
  const RecordData graph_record = epair_archive.ReadRecord(kGraphSource.c_str());
  ParsedModel model;
  ASSERT_EQ(ParsedModel::Parse(graph_record.AsString(), model), SUCCESS);

  GeWeightLoader loader;
  ASSERT_EQ(GeWeightLoader::Create(epair_archive, model, loader), SUCCESS);

  ASSERT_TRUE(loader.HasTensor("p_weight"));
  const ParsedTensorMeta *meta = loader.GetTensorMeta("p_weight");
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->sizes.size(), 2u);
  EXPECT_EQ(meta->sizes[0], 3);
  EXPECT_EQ(meta->sizes[1], 3);

  TensorHolder holder;
  ASSERT_EQ(loader.ReadTensorData("p_weight", holder), SUCCESS);
  ASSERT_EQ(holder.dims.size(), 2u);
  EXPECT_EQ(holder.dims[0], 3);
  EXPECT_EQ(holder.dims[1], 3);
  EXPECT_EQ(holder.datatype, 7);
  EXPECT_EQ(holder.format, 7);
  EXPECT_EQ(holder.ByteSize(), 36u);
  EXPECT_TRUE(holder.HasValue());
  EXPECT_TRUE(holder.requires_grad);

  TensorHolder bias;
  ASSERT_EQ(loader.ReadTensorData("b_bias", bias), SUCCESS);
  ASSERT_EQ(bias.dims.size(), 1u);
  EXPECT_EQ(bias.dims[0], 3);
  EXPECT_EQ(bias.datatype, 7);
  EXPECT_EQ(bias.ByteSize(), 12u);
}

TEST_F(GeWeightLoaderTest, MissingTensorNameReportsError) {
  EpairArchive epair_archive(kAnnotated.c_str());
  const RecordData graph_record = epair_archive.ReadRecord(kGraphSource.c_str());
  ParsedModel model;
  ASSERT_EQ(ParsedModel::Parse(graph_record.AsString(), model), SUCCESS);

  GeWeightLoader loader;
  ASSERT_EQ(GeWeightLoader::Create(epair_archive, model, loader), SUCCESS);

  TensorHolder holder;
  ASSERT_NE(loader.ReadTensorData("does_not_exist", holder), SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, MissingWeightRecordReportsError) {
  EpairArchive epair_archive(kMissingWeightRecord.c_str());
  const RecordData graph_record = epair_archive.ReadRecord(kGraphSource.c_str());
  ParsedModel model;
  ASSERT_EQ(ParsedModel::Parse(graph_record.AsString(), model), SUCCESS);

  GeWeightLoader loader;
  ASSERT_EQ(GeWeightLoader::Create(epair_archive, model, loader), SUCCESS);

  TensorHolder holder;
  ASSERT_NE(loader.ReadTensorData("p_weight", holder), SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, WeightSizeMismatchReportsError) {
  EpairArchive epair_archive(kBadWeightSize.c_str());
  const RecordData graph_record = epair_archive.ReadRecord(kGraphSource.c_str());
  ParsedModel model;
  ASSERT_EQ(ParsedModel::Parse(graph_record.AsString(), model), SUCCESS);

  GeWeightLoader loader;
  ASSERT_EQ(GeWeightLoader::Create(epair_archive, model, loader), SUCCESS);

  TensorHolder holder;
  ASSERT_NE(loader.ReadTensorData("p_weight", holder), SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorRejectsNonTensorPayload) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_NE(GeWeightLoader::RestoreInlineConstTensor(nlohmann::json{{"i", "1"}}, mapper, "Const_0", holder), SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorAcceptsScalarPayload) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_EQ(
      GeWeightLoader::RestoreInlineConstTensor(
          nlohmann::json{{"t",
                          {{"desc", {{"dtype", "DT_FLOAT"}, {"shape", nlohmann::json::object()}, {"layout", "ND"}}},
                           {"data", "8wS1PQ=="}}}},
          mapper, "ConstScalar", holder),
      SUCCESS);
  ASSERT_TRUE(holder.HasValue());
  EXPECT_TRUE(holder.dims.empty());
  EXPECT_EQ(holder.datatype, 7);
  EXPECT_EQ(holder.format, 7);
  EXPECT_EQ(holder.ByteSize(), sizeof(float));
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorDecodesBase64Payload) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_EQ(GeWeightLoader::RestoreInlineConstTensor(
                nlohmann::json{{"t",
                                {{"desc", {{"dtype", "DT_INT64"}, {"shape", {{"dim", {"2"}}}}, {"layout", "ND"}}},
                                 {"data", "AQAAAAAAAAACAAAAAAAAAA=="}}}},
                mapper, "ConstInt64", holder),
            SUCCESS);
  ASSERT_TRUE(holder.HasValue());
  const auto *values = std::get_if<std::vector<int64_t>>(&holder.value);
  ASSERT_NE(values, nullptr);
  ASSERT_EQ(values->size(), 2u);
  EXPECT_EQ((*values)[0], 1);
  EXPECT_EQ((*values)[1], 2);
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorRejectsUnsupportedDtype) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_NE(GeWeightLoader::RestoreInlineConstTensor(
                nlohmann::json{{"t",
                                {{"desc", {{"dtype", "DT_COMPLEX64"}, {"shape", {{"dim", {"2"}}}}, {"layout", "ND"}}},
                                 {"data", "AQAAAAAAAAACAAAAAAAAAA=="}}}},
                mapper, "Const_1", holder),
            SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorRejectsUnsupportedLayout) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_NE(GeWeightLoader::RestoreInlineConstTensor(
                nlohmann::json{{"t",
                                {{"desc", {{"dtype", "DT_INT64"}, {"shape", {{"dim", {"2"}}}}, {"layout", "NCHW"}}},
                                 {"data", "AQAAAAAAAAACAAAAAAAAAA=="}}}},
                mapper, "Const_2", holder),
            SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorRejectsPackedData) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_NE(GeWeightLoader::RestoreInlineConstTensor(
                nlohmann::json{{"t",
                                {{"desc", {{"dtype", "DT_INT64"}, {"shape", {{"dim", {"2"}}}}, {"layout", "ND"}}},
                                 {"packed_data", true}}}},
                mapper, "Const_3", holder),
            SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorAcceptsEmptyTensorWithoutData) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_EQ(
      GeWeightLoader::RestoreInlineConstTensor(
          nlohmann::json{{"t", {{"desc", {{"dtype", "DT_FLOAT"}, {"shape", {{"dim", {"0"}}}}, {"layout", "ND"}}}}}},
          mapper, "ConstEmpty", holder),
      SUCCESS);
  ASSERT_TRUE(holder.HasValue());
  ASSERT_EQ(holder.dims.size(), 1u);
  EXPECT_EQ(holder.dims[0], 0);
  EXPECT_EQ(holder.datatype, 7);
  EXPECT_EQ(holder.format, 7);
  EXPECT_EQ(holder.ByteSize(), 0u);
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorAcceptsEmptyTensorWithEmptyData) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_EQ(GeWeightLoader::RestoreInlineConstTensor(
                nlohmann::json{{"t",
                                {{"desc", {{"dtype", "DT_INT64"}, {"shape", {{"dim", {"3", "0"}}}}, {"layout", "ND"}}},
                                 {"data", ""}}}},
                mapper, "ConstEmptyData", holder),
            SUCCESS);
  ASSERT_TRUE(holder.HasValue());
  ASSERT_EQ(holder.dims.size(), 2u);
  EXPECT_EQ(holder.dims[0], 3);
  EXPECT_EQ(holder.dims[1], 0);
  EXPECT_EQ(holder.datatype, 5);
  EXPECT_EQ(holder.ByteSize(), 0u);
}

TEST_F(GeWeightLoaderTest, RestoreInlineConstTensorRejectsNonEmptyTensorWithoutData) {
  AtenTensorDescMapper mapper;
  TensorHolder holder;
  ASSERT_NE(
      GeWeightLoader::RestoreInlineConstTensor(
          nlohmann::json{{"t", {{"desc", {{"dtype", "DT_FLOAT"}, {"shape", {{"dim", {"2"}}}}, {"layout", "ND"}}}}}},
          mapper, "ConstMissingData", holder),
      SUCCESS);
  EXPECT_FALSE(holder.HasValue());
}
}  // namespace td
