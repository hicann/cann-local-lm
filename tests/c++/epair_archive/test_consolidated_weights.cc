#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "backend/epair_model_parser.h"
#include "backend/ge_weight_loader.h"
#include "common/abi_compat_utils.h"
#include "common/epair_archive_constants.h"
#include "common/logging/logging.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "fixtures"
#endif

namespace {

const std::string kConsolidated = std::string(FIXTURES_DIR) + "/annotated_consolidated_mm_add";
const std::string kConsolidatedMissingMeta = std::string(FIXTURES_DIR) + "/annotated_consolidated_missing_meta";
const std::string kConsolidatedNoIndex = std::string(FIXTURES_DIR) + "/annotated_consolidated_no_index";

std::string ReadFile(const std::string &path) {
  std::ifstream file(path);
  if (!file) return {};
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

void AssertWeightMatches(const td::TensorHolder &holder, const std::vector<int64_t> &expected_dims,
                          int64_t expected_dtype, int64_t expected_format, size_t expected_bytes,
                          bool expected_requires_grad) {
  ASSERT_EQ(holder.dims.size(), expected_dims.size());
  for (size_t i = 0; i < expected_dims.size(); ++i) {
    EXPECT_EQ(holder.dims[i], expected_dims[i]);
  }
  EXPECT_EQ(holder.datatype, expected_dtype);
  EXPECT_EQ(holder.format, expected_format);
  EXPECT_EQ(holder.ByteSize(), expected_bytes);
  EXPECT_TRUE(holder.HasValue());
  EXPECT_EQ(holder.requires_grad, expected_requires_grad);
}

td::GeWeightLoader CreateLoader(const std::string &dir_path, td::ValidationResult &result) {
  const std::string model_json = ReadFile(dir_path + "/extra/ep_json_with_ascend_ir.json");
  td::EpairModelParser parser = td::EpairModelParser::Parse(model_json, result);
  if (!result.ok) return {};
  return td::GeWeightLoader::Create(dir_path + "/data/weights_index.json", parser.Model(), result);
}

}  // namespace

namespace td {

TEST(GeWeightLoaderConsolidatedTest, LoadsTensorHolderCorrectly) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  ASSERT_TRUE(loader.HasTensor("p_weight"));
  const ParsedTensorMeta *meta = loader.GetTensorMeta("p_weight");
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->sizes.size(), 2u);
  EXPECT_EQ(meta->sizes[0], 3);
  EXPECT_EQ(meta->sizes[1], 3);

  TensorHolder holder = loader.ReadTensorData("p_weight", result);
  ASSERT_TRUE(result.ok);
  AssertWeightMatches(holder, {3, 3}, 7, 7, 36, true);

  TensorHolder bias = loader.ReadTensorData("b_bias", result);
  ASSERT_TRUE(result.ok);
  AssertWeightMatches(bias, {3}, 7, 7, 12, false);
}

TEST(GeWeightLoaderConsolidatedTest, MissingTensorNameReportsError) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  ValidationResult bad_result{true, {}};
  TensorHolder holder = loader.ReadTensorData("does_not_exist", bad_result);
  EXPECT_FALSE(bad_result.ok);
  EXPECT_FALSE(holder.HasValue());
}

TEST(GeWeightLoaderConsolidatedTest, MissingTensorMetaReportsError) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidatedMissingMeta, result);
  EXPECT_FALSE(result.ok);
  const bool found =
      std::any_of(result.issues.begin(), result.issues.end(),
                   [](const auto &issue) { return issue.code == "MISSING_TENSOR_META"; });
  EXPECT_TRUE(found);
}

TEST(GeWeightLoaderConsolidatedTest, NoIndexReportsError) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidatedNoIndex, result);
  EXPECT_FALSE(result.ok);
  const bool found =
      std::any_of(result.issues.begin(), result.issues.end(),
                   [](const auto &issue) { return issue.code == "MISSING_WEIGHTS_INDEX"; });
  EXPECT_TRUE(found);
}

TEST(GeWeightLoaderConsolidatedTest, TensorBinaryValuesCorrect) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  TensorHolder holder = loader.ReadTensorData("p_weight", result);
  ASSERT_TRUE(result.ok);
  ASSERT_EQ(holder.ByteSize(), 36u);

  const auto *values = std::get_if<std::vector<float>>(&holder.value);
  ASSERT_NE(values, nullptr);
  ASSERT_EQ(values->size(), 9u);
  EXPECT_FLOAT_EQ((*values)[0], 1.0f);
  EXPECT_FLOAT_EQ((*values)[4], 5.0f);
  EXPECT_FLOAT_EQ((*values)[8], 9.0f);

  TensorHolder bias = loader.ReadTensorData("b_bias", result);
  ASSERT_TRUE(result.ok);
  const auto *bias_values = std::get_if<std::vector<float>>(&bias.value);
  ASSERT_NE(bias_values, nullptr);
  ASSERT_EQ(bias_values->size(), 3u);
  EXPECT_FLOAT_EQ((*bias_values)[0], 0.5f);
  EXPECT_FLOAT_EQ((*bias_values)[1], 1.5f);
  EXPECT_FLOAT_EQ((*bias_values)[2], 2.5f);
}

TEST(GeWeightLoaderConsolidatedTest, IsParameterCorrectForKind) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  const TensorRecordInfo *weight_info = loader.GetTensorRecordInfo("p_weight");
  ASSERT_NE(weight_info, nullptr);
  EXPECT_TRUE(weight_info->is_parameter);

  const TensorRecordInfo *bias_info = loader.GetTensorRecordInfo("b_bias");
  ASSERT_NE(bias_info, nullptr);
  EXPECT_FALSE(bias_info->is_parameter);
}

TEST(GeWeightLoaderConsolidatedTest, RecordInfoHasOffsetAndSize) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  const TensorRecordInfo *weight_info = loader.GetTensorRecordInfo("p_weight");
  ASSERT_NE(weight_info, nullptr);
  EXPECT_EQ(weight_info->offset, 0u);
  EXPECT_EQ(weight_info->data_size, 36u);

  const TensorRecordInfo *bias_info = loader.GetTensorRecordInfo("b_bias");
  ASSERT_NE(bias_info, nullptr);
  EXPECT_EQ(bias_info->offset, 36u);
  EXPECT_EQ(bias_info->data_size, 12u);
}

TEST(GeWeightLoaderConsolidatedTest, GetTensorMetaWorks) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  const ParsedTensorMeta *meta = loader.GetTensorMeta("p_weight");
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->sizes.size(), 2u);
  EXPECT_EQ(meta->sizes[0], 3);
  EXPECT_EQ(meta->sizes[1], 3);
  EXPECT_EQ(meta->dtype, 7);
  EXPECT_TRUE(meta->requires_grad);

  const ParsedTensorMeta *bias_meta = loader.GetTensorMeta("b_bias");
  ASSERT_NE(bias_meta, nullptr);
  ASSERT_EQ(bias_meta->sizes.size(), 1u);
  EXPECT_EQ(bias_meta->sizes[0], 3);
  EXPECT_EQ(bias_meta->dtype, 7);
  EXPECT_FALSE(bias_meta->requires_grad);
}

TEST(GeWeightLoaderConsolidatedTest, HasTensorReturnsCorrectly) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  EXPECT_TRUE(loader.HasTensor("p_weight"));
  EXPECT_TRUE(loader.HasTensor("b_bias"));
  EXPECT_FALSE(loader.HasTensor("does_not_exist"));
}

TEST(GeWeightLoaderConsolidatedTest, FormatStrReadFromWeightInfoNZ) {
  // Verify format_str is read from weight_info.json "format" field.
  // The test fixture doesn't have "format" field → format_str should be empty (default ND).
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  const TensorRecordInfo *weight_info = loader.GetTensorRecordInfo("p_weight");
  ASSERT_NE(weight_info, nullptr);
  // Fixture doesn't have "format" field → empty string (not "NZ")
  EXPECT_EQ(weight_info->format_str, "");

  // ReadTensorData should propagate format_str to holder
  TensorHolder holder = loader.ReadTensorData("p_weight", result);
  ASSERT_TRUE(result.ok);
  EXPECT_EQ(holder.format_str, "");
}

}  // namespace td
