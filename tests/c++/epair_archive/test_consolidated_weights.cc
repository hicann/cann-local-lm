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
const std::string kConsolidatedNoField =
    std::string(FIXTURES_DIR) + "/annotated_consolidated_mm_add_no_field";  // index without _is_weight (fallback path)
const std::string kConsolidatedMissingMeta = std::string(FIXTURES_DIR) + "/annotated_consolidated_missing_meta";
const std::string kConsolidatedNoIndex = std::string(FIXTURES_DIR) + "/annotated_consolidated_no_index";
const std::string kV1NoGraphs = std::string(FIXTURES_DIR) + "/annotated_v1_no_graphs";
const std::string kPdDedup = std::string(FIXTURES_DIR) + "/annotated_pd_dedup";

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

td::GeWeightLoader CreateLoader(const std::string &dir_path, td::ValidationResult &result,
                                const std::string &graph_name = "single") {
  const std::string model_json = ReadFile(dir_path + "/extra/ep_json_with_ascend_ir.json");
  td::EpairModelParser parser = td::EpairModelParser::Parse(model_json, result);
  if (!result.ok) return {};
  return td::GeWeightLoader::Create(dir_path + "/data/weights_index.json", parser.Model(),
                                    result, graph_name);
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

// TensorRecordInfo.is_parameter mirrors weight_info.json "_is_weight": True for
// nn.Module state (parameters + buffers, shared across prefill/decode), False for
// lifted tensor constants. This fixture's index carries the _is_weight field, so
// LoadConsolidated reads it directly (entry-read path). Both p_weight (parameter)
// and b_bias (buffer) are module state → is_parameter=true.
TEST(GeWeightLoaderConsolidatedTest, IsWeightFieldReadFromEntry) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  const TensorRecordInfo *weight_info = loader.GetTensorRecordInfo("p_weight");
  ASSERT_NE(weight_info, nullptr);
  EXPECT_TRUE(weight_info->is_parameter);

  const TensorRecordInfo *bias_info = loader.GetTensorRecordInfo("b_bias");
  ASSERT_NE(bias_info, nullptr);
  EXPECT_TRUE(bias_info->is_parameter);  // buffer is module state → _is_weight=true
}

// Fallback path: old weight_info.json without _is_weight field. LoadConsolidated
// derives is_parameter from model.input_specs PlaceholderKind (Parameter/Buffer →
// true, TensorConstant → false). Back-compat for indices produced by old packer.
TEST(GeWeightLoaderConsolidatedTest, IsWeightFallbackFromInputSpecsWithoutField) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidatedNoField, result);
  ASSERT_TRUE(result.ok);

  const TensorRecordInfo *weight_info = loader.GetTensorRecordInfo("p_weight");
  ASSERT_NE(weight_info, nullptr);
  EXPECT_TRUE(weight_info->is_parameter);  // Parameter → true

  const TensorRecordInfo *bias_info = loader.GetTensorRecordInfo("b_bias");
  ASSERT_NE(bias_info, nullptr);
  EXPECT_TRUE(bias_info->is_parameter);  // Buffer → true (module state, same as entry path)
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

TEST(GeWeightLoaderConsolidatedTest, FormatStrReadFromWeightInfo) {
  // v2 fixture carries "format": "nd" → format_str propagated to record + holder.
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);

  const TensorRecordInfo *weight_info = loader.GetTensorRecordInfo("p_weight");
  ASSERT_NE(weight_info, nullptr);
  EXPECT_EQ(weight_info->format_str, "nd");

  // ReadTensorData should propagate format_str to holder
  TensorHolder holder = loader.ReadTensorData("p_weight", result);
  ASSERT_TRUE(result.ok);
  EXPECT_EQ(holder.format_str, "nd");
}

// ── CLI-014 B: v2 two-level index (graphs/blobs, no version, v1 rejected) ──

// v1 single-level index (no graphs/blobs) is rejected — must re-export with v2.
TEST(GeWeightLoaderConsolidatedTest, V1IndexRejectedReportsError) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kV1NoGraphs, result);
  EXPECT_FALSE(result.ok);
  const bool found =
      std::any_of(result.issues.begin(), result.issues.end(),
                   [](const auto &issue) { return issue.code == "INVALID_CONSOLIDATED_INDEX"; });
  EXPECT_TRUE(found);
}

// PD fixture: graphs["prefill"] loads the shared weight blob.
TEST(GeWeightLoaderConsolidatedTest, PdPrefillGraphLoadsWeight) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kPdDedup, result, "prefill");
  ASSERT_TRUE(result.ok);
  ASSERT_TRUE(loader.HasTensor("p_weight"));

  TensorHolder holder = loader.ReadTensorData("p_weight", result);
  ASSERT_TRUE(result.ok);
  AssertWeightMatches(holder, {3}, 7, 7, 12, true);
  const auto *values = std::get_if<std::vector<float>>(&holder.value);
  ASSERT_NE(values, nullptr);
  EXPECT_FLOAT_EQ((*values)[0], 1.0f);
  EXPECT_FLOAT_EQ((*values)[2], 3.0f);
}

// PD fixture: graphs["decode"] loads the same shared blob (hash dedup).
TEST(GeWeightLoaderConsolidatedTest, PdDecodeGraphLoadsSharedWeight) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kPdDedup, result, "decode");
  ASSERT_TRUE(result.ok);
  ASSERT_TRUE(loader.HasTensor("p_weight"));

  TensorHolder holder = loader.ReadTensorData("p_weight", result);
  ASSERT_TRUE(result.ok);
  const auto *values = std::get_if<std::vector<float>>(&holder.value);
  ASSERT_NE(values, nullptr);
  EXPECT_FLOAT_EQ((*values)[0], 1.0f);  // same blob as prefill
}

// PD fixture: nonexistent graph_name → error (graphs missing graph).
TEST(GeWeightLoaderConsolidatedTest, MissingGraphNameReportsError) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kPdDedup, result, "nonexistent");
  EXPECT_FALSE(result.ok);
  const bool found =
      std::any_of(result.issues.begin(), result.issues.end(),
                   [](const auto &issue) { return issue.code == "INVALID_CONSOLIDATED_INDEX"; });
  EXPECT_TRUE(found);
}

// v2 index must not carry a "version" field — detection is by graphs/blobs presence.
TEST(GeWeightLoaderConsolidatedTest, V2IndexHasNoVersionField) {
  ValidationResult result{true, {}};
  GeWeightLoader loader = CreateLoader(kConsolidated, result);
  ASSERT_TRUE(result.ok);
  // If we got here, v2 detection (graphs/blobs) worked without a version field.
  ASSERT_TRUE(loader.HasTensor("p_weight"));
}

}  // namespace td
