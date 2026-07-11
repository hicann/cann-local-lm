#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "common/epair_archive_constants.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "fixtures"
#endif

namespace {

const std::string kConsolidated = std::string(FIXTURES_DIR) + "/annotated_consolidated_mm_add";

std::string ReadFile(const std::string &path) {
  std::ifstream file(path);
  if (!file) return {};
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

}  // namespace

TEST(EpairArchiveDirectoryTest, ModelJsonExists) {
  const std::string path = kConsolidated + "/extra/ep_json_with_ascend_ir.json";
  std::ifstream file(path);
  EXPECT_TRUE(file.good());
}

TEST(EpairArchiveDirectoryTest, WeightsBinExists) {
  const std::string path = kConsolidated + "/data/weights.bin";
  std::ifstream file(path, std::ios::binary);
  EXPECT_TRUE(file.good());
}

TEST(EpairArchiveDirectoryTest, WeightsIndexExists) {
  const std::string path = kConsolidated + "/data/weights_index.json";
  std::ifstream file(path);
  EXPECT_TRUE(file.good());
}

TEST(EpairArchiveDirectoryTest, ModelJsonNonEmpty) {
  const std::string content = ReadFile(kConsolidated + "/extra/ep_json_with_ascend_ir.json");
  EXPECT_FALSE(content.empty());
}

TEST(EpairArchiveDirectoryTest, MissingDirFails) {
  std::ifstream file("/nonexistent/path/ep_json_with_ascend_ir.json");
  EXPECT_FALSE(file.good());
}
