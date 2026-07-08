#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "backend/epair_archive.h"
#include "common/epair_archive_constants.h"

#ifndef FIXTURES_DIR
#define FIXTURES_DIR "fixtures"
#endif

namespace {

const std::string kAnnotated = std::string(FIXTURES_DIR) + "/annotated_mm_add.epair";
const std::string kMissingGraphSource = std::string(FIXTURES_DIR) + "/missing_graph_source.epair";

}  // namespace

TEST(EpairArchiveTest, OpensAnnotatedFixture) {
  td::EpairArchive archive(kAnnotated.c_str());
  EXPECT_TRUE(archive.HasRecord(td::epair_spec::kAnnotatedManifestJson));
  EXPECT_TRUE(archive.HasRecord(td::epair_spec::kAnnotatedModelJson));
}

TEST(EpairArchiveTest, MissingArchiveThrows) {
  EXPECT_THROW({ td::EpairArchive archive("/tmp/does_not_exist_98765.epair"); }, c10::Error);
}

TEST(EpairArchiveTest, MissingGraphSourceDetected) {
  td::EpairArchive archive(kMissingGraphSource.c_str());
  EXPECT_FALSE(archive.HasRecord(td::epair_spec::kAnnotatedModelJson));
}

TEST(EpairArchiveTest, TryCreateReturnsNullForInvalidPath) {
  EXPECT_EQ(td::EpairArchive::TryCreate("/nonexistent/path.epair"), nullptr);
}

TEST(EpairArchiveTest, TryCreateSucceedsForValidPath) {
  auto archive = td::EpairArchive::TryCreate(kAnnotated.c_str());
  EXPECT_NE(archive, nullptr);
  EXPECT_TRUE(archive->HasRecord(td::epair_spec::kAnnotatedModelJson));
}

TEST(EpairArchiveTest, TryCreateReturnsNullForNullPath) {
  EXPECT_EQ(td::EpairArchive::TryCreate(nullptr), nullptr);
}

TEST(EpairArchiveTest, ReadRecordReturnsNonEmpty) {
  td::EpairArchive archive(kAnnotated.c_str());
  td::RecordData record = archive.ReadRecord(td::epair_spec::kAnnotatedManifestJson);
  EXPECT_FALSE(record.Empty());
  EXPECT_GT(record.size, 0u);
  EXPECT_NE(record.data, nullptr);
}
