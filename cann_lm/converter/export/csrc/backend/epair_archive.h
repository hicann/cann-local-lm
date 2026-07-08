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
