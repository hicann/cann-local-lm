#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <caffe2/serialize/inline_container.h>

namespace td {

struct RecordData {
  std::unique_ptr<uint8_t[]> data;
  size_t size = 0;

  bool Empty() const {
    return data == nullptr || size == 0;
  }

  std::string AsString() const {
    return Empty() ? std::string() : std::string(reinterpret_cast<const char *>(data.get()), size);
  }
};

/* EpairArchiveReader wraps caffe2::serialize::PyTorchStreamReader for epair archive access.
 * NOTE: The constructor may throw exceptions because PyTorchStreamReader throws when:
 * - The archive file does not exist
 * - The archive file cannot be opened (permission denied, corrupted, etc.)
 * - The archive format is invalid
 *  Callers should use EpairArchive::TryCreate() which catches these exceptions and returns nullptr.
 **/
class EpairArchiveReader {
 public:
  explicit EpairArchiveReader(std::string epair_path);
  ~EpairArchiveReader();

  EpairArchiveReader(const EpairArchiveReader &) = delete;
  EpairArchiveReader &operator=(const EpairArchiveReader &) = delete;
  EpairArchiveReader(EpairArchiveReader &&) = delete;
  EpairArchiveReader &operator=(EpairArchiveReader &&) = delete;

  RecordData ReadRecord(const std::string &name) const;
  std::vector<std::string> ListRecords() const;
  bool HasRecord(const std::string &name) const;

 private:
  std::string archive_path_;
  mutable caffe2::serialize::PyTorchStreamReader reader_;
};

}  // namespace td
