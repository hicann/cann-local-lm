#include "common/epair_archive_reader.h"

#include <cstring>
#include <string>

#include "common/abi_compat_utils.h"
#include "common/logging/logging.h"
#include "common/memory_utils.h"
#include "logging/assert_macros.h"

namespace td {

EpairArchiveReader::EpairArchiveReader(std::string epair_path)
    : archive_path_(std::move(epair_path)), reader_(archive_path_) {}

EpairArchiveReader::~EpairArchiveReader() = default;

RecordData EpairArchiveReader::ReadRecord(const std::string &name) const {
  if (!reader_.hasRecord(name)) {
    TDLOGE("EpairArchiveReader::ReadRecord failed: record not found: %s", name.c_str());
    return {};
  }
  auto [data, size] = reader_.getRecord(name);
  RecordData record;
  record.size = size;
  if (size == 0) {
    return record;
  }
  record.data = MakeUnique<uint8_t[]>(size);
  if (record.data == nullptr) {
    TDLOGE("EpairArchiveReader::ReadRecord failed: buffer allocation failed, size=%zu", size);
    return {};
  }
  std::memcpy(record.data.get(), data.get(), size);
  return record;
}

std::vector<std::string> EpairArchiveReader::ListRecords() const {
  return reader_.getAllRecords();
}

bool EpairArchiveReader::HasRecord(const std::string &name) const {
  return reader_.hasRecord(name);
}

}  // namespace td
