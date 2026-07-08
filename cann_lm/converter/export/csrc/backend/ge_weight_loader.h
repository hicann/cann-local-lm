#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/model_types.h"
#include "epair_archive.h"
#include "common/logging/logging.h"

namespace td {

class GeWeightLoader {
 public:
  static GeWeightLoader Create(const EpairArchive &archive, const ParsedModel &model, ValidationResult &result);

  static GeWeightLoader CreateFromDirectory(const std::string &dir_path, const ParsedModel &model,
                                            ValidationResult &result);

  bool HasTensor(const std::string &name) const;
  const TensorRecordInfo *GetTensorRecordInfo(const std::string &name) const;
  const ParsedTensorMeta *GetTensorMeta(const std::string &name) const;

  TensorHolder ReadTensorData(const std::string &name, ValidationResult &result) const;

  static TensorHolder RestoreInlineConstTensor(const OrderedJson &value_attr, ValidationResult &result,
                                               const std::string &context);

 private:
  void LoadConsolidated(const EpairArchive &archive, const ParsedModel &model, ValidationResult &result);
  void LoadConsolidatedFromDir(const std::string &dir_path, const ParsedModel &model, ValidationResult &result);
  void LoadLegacy(const EpairArchive &archive, const ParsedModel &model, ValidationResult &result);

  const EpairArchive *archive_ = nullptr;
  bool consolidated_ = false;
  std::string dir_path_;
  std::unordered_map<std::string, std::string> placeholder_to_logical_name_;
  std::unordered_map<std::string, TensorRecordInfo> logical_name_to_record_;
  mutable std::vector<uint8_t> consolidated_data_;
};

}  // namespace td
