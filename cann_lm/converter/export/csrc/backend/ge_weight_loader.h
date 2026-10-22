#pragma once

#include <string>
#include <unordered_map>

#include "backend/model_types.h"
#include "common/logging/logging.h"

namespace td {

class GeWeightLoader {
 public:
  static GeWeightLoader Create(const std::string &weights_index_path, const ParsedModel &model,
                               ValidationResult &result, const std::string &graph_name = "");

  bool HasTensor(const std::string &name) const noexcept;
  const ParsedTensorMeta *GetTensorMeta(const std::string &name) const noexcept;
  const TensorRecordInfo *GetTensorRecordInfo(const std::string &name) const noexcept;
  TensorHolder ReadTensorData(const std::string &name, ValidationResult &result) const;
  TensorHolder RestoreInlineConstTensor(const OrderedJson &value_ref, ValidationResult &result,
                                         const std::string &context) const;

 private:
  void LoadConsolidated(const std::string &weights_index_path, const ParsedModel &model,
                        ValidationResult &result, const std::string &graph_name);

  std::unordered_map<std::string, std::string> placeholder_to_logical_name_;
  std::unordered_map<std::string, TensorRecordInfo> logical_name_to_record_;
  mutable std::unordered_map<std::string, std::vector<uint8_t>> bin_files_cache_;
};

}  // namespace td
