#pragma once

#include <memory>

#include <graph/graph.h>

#include "backend/aten_schema_resolver.h"
#include "backend/aten_tensor_desc_mapper.h"
#include "epair_archive.h"
#include "backend/epair_model_parser.h"
#include "backend/ge_weight_loader.h"
#include "backend/value_registry.h"
#include "common/logging/logging.h"

namespace td {

class GeGraphBuilder {
 public:
  explicit GeGraphBuilder(const EpairArchive &archive, ValidationResult &result);

  GeGraphBuilder(const std::string &model_json, const std::string &dir_path,
                 ValidationResult &result);

  std::unique_ptr<ge::Graph> Build(ValidationResult &result);
  bool IsFullyOffloaded() const noexcept;

 private:
  const EpairArchive *archive_ = nullptr;
  std::string dir_path_;
  EpairModelParser parser_;
  GeWeightLoader loader_;
  AtenSchemaResolver schema_resolver_;
  AtenTensorDescMapper tensor_desc_mapper_;
  bool initialized_ = false;
};

}  // namespace td
