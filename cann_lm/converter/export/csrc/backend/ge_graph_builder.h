#pragma once

#include <memory>
#include <string>
#include <vector>

#include <graph/graph.h>

#include "backend/aten_schema_resolver.h"
#include "backend/aten_tensor_desc_mapper.h"
#include "backend/epair_model_parser.h"
#include "backend/ge_weight_loader.h"
#include "backend/value_registry.h"
#include "common/logging/logging.h"

namespace td {

class GeGraphBuilder {
 public:
  GeGraphBuilder(const std::string &model_json, const std::string &weights_index_path,
                 ValidationResult &result);

  // Tap 中间 FX tensor 到 graph 输出: 调用于 Build() 前, 传入要暴露为 graph 额外输出的
  // FX 输出 tensor 名 (单输出 node 用 node.name, 如 "mm"/"add_1"; 多输出用 "name_0").
  // FinalizeGraph 经 ValueRegistry 的 alias (ascend_ir.mapping.outputs: FX->GE) 解析到
  // 对应 GE GNode, append 到 graph outputs. 使被 GE 死代码消除的中间值保活, 供 OM 逐算子比对.
  void SetTapOutputs(std::vector<std::string> names) { tap_outputs_ = std::move(names); }

  std::unique_ptr<ge::Graph> Build(ValidationResult &result);
  bool IsFullyOffloaded() const noexcept;

 private:
  std::string weights_index_path_;
  EpairModelParser parser_;
  GeWeightLoader loader_;
  AtenSchemaResolver schema_resolver_;
  AtenTensorDescMapper tensor_desc_mapper_;
  std::vector<std::string> tap_outputs_;  // tap 到 graph 输出的中间 FX tensor 名
  bool initialized_ = false;
};

}  // namespace td
