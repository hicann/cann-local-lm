/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/ge_graph_builder.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ge/ge_api_v2.h>
#include <graph/attr_value.h>
#include <graph/graph.h>
#include <graph/tensor.h>
#include <graph/utils/type_utils.h>
#include <nlohmann/json.hpp>

#include "common/base64.h"
#include "common/epair_archive_constants.h"
#include "common/json_parse_utils.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "common/memory_utils.h"
#include "common/pt2_type_constants.h"
#include "backend/ge_graph_optimizer.h"
#include "backend/graph_build_utils.h"
#include "backend/plan_converter_node.h"
#include "backend/plan_fallback_node.h"
#include "fallback_codegen/fallback_codegen.h"

namespace td {
namespace {

std::vector<std::pair<int64_t, int64_t>> BuildShapeRange(const ParsedTensorMeta &meta, const ParsedModel &model) {
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  shape_range.reserve(meta.sizes.size());
  for (size_t i = 0; i < meta.sizes.size(); ++i) {
    if (i < meta.dynamic_dim_symbols.size() && !meta.dynamic_dim_symbols[i].empty()) {
      const std::string &symbol = meta.dynamic_dim_symbols[i];
      const auto *range = model.FindRangeConstraint(symbol);
      if (range != nullptr) {
        shape_range.push_back(*range);
        TDLOGI("BuildShapeRange: dim %zu is dynamic, symbol=%s, range=[%ld, %ld]", i, symbol.c_str(),
               static_cast<long>(range->first), static_cast<long>(range->second));
      } else {
        shape_range.push_back({0, -1});
        TDLOGW(
            "BuildShapeRange: dim %zu is dynamic but no range constraint found for symbol=%s, "
            "using default [0, -1]",
            i, symbol.c_str());
      }
    } else {
      shape_range.push_back({meta.sizes[i], meta.sizes[i]});
    }
  }
  return shape_range;
}

Status PrepareTensorInput(GraphBuildContext &ctx, const ParsedModel &model, const std::string &input_name,
                          int64_t &user_input_index, const td::AtenTensorDescMapper &mapper) {
  TD_ASSERT_NOTNULL(ctx.graph, "PrepareTensorInput failed: graph is null");
  const ParsedTensorMeta *meta = model.FindTensorMeta(input_name);
  if (meta == nullptr) {
    TDLOGE("PrepareTensorInput failed: missing tensor meta for %s", input_name.c_str());
    return INVALID_PARAM;
  }
  GeTensorDescInfo desc_info;
  TD_CHK_STATUS(mapper.MapPt2TensorMeta(*meta, input_name, desc_info));
  ge::AttrValue index_attr;
  index_attr.SetAttrValue(user_input_index++);
  std::vector<std::pair<std::string, ge::TensorDesc>> data_inputs = {{"x", MakeTensorDesc(desc_info)}};
  std::vector<std::pair<std::string, ge::TensorDesc>> data_outputs = {{"y", MakeTensorDesc(desc_info)}};
  data_inputs[0].second.SetPlacement(ge::kPlacementDevice);
  data_outputs[0].second.SetPlacement(ge::kPlacementDevice);
  const auto shape_range = BuildShapeRange(*meta, model);
  for (auto &[name, desc] : data_inputs) {
    desc.SetShapeRange(shape_range);
  }
  for (auto &[name, desc] : data_outputs) {
    desc.SetShapeRange(shape_range);
  }
  std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{"index", index_attr}};
  std::unique_ptr<ge::GNode> data_node;
  TD_CHK_STATUS(
      BuildRecoverableNode(*ctx.graph, {"Data", input_name, data_inputs, data_outputs, attrs, input_name}, data_node));
  int32_t output_index = 0;
  TD_CHK_STATUS(LookupOutputIndex(*data_node, "y", output_index, input_name));
  TD_CHK_STATUS(ctx.registry.RegisterPlaceholderSource(input_name, {*data_node, output_index}));
  TDLOGI("PrepareTensorInput: created Data node for %s", input_name.c_str());
  return SUCCESS;
}

Status PrepareSymInput(GraphBuildContext &ctx, const ParsedModel &model, const std::string &input_name,
                       NodeInputKind kind, int64_t &user_input_index) {
  TD_ASSERT_NOTNULL(ctx.graph, "PrepareSymInput failed: graph is null");
  ge::DataType dtype = ge::DT_UNDEFINED;
  TensorHolder holder;
  holder.dims = {};
  holder.format = pt2::kLayoutStrided;
  if (kind == NodeInputKind::SymInt) {
    auto value_opt = model.FindSymInt(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing sym_int value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    dtype = ge::DT_INT64;
    holder.datatype = pt2::kDtypeLong;
    holder.value = std::vector<int64_t>{value_opt.value()};
  } else if (kind == NodeInputKind::SymFloat) {
    auto value_opt = model.FindSymFloat(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing sym_float value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    dtype = ge::DT_FLOAT;
    holder.datatype = pt2::kDtypeFloat;
    holder.value = std::vector<float>{static_cast<float>(value_opt.value())};
  } else if (kind == NodeInputKind::SymBool) {
    auto value_opt = model.FindSymBool(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing sym_bool value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    dtype = ge::DT_BOOL;
    holder.datatype = pt2::kDtypeBool;
    holder.value = std::vector<uint8_t>{static_cast<uint8_t>(value_opt.value() ? 1 : 0)};
  } else {
    TDLOGE("PrepareSymInput failed: unsupported sym kind=%s, input=%s", NodeInputKindToString(kind),
           input_name.c_str());
    return UNIMPLEMENTED;
  }
  ge::TensorDesc desc;
  desc.SetDataType(dtype);
  desc.SetShape(ge::Shape(holder.dims));
  desc.SetFormat(ge::FORMAT_ND);
  desc.SetPlacement(ge::kPlacementHost);
  ge::AttrValue index_attr;
  index_attr.SetAttrValue(user_input_index++);
  std::vector<std::pair<std::string, ge::TensorDesc>> data_inputs = {{"x", desc}};
  std::vector<std::pair<std::string, ge::TensorDesc>> data_outputs = {{"y", desc}};
  std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{"index", index_attr}};
  std::unique_ptr<ge::GNode> data_node;
  TD_CHK_STATUS(
      BuildRecoverableNode(*ctx.graph, {"Data", input_name, data_inputs, data_outputs, attrs, input_name}, data_node));
  int32_t output_index = 0;
  TD_CHK_STATUS(LookupOutputIndex(*data_node, "y", output_index, input_name));
  TD_CHK_STATUS(ctx.registry.RegisterPlaceholderSource(input_name, {*data_node, output_index}));
  TDLOGI("PrepareSymInput: created Data node for %s (dtype=%d, placement=Host)", input_name.c_str(),
         static_cast<int>(dtype));
  return SUCCESS;
}

Status PrepareScalarInput(GraphBuildContext &ctx, const ParsedModel &model, const std::string &input_name,
                          NodeInputKind kind, const td::AtenTensorDescMapper &mapper) {
  TD_ASSERT_NOTNULL(ctx.graph, "PrepareScalarInput failed: graph is null");
  TensorHolder holder;
  holder.dims = {1};
  holder.format = pt2::kLayoutStrided;
  if (kind == NodeInputKind::ScalarInt) {
    auto value_opt = model.FindInt(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing int value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    holder.datatype = pt2::kDtypeLong;
    holder.value = std::vector<int64_t>{value_opt.value()};
  } else if (kind == NodeInputKind::ScalarFloat) {
    auto value_opt = model.FindFloat(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing float value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    holder.datatype = pt2::kDtypeFloat;
    holder.value = std::vector<float>{static_cast<float>(value_opt.value())};
  } else if (kind == NodeInputKind::ScalarBool) {
    auto value_opt = model.FindBool(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing bool value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    holder.datatype = pt2::kDtypeBool;
    holder.value = std::vector<uint8_t>{static_cast<uint8_t>(value_opt.value() ? 1 : 0)};
  } else if (kind == NodeInputKind::ScalarString) {
    auto value_opt = model.FindString(input_name);
    if (!value_opt.has_value()) {
      TDLOGE("Missing string value, input=%s", input_name.c_str());
      return INVALID_PARAM;
    }
    holder.datatype = pt2::kDtypeByte;
    const std::string &str = value_opt.value();
    holder.value = std::vector<uint8_t>(str.begin(), str.end());
    holder.dims = {static_cast<int64_t>(str.size())};
  } else {
    TDLOGE("PrepareScalarInput failed: unsupported scalar kind=%s, input=%s", NodeInputKindToString(kind),
           input_name.c_str());
    return UNIMPLEMENTED;
  }
  RegisteredValue const_value;
  TD_CHK_STATUS(CreateConstNode(*ctx.graph, {input_name, holder}, mapper, const_value));
  TD_CHK_STATUS(ctx.registry.RegisterPlaceholderSource(input_name, const_value));
  TDLOGI("PrepareScalarInput: created Const node for %s", input_name.c_str());
  return SUCCESS;
}

Status PrepareUserInput(GraphBuildContext &ctx, const ParsedModel &model, const std::string &input_name,
                        int64_t &user_input_index, const td::AtenTensorDescMapper &mapper) {
  const ParsedInputSpec *input_spec = model.FindInputSpec(input_name);
  TD_ASSERT_NOTNULL(input_spec, "PrepareUserInput failed: missing input spec for %s", input_name.c_str());
  const NodeInputKind kind = input_spec->input_kind;
  if (kind == NodeInputKind::Tensor) {
    return PrepareTensorInput(ctx, model, input_name, user_input_index, mapper);
  }
  if (IsSymKind(kind)) {
    return PrepareSymInput(ctx, model, input_name, kind, user_input_index);
  }
  if (IsScalarKind(kind)) {
    return PrepareScalarInput(ctx, model, input_name, kind, mapper);
  }
  TDLOGE("PrepareUserInput failed: unsupported input_kind=%s for %s", NodeInputKindToString(kind), input_name.c_str());
  return UNIMPLEMENTED;
}

Status PrepareWeightInput(GraphBuildContext &ctx, const GeWeightLoader &loader, const std::string &input_name,
                          const td::AtenTensorDescMapper &mapper) {
  TD_ASSERT_NOTNULL(ctx.graph, "PrepareWeightInput failed: graph is null");
  TensorHolder holder;
  TD_CHK_STATUS(loader.ReadTensorData(input_name, holder));
  RegisteredValue const_value;
  TD_CHK_STATUS(CreateConstNode(*ctx.graph, {input_name, holder}, mapper, const_value));
  TD_CHK_STATUS(ctx.registry.RegisterPlaceholderSource(input_name, const_value));
  TDLOGI("PrepareWeightInput: created Const node for %s (bytes=%zu)", input_name.c_str(), holder.ByteSize());
  return SUCCESS;
}

void NormalizeDescDefaults(ge::GNode &gnode) {
  for (size_t i = 0; i < gnode.GetInputsSize(); ++i) {
    ge::TensorDesc desc;
    if (gnode.GetInputDesc(static_cast<int32_t>(i), desc) != ge::GRAPH_SUCCESS) {
      continue;
    }
    desc.SetFormat(ge::FORMAT_ND);
    if (desc.GetDataType() == ge::DT_UNDEFINED) {
      desc.SetDataType(ge::DT_FLOAT);
    }
    gnode.UpdateInputDesc(static_cast<int32_t>(i), desc);
  }
  for (size_t i = 0; i < gnode.GetOutputsSize(); ++i) {
    ge::TensorDesc desc;
    if (gnode.GetOutputDesc(static_cast<int32_t>(i), desc) != ge::GRAPH_SUCCESS) {
      continue;
    }
    desc.SetFormat(ge::FORMAT_ND);
    if (desc.GetDataType() == ge::DT_UNDEFINED) {
      desc.SetDataType(ge::DT_FLOAT);
    }
    gnode.UpdateOutputDesc(static_cast<int32_t>(i), desc);
  }
}

bool ParseLayoutInfo(const OrderedJson &attrs, const std::string &attr_name, std::vector<int64_t> &indices,
                     std::vector<ge::AscendString> &formats) {
  if (!attrs.is_object() || !attrs.contains(attr_name)) {
    return false;
  }
  const auto &layout_info = attrs[attr_name];
  if (!layout_info.is_object() || !layout_info.contains("list")) {
    return false;
  }
  const auto &list = layout_info["list"];
  indices.clear();
  formats.clear();
  if (list.contains("i") && list["i"].is_array()) {
    for (const auto &item : list["i"]) {
      const std::optional<int64_t> val = ParseJsonIntLike(item);
      if (val.has_value()) {
        indices.push_back(val.value());
      }
    }
  }
  if (list.contains("s") && list["s"].is_array()) {
    for (const auto &item : list["s"]) {
      if (!item.is_string()) {
        continue;
      }
      std::string decoded;
      if (td::internal::DecodeBase64ToString(item.get<std::string>(), decoded)) {
        formats.emplace_back(decoded.c_str());
      }
    }
  }
  return !indices.empty() && !formats.empty();
}

using OpNameMap = std::unordered_map<std::string, const ParsedAscendIrOp *>;

void ApplyLayoutInfo(ge::GNode &gnode, const OpNameMap &op_map) {
  ge::AscendString node_name;
  if (gnode.GetName(node_name) != ge::GRAPH_SUCCESS || node_name.GetString() == nullptr) {
    return;
  }
  const auto it = op_map.find(node_name.GetString());
  if (it == op_map.end() || it->second == nullptr) {
    return;
  }
  const auto &op = *it->second;
  std::vector<int64_t> in_indices;
  std::vector<ge::AscendString> in_formats;
  if (ParseLayoutInfo(op.attrs, "input_layout_info", in_indices, in_formats)) {
    for (size_t k = 0; k < in_indices.size() && k < in_formats.size(); ++k) {
      const int32_t idx = static_cast<int32_t>(in_indices[k]);
      if (idx < 0 || static_cast<size_t>(idx) >= gnode.GetInputsSize()) {
        TDLOGW("ApplyLayoutInfo: input index %d out of range %zu, node=%s", idx, gnode.GetInputsSize(),
               op.name.c_str());
        continue;
      }
      ge::TensorDesc desc;
      if (gnode.GetInputDesc(idx, desc) == ge::GRAPH_SUCCESS) {
        desc.SetFormat(ge::TypeUtils::AscendStringToFormat(in_formats[k]));
        gnode.UpdateInputDesc(idx, desc);
      }
    }
  }
  std::vector<int64_t> out_indices;
  std::vector<ge::AscendString> out_formats;
  if (ParseLayoutInfo(op.attrs, "output_layout_info", out_indices, out_formats)) {
    for (size_t k = 0; k < out_indices.size() && k < out_formats.size(); ++k) {
      const int32_t idx = static_cast<int32_t>(out_indices[k]);
      if (idx < 0 || static_cast<size_t>(idx) >= gnode.GetOutputsSize()) {
        TDLOGW("ApplyLayoutInfo: output index %d out of range %zu, node=%s", idx, gnode.GetOutputsSize(),
               op.name.c_str());
        continue;
      }
      ge::TensorDesc desc;
      if (gnode.GetOutputDesc(idx, desc) == ge::GRAPH_SUCCESS) {
        desc.SetFormat(ge::TypeUtils::AscendStringToFormat(out_formats[k]));
        gnode.UpdateOutputDesc(idx, desc);
      }
    }
  }
}

}  // namespace

GeGraphBuilder::GeGraphBuilder(const EpairArchive &archive, Status &init_status)
    : archive_(archive), ctx_("epair_graph") {
  TDLOGI("GeGraphBuilder enter: path=%s", archive.Path());

  init_status = FAILED;
  if (!archive.HasRecord(epair_spec::kAnnotatedModelJson)) {
    TDLOGE("GeGraphBuilder failed: annotated archive payload is required, path=%s", archive.Path());
    return;
  }

  TDLOGI("GeGraphBuilder step: reading model json");
  const RecordData graph_record = archive.ReadRecord(epair_spec::kAnnotatedModelJson);
  const std::string graph_json = graph_record.AsString();
  TDLOGI("GeGraphBuilder step: model json loaded, size=%zu", graph_json.size());

  const char *enable_codegen_env = std::getenv("TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN");
  bool enable_codegen = true;
  if (enable_codegen_env != nullptr) {
    enable_codegen = (std::string(enable_codegen_env) == "1");
  }
  TDLOGI("GeGraphBuilder config: TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=%s, fallback codegen %s",
         enable_codegen_env ? enable_codegen_env : "(not set, default enabled)",
         enable_codegen ? "enabled" : "disabled");

  if (enable_codegen) {
    TDLOGI("GeGraphBuilder step: running fallback codegen");
    auto codegen_result = torch_delegate::fallback::FallbackCodegen::GenerateFromModelJson(graph_json);
    if (!codegen_result.success) {
      TDLOGE("GeGraphBuilder failed: codegen error (see fallback codegen logs above for details)");
      return;
    } else if (codegen_result.skipped) {
      TDLOGI("GeGraphBuilder step: codegen skipped, %s", codegen_result.error_message.c_str());
    } else {
      TDLOGI("GeGraphBuilder step: codegen success, ops=%zu", codegen_result.generated_op_names.size());
    }
  }

  TDLOGI("GeGraphBuilder step: parsing model");
  Status parse_status = ParsedModel::Parse(graph_json, model_);
  if (parse_status != SUCCESS) {
    TDLOGE("GeGraphBuilder failed: parse error, status=%s", StatusName(parse_status));
    return;
  }
  TDLOGI("GeGraphBuilder step: parsed, nodes=%zu, inputs=%zu, outputs=%zu", model_.Nodes().size(),
         model_.GraphInputs().size(), model_.GraphOutputs().size());

  TDLOGI("GeGraphBuilder step: loading weights");
  Status loader_status = GeWeightLoader::Create(archive, model_, loader_);
  if (loader_status != SUCCESS) {
    TDLOGE("GeGraphBuilder failed: weight load error, status=%s", StatusName(loader_status));
    return;
  }
  TDLOGI("GeGraphBuilder step: weights loaded");

  initialized_ = true;
  init_status = SUCCESS;
  TDLOGI("GeGraphBuilder exit: success, initialized=true");
}

bool GeGraphBuilder::IsFullyOffloaded() const noexcept {
  for (const ParsedNode &node : model_.Nodes()) {
    if (!node.ascend_ir.has_value()) {
      return false;
    }
  }
  return true;
}

Status GeGraphBuilder::PreparePlaceholders() {
  TDLOGI("PreparePlaceholders enter: inputs=%zu", model_.GraphInputs().size());
  int64_t user_input_index = 0;
  for (const auto &input : model_.GraphInputs()) {
    const PlaceholderKind kind = model_.PlaceholderKindOf(input.name);
    TDLOGI("PreparePlaceholders: processing %s (kind=%d)", input.name.c_str(), static_cast<int>(kind));
    if (kind == PlaceholderKind::UserInput) {
      // 用户输入：创建 GE Data 节点，运行时由框架喂数据
      TD_CHK_STATUS(PrepareUserInput(ctx_, model_, input.name, user_input_index, tensor_desc_mapper_));
    } else if (kind == PlaceholderKind::Parameter || kind == PlaceholderKind::Buffer ||
               kind == PlaceholderKind::TensorConstant) {
      // 持久化张量（模型参数/Buffer/张量常量）：从归档读取权重数据，创建 GE Const 节点
      TD_CHK_STATUS(PrepareWeightInput(ctx_, loader_, input.name, tensor_desc_mapper_));
    } else if (kind == PlaceholderKind::ConstantInput) {
      // 标量常量：导出时已 specialize 到图 IR 中，归档无权重记录，跳过
      TDLOGW("PreparePlaceholders: skipping ConstantInput %s (value is specialized and inlined in graph)",
             input.name.c_str());
      continue;
    } else {
      TDLOGE("PreparePlaceholders failed: unknown kind for %s", input.name.c_str());
      return INVALID_PARAM;
    }
  }
  TDLOGI("PreparePlaceholders exit: success");
  return SUCCESS;
}

Status GeGraphBuilder::PlanNodes() {
  TDLOGI("PlanNodes enter: nodes=%zu", model_.Nodes().size());
  const GraphPlanningDeps deps{model_, schema_resolver_, tensor_desc_mapper_};
  for (const auto &node : model_.Nodes()) {
    if (IsAssertTensorMetadataNode(node)) {
      TDLOGI("PlanNodes skipping _assert_tensor_metadata node: %s", node.name.c_str());
      continue;
    }
    TDLOGI("PlanNodes visiting: node=%s, target=%s, source=%s, outputs=%zu", node.name.c_str(), node.target.c_str(),
           node.source_info.source.c_str(), node.outputs.size());
    if (IsConverterNode(node)) {
      TD_ASSERT_TRUE(IsAscendIrPresent(node),
                     "Parsed node [%s] does not contain ascend_ir field, "
                     "json node index [%zu]",
                     node.name.c_str(), node.json_node_index);
      if (HasBuildableConverterOps(node)) {
        TD_CHK_STATUS(ConverterNodePlanner(ctx_, deps).Plan(node));
      } else if (IsConverterNoOpNode(node)) {
        TD_CHK_STATUS(ConverterNodePlanner(ctx_, deps).PlanNoOp(node));
      } else {
        TDLOGE("PlanNodes failed: unsupported scenario, node=%s", node.name.c_str());
        return UNIMPLEMENTED;
      }
    } else {
      TD_ASSERT_TRUE(!node.effective_schema_target.empty(),
                     "Fallback node has no effective schema target, node=%s, effective_target=%s", node.name.c_str(),
                     node.effective_schema_target.c_str());
      TD_CHK_STATUS(FallbackNodePlanner(ctx_, deps).Plan(node));
    }
  }
  TDLOGI("PlanNodes exit: success");
  return SUCCESS;
}

Status GeGraphBuilder::BindDeferredEdges() {
  TD_ASSERT_NOTNULL(ctx_.graph, "BindDeferredEdges failed: graph is null");
  TDLOGI("BindDeferredEdges enter: count=%zu", ctx_.deferred_edges.size());
  for (auto &edge : ctx_.deferred_edges) {
    TDLOGI("BindDeferredEdges processing: source_ref=%s, dst_node=%s, dst_index=%d", edge.src_ref.c_str(),
           GetNodeNameForLog(edge.dst_node).c_str(), edge.dst_index);
    const RegisteredValue *source = ctx_.registry.Resolve(edge.src_ref);
    if (source == nullptr) {
      TDLOGE("BindDeferredEdges failed: source_ref not found, source_ref=%s", edge.src_ref.c_str());
      return INVALID_PARAM;
    }
    ge::GNode source_node = source->producer;
    TDLOGI("BindDeferredEdges resolved: source_node=%s, source_index=%d, dst_node=%s, dst_index=%d",
           GetNodeNameForLog(source_node).c_str(), source->output_index, GetNodeNameForLog(edge.dst_node).c_str(),
           edge.dst_index);
    if (ctx_.graph->AddDataEdge(source_node, source->output_index, edge.dst_node, edge.dst_index) !=
        ge::GRAPH_SUCCESS) {
      TDLOGE("BindDeferredEdges failed: AddDataEdge error, source_node=%s, source_index=%d, dst_node=%s, dst_index=%d",
             GetNodeNameForLog(source_node).c_str(), source->output_index, GetNodeNameForLog(edge.dst_node).c_str(),
             edge.dst_index);
      return FAILED;
    }
  }
  TDLOGI("BindDeferredEdges exit: success");
  return SUCCESS;
}

Status GeGraphBuilder::BindDeferredOutputs() {
  TDLOGI("BindDeferredOutputs enter: count=%zu", ctx_.deferred_outputs.size());
  for (const auto &binding : ctx_.deferred_outputs) {
    TDLOGI("BindDeferredOutputs processing: alias=%s, source_ref=%s", binding.alias.c_str(),
           binding.source_ref.c_str());
    const RegisteredValue *source = ctx_.registry.Resolve(binding.source_ref);
    if (source == nullptr) {
      TDLOGE("BindDeferredOutputs failed: source_ref not found, alias=%s, source_ref=%s", binding.alias.c_str(),
             binding.source_ref.c_str());
      return INVALID_PARAM;
    }
    if (ctx_.registry.RegisterAlias(binding.alias, binding.source_ref) != SUCCESS) {
      TDLOGE("BindDeferredOutputs failed: register alias error, alias=%s, source_ref=%s", binding.alias.c_str(),
             binding.source_ref.c_str());
      return FAILED;
    }
    TDLOGI("BindDeferredOutputs success: alias=%s -> source_ref=%s", binding.alias.c_str(), binding.source_ref.c_str());
  }
  TDLOGI("BindDeferredOutputs exit: success");
  return SUCCESS;
}

Status GeGraphBuilder::FinalizeGraph() {
  TD_ASSERT_NOTNULL(ctx_.graph, "FinalizeGraph failed: graph is null");
  TDLOGI("FinalizeGraph enter: outputs=%zu", model_.GraphOutputs().size());
  std::vector<std::pair<ge::GNode, int32_t>> outputs;
  outputs.reserve(model_.GraphOutputs().size());
  for (size_t index = 0; index < model_.GraphOutputs().size(); ++index) {
    TDLOGI("FinalizeGraph processing output: graph_output_index=%zu, tensor_name=%s", index,
           model_.GraphOutputs()[index].name.c_str());
    const RegisteredValue *output = ctx_.registry.Resolve(model_.GraphOutputs()[index].name);
    if (output == nullptr) {
      TDLOGE("FinalizeGraph failed: graph output tensor not found, graph_output_index=%zu, tensor_name=%s", index,
             model_.GraphOutputs()[index].name.c_str());
      return INVALID_PARAM;
    }
    TDLOGI("FinalizeGraph resolved output: graph_output_index=%zu, tensor_name=%s, producer_node=%s, producer_index=%d",
           index, model_.GraphOutputs()[index].name.c_str(), GetNodeNameForLog(output->producer).c_str(),
           output->output_index);
    outputs.push_back({output->producer, output->output_index});
  }
  if (ctx_.graph->SetOutputs(outputs) != ge::GRAPH_SUCCESS) {
    TDLOGE("FinalizeGraph failed: SetOutputs error, output_count=%zu", outputs.size());
    return FAILED;
  }
  TDLOGI("FinalizeGraph success: output_count=%zu", outputs.size());
  return SUCCESS;
}

Status GeGraphBuilder::NormalizeGraph() {
  TD_ASSERT_NOTNULL(ctx_.graph, "NormalizeGraph failed: graph is null");
  TDLOGI("NormalizeGraph enter");

  OpNameMap op_map;
  for (const auto &node : model_.Nodes()) {
    if (!node.ascend_ir.has_value()) {
      continue;
    }
    for (const auto &op : node.ascend_ir->ops) {
      if (!op.name.empty()) {
        op_map.emplace(op.name, &op);
      }
    }
  }

  for (auto &gnode : ctx_.graph->GetAllNodes()) {
    ge::AscendString type;
    if (gnode.GetType(type) != ge::GRAPH_SUCCESS) {
      continue;
    }
    if (type == "Data" || type == "RefData") {
      continue;
    }
    NormalizeDescDefaults(gnode);
    ApplyLayoutInfo(gnode, op_map);
  }
  TDLOGI("NormalizeGraph exit: success");
  return SUCCESS;
}

std::shared_ptr<ge::Graph> GeGraphBuilder::Build(InputBindingMap &binding_map) {
  TDLOGI("Build enter: path=%s", archive_.Path());
  TD_ASSERT_TRUE(initialized_, "Build failed: builder not initialized");

  if (!IsFullyOffloaded()) {
    TDLOGE("Build skipped: model is not fully offloaded");
    return nullptr;
  }

  TD_ASSERT_TRUE(PreparePlaceholders() == SUCCESS, "Build failed at PreparePlaceholders");

  TD_ASSERT_TRUE(PlanNodes() == SUCCESS, "Build failed at PlanNodes");

  TD_ASSERT_TRUE(BindDeferredOutputs() == SUCCESS, "Build failed at BindDeferredOutputs");

  TD_ASSERT_TRUE(BindDeferredEdges() == SUCCESS, "Build failed at BindDeferredEdges");

  TD_ASSERT_TRUE(FinalizeGraph() == SUCCESS, "Build failed at FinalizeGraph");

  TD_ASSERT_TRUE(NormalizeGraph() == SUCCESS, "Build failed at NormalizeGraph");

  TD_ASSERT_TRUE(GeGraphOptimizer::Optimize(*ctx_.graph, binding_map) == SUCCESS, "Build failed at OptimizeGraph");

  TDLOGI("Build exit: success");
  return ctx_.graph;
}

}  // namespace td
