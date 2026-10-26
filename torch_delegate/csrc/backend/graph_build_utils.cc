/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/graph_build_utils.h"

#include <sstream>
#include <type_traits>

#include <ge/ge_api_v2.h>
#include <graph/named_io_node_builder.h>

#include "common/json_parse_utils.h"
#include "common/logging/assert_macros.h"

namespace td {

size_t ElementSizeForGeDtype(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_DOUBLE:
      return sizeof(double);
    case ge::DT_FLOAT:
      return sizeof(float);
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
    case ge::DT_UINT16:
      return sizeof(uint16_t);
    case ge::DT_INT64:
      return sizeof(int64_t);
    case ge::DT_INT32:
      return sizeof(int32_t);
    case ge::DT_INT16:
      return sizeof(int16_t);
    case ge::DT_INT8:
      return sizeof(int8_t);
    case ge::DT_UINT8:
    case ge::DT_BOOL:
      return sizeof(uint8_t);
    default:
      return 0;
  }
}

ge::TensorDesc MakeTensorDesc(const td::GeTensorDescInfo &info) {
  ge::TensorDesc desc;
  if (info.has_shape) {
    desc.SetShape(ge::Shape(info.shape));
  }
  if (info.has_format) {
    desc.SetFormat(info.format);
  }
  if (info.has_dtype) {
    desc.SetDataType(info.dtype);
  }
  return desc;
}

std::vector<uint8_t> TensorBytes(const TensorHolder &holder) {
  return std::visit(
      [](const auto &values) -> std::vector<uint8_t> {
        using ValueType = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<ValueType, std::monostate>) {
          return {};
        } else {
          const auto *begin = reinterpret_cast<const uint8_t *>(values.data());
          return {begin, begin + values.size() * sizeof(typename ValueType::value_type)};
        }
      },
      holder.value);
}

Status BuildRecoverableNode(ge::Graph &graph, const NodeBuildSpec &spec, std::unique_ptr<ge::GNode> &out) {
  ge::NamedIoNodeBuilder builder(graph);
  builder.Type(spec.type.c_str()).Name(spec.name.c_str());
  for (const auto &input : spec.inputs) {
    builder.AddInput(input.first.c_str(), input.second);
  }
  for (const auto &output : spec.outputs) {
    builder.AddOutput(output.first.c_str(), output.second);
  }
  for (const auto &attr : spec.attrs) {
    builder.Attr(attr.first.c_str(), attr.second);
  }
  ge::AscendString error_message;
  out = builder.Build(error_message);
  if (out == nullptr) {
    TDLOGE("NamedIoNodeBuilder build failed, context=%s, error=%s", spec.context.c_str(),
           error_message.GetString() == nullptr ? "NamedIoNodeBuilder build failed" : error_message.GetString());
    return FAILED;
  }
  return SUCCESS;
}

Status LookupInputIndex(ge::GNode &node, const std::string &input_name, int32_t &input_index,
                        const std::string &context) {
  if (node.GetInputIndexByName(ge::AscendString(input_name.c_str()), input_index) != ge::GRAPH_SUCCESS) {
    TDLOGE("Failed to resolve GE input index by name, context=%s, input_name=%s", context.c_str(), input_name.c_str());
    return INVALID_PARAM;
  }
  return SUCCESS;
}

Status LookupOutputIndex(ge::GNode &node, const std::string &output_name, int32_t &output_index,
                         const std::string &context) {
  if (node.GetOutputIndexByName(ge::AscendString(output_name.c_str()), output_index) != ge::GRAPH_SUCCESS) {
    TDLOGE("Failed to resolve GE output index by name, context=%s, output_name=%s", context.c_str(),
           output_name.c_str());
    return INVALID_PARAM;
  }
  return SUCCESS;
}

Status CreateConstNode(ge::Graph &graph, const ConstNodeSpec &spec, const td::AtenTensorDescMapper &mapper,
                       RegisteredValue &out) {
  const std::string &context = spec.node_name;
  ge::DataType dtype = ge::DT_UNDEFINED;
  TD_CHK_STATUS(mapper.MapPt2Dtype(spec.holder.datatype, context, dtype));
  ge::Format format = ge::FORMAT_ND;
  TD_CHK_STATUS(mapper.MapPt2Layout(spec.holder.format, context, format));
  const size_t element_size = ElementSizeForGeDtype(dtype);
  if (element_size == 0 || spec.holder.ByteSize() != Product(spec.holder.dims) * element_size) {
    TDLOGE("Const payload does not match tensor metadata, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  ge::TensorDesc desc(ge::Shape(spec.holder.dims), format, dtype);
  desc.SetPlacement(spec.placement);
  ge::Tensor tensor(desc);
  const std::vector<uint8_t> data = TensorBytes(spec.holder);
  if (tensor.SetData(data) != ge::GRAPH_SUCCESS) {
    TDLOGE("Failed to set const tensor data, context=%s", context.c_str());
    return FAILED;
  }
  ge::AttrValue value_attr;
  value_attr.SetAttrValue(tensor);
  std::vector<std::pair<std::string, ge::TensorDesc>> outputs = {{"y", desc}};
  std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{"value", value_attr}};
  std::unique_ptr<ge::GNode> node;
  TD_CHK_STATUS(BuildRecoverableNode(graph, {"Const", spec.node_name, {}, outputs, attrs, context}, node));
  int32_t output_index = 0;
  TD_CHK_STATUS(LookupOutputIndex(*node, "y", output_index, context));
  out = RegisteredValue{*node, output_index};
  return SUCCESS;
}

Status ValidateConstOutputDesc(const ParsedAscendIrOp &op, const TensorHolder &holder,
                               const td::AtenTensorDescMapper &mapper) {
  if (op.output_descs.size() != 1) {
    TDLOGE("Inline const must expose exactly one output_desc entry, op=%s", op.name.c_str());
    return INVALID_PARAM;
  }
  td::GeTensorDescInfo output_desc;
  TD_CHK_STATUS(mapper.MapAscendIrDesc(&op.output_descs[0], nullptr, op.name, output_desc));
  ge::DataType holder_dtype = ge::DT_UNDEFINED;
  TD_CHK_STATUS(mapper.MapPt2Dtype(holder.datatype, op.name, holder_dtype));
  ge::Format holder_format = ge::FORMAT_ND;
  TD_CHK_STATUS(mapper.MapPt2Layout(holder.format, op.name, holder_format));
  if (!output_desc.has_dtype || output_desc.dtype != holder_dtype) {
    TDLOGE("Inline const dtype does not match output_desc, op=%s", op.name.c_str());
    return INVALID_PARAM;
  }
  if (!output_desc.has_format || output_desc.format != holder_format) {
    TDLOGE("Inline const format does not match output_desc, op=%s", op.name.c_str());
    return INVALID_PARAM;
  }
  if (!output_desc.has_shape || output_desc.shape != holder.dims) {
    TDLOGE("Inline const shape does not match output_desc, op=%s", op.name.c_str());
    return INVALID_PARAM;
  }
  return SUCCESS;
}

bool IsConstOp(const ParsedAscendIrOp &op) {
  return op.type == "Const";
}

bool IsAscendIrPresent(const ParsedNode &node) {
  return node.ascend_ir.has_value();
}

bool IsConverterNode(const ParsedNode &node) {
  return node.ascend_ir.has_value() && node.ascend_ir->source == "converter";
}

bool HasBuildableConverterOps(const ParsedNode &node) {
  if (!node.ascend_ir.has_value()) {
    return false;
  }
  for (const auto &op : node.ascend_ir->ops) {
    if (!op.name.empty() && !op.type.empty()) {
      return true;
    }
  }
  return false;
}

bool IsConverterNoOpNode(const ParsedNode &node) {
  return IsConverterNode(node) && !HasBuildableConverterOps(node);
}

bool IsAssertTensorMetadataNode(const ParsedNode &node) {
  return node.target == "torch.ops.aten._assert_tensor_metadata.default";
}

bool IsSameTensorMeta(const ParsedTensorMeta &lhs, const ParsedTensorMeta &rhs) {
  return lhs.dtype == rhs.dtype && lhs.sizes == rhs.sizes && lhs.requires_grad == rhs.requires_grad &&
         lhs.device.type == rhs.device.type && lhs.device.index == rhs.device.index && lhs.strides == rhs.strides &&
         lhs.storage_offset == rhs.storage_offset && lhs.layout == rhs.layout;
}

std::vector<std::string> CollectOutputTensorNames(const ParsedNodeOutput &output) {
  if (output.is_tensor) {
    return {output.tensor_name};
  }
  if (output.is_tensor_list) {
    return output.tensor_names;
  }
  return {};
}

std::vector<std::string> CollectNodeOutputTensorNames(const ParsedNode &node) {
  std::vector<std::string> names;
  for (const auto &output : node.outputs) {
    const std::vector<std::string> output_names = CollectOutputTensorNames(output);
    names.insert(names.end(), output_names.begin(), output_names.end());
  }
  return names;
}

std::string ResolveBoundaryInputSourceRef(const ParsedNode &node, const std::string &input_ref) {
  if (!node.ascend_ir.has_value()) {
    return input_ref;
  }
  const auto mapped_input = node.ascend_ir->input_mappings.find(input_ref);
  if (mapped_input != node.ascend_ir->input_mappings.end()) {
    return mapped_input->second;
  }
  for (const auto &boundary_input : node.ascend_ir->boundary_inputs) {
    if (input_ref == boundary_input.fx_tensor_name) {
      return boundary_input.ge_tensor_name;
    }
  }
  return input_ref;
}

std::string GetNodeNameForLog(const ge::GNode &node) {
  ge::AscendString name;
  if (node.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
    return "<unknown>";
  }
  return name.GetString();
}

std::string MakeConverterNodeError(const ParsedNode &node, const ParsedAscendIrOp *op, const std::string &reason,
                                   const NodeErrorContext &ctx) {
  std::ostringstream oss;
  oss << "PlanConverterNode failed: node_name=" << node.name << ", op_name=" << (op == nullptr ? "" : op->name)
      << ", op_type=" << (op == nullptr ? "" : op->type) << ", fx_tensor=" << ctx.fx_tensor
      << ", ge_tensor=" << ctx.ge_tensor << ", source_ref=" << ctx.source_ref << ", dst_index=" << ctx.dst_index
      << ", reason=" << reason;
  return oss.str();
}

std::string MakeFallbackNodeError(const ParsedNode &node, const std::string &op_type, const std::string &reason,
                                  const NodeErrorContext &ctx) {
  std::ostringstream oss;
  oss << "PlanFallbackNode failed: node_name=" << node.name << ", op_type=" << op_type
      << ", fx_tensor=" << ctx.fx_tensor << ", ge_tensor=" << ctx.ge_tensor << ", source_ref=" << ctx.source_ref
      << ", dst_index=" << ctx.dst_index << ", reason=" << reason;
  return oss.str();
}

bool IsScalarInput(const ParsedNodeInput &input) {
  return IsScalarKind(input.input_kind);
}

bool IsSymInput(const ParsedNodeInput &input) {
  return IsSymKind(input.input_kind);
}

ge::DataType GetScalarDtype(const ParsedNodeInput &input) {
  switch (input.input_kind) {
    case NodeInputKind::ScalarInt:
    case NodeInputKind::ScalarListInt:
      return ge::DT_INT64;
    case NodeInputKind::ScalarFloat:
    case NodeInputKind::ScalarListFloat:
      return ge::DT_FLOAT;
    case NodeInputKind::ScalarBool:
    case NodeInputKind::ScalarListBool:
      return ge::DT_BOOL;
    default:
      return ge::DT_INT64;
  }
}

std::string NodeDebugContext(const ParsedNode &node) {
  std::ostringstream stream;
  stream << "graph_module.graph.nodes[" << node.json_node_index << "]";
  if (!node.name.empty()) {
    stream << " name=" << node.name;
  }
  if (!node.target.empty()) {
    stream << " target=" << node.target;
  }
  if (!node.effective_schema_target.empty() && node.effective_schema_target != node.target) {
    stream << " effective_target=" << node.effective_schema_target;
  }
  if (node.ascend_ir.has_value()) {
    stream << " source=" << (node.ascend_ir->source.empty() ? "<empty>" : node.ascend_ir->source);
  }
  if (!node.outputs.empty()) {
    stream << " output0=" << node.outputs.front().tensor_name;
  }
  return stream.str();
}

}  // namespace td
