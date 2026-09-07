#include "backend/ge_graph_builder.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ge/ge_api_v2.h>
#include <graph/gnode.h>
#include <graph/graph.h>
#include <graph/named_io_node_builder.h>
#include <graph/tensor.h>
#include <nlohmann/json.hpp>

#include "common/abi_compat_utils.h"
#include "common/base64.h"
#include "common/logging/assert_macros.h"
#include "common/memory_utils.h"

// todo (@AGENTS) 本文件超过 1000 行，下一步可以拆解重构，精简代码源文件

namespace td {
namespace {

struct PlannedEdgeBinding {
  std::string src_ref;
  std::string input_name;
  int32_t dst_index = 0;
};

struct DeferredEdge {
  std::string src_ref;
  ge::GNode dst_node;
  int32_t dst_index = 0;
};

struct DeferredOutputBinding {
  std::string alias;
  std::string source_ref;
};

struct GraphBuildContext {
  explicit GraphBuildContext(const std::string &graph_name) : graph(MakeUnique<ge::Graph>(graph_name.c_str())) {}

  std::unique_ptr<ge::Graph> graph;
  td::ValueRegistry registry;
  std::vector<DeferredEdge> deferred_edges;
  std::unordered_set<std::string> deferred_nz_transpose_x2_ops;
  std::unordered_map<std::string, std::pair<std::string, bool>> nz_transpose_passthrough;
  std::vector<DeferredOutputBinding> deferred_outputs;
};

struct GraphPlanningDeps {
  const EpairModelParser &parser;
  const td::GeWeightLoader &weight_loader;
  const td::AtenSchemaResolver &schema_resolver;
  const td::AtenTensorDescMapper &tensor_desc_mapper;
};

struct RegisteredIrDef {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> attr_names;
};

std::optional<int64_t> ParseJsonIntLike(const OrderedJson &value) {
  if (value.is_number_integer()) {
    return value.get<int64_t>();
  }
  if (value.is_string()) {
    try {
      return std::stoll(value.get<std::string>());
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

bool IsOptionalUnfed(const ParsedAscendIrOpDesc &desc) {
  if (!desc.attr.is_object() || !desc.attr.contains("_is_unfed_optional")) {
    return false;
  }
  const auto &flag = desc.attr["_is_unfed_optional"];
  return flag.is_object() && flag.contains("i") && ParseJsonIntLike(flag["i"]) == 1;
}

bool IsConstOp(const ParsedAscendIrOp &op) {
  return op.type == "Const";
}

enum class ListAttrKind {
  kInt,
  kFloat,
  kBool,
  kString,
  kDataType,
};

struct ListAttrTypeInfo {
  const char *attr_type;
  const char *payload_key;
  ListAttrKind kind;
};

constexpr std::array<ListAttrTypeInfo, 5> kSupportedListAttrTypes = {{
    {"VT_LIST_INT", "i", ListAttrKind::kInt},
    {"VT_LIST_FLOAT", "f", ListAttrKind::kFloat},
    {"VT_LIST_BOOL", "b", ListAttrKind::kBool},
    {"VT_LIST_STRING", "s", ListAttrKind::kString},
    {"VT_LIST_DATA_TYPE", "dt", ListAttrKind::kDataType},
}};

const OrderedJson *FindField(const OrderedJson &value, std::initializer_list<const char *> names) {
  for (const char *name : names) {
    auto it = value.find(name);
    if (it != value.end()) {
      return &(*it);
    }
  }
  return nullptr;
}

const ListAttrTypeInfo *FindListAttrTypeInfo(const std::string &attr_type) {
  for (const auto &info : kSupportedListAttrTypes) {
    if (attr_type == info.attr_type) {
      return &info;
    }
  }
  return nullptr;
}

std::optional<double> ParseJsonFloatLike(const OrderedJson &value) {
  if (value.is_number()) {
    return value.get<double>();
  }
  if (value.is_string()) {
    try {
      return std::stod(value.get<std::string>());
    } catch (const std::exception &) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<ge::DataType> ParseJsonDataTypeLike(const OrderedJson &value, ValidationResult &result,
                                                  const std::string &context) {
  if (const std::optional<int64_t> dtype = ParseJsonIntLike(value); dtype.has_value()) {
    if (dtype.value() < 0 || dtype.value() > static_cast<int64_t>(ge::DT_HIFLOAT4)) {
      AddError(result, "UNSUPPORTED_ATTR_VALUE", "Unsupported ge::DataType value", context);
      return std::nullopt;
    }
    return static_cast<ge::DataType>(dtype.value());
  }
  if (value.is_string()) {
    ValidationResult dtype_result{true, {}};
    const td::AtenTensorDescMapper mapper;
    const ge::DataType dtype = mapper.MapGeDtypeString(value.get<std::string>(), dtype_result, context);
    if (!dtype_result.ok) {
      AddError(result, "UNSUPPORTED_ATTR_VALUE", "Unsupported ge::DataType value", context);
      return std::nullopt;
    }
    return dtype;
  }
  AddError(result, "UNSUPPORTED_ATTR_VALUE", "DataType attribute must be int-like or string", context);
  return std::nullopt;
}

std::optional<ge::AttrValue> ParseListAttrValue(const OrderedJson &value, ValidationResult &result, const std::string &context) {
  ge::AttrValue attr_value;
  const OrderedJson *val_type = FindField(value, {"valType", "val_type"});
  const ListAttrTypeInfo *info =
      val_type != nullptr && val_type->is_string() ? FindListAttrTypeInfo(val_type->get<std::string>()) : nullptr;
  if (info == nullptr) {
    TDLOGW("ParseListAttrValue: unsupported valType, skipping attr, context=%s, json=%s", context.c_str(), value.dump().c_str());
    return std::nullopt;
  }
  
  const OrderedJson *payload = FindField(value, {info->payload_key});
  if (payload == nullptr) {
    switch (info->kind) {
      case ListAttrKind::kInt:
        attr_value.SetAttrValue(std::vector<int64_t>{});
        TDLOGD("ParseListAttrValue: empty int list, context=%s", context.c_str());
        return attr_value;
      case ListAttrKind::kFloat:
        attr_value.SetAttrValue(std::vector<float>{});
        TDLOGD("ParseListAttrValue: empty float list, context=%s", context.c_str());
        return attr_value;
      case ListAttrKind::kBool:
        attr_value.SetAttrValue(std::vector<bool>{});
        TDLOGD("ParseListAttrValue: empty bool list, context=%s", context.c_str());
        return attr_value;
      case ListAttrKind::kString:
        attr_value.SetAttrValue(std::vector<ge::AscendString>{});
        TDLOGD("ParseListAttrValue: empty string list, context=%s", context.c_str());
        return attr_value;
      case ListAttrKind::kDataType:
        attr_value.SetAttrValue(std::vector<ge::DataType>{});
        TDLOGD("ParseListAttrValue: empty datatype list, context=%s", context.c_str());
        return attr_value;
    }
  }
  if (!payload->is_array()) {
    AddError(result, "UNSUPPORTED_ATTR_VALUE", "List attribute payload must be array", context);
    TDLOGE("ParseListAttrValue failed: payload is not array, context=%s", context.c_str());
    return attr_value;
  }
  switch (info->kind) {
    case ListAttrKind::kInt: {
      std::vector<int64_t> values;
      values.reserve(payload->size());
      for (const auto &entry : *payload) {
        const std::optional<int64_t> item = ParseJsonIntLike(entry);
        if (!item.has_value()) {
          AddError(result, "UNSUPPORTED_ATTR_VALUE", "ListInt item must be int-like", context);
          TDLOGE("ParseListAttrValue failed: ListInt item not int-like, context=%s", context.c_str());
          return attr_value;
        }
        values.push_back(item.value());
      }
      attr_value.SetAttrValue(values);
      TDLOGD("ParseListAttrValue: int list size=%zu, context=%s", values.size(), context.c_str());
      return attr_value;
    }
    case ListAttrKind::kFloat: {
      std::vector<float> values;
      values.reserve(payload->size());
      for (const auto &entry : *payload) {
        const std::optional<double> item = ParseJsonFloatLike(entry);
        if (!item.has_value()) {
          AddError(result, "UNSUPPORTED_ATTR_VALUE", "ListFloat item must be number-like", context);
          TDLOGE("ParseListAttrValue failed: ListFloat item not number-like, context=%s", context.c_str());
          return attr_value;
        }
        values.push_back(static_cast<float>(item.value()));
      }
      attr_value.SetAttrValue(values);
      TDLOGD("ParseListAttrValue: float list size=%zu, context=%s", values.size(), context.c_str());
      return attr_value;
    }
    case ListAttrKind::kBool: {
      std::vector<bool> values;
      values.reserve(payload->size());
      for (const auto &entry : *payload) {
        if (!entry.is_boolean()) {
          AddError(result, "UNSUPPORTED_ATTR_VALUE", "ListBool item must be boolean", context);
          TDLOGE("ParseListAttrValue failed: ListBool item not boolean, context=%s", context.c_str());
          return attr_value;
        }
        values.push_back(entry.get<bool>());
      }
      attr_value.SetAttrValue(values);
      TDLOGD("ParseListAttrValue: bool list size=%zu, context=%s", values.size(), context.c_str());
      return attr_value;
    }
    case ListAttrKind::kString: {
      std::vector<ge::AscendString> values;
      values.reserve(payload->size());
      for (size_t index = 0; index < payload->size(); ++index) {
        const auto &entry = (*payload)[index];
        if (!entry.is_string()) {
          AddError(result, "UNSUPPORTED_ATTR_VALUE", "ListString item must be string", context);
          TDLOGE("ParseListAttrValue failed: ListString item not string, context=%s", context.c_str());
          return attr_value;
        }
        std::string decoded;
        if (!td::internal::DecodeBase64ToString(entry.get<std::string>(), decoded)) {
          AddError(result, "INVALID_ATTR_BASE64", "ListString item is not valid base64",
                   context + ":s[" + std::to_string(index) + "]");
          TDLOGE("ParseListAttrValue failed: ListString base64 decode error, context=%s, index=%zu", context.c_str(), index);
          return attr_value;
        }
        values.emplace_back(decoded.c_str());
      }
      attr_value.SetAttrValue(values);
      TDLOGD("ParseListAttrValue: string list size=%zu, context=%s", values.size(), context.c_str());
      return attr_value;
    }
    case ListAttrKind::kDataType: {
      std::vector<ge::DataType> values;
      values.reserve(payload->size());
      for (size_t index = 0; index < payload->size(); ++index) {
        const std::optional<ge::DataType> item =
            ParseJsonDataTypeLike((*payload)[index], result, context + ":dt[" + std::to_string(index) + "]");
        if (!item.has_value()) {
          TDLOGE("ParseListAttrValue failed: DataType parse error, context=%s, index=%zu", context.c_str(), index);
          return attr_value;
        }
        values.push_back(item.value());
      }
      attr_value.SetAttrValue(values);
      TDLOGD("ParseListAttrValue: datatype list size=%zu, context=%s", values.size(), context.c_str());
      return attr_value;
    }
  }
  return attr_value;
}

std::optional<ge::AttrValue> ParseAttrValue(const OrderedJson &value, ValidationResult &result, const std::string &context) {
  ge::AttrValue attr_value;
  if (!value.is_object()) {
    TDLOGW("ParseAttrValue: payload is not object, skipping, context=%s, json=%s", context.c_str(), value.dump().c_str());
    return std::nullopt;
  }
  if (value.contains("list")) {
    auto result_value = ParseListAttrValue(value["list"], result, context);
    if (result.ok && result_value.has_value()) {
      TDLOGD("ParseAttrValue: list attribute parsed, context=%s", context.c_str());
    }
    return result_value;
  }
  if (value.contains("b") && value["b"].is_boolean()) {
    attr_value.SetAttrValue(value["b"].get<bool>());
    TDLOGD("ParseAttrValue: bool attribute, context=%s, value=%s", context.c_str(), value["b"].get<bool>() ? "true" : "false");
    return attr_value;
  }
  if (value.contains("i") && ParseJsonIntLike(value["i"]).has_value()) {
    int64_t int_val = ParseJsonIntLike(value["i"]).value();
    attr_value.SetAttrValue(int_val);
    TDLOGD("ParseAttrValue: int attribute, context=%s, value=%ld", context.c_str(), int_val);
    return attr_value;
  }
  if (value.contains("f") && ParseJsonFloatLike(value["f"]).has_value()) {
    float float_val = static_cast<float>(ParseJsonFloatLike(value["f"]).value());
    attr_value.SetAttrValue(float_val);
    TDLOGD("ParseAttrValue: float attribute, context=%s, value=%f", context.c_str(), float_val);
    return attr_value;
  }
  if (value.contains("s") && value["s"].is_string()) {
    std::string decoded;
    if (!td::internal::DecodeBase64ToString(value["s"].get<std::string>(), decoded)) {
      AddError(result, "INVALID_ATTR_BASE64", "String attribute is not valid base64", context);
      TDLOGE("ParseAttrValue failed: base64 decode error, context=%s", context.c_str());
      return attr_value;
    }
    attr_value.SetAttrValue(ge::AscendString(decoded.c_str()));
    TDLOGD("ParseAttrValue: string attribute, context=%s, value=%s", context.c_str(), decoded.c_str());
    return attr_value;
  }
  if (value.contains("dt")) {
    const std::optional<ge::DataType> dtype = ParseJsonDataTypeLike(value["dt"], result, context);
    if (dtype.has_value()) {
      attr_value.SetAttrValue(dtype.value());
      TDLOGD("ParseAttrValue: datatype attribute, context=%s, value=%d", context.c_str(), static_cast<int>(dtype.value()));
    } else {
      TDLOGE("ParseAttrValue failed: datatype parse error, context=%s", context.c_str());
    }
    return attr_value;
  }
  // "t" 类型在GeWeightLoader中统一处理

  TDLOGW("ParseAttrValue: unsupported attribute payload, skipping, context=%s, json=%s", context.c_str(), value.dump().c_str());
  return std::nullopt;
}

size_t Product(const std::vector<int64_t> &dims) {
  size_t product = 1;
  for (const int64_t dim : dims) {
    if (dim < 0) {
      return 0;
    }
    product *= static_cast<size_t>(dim);
  }
  return product;
}

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

bool LookupInputIndex(ge::GNode &node, const std::string &input_name, int32_t &input_index, ValidationResult &result,
                      const std::string &context) {
  if (node.GetInputIndexByName(ge::AscendString(input_name.c_str()), input_index) != ge::GRAPH_SUCCESS) {
    AddError(result, "MISSING_INPUT_DESC", "Failed to resolve GE input index by name", context + ":" + input_name);
    return false;
  }
  return true;
}

bool LookupOutputIndex(ge::GNode &node, const std::string &output_name, int32_t &output_index, ValidationResult &result,
                       const std::string &context) {
  if (node.GetOutputIndexByName(ge::AscendString(output_name.c_str()), output_index) != ge::GRAPH_SUCCESS) {
    AddError(result, "MISSING_OUTPUT_DESC", "Failed to resolve GE output index by name", context + ":" + output_name);
    return false;
  }
  return true;
}

std::unique_ptr<ge::GNode> BuildRecoverableNode(ge::Graph &graph, const std::string &type, const std::string &name,
                                                const std::vector<std::pair<std::string, ge::TensorDesc>> &inputs,
                                                const std::vector<std::pair<std::string, ge::TensorDesc>> &outputs,
                                                const std::vector<std::pair<std::string, ge::AttrValue>> &attrs,
                                                ValidationResult &result, const std::string &context) {
  ge::NamedIoNodeBuilder builder(graph);
  builder.Type(type.c_str()).Name(name.c_str());
  for (const auto &input : inputs) {
    builder.AddInput(input.first.c_str(), input.second);
  }
  for (const auto &output : outputs) {
    builder.AddOutput(output.first.c_str(), output.second);
  }
  for (const auto &attr : attrs) {
    builder.Attr(attr.first.c_str(), attr.second);
  }
  ge::AscendString error_message;
  std::unique_ptr<ge::GNode> node = builder.Build(error_message);
  if (node == nullptr) {
    AddError(result, "IR_RECOVERABLE_BUILD_FAILED",
             error_message.GetString() == nullptr ? "NamedIoNodeBuilder build failed" : error_message.GetString(),
             context);
  }
  return node;
}

std::optional<RegisteredValue> CreateConstNode(ge::Graph &graph, const std::string &node_name,
                                               const TensorHolder &holder, const td::AtenTensorDescMapper &mapper,
                                               ValidationResult &result, const std::string &context,
                                               ge::Placement placement = ge::kPlacementDevice) {
  const ge::DataType dtype = mapper.MapPt2Dtype(holder.datatype, result, context);
  const bool is_nz = (holder.format_str == "NZ");
  const ge::Format format = is_nz
      ? mapper.MapGeLayoutString("NZ", result, context)
      : mapper.MapPt2Layout(holder.format, result, context);
  if (!result.ok) {
    return std::nullopt;
  }
  const size_t element_size = ElementSizeForGeDtype(dtype);
  // For NZ, use the 4D FRACTAL_NZ storage shape [ceil(M/16), ceil(K/16), 16, 16] directly.
  // GE's CheckTensorIsValid checks data_size == Product(shape) * element_size — for non-aligned
  // shapes, the NZ-padded bytes (41472) differ from logical 2D (33800), so 2D fails the check.
  // 4D passes (Product(4D) == padded bytes). The 6D issue (TransShapeToFracNz re-applying on 4D)
  // only occurs with a Transpose op — which we skip for NZ weights (transpose_x2=True instead).
  std::vector<int64_t> desc_dims = holder.dims;
  if (is_nz && holder.dims.size() == 2) {
    constexpr int64_t kBlock = 16;
    desc_dims = {((holder.dims[0] + kBlock - 1) / kBlock),
                 ((holder.dims[1] + kBlock - 1) / kBlock), kBlock, kBlock};
  }
  if (element_size == 0 || holder.ByteSize() != Product(desc_dims) * element_size) {
    AddError(result, "CONST_SIZE_MISMATCH", "Const payload does not match tensor metadata", context);
    return std::nullopt;
  }
  ge::TensorDesc desc(ge::Shape(desc_dims), format, dtype);
  desc.SetPlacement(placement);
  if (is_nz) {
    desc.SetOriginShape(ge::Shape(holder.dims));  // logical [M, K] (non-padded)
    desc.SetOriginFormat(ge::FORMAT_ND);
    // NOTE: ATTR_NAME_STORAGE_FORMAT / FORMAT_LOCKED would lock the storage format against
    // FE FormatSelector override, but those symbols live in ge_attr_define.h which the CANN
    // SDK does not ship. Rely on the consistent (storage [m,k,16,16] + origin [M,K] + FRACTAL_NZ)
    // representation; if GE still reverts, a GE option or shipped attr path is needed.
  }
  ge::Tensor tensor(desc);
  const std::vector<uint8_t> data = TensorBytes(holder);
  if (tensor.SetData(data) != ge::GRAPH_SUCCESS) {
    AddError(result, "CONST_DATA_ERROR", "Failed to set const tensor data", context);
    return std::nullopt;
  }
  ge::AttrValue value_attr;
  value_attr.SetAttrValue(tensor);
  std::vector<std::pair<std::string, ge::TensorDesc>> outputs = {{"y", desc}};
  std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{"value", value_attr}};
  std::unique_ptr<ge::GNode> node =
      BuildRecoverableNode(graph, "Const", node_name, {}, outputs, attrs, result, context);
  if (node == nullptr) {
    return std::nullopt;
  }
  int32_t output_index = 0;
  if (!LookupOutputIndex(*node, "y", output_index, result, context)) {
    return std::nullopt;
  }
  return RegisteredValue{*node, output_index};
}

bool ValidateConstOutputDesc(const ParsedAscendIrOp &op, const TensorHolder &holder,
                             const td::AtenTensorDescMapper &mapper, ValidationResult &result) {
  if (op.output_descs.size() != 1) {
    AddError(result, "INVALID_CONST_OUTPUT_DESC", "Inline const must expose exactly one output_desc entry", op.name);
    return false;
  }
  const td::GeTensorDescInfo output_desc = mapper.MapAscendIrDesc(&op.output_descs[0], nullptr, result, op.name);
  const ge::DataType holder_dtype = mapper.MapPt2Dtype(holder.datatype, result, op.name);
  const ge::Format holder_format = mapper.MapPt2Layout(holder.format, result, op.name);
  if (!result.ok) {
    return false;
  }
  if (!output_desc.has_dtype || output_desc.dtype != holder_dtype) {
    AddError(result, "CONST_DESC_MISMATCH", "Inline const dtype does not match output_desc", op.name);
    return false;
  }
  if (!output_desc.has_format || output_desc.format != holder_format) {
    AddError(result, "CONST_DESC_MISMATCH", "Inline const format does not match output_desc", op.name);
    return false;
  }
  if (!output_desc.has_shape || output_desc.shape != holder.dims) {
    AddError(result, "CONST_DESC_MISMATCH", "Inline const shape does not match output_desc", op.name);
    return false;
  }
  return true;
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
  // torch.export + torchair decomposition 会插入 _assert_tensor_metadata 节点
  // （schema: () -> ()，无输出、无 GE 算子映射），用于运行时校验中间张量的
  // shape/dtype/stride。它们没有计算语义，GE 编译阶段不需要，会在 PlanNodes 里
  // 被当作 "no ascend_ir mapping" 的未映射算子而报错（UNMAPPED_OPERATOR）导致
  // 构图失败。这里识别后在 PlanNodes 顶部直接跳过。
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

std::optional<RegisteredIrDef> QueryRegisteredIrDef(const std::string &op_type, ValidationResult &result,
                                                    const std::string &context) {
  std::vector<std::pair<ge::AscendString, ge::AscendString>> inputs;
  std::vector<std::pair<ge::AscendString, ge::AscendString>> outputs;
  std::vector<std::pair<ge::AscendString, ge::AscendString>> attrs;
  if (::GetRegisteredIrDef(op_type.c_str(), inputs, outputs, attrs) != ge::GRAPH_SUCCESS) {
    AddError(result, "REGISTERED_IR_QUERY_FAILED", "GetRegisteredIrDef failed", context + ":" + op_type);
    return std::nullopt;
  }
  RegisteredIrDef ir_def;
  ir_def.input_names.reserve(inputs.size());
  ir_def.output_names.reserve(outputs.size());
  ir_def.attr_names.reserve(attrs.size());
  for (const auto &input : inputs) {
    const std::string ir_name = td::internal::AscendStringToString(input.first);
    if (ir_name.empty()) {
      AddError(result, "INVALID_REGISTERED_IR", "Registered IR input name is empty", context + ":" + op_type);
      return std::nullopt;
    }
    ir_def.input_names.push_back(ir_name);
  }
  for (const auto &output : outputs) {
    const std::string ir_name = td::internal::AscendStringToString(output.first);
    if (ir_name.empty()) {
      AddError(result, "INVALID_REGISTERED_IR", "Registered IR output name is empty", context + ":" + op_type);
      return std::nullopt;
    }
    ir_def.output_names.push_back(ir_name);
  }
  for (const auto &attr : attrs) {
    const std::string ir_name = td::internal::AscendStringToString(attr.first);
    if (!ir_name.empty()) {
      ir_def.attr_names.push_back(ir_name);
    }
  }
  return ir_def;
}

std::string ResolveBoundaryInputSourceRef(const ParsedNode &node, const std::string &input_ref) {
  if (!node.ascend_ir.has_value()) {
    return input_ref;
  }
  // New mapping contract: mapping.inputs directly tells us how an FX input tensor
  // maps to its peer GE tensor, so we resolve that first before falling back to
  // locally reconstructed boundary aliases.
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
                                   const std::string &fx_tensor = "", const std::string &ge_tensor = "",
                                   const std::string &source_ref = "", int32_t dst_index = -1) {
  std::ostringstream oss;
  oss << "PlanConverterNode failed: node_name=" << node.name << ", op_name=" << (op == nullptr ? "" : op->name)
      << ", op_type=" << (op == nullptr ? "" : op->type) << ", fx_tensor=" << fx_tensor << ", ge_tensor=" << ge_tensor
      << ", source_ref=" << source_ref << ", dst_index=" << dst_index << ", reason=" << reason;
  return oss.str();
}

std::string MakeFallbackNodeError(const ParsedNode &node, const std::string &op_type, const std::string &reason,
                                  const std::string &fx_tensor = "", const std::string &ge_tensor = "",
                                  const std::string &source_ref = "", int32_t dst_index = -1) {
  std::ostringstream oss;
  oss << "PlanFallbackNode failed: node_name=" << node.name << ", op_type=" << op_type << ", fx_tensor=" << fx_tensor
      << ", ge_tensor=" << ge_tensor << ", source_ref=" << source_ref << ", dst_index=" << dst_index
      << ", reason=" << reason;
  return oss.str();
}

OrderedJson CollectFallbackAttrs(const ParsedNode &node, ValidationResult &result) {
  if (!node.ascend_ir.has_value()) {
    return OrderedJson::object();
  }
  OrderedJson attrs = OrderedJson::object();
  for (const auto &op : node.ascend_ir->ops) {
    if (!op.attrs.is_object()) {
      continue;
    }
    for (const auto &[name, value] : op.attrs.items()) {
      if (attrs.contains(name)) {
        AddError(result, "DUPLICATE_ATTR", "Fallback attribute appears multiple times",
                 NodeDebugContext(node) + ":" + name);
        return {};
      }
      attrs[name] = value;
    }
  }
  return attrs;
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

std::optional<TensorHolder> BuildScalarConstHolder(const ParsedNodeInput &input, const std::string &context,
                                                    ValidationResult &result) {
  TensorHolder holder;
  holder.format = pt2::kLayoutStrided;

  switch (input.input_kind) {
    case NodeInputKind::ScalarInt: {
      holder.datatype = pt2::kDtypeLong;
      int64_t val = input.raw_arg["as_int"].get<int64_t>();
      holder.value = std::vector<int64_t>{val};
      break;
    }
    case NodeInputKind::ScalarFloat: {
      holder.datatype = pt2::kDtypeFloat;
      float val = static_cast<float>(input.raw_arg["as_float"].get<double>());
      holder.value = std::vector<float>{val};
      break;
    }
    case NodeInputKind::ScalarBool: {
      holder.datatype = pt2::kDtypeBool;
      uint8_t val = input.raw_arg["as_bool"].get<bool>() ? 1 : 0;
      holder.value = std::vector<uint8_t>{val};
      break;
    }
    case NodeInputKind::ScalarListInt: {
      holder.datatype = pt2::kDtypeLong;
      std::vector<int64_t> vals;
      for (const auto &v : input.raw_arg["as_ints"]) {
        vals.push_back(v.get<int64_t>());
      }
      holder.dims = {static_cast<int64_t>(vals.size())};
      holder.value = std::move(vals);
      return holder;
    }
    case NodeInputKind::ScalarListFloat: {
      holder.datatype = pt2::kDtypeFloat;
      std::vector<float> vals;
      for (const auto &v : input.raw_arg["as_floats"]) {
        vals.push_back(static_cast<float>(v.get<double>()));
      }
      holder.dims = {static_cast<int64_t>(vals.size())};
      holder.value = std::move(vals);
      return holder;
    }
    case NodeInputKind::ScalarListBool: {
      holder.datatype = pt2::kDtypeBool;
      std::vector<uint8_t> vals;
      for (const auto &v : input.raw_arg["as_bools"]) {
        vals.push_back(v.get<bool>() ? 1 : 0);
      }
      holder.dims = {static_cast<int64_t>(vals.size())};
      holder.value = std::move(vals);
      return holder;
    }
    case NodeInputKind::Enum: {
      holder.datatype = pt2::kDtypeLong;
      int64_t val = input.raw_arg["as_scalar_type"].get<int64_t>();
      holder.value = std::vector<int64_t>{val};
      break;
    }
    case NodeInputKind::ScalarString: {
      holder.datatype = pt2::kDtypeByte;
      std::string val = input.raw_arg["as_string"].get<std::string>();
      std::vector<uint8_t> bytes(val.begin(), val.end());
      holder.dims = {static_cast<int64_t>(bytes.size())};
      holder.value = std::move(bytes);
      return holder;
    }
    default:
      AddError(result, "UNSUPPORTED_SCALAR_TYPE", "Unsupported scalar input type",
               context + ":" + input.name + ":" + NodeInputKindToString(input.input_kind));
      return std::nullopt;
  }
  return holder;
}

std::vector<std::pair<std::string, ge::TensorDesc>> BuildFallbackInputDefs(
    const ParsedNode &node, const EpairModelParser &parser, const RegisteredIrDef &ir_def,
    const td::AtenTensorDescMapper &mapper, std::vector<PlannedEdgeBinding> &bindings, ValidationResult &result) {
  const std::string context = NodeDebugContext(node);
  for (const auto &input : node.inputs) {
    if (input.input_kind == NodeInputKind::Tensor || input.input_kind == NodeInputKind::TensorList ||
        IsScalarInput(input) || IsSymInput(input) || input.input_kind == NodeInputKind::None) {
      continue;
    }
    TDLOGE("Unsupported fallback input: name=%s, input_kind=%s",
           input.name.c_str(), NodeInputKindToString(input.input_kind));
    AddError(result, "UNSUPPORTED_FALLBACK_INPUT",
             "Fallback node input type not supported", context + ":" + input.name);
    return {};
  }

  size_t non_none_count = 0;
  for (const auto &input : node.inputs) {
    if (input.input_kind != NodeInputKind::None) non_none_count++;
  }
  if (non_none_count != ir_def.input_names.size()) {
    AddError(result, "INVALID_FALLBACK_INPUTS", "Fallback JSON inputs do not match registered IR input count", context);
    return {};
  }
  std::vector<std::pair<std::string, ge::TensorDesc>> inputs;
  size_t ir_index = 0;
  for (size_t input_index = 0; input_index < node.inputs.size(); ++input_index) {
    const ParsedNodeInput &input = node.inputs[input_index];
    if (input.input_kind == NodeInputKind::None) continue;
    const std::string &ir_name = ir_def.input_names[ir_index++];

    if (input.input_kind == NodeInputKind::Tensor) {
      const ParsedTensorMeta *input_meta = parser.FindTensorValue(input.tensor_name);
      if (input_meta == nullptr) {
        AddError(result, "MISSING_INPUT_META", "Fallback input tensor meta missing", context + ":" + input.tensor_name);
        return {};
      }
      const td::GeTensorDescInfo desc_info = mapper.MapPt2TensorMeta(*input_meta, result, context + ":" + ir_name);
      if (!result.ok) return {};
      ge::TensorDesc desc = MakeTensorDesc(desc_info);
      desc.SetPlacement(ge::kPlacementDevice);
      inputs.push_back({ir_name, desc});
      bindings.push_back({input.tensor_name, ir_name, -1});
    } else if (input.input_kind == NodeInputKind::TensorList) {
      for (size_t tensor_index = 0; tensor_index < input.tensor_names.size(); ++tensor_index) {
        const std::string &tensor_name = input.tensor_names[tensor_index];
        const ParsedTensorMeta *input_meta = parser.FindTensorValue(tensor_name);
        if (input_meta == nullptr) {
          AddError(result, "MISSING_INPUT_META", "Fallback input tensor meta missing", context + ":" + tensor_name);
          return {};
        }
        const td::GeTensorDescInfo desc_info = mapper.MapPt2TensorMeta(*input_meta, result, context + ":" + ir_name);
        if (!result.ok) return {};
        const std::string instance_name = ir_name + std::to_string(tensor_index);
        ge::TensorDesc desc = MakeTensorDesc(desc_info);
        desc.SetPlacement(ge::kPlacementDevice);
        inputs.push_back({instance_name, desc});
        bindings.push_back({tensor_name, instance_name, -1});
      }
    } else if (IsScalarInput(input)) {
      ge::DataType dtype = GetScalarDtype(input);
      std::vector<int64_t> shape_dims;
      // For list types, create 1D shape; for scalar types, create scalar shape (empty dims)
      if (input.input_kind == NodeInputKind::ScalarListInt || input.input_kind == NodeInputKind::ScalarListFloat ||
          input.input_kind == NodeInputKind::ScalarListBool) {
        if (input.input_kind == NodeInputKind::ScalarListInt && input.raw_arg.contains("as_ints") &&
            input.raw_arg["as_ints"].is_array()) {
          shape_dims = {static_cast<int64_t>(input.raw_arg["as_ints"].size())};
        } else if (input.input_kind == NodeInputKind::ScalarListFloat && input.raw_arg.contains("as_floats") &&
                   input.raw_arg["as_floats"].is_array()) {
          shape_dims = {static_cast<int64_t>(input.raw_arg["as_floats"].size())};
        } else if (input.input_kind == NodeInputKind::ScalarListBool && input.raw_arg.contains("as_bools") &&
                   input.raw_arg["as_bools"].is_array()) {
          shape_dims = {static_cast<int64_t>(input.raw_arg["as_bools"].size())};
        }
      }
      ge::TensorDesc desc(ge::Shape(shape_dims), ge::FORMAT_ND, dtype);
      inputs.push_back({ir_name, desc});
      std::string const_ref = node.name + "_const_" + input.name;
      bindings.push_back({const_ref, ir_name, -1});
    } else if (IsSymInput(input)) {
      ge::DataType dtype = ge::DT_INT64;
      if (input.input_kind == NodeInputKind::SymFloat || input.input_kind == NodeInputKind::SymListFloat) {
        dtype = ge::DT_FLOAT;
      } else if (input.input_kind == NodeInputKind::SymBool || input.input_kind == NodeInputKind::SymListBool) {
        dtype = ge::DT_BOOL;
      }
      std::vector<int64_t> shape_dims;
      if (input.input_kind == NodeInputKind::SymListInt || input.input_kind == NodeInputKind::SymListFloat ||
          input.input_kind == NodeInputKind::SymListBool) {
        shape_dims = {static_cast<int64_t>(input.tensor_names.size())};
      }
      ge::TensorDesc desc(ge::Shape(shape_dims), ge::FORMAT_ND, dtype);
      inputs.push_back({ir_name, desc});
      if (input.input_kind == NodeInputKind::SymListInt || input.input_kind == NodeInputKind::SymListFloat ||
          input.input_kind == NodeInputKind::SymListBool) {
        for (size_t tensor_index = 0; tensor_index < input.tensor_names.size(); ++tensor_index) {
          const std::string &tensor_name = input.tensor_names[tensor_index];
          const std::string instance_name = ir_name + std::to_string(tensor_index);
          bindings.push_back({tensor_name, instance_name, -1});
        }
      } else {
        bindings.push_back({input.tensor_name, ir_name, -1});
      }
    }
  }
  return inputs;
}

std::vector<std::pair<std::string, ge::TensorDesc>> BuildFallbackOutputDefs(const ParsedNode &node,
                                                                            const EpairModelParser &parser,
                                                                            const RegisteredIrDef &ir_def,
                                                                            const td::AtenTensorDescMapper &mapper,
                                                                            ValidationResult &result) {
  const std::string context = NodeDebugContext(node);
  if (node.outputs.size() != ir_def.output_names.size()) {
    AddError(result, "INVALID_FALLBACK_OUTPUTS", "Fallback JSON outputs do not match registered IR output count",
             context);
    return {};
  }
  std::vector<std::pair<std::string, ge::TensorDesc>> outputs;
  for (size_t output_index = 0; output_index < node.outputs.size(); ++output_index) {
    const ParsedNodeOutput &output = node.outputs[output_index];
    const std::string &ir_name = ir_def.output_names[output_index];
    if (!output.is_tensor && !output.is_tensor_list) {
      // todo(@AGENTS): Fallback 仅支持 tensor / tensor_list；其他 JSON 输出类型直接报错。
      AddError(result, "UNSUPPORTED_FALLBACK_OUTPUT",
               "Fallback node currently only supports tensor or tensor-list JSON outputs", context + ":" + ir_name);
      return {};
    }
    const std::vector<std::string> tensor_names = CollectOutputTensorNames(output);
    for (size_t tensor_index = 0; tensor_index < tensor_names.size(); ++tensor_index) {
      const std::string &tensor_name = tensor_names[tensor_index];
      const ParsedTensorMeta *output_meta = parser.FindTensorValue(tensor_name);
      if (output_meta == nullptr) {
        AddError(result, "MISSING_OUTPUT_META", "Fallback output tensor meta missing", tensor_name);
        return {};
      }
      const td::GeTensorDescInfo desc_info = mapper.MapPt2TensorMeta(*output_meta, result, context + ":" + ir_name);
      if (!result.ok) {
        return {};
      }
      const std::string instance_name = output.is_tensor ? ir_name : ir_name + std::to_string(tensor_index);
      outputs.push_back({instance_name, MakeTensorDesc(desc_info)});
    }
  }
  return outputs;
}

std::vector<std::pair<std::string, ge::AttrValue>> BuildFallbackAttrs(const ParsedNode &node,
                                                                      ValidationResult &result) {
  const OrderedJson attrs_json = CollectFallbackAttrs(node, result);
  if (!result.ok || !attrs_json.is_object()) {
    return {};
  }
  std::vector<std::pair<std::string, ge::AttrValue>> attrs;
  attrs.reserve(attrs_json.size());
  for (const auto &[name, value] : attrs_json.items()) {
    auto parsed = ParseAttrValue(value, result, NodeDebugContext(node) + ":" + name);
    if (!result.ok) {
      return {};
    }
    if (parsed.has_value()) {
      attrs.push_back({name, parsed.value()});
    }
  }
  return attrs;
}

bool PrepareContext(GraphBuildContext &ctx, ValidationResult &result) {
  if (ctx.graph == nullptr) {
    TDLOGE("PrepareContext failed: graph is null");
    AddError(result, "GRAPH_BUILDER_ERROR", "Graph initialization returned null graph");
    return false;
  }
  ge::AscendString graph_name;
  const char *graph_name_cstr = "<unknown>";
  if (ctx.graph->GetName(graph_name) == ge::GRAPH_SUCCESS && graph_name.GetString() != nullptr) {
    graph_name_cstr = graph_name.GetString();
  }
  TDLOGI("PrepareContext success: graph_name=%s", graph_name_cstr);
  return true;
}

bool PreparePlaceholders(GraphBuildContext &ctx, const EpairModelParser &parser, const GeWeightLoader &loader,
                         const td::AtenTensorDescMapper &mapper, ValidationResult &result) {
  int64_t user_input_index = 0;
  for (const auto &input : parser.GraphInputs()) {
    const PlaceholderKind kind = parser.PlaceholderKindOf(input.name);
    TDLOGI("PreparePlaceholders processing: placeholder=%s, kind=%d", input.name.c_str(), static_cast<int>(kind));
    if (kind == PlaceholderKind::UserInput) {
      // Look up the InputSpec to get value_type
      const ParsedInputSpec *input_spec = nullptr;
      for (const auto &spec : parser.Model().input_specs) {
        if (spec.placeholder_name == input.name) {
          input_spec = &spec;
          break;
        }
      }
      
      if (input_spec == nullptr) {
        TDLOGE("PreparePlaceholders failed: missing input spec for %s", input.name.c_str());
        AddError(result, "MISSING_INPUT_SPEC", "Missing input spec", input.name);
        return false;
      }
      
      switch (input_spec->input_kind) {
        case NodeInputKind::Tensor: {
          // Existing logic: create Data node for tensor
          const ParsedTensorMeta *meta = parser.FindTensorValue(input.name);
          if (meta == nullptr) {
            TDLOGE("PreparePlaceholders failed: missing tensor meta for user input %s", input.name.c_str());
            AddError(result, "MISSING_INPUT_META", "Missing tensor metadata for user input", input.name);
            return false;
          }
          const auto desc_info = mapper.MapPt2TensorMeta(*meta, result, input.name);
          if (!result.ok) {
            return false;
          }
          ge::AttrValue index_attr;
          index_attr.SetAttrValue(user_input_index++);
          std::vector<std::pair<std::string, ge::TensorDesc>> data_inputs = {{"x", MakeTensorDesc(desc_info)}};
          std::vector<std::pair<std::string, ge::TensorDesc>> data_outputs = {{"y", MakeTensorDesc(desc_info)}};
          
          data_inputs[0].second.SetPlacement(ge::kPlacementDevice);
          data_outputs[0].second.SetPlacement(ge::kPlacementDevice);
          
          // Build shape_range: static dims use {size, size}, dynamic dims use range_constraints
          std::vector<std::pair<int64_t, int64_t>> shape_range;
          shape_range.reserve(meta->sizes.size());
          for (size_t i = 0; i < meta->sizes.size(); ++i) {
            if (i < meta->dynamic_dim_symbols.size() && !meta->dynamic_dim_symbols[i].empty()) {
              const std::string &symbol = meta->dynamic_dim_symbols[i];
              auto it = parser.Model().range_constraints.find(symbol);
              if (it != parser.Model().range_constraints.end()) {
                shape_range.push_back(it->second);
                TDLOGI("PreparePlaceholders: shape_range dim %zu is dynamic, symbol=%s, range=[%ld, %ld]",
                       i, symbol.c_str(), static_cast<long>(it->second.first), static_cast<long>(it->second.second));
              } else {
                shape_range.push_back({-1, -1});
              }
            } else {
              shape_range.push_back({meta->sizes[i], meta->sizes[i]});
            }
          }
          for (auto &[name, desc] : data_inputs) {
            desc.SetShapeRange(shape_range);
          }
          for (auto &[name, desc] : data_outputs) {
            desc.SetShapeRange(shape_range);
          }
          
          std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{"index", index_attr}};
          std::unique_ptr<ge::GNode> data_node =
              BuildRecoverableNode(*ctx.graph, "Data", input.name, data_inputs, data_outputs, attrs, result, input.name);
          if (data_node == nullptr) {
            return false;
          }
          int32_t output_index = 0;
          if (!LookupOutputIndex(*data_node, "y", output_index, result, input.name)) {
            return false;
          }
          if (!ctx.registry.RegisterPlaceholderSource(input.name, {*data_node, output_index}, result)) {
            return false;
          }
          TDLOGI("PreparePlaceholders created Data node: placeholder=%s, output_ref=%s:0", input.name.c_str(),
                 input.name.c_str());
          break;
        }
        
        case NodeInputKind::SymInt:
        case NodeInputKind::SymFloat:
        case NodeInputKind::SymBool: {
          ge::DataType dtype;
          TensorHolder holder;
          holder.dims = {1};
          holder.format = pt2::kLayoutStrided;
          
          if (input_spec->input_kind == NodeInputKind::SymInt) {
            auto value_opt = parser.FindSymIntValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing sym_int value for %s", input.name.c_str());
              AddError(result, "MISSING_SYM_INT_VALUE", "Missing sym_int value", input.name);
              return false;
            }
            dtype = ge::DT_INT64;
            holder.datatype = pt2::kDtypeLong;
            holder.value = std::vector<int64_t>{value_opt.value()};
          } else if (input_spec->input_kind == NodeInputKind::SymFloat) {
            auto value_opt = parser.FindSymFloatValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing sym_float value for %s", input.name.c_str());
              AddError(result, "MISSING_SYM_FLOAT_VALUE", "Missing sym_float value", input.name);
              return false;
            }
            dtype = ge::DT_FLOAT;
            holder.datatype = pt2::kDtypeFloat;
            holder.value = std::vector<float>{static_cast<float>(value_opt.value())};
          } else {
            auto value_opt = parser.FindSymBoolValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing sym_bool value for %s", input.name.c_str());
              AddError(result, "MISSING_SYM_BOOL_VALUE", "Missing sym_bool value", input.name);
              return false;
            }
            dtype = ge::DT_BOOL;
            holder.datatype = pt2::kDtypeBool;
            holder.value = std::vector<uint8_t>{static_cast<uint8_t>(value_opt.value() ? 1 : 0)};
          }
          
          // Create Data node for sym type
          ge::TensorDesc desc;
          desc.SetDataType(dtype);
          desc.SetShape(ge::Shape(holder.dims));
          desc.SetFormat(ge::FORMAT_ND);
          desc.SetPlacement(ge::kPlacementDevice);
          
          ge::AttrValue index_attr;
          index_attr.SetAttrValue(user_input_index++);
          
          std::vector<std::pair<std::string, ge::TensorDesc>> data_inputs = {{"x", desc}};
          std::vector<std::pair<std::string, ge::TensorDesc>> data_outputs = {{"y", desc}};
          std::vector<std::pair<std::string, ge::AttrValue>> attrs = {{"index", index_attr}};
          
          std::unique_ptr<ge::GNode> data_node =
              BuildRecoverableNode(*ctx.graph, "Data", input.name, data_inputs, data_outputs, attrs, result, input.name);
          if (data_node == nullptr) {
            return false;
          }
          
          int32_t output_index = 0;
          if (!LookupOutputIndex(*data_node, "y", output_index, result, input.name)) {
            return false;
          }
          if (!ctx.registry.RegisterPlaceholderSource(input.name, {*data_node, output_index}, result)) {
            return false;
          }
          TDLOGI("PreparePlaceholders created Data node for sym type: placeholder=%s, dtype=%d", 
                 input.name.c_str(), static_cast<int>(dtype));
          break;
        }
        
        case NodeInputKind::ScalarInt:
        case NodeInputKind::ScalarFloat:
        case NodeInputKind::ScalarBool:
        case NodeInputKind::ScalarString: {
          TensorHolder holder;
          holder.dims = {1};
          holder.format = pt2::kLayoutStrided;
          
          if (input_spec->input_kind == NodeInputKind::ScalarInt) {
            auto value_opt = parser.FindIntValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing int value for %s", input.name.c_str());
              AddError(result, "MISSING_INT_VALUE", "Missing int value", input.name);
              return false;
            }
            holder.datatype = pt2::kDtypeLong;
            holder.value = std::vector<int64_t>{value_opt.value()};
          } else if (input_spec->input_kind == NodeInputKind::ScalarFloat) {
            auto value_opt = parser.FindFloatValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing float value for %s", input.name.c_str());
              AddError(result, "MISSING_FLOAT_VALUE", "Missing float value", input.name);
              return false;
            }
            holder.datatype = pt2::kDtypeFloat;
            holder.value = std::vector<float>{static_cast<float>(value_opt.value())};
          } else if (input_spec->input_kind == NodeInputKind::ScalarBool) {
            auto value_opt = parser.FindBoolValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing bool value for %s", input.name.c_str());
              AddError(result, "MISSING_BOOL_VALUE", "Missing bool value", input.name);
              return false;
            }
            holder.datatype = pt2::kDtypeBool;
            holder.value = std::vector<uint8_t>{static_cast<uint8_t>(value_opt.value() ? 1 : 0)};
          } else {
            auto value_opt = parser.FindStringValue(input.name);
            if (!value_opt.has_value()) {
              TDLOGE("PreparePlaceholders failed: missing string value for %s", input.name.c_str());
              AddError(result, "MISSING_STRING_VALUE", "Missing string value", input.name);
              return false;
            }
            holder.datatype = pt2::kDtypeByte;
            const std::string &str = value_opt.value();
            holder.value = std::vector<uint8_t>(str.begin(), str.end());
            holder.dims = {static_cast<int64_t>(str.size())};
          }
          
          const std::optional<RegisteredValue> const_value =
              CreateConstNode(*ctx.graph, input.name, holder, mapper, result, input.name);
          if (!result.ok || !const_value.has_value()) {
            TDLOGE("PreparePlaceholders failed: create const node error for %s", input.name.c_str());
            return false;
          }
          if (!ctx.registry.RegisterPlaceholderSource(input.name, const_value.value(), result)) {
            return false;
          }
          TDLOGI("PreparePlaceholders created Const node for scalar: placeholder=%s", input.name.c_str());
          break;
        }
        
        default:
          TDLOGE("PreparePlaceholders failed: unknown value type for %s", input.name.c_str());
          AddError(result, "UNKNOWN_VALUE_TYPE", "Unknown value type", input.name);
          return false;
      }
      
      continue;
    }
    if (kind == PlaceholderKind::Parameter || kind == PlaceholderKind::Buffer ||
        kind == PlaceholderKind::TensorConstant) {
      TensorHolder holder = loader.ReadTensorData(input.name, result);
      if (!result.ok) {
        TDLOGE("PreparePlaceholders failed: load tensor data error, placeholder=%s", input.name.c_str());
        return false;
      }
      const std::optional<RegisteredValue> const_value =
          CreateConstNode(*ctx.graph, input.name, holder, mapper, result, input.name);
      if (!result.ok || !const_value.has_value()) {
        TDLOGE("PreparePlaceholders failed: create const node error, placeholder=%s", input.name.c_str());
        return false;
      }
      if (!ctx.registry.RegisterPlaceholderSource(input.name, const_value.value(), result)) {
        return false;
      }
      TDLOGI("PreparePlaceholders created Const node: placeholder=%s, bytes=%zu", input.name.c_str(),
             holder.ByteSize());
      continue;
    }
    if (kind == PlaceholderKind::ConstantInput) {
      // Find the input_spec to get the value
      const ParsedInputSpec *input_spec = nullptr;
      for (const auto &spec : parser.Model().input_specs) {
        if (spec.placeholder_name == input.name) {
          input_spec = &spec;
          break;
        }
      }
      if (input_spec == nullptr) {
        TDLOGE("PreparePlaceholders failed: missing input spec for constant_input %s", input.name.c_str());
        AddError(result, "MISSING_INPUT_SPEC", "Missing input spec for constant_input", input.name);
        return false;
      }
      
      TensorHolder holder;
      holder.dims = {};
      holder.format = pt2::kLayoutStrided;
      
      if (input_spec->input_kind == NodeInputKind::ScalarInt) {
        auto value_opt = parser.FindIntValue(input.name);
        if (!value_opt.has_value()) {
          TDLOGE("PreparePlaceholders failed: missing int value for %s", input.name.c_str());
          AddError(result, "MISSING_INT_VALUE", "Missing int value", input.name);
          return false;
        }
        holder.datatype = pt2::kDtypeLong;
        holder.value = std::vector<int64_t>{value_opt.value()};
      } else if (input_spec->input_kind == NodeInputKind::ScalarFloat) {
        auto value_opt = parser.FindFloatValue(input.name);
        if (!value_opt.has_value()) {
          TDLOGE("PreparePlaceholders failed: missing float value for %s", input.name.c_str());
          AddError(result, "MISSING_FLOAT_VALUE", "Missing float value", input.name);
          return false;
        }
        holder.datatype = pt2::kDtypeFloat;
        holder.value = std::vector<float>{static_cast<float>(value_opt.value())};
      } else if (input_spec->input_kind == NodeInputKind::ScalarBool) {
        auto value_opt = parser.FindBoolValue(input.name);
        if (!value_opt.has_value()) {
          TDLOGE("PreparePlaceholders failed: missing bool value for %s", input.name.c_str());
          AddError(result, "MISSING_BOOL_VALUE", "Missing bool value", input.name);
          return false;
        }
        holder.datatype = pt2::kDtypeBool;
        holder.value = std::vector<uint8_t>{static_cast<uint8_t>(value_opt.value())};
      } else if (input_spec->input_kind == NodeInputKind::None) {
        // as_none 的 constant_input（如 past_key_values=None, embeddings=None）
        // 是 Python None，不是真实 tensor → 跳过，不创建 Const 节点。
        // 下游 PlanConverterNode (line ~830) 已用 if (None) continue 跳过此类输入。
        TDLOGI("PreparePlaceholders skipping as_none constant_input: %s", input.name.c_str());
        continue;
      } else {
        TDLOGE("PreparePlaceholders failed: unsupported constant_input value type for %s", input.name.c_str());
        AddError(result, "UNSUPPORTED_CONSTANT_TYPE", "Unsupported constant_input value type", input.name);
        return false;
      }
      
      const std::optional<RegisteredValue> const_value =
          CreateConstNode(*ctx.graph, input.name, holder, mapper, result, input.name);
      if (!result.ok || !const_value.has_value()) {
        TDLOGE("PreparePlaceholders failed: create const node error for constant_input %s", input.name.c_str());
        return false;
      }
      if (!ctx.registry.RegisterPlaceholderSource(input.name, const_value.value(), result)) {
        return false;
      }
      TDLOGI("PreparePlaceholders created Const node for constant_input: placeholder=%s", input.name.c_str());
    }
  }
  return result.ok;
}

bool PlanConverterNode(GraphBuildContext &ctx, const ParsedNode &node, const GraphPlanningDeps &deps,
                       ValidationResult &result) {
  const ParsedAscendIr &ascend_ir = node.ascend_ir.value();
  // Pre-pass: a Transpose whose data input (input[0]) is an NZ weight must be skipped — its 2D perm
  // can't match GE's 4D FRACTAL_NZ storage. Instead: pass the NZ Const directly to the downstream
  // MatMul and set transpose_x2=True (GE MatMul native attr, internal transpose, no Transpose op).
  // Map: {Transpose output ref -> (Const ref, needs_transpose_x2)}. Stored in ctx (cross-node).
  for (const auto &pre_op : ascend_ir.ops) {
    if (pre_op.type != "Transpose" || pre_op.inputs.size() < 2) continue;
    std::string data_ref = pre_op.inputs[0];
    const size_t colon = data_ref.rfind(':');
    if (colon != std::string::npos) data_ref = data_ref.substr(0, colon);
    const auto *w_info = deps.weight_loader.GetTensorRecordInfo(data_ref);
    if (w_info != nullptr && w_info->format_str == "NZ") {
      ctx.nz_transpose_passthrough[pre_op.name + ":0"] = {pre_op.inputs[0], true};
    }
  }
  for (const auto &op : ascend_ir.ops) {
    if (op.name.empty() || op.type.empty()) {
      continue;
    }
    if (IsConstOp(op)) {
      TDLOGI("Restore Const op in ascend_ir: name[%s]", op.name.c_str());
      if (!op.attrs.is_object() || !op.attrs.contains("value")) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "inline const missing value attr", "", op.name).c_str());
        AddError(result, "MISSING_CONST_VALUE", "Inline const op missing value attr", op.name);
        return false;
      }
      TensorHolder holder = deps.weight_loader.RestoreInlineConstTensor(op.attrs["value"], result, op.name);
      if (!result.ok || !ValidateConstOutputDesc(op, holder, deps.tensor_desc_mapper, result)) {
        if (!result.ok) {
          TDLOGE("%s",
                 MakeConverterNodeError(node, &op, "inline const restore or output desc validation error", "", op.name)
                     .c_str());
        }
        return false;
      }
      const std::optional<RegisteredValue> const_value =
          CreateConstNode(*ctx.graph, op.name, holder, deps.tensor_desc_mapper, result, op.name);
      if (!result.ok || !const_value.has_value()) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "inline const node creation error", "", op.name).c_str());
        return false;
      }
      if (!ctx.registry.RegisterProducerOutput(op.name + ":0", const_value.value(), result)) {
        TDLOGE("%s",
               MakeConverterNodeError(node, &op, "register const output error", "", op.name, op.name + ":0").c_str());
        return false;
      }
      continue;
    }
    // Skip Transpose ops on NZ weights — pass through to the downstream MatMul with transpose_x2=True.
    if (op.type == "Transpose" && ctx.nz_transpose_passthrough.count(op.name + ":0") > 0) {
      TDLOGI("[NZ] Skipping Transpose %s on NZ weight — downstream MatMul gets transpose_x2=True", op.name.c_str());
      // Register the Transpose's output as a pass-through to the Const's output,
      // so downstream bindings (MatMul input[1] → Transpose:0) resolve to the Const.
      const auto &passthrough = ctx.nz_transpose_passthrough.at(op.name + ":0");
      const auto *src_value = ctx.registry.Resolve(passthrough.first);
      if (src_value != nullptr) {
        ctx.registry.RegisterProducerOutput(op.name + ":0", *src_value, result);
      }
      continue;
    }
    std::vector<std::pair<std::string, ge::TensorDesc>> inputs;
    std::vector<std::string> input_names;
    std::vector<PlannedEdgeBinding> bindings;
    inputs.reserve(op.inputs.size());
    input_names.reserve(op.inputs.size());
    bindings.reserve(op.inputs.size());
    for (size_t input_index = 0; input_index < op.inputs.size(); ++input_index) {
      const ParsedAscendIrOpDesc *desc = input_index < op.input_descs.size() ? &op.input_descs[input_index] : nullptr;
      if (desc == nullptr || desc->name.empty()) {
        TDLOGE("%s",
               MakeConverterNodeError(node, &op, "missing converter inputDesc name", "", "", op.inputs[input_index])
                   .c_str());
        AddError(result, "MISSING_INPUT_DESC", "Converter inputDesc is required for each input", op.name);
        return false;
      }
      if (op.inputs[input_index].empty()) {
        if (IsOptionalUnfed(*desc)) {
          continue;
        }
        TDLOGE("%s",
               MakeConverterNodeError(node, &op, "converter input tensor ref is empty", "", desc->name, "").c_str());
        AddError(result, "MISSING_INPUT_TENSOR", "Converter input tensor ref is empty", op.name + ":" + desc->name);
        return false;
      }
      // Fallback: 对边界输入(FX 张量名)从 tensor_values 取 shape/dtype, 让 GE infershape
      // 能沿 Data 边推断下游 op 的 input (如 Unsqueeze->Conv2D). 内部 GE ref 找不到返回 nullptr.
      const ParsedTensorMeta *input_fallback = deps.parser.FindTensorValue(op.inputs[input_index]);
      td::GeTensorDescInfo desc_info =
          deps.tensor_desc_mapper.MapAscendIrDesc(desc, input_fallback, result, op.name + ":" + desc->name);
      if (!result.ok) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "converter input desc mapping error", "", desc->name,
                                            op.inputs[input_index])
                         .c_str());
        return false;
      }
      // NZ weight input: if this input refs a weight whose weight_info.format == "NZ", set the input
      // desc format to FRACTAL_NZ so GE selects the NZ kernel (e.g. MatMulV2_NDNZ_ND) and the
      // weight Const (FRACTAL_NZ) matches — avoids the format-transfer Transpose (4D-vs-2D perm).
      // Also: if this input was supposed to come from a skipped NZ-Transpose, bind to the Const
      // directly and flag transpose_x2=True for the downstream MatMul.
      bool need_transpose_x2 = false;
      {
        // Resolve boundary alias (e.g., 't' → 'Transpose:0') before checking passthrough map.
        std::string resolved_ref = ResolveBoundaryInputSourceRef(node, op.inputs[input_index]);
        std::string ref = resolved_ref;
        // If this input refs a skipped Transpose's output, replace with the Const ref.
        auto pt_it = ctx.nz_transpose_passthrough.find(ref);
        if (pt_it != ctx.nz_transpose_passthrough.end()) {
          ref = pt_it->second.first;  // the Const ref (e.g., "p_q_proj_weight:0")
          need_transpose_x2 = pt_it->second.second;
        }
        const size_t colon = ref.rfind(':');
        if (colon != std::string::npos) ref = ref.substr(0, colon);
        const auto *w_info = deps.weight_loader.GetTensorRecordInfo(ref);
        if (w_info != nullptr && w_info->format_str == "NZ") {
          desc_info.format = deps.tensor_desc_mapper.MapGeLayoutString("NZ", result, op.name + ":" + desc->name);
          desc_info.has_format = true;
          if (!result.ok) {
            TDLOGE("%s", MakeConverterNodeError(node, &op, "NZ input format override error", "", desc->name,
                                                op.inputs[input_index])
                             .c_str());
            return false;
          }
        }
        // Binding: use the passthrough Const ref if available, else the original resolved ref.
        const std::string &binding_ref = (pt_it != ctx.nz_transpose_passthrough.end()) ? pt_it->second.first : resolved_ref;
        bindings.push_back({ResolveBoundaryInputSourceRef(node, binding_ref), desc->name, -1});
        if (need_transpose_x2) {
          ctx.deferred_nz_transpose_x2_ops.insert(op.name);
        }
      }
      inputs.push_back({desc->name, MakeTensorDesc(desc_info)});
      input_names.push_back(desc->name);
    }
    std::vector<std::pair<std::string, ge::TensorDesc>> outputs;
    outputs.reserve(op.output_descs.size());
    for (const auto &desc : op.output_descs) {
      if (desc.name.empty()) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "missing converter outputDesc name").c_str());
        AddError(result, "MISSING_OUTPUT_DESC", "Converter outputDesc name is required", op.name);
        return false;
      }
      const td::GeTensorDescInfo desc_info =
          deps.tensor_desc_mapper.MapAscendIrDesc(&desc, nullptr, result, op.name + ":" + desc.name);
      if (!result.ok) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "converter output desc mapping error", "", desc.name).c_str());
        return false;
      }
      outputs.push_back({desc.name, MakeTensorDesc(desc_info)});
    }
    std::vector<std::pair<std::string, ge::AttrValue>> attrs;
    if (op.attrs.is_object()) {
      attrs.reserve(op.attrs.size());
      TDLOGD("PlanConverterNode: parsing %zu attributes for op=%s, type=%s", op.attrs.size(), op.name.c_str(), op.type.c_str());
      for (const auto &[attr_name, attr_value_json] : op.attrs.items()) {
        auto parsed = ParseAttrValue(attr_value_json, result, op.name + ":" + attr_name);
        if (!result.ok) {
          TDLOGE("%s", MakeConverterNodeError(node, &op, "converter attr parse error", "", "", attr_name).c_str());
          return false;
        }
        if (parsed.has_value()) {
          attrs.push_back({attr_name, parsed.value()});
        }
      }
    }
    // Override transpose_x2=True for MatMul ops whose weight input was a skipped NZ-Transpose.
    if (ctx.deferred_nz_transpose_x2_ops.count(op.name) > 0) {
      ge::AttrValue tx2_attr;
      tx2_attr.SetAttrValue(true);
      attrs.push_back({"transpose_x2", tx2_attr});
      TDLOGI("[NZ] Set transpose_x2=True on %s (skipped NZ-Transpose passthrough)", op.name.c_str());
    }
    std::unique_ptr<ge::GNode> node_ptr =
        BuildRecoverableNode(*ctx.graph, op.type, op.name, inputs, outputs, attrs, result, op.name);
    if (node_ptr == nullptr) {
      TDLOGE("%s", MakeConverterNodeError(node, &op, "recoverable build error").c_str());
      return false;
    }
    ge::GNode ge_node = *node_ptr;
    for (auto &binding : bindings) {
      if (!LookupInputIndex(ge_node, binding.input_name, binding.dst_index, result, op.name)) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "lookup input index error", "", binding.input_name,
                                            binding.src_ref, binding.dst_index)
                          .c_str());
        return false;
      }
      ctx.deferred_edges.push_back({binding.src_ref, ge_node, binding.dst_index});
    }
    for (const auto &output : outputs) {
      int32_t output_index = 0;
      if (!LookupOutputIndex(ge_node, output.first, output_index, result, op.name)) {
        TDLOGE("%s", MakeConverterNodeError(node, &op, "lookup output index error", "", output.first).c_str());
        return false;
      }
      const std::string output_ref = op.name + ":" + std::to_string(output_index);
      if (!ctx.registry.RegisterProducerOutput(output_ref, {ge_node, output_index}, result)) {
        TDLOGE("%s",
               MakeConverterNodeError(node, &op, "register output error", "", output.first, output_ref, output_index)
                   .c_str());
        return false;
      }
    }
  }
  const std::vector<std::string> node_output_names = CollectNodeOutputTensorNames(node);
  for (size_t index = 0; index < node_output_names.size(); ++index) {
    const std::string &fx_tensor_name = node_output_names[index];
    auto output_it = ascend_ir.output_mappings.find(fx_tensor_name);
    if (output_it == ascend_ir.output_mappings.end()) {
      output_it = ascend_ir.output_mappings.find(std::to_string(index));
    }
    if (output_it == ascend_ir.output_mappings.end()) {
      TDLOGE("%s", MakeConverterNodeError(node, nullptr, "mapping.outputs is incomplete", fx_tensor_name).c_str());
      AddError(result, "MISSING_MAPPED_OUTPUT", "Converter mapping.outputs is incomplete", NodeDebugContext(node));
      return false;
    }
    ctx.deferred_outputs.push_back({fx_tensor_name, output_it->second});
  }
  return result.ok;
}

bool PlanConverterNoOpNode(GraphBuildContext &ctx, const ParsedNode &node, const GraphPlanningDeps &deps,
                           ValidationResult &result) {
  if (node.ascend_ir.has_value() && !node.ascend_ir->output_mappings.empty()) {
    const std::vector<std::string> node_output_names = CollectNodeOutputTensorNames(node);
    for (const auto &output_name : node_output_names) {
      auto output_it = node.ascend_ir->output_mappings.find(output_name);
      if (output_it == node.ascend_ir->output_mappings.end()) {
        TDLOGE(
            "%s",
            MakeConverterNodeError(node, nullptr, "NoOp converter mapping.outputs is incomplete", output_name).c_str());
        AddError(result, "MISSING_MAPPED_OUTPUT", "NoOp converter mapping.outputs is incomplete",
                 NodeDebugContext(node));
        return false;
      }
      ctx.deferred_outputs.push_back({output_name, output_it->second});
    }
    return true;
  }

  if (!node.ascend_ir.has_value() || !node.ascend_ir->input_mappings.empty()) {
    TDLOGE(
        "%s",
        MakeConverterNodeError(node, nullptr, "NoOp converter with non-empty mapping.inputs is unsupported").c_str());
    AddError(result, "UNSUPPORTED_NOOP_CONVERTER",
             "NoOp converter currently requires empty mapping.inputs when mapping.outputs is absent",
             NodeDebugContext(node));
    return false;
  }

  // NoOp converter means exporter intentionally skipped GE node creation.
  // Under the current exporter contract, the FX output is a passthrough alias
  // of the first tensor input (typically schema `self`), so we recover that
  // alias directly through the registry instead of fabricating a GE node.
  if (node.outputs.size() != 1 || !node.outputs[0].is_tensor) {
    TDLOGE(
        "%s",
        MakeConverterNodeError(node, nullptr, "NoOp converter currently only supports a single tensor output").c_str());
    AddError(result, "UNSUPPORTED_NOOP_CONVERTER", "NoOp converter currently only supports a single tensor output",
             NodeDebugContext(node));
    return false;
  }
  if (node.inputs.empty()) {
    TDLOGE("%s", MakeConverterNodeError(node, nullptr, "NoOp converter has no inputs").c_str());
    AddError(result, "UNSUPPORTED_NOOP_CONVERTER", "NoOp converter must have at least one input",
             NodeDebugContext(node));
    return false;
  }

  const ParsedNodeInput &passthrough_input = node.inputs.front();
  if (passthrough_input.input_kind != NodeInputKind::Tensor || passthrough_input.tensor_name.empty()) {
    TDLOGE("%s", MakeConverterNodeError(node, nullptr, "NoOp converter first input must be a tensor", "",
                                        passthrough_input.name)
                     .c_str());
    AddError(result, "UNSUPPORTED_NOOP_CONVERTER",
             "NoOp converter currently requires the first input to be a tensor passthrough source",
             NodeDebugContext(node));
    return false;
  }

  const std::string &output_name = node.outputs[0].tensor_name;
  const ParsedTensorMeta *input_meta = deps.parser.FindTensorValue(passthrough_input.tensor_name);
  const ParsedTensorMeta *output_meta = deps.parser.FindTensorValue(output_name);
  if (input_meta == nullptr || output_meta == nullptr) {
    TDLOGE("%s", MakeConverterNodeError(node, nullptr, "NoOp converter tensor meta missing", output_name,
                                        passthrough_input.tensor_name)
                     .c_str());
    AddError(result, "MISSING_INPUT_META", "NoOp converter passthrough tensor metadata is missing",
             NodeDebugContext(node));
    return false;
  }
  if (!IsSameTensorMeta(*input_meta, *output_meta)) {
    TDLOGE("%s", MakeConverterNodeError(node, nullptr, "NoOp converter input/output tensor meta mismatch", output_name,
                                        passthrough_input.tensor_name)
                     .c_str());
    AddError(result, "INVALID_NOOP_CONVERTER", "NoOp converter output tensor meta must match the first tensor input",
             NodeDebugContext(node));
    return false;
  }

  ctx.deferred_outputs.push_back({output_name, passthrough_input.tensor_name});
  return true;
}

[[maybe_unused]] bool PlanFallbackNode(GraphBuildContext &ctx, const ParsedNode &node, const GraphPlanningDeps &deps,
                      ValidationResult &result) {
  const std::string context = NodeDebugContext(node);
  if (node.outputs.empty()) {
    TDLOGE("%s", MakeFallbackNodeError(node, "", "no outputs").c_str());
    return false;
  }

  const td::AtenTargetInfo target = deps.schema_resolver.ParseTarget(node.effective_schema_target, result, context);
  if (!result.ok) {
    TDLOGE("%s", MakeFallbackNodeError(node, "", "target parse error").c_str());
    return false;
  }

  const std::string op_type = deps.schema_resolver.MakeFallbackOpType(target);
  const std::optional<RegisteredIrDef> ir_def = QueryRegisteredIrDef(op_type, result, context);
  if (!result.ok || !ir_def.has_value()) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type, "registered IR query error").c_str());
    return false;
  }

  for (const auto &input : node.inputs) {
    if (!IsScalarInput(input)) continue;
    std::string const_name = node.name + "_const_" + input.name;
    auto holder = BuildScalarConstHolder(input, context, result);
    if (!result.ok) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type, "scalar const build error", input.name).c_str());
      return false;
    }
    auto const_value = CreateConstNode(*ctx.graph, const_name, holder.value(), deps.tensor_desc_mapper, result, context,
                                       ge::kPlacementHost);
    if (!result.ok || !const_value.has_value()) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type, "scalar const node creation error", input.name).c_str());
      return false;
    }
    std::string const_ref = const_name;
    if (!ctx.registry.RegisterProducerOutput(const_ref, const_value.value(), result)) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type, "register scalar const output error", input.name).c_str());
      return false;
    }
  }

  std::vector<PlannedEdgeBinding> bindings;
  std::vector<std::pair<std::string, ge::TensorDesc>> inputs =
      BuildFallbackInputDefs(node, deps.parser, ir_def.value(), deps.tensor_desc_mapper, bindings, result);
  if (!result.ok) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type, "fallback input definition error").c_str());
    return false;
  }

  std::vector<std::pair<std::string, ge::TensorDesc>> outputs =
      BuildFallbackOutputDefs(node, deps.parser, ir_def.value(), deps.tensor_desc_mapper, result);
  if (!result.ok) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type, "fallback output definition error").c_str());
    return false;
  }

  std::vector<std::pair<std::string, ge::AttrValue>> attrs = BuildFallbackAttrs(node, result);
  if (!result.ok) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type, "fallback attr parse error").c_str());
    return false;
  }

  std::unique_ptr<ge::GNode> node_ptr =
      BuildRecoverableNode(*ctx.graph, op_type, node.name, inputs, outputs, attrs, result, context);
  if (node_ptr == nullptr) {
    TDLOGE("%s", MakeFallbackNodeError(node, op_type, "recoverable build error").c_str());
    return false;
  }
  ge::GNode ge_node = *node_ptr;

  for (auto &binding : bindings) {
    if (!LookupInputIndex(ge_node, binding.input_name, binding.dst_index, result, context)) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type, "lookup input index error", "", binding.input_name,
                                         binding.src_ref, binding.dst_index)
                        .c_str());
      return false;
    }
  }
  for (const auto &binding : bindings) {
    ctx.deferred_edges.push_back({binding.src_ref, ge_node, binding.dst_index});
  }
  const std::vector<std::string> output_tensor_names = CollectNodeOutputTensorNames(node);
  for (size_t index = 0; index < output_tensor_names.size(); ++index) {
    int32_t output_index = 0;
    if (!LookupOutputIndex(ge_node, outputs[index].first, output_index, result, context)) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type, "lookup output index error", output_tensor_names[index],
                                         outputs[index].first)
                       .c_str());
      return false;
    }
    const std::string output_ref = node.name + ":" + std::to_string(output_index);
    if (!ctx.registry.RegisterProducerOutput(output_ref, {ge_node, output_index}, result)) {
      TDLOGE("%s", MakeFallbackNodeError(node, op_type, "register output error", output_tensor_names[index],
                                         outputs[index].first, output_ref, output_index)
                       .c_str());
      return false;
    }
    ctx.deferred_outputs.push_back({output_tensor_names[index], output_ref});
    if (node.ascend_ir.has_value()) {
      auto output_it = node.ascend_ir->output_mappings.find(output_tensor_names[index]);
      if (output_it != node.ascend_ir->output_mappings.end() && !output_it->second.empty()) {
        ctx.deferred_outputs.push_back({output_it->second, output_ref});
      }
    }
  }
  return result.ok;
}

bool PlanNodes(GraphBuildContext &ctx, const EpairModelParser &parser, const td::GeWeightLoader &weight_loader,
               const td::AtenSchemaResolver &resolver, const td::AtenTensorDescMapper &mapper, ValidationResult &result) {
  const GraphPlanningDeps deps{parser, weight_loader, resolver, mapper};
  for (const auto &node : parser.Nodes()) {
    // _assert_tensor_metadata 是 torch.export/torchair decomposition 插入的运行时
    // 校验节点（无输出、无 GE 算子映射），跳过以免被当成未映射算子导致构图失败。
    if (IsAssertTensorMetadataNode(node)) {
      TDLOGI("PlanNodes skipping _assert_tensor_metadata node: %s", node.name.c_str());
      continue;
    }
    TDLOGI("PlanNodes visiting: node=%s, target=%s, source=%s, outputs=%zu", node.name.c_str(), node.target.c_str(),
           node.source_info.source.c_str(), node.outputs.size());
    // 当前如果模型 JSON 中不包含 ascend_ir 字段，报错退出
    TD_ASSERT_TRUE(IsAscendIrPresent(node),
                   "Parsed node [%s] does not contain ascend_ir field, "
                   "json node index [%zu]",
                   node.name.c_str(), node.json_node_index);
    if (IsConverterNode(node)) {
      // 处理 Converter Node 场景
      if (HasBuildableConverterOps(node)) {
        TD_ASSERT_TRUE(PlanConverterNode(ctx, node, deps, result),
                       "PlanNodes failed: converter node build error, node=%s", node.name.c_str());
      } else if (IsConverterNoOpNode(node)) {
        // 处理 Converter Node 可能作了性能优化不产生 GE Op
        TD_ASSERT_TRUE(PlanConverterNoOpNode(ctx, node, deps, result),
                       "PlanNodes failed: converter NoOp node error, node=%s", node.name.c_str());
      } else {
        TDLOGE("PlanNodes failed: unsupported scenario, node=%s", node.name.c_str());
        return false;
      }
    } else {
      // 算子无法映射到 Ascend IR，直接报错退出
      TDLOGE("PlanNodes failed: node [%s] has no ascend_ir mapping (source is empty), "
             "target=%s, effective_target=%s. "
             "This operator cannot be converted to Ascend IR. "
             "Please register the GE operator or decompose the operation in the model definition.",
             node.name.c_str(), node.target.c_str(), node.effective_schema_target.c_str());
      result.ok = false;
      result.issues.push_back({
          IssueSeverity::Error,
          "UNMAPPED_OPERATOR",
          "Operator has no Ascend IR mapping: " + node.name,
          node.target
      });
      return false;
    }
  }
  return true;
}

bool BindDeferredEdges(GraphBuildContext &ctx, ValidationResult &result) {
  for (auto &edge : ctx.deferred_edges) {
    TDLOGI("BindDeferredEdges processing: source_ref=%s, dst_node=%s, dst_index=%d", edge.src_ref.c_str(),
           GetNodeNameForLog(edge.dst_node).c_str(), edge.dst_index);
    const RegisteredValue *source = ctx.registry.Resolve(edge.src_ref);
    if (source == nullptr) {
      TDLOGE("BindDeferredEdges failed: source_ref not found, source_ref=%s", edge.src_ref.c_str());
      AddError(result, "MISSING_INPUT_TENSOR", "Input tensor ref not found", edge.src_ref);
      return false;
    }
    ge::GNode source_node = source->producer;
    TDLOGI("BindDeferredEdges resolved: source_node=%s, source_index=%d, dst_node=%s, dst_index=%d",
           GetNodeNameForLog(source_node).c_str(), source->output_index, GetNodeNameForLog(edge.dst_node).c_str(),
           edge.dst_index);
    if (ctx.graph->AddDataEdge(source_node, source->output_index, edge.dst_node, edge.dst_index) != ge::GRAPH_SUCCESS) {
      TDLOGE("BindDeferredEdges failed: AddDataEdge error, source_node=%s, source_index=%d, dst_node=%s, dst_index=%d",
             GetNodeNameForLog(source_node).c_str(), source->output_index, GetNodeNameForLog(edge.dst_node).c_str(),
             edge.dst_index);
      AddError(result, "ADD_EDGE_FAILED", "Failed to connect GE edge", edge.src_ref);
      return false;
    }
  }
  return true;
}

bool BindDeferredOutputs(GraphBuildContext &ctx, ValidationResult &result) {
  for (const auto &binding : ctx.deferred_outputs) {
    TDLOGI("BindDeferredOutputs processing: alias=%s, source_ref=%s", binding.alias.c_str(),
           binding.source_ref.c_str());
    const RegisteredValue *source = ctx.registry.Resolve(binding.source_ref);
    if (source == nullptr) {
      TDLOGE("BindDeferredOutputs failed: source_ref not found, alias=%s, source_ref=%s", binding.alias.c_str(),
             binding.source_ref.c_str());
      AddError(result, "MISSING_MAPPED_OUTPUT", "Output binding source not found", binding.source_ref);
      return false;
    }
    if (!ctx.registry.RegisterAlias(binding.alias, binding.source_ref, result)) {
      TDLOGE("BindDeferredOutputs failed: register alias error, alias=%s, source_ref=%s", binding.alias.c_str(),
             binding.source_ref.c_str());
      return false;
    }
    TDLOGI("BindDeferredOutputs success: alias=%s -> source_ref=%s", binding.alias.c_str(), binding.source_ref.c_str());
  }
  return true;
}

bool FinalizeGraph(GraphBuildContext &ctx, const EpairModelParser &parser, ValidationResult &result) {
  std::vector<std::pair<ge::GNode, int32_t>> outputs;
  outputs.reserve(parser.GraphOutputs().size());
  for (size_t index = 0; index < parser.GraphOutputs().size(); ++index) {
    TDLOGI("FinalizeGraph processing output: graph_output_index=%zu, tensor_name=%s", index,
           parser.GraphOutputs()[index].name.c_str());
    const RegisteredValue *output = ctx.registry.Resolve(parser.GraphOutputs()[index].name);
    if (output == nullptr) {
      TDLOGE("FinalizeGraph failed: graph output tensor not found, graph_output_index=%zu, tensor_name=%s", index,
             parser.GraphOutputs()[index].name.c_str());
      AddError(result, "MISSING_GRAPH_OUTPUT", "Graph output tensor not found", parser.GraphOutputs()[index].name);
      return false;
    }
    TDLOGI("FinalizeGraph resolved output: graph_output_index=%zu, producer_node=%s, producer_index=%d", index,
           GetNodeNameForLog(output->producer).c_str(), output->output_index);
    outputs.push_back({output->producer, output->output_index});
  }
  if (ctx.graph->SetOutputs(outputs) != ge::GRAPH_SUCCESS) {
    TDLOGE("FinalizeGraph failed: SetOutputs error, output_count=%zu", outputs.size());
    AddError(result, "SET_GRAPH_OUTPUT_FAILED", "Failed to bind graph outputs");
    return false;
  }
  TDLOGI("FinalizeGraph success: output_count=%zu", outputs.size());
  return true;
}

}  // namespace

GeGraphBuilder::GeGraphBuilder(const std::string &model_json, const std::string &weights_index_path,
                                ValidationResult &result)
    : weights_index_path_(weights_index_path) {
  TDLOGI("GeGraphBuilder enter: weights_index=%s", weights_index_path.c_str());

  TDLOGI("GeGraphBuilder step: parsing model");
  parser_ = EpairModelParser::Parse(model_json, result);
  if (!result.ok) {
    TDLOGE("GeGraphBuilder failed: parse error, issues=%zu", result.issues.size());
    for (const auto &issue : result.issues) {
      if (issue.severity == IssueSeverity::Error) {
        TDLOGE("  parse issue: [%s] %s", issue.code.c_str(), issue.message.c_str());
      }
    }
    return;
  }
  TDLOGI("GeGraphBuilder step: parsed, nodes=%zu, inputs=%zu, outputs=%zu",
         parser_.Nodes().size(), parser_.GraphInputs().size(), parser_.GraphOutputs().size());

  TDLOGI("GeGraphBuilder step: loading weights");
  loader_ = GeWeightLoader::Create(weights_index_path_, parser_.Model(), result);
  if (!result.ok) {
    for (const auto &issue : result.issues) {
      TDLOGE("GeGraphBuilder weight issue: code=%s, message=%s, context=%s",
             issue.code.c_str(), issue.message.c_str(), issue.context.c_str());
    }
    TDLOGE("GeGraphBuilder failed: weight load error");
    return;
  }
  TDLOGI("GeGraphBuilder step: weights loaded");

  initialized_ = result.ok;
  TDLOGI("GeGraphBuilder exit: success, initialized=%s", initialized_ ? "true" : "false");
}

bool GeGraphBuilder::IsFullyOffloaded() const noexcept {
  for (const ParsedNode &node : parser_.Nodes()) {
    if (!node.ascend_ir.has_value()) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<ge::Graph> GeGraphBuilder::Build(ValidationResult &result) {
  TDLOGI("Build enter: weights_index=%s", weights_index_path_.c_str());

  if (!initialized_) {
    TDLOGE("Build failed: builder not initialized");
    AddError(result, "GRAPH_BUILDER_NOT_INITIALIZED", "GeGraphBuilder was not initialized");
    return nullptr;
  }

  GraphBuildContext ctx("epair_graph");

  TDLOGI("Build stage: PrepareContext");
  if (!PrepareContext(ctx, result)) {
    TDLOGE("Build failed at PrepareContext");
    return nullptr;
  }

  TDLOGI("Build stage: PreparePlaceholders, inputs=%zu", parser_.GraphInputs().size());
  if (!PreparePlaceholders(ctx, parser_, loader_, tensor_desc_mapper_, result)) {
    TDLOGE("Build failed at PreparePlaceholders");
    return nullptr;
  }

  TDLOGI("Build stage: PlanNodes, nodes=%zu", parser_.Nodes().size());
  if (!PlanNodes(ctx, parser_, loader_, schema_resolver_, tensor_desc_mapper_, result)) {
    TDLOGE("Build failed at PlanNodes");
    return nullptr;
  }

  TDLOGI("Build stage: BindDeferredOutputs, count=%zu", ctx.deferred_outputs.size());
  if (!BindDeferredOutputs(ctx, result)) {
    TDLOGE("Build failed at BindDeferredOutputs");
    return nullptr;
  }

  TDLOGI("Build stage: BindDeferredEdges, count=%zu", ctx.deferred_edges.size());
  if (!BindDeferredEdges(ctx, result)) {
    TDLOGE("Build failed at BindDeferredEdges");
    return nullptr;
  }

  TDLOGI("Build stage: FinalizeGraph, outputs=%zu", parser_.GraphOutputs().size());
  if (!FinalizeGraph(ctx, parser_, result)) {
    TDLOGE("Build failed at FinalizeGraph");
    return nullptr;
  }

  TDLOGI("Build exit: success");
  return std::move(ctx.graph);
}

}  // namespace td
