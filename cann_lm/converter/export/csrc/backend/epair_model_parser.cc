#include "backend/epair_model_parser.h"

#include <cctype>
#include <sstream>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "common/logging/logging.h"

namespace td {
namespace {

std::optional<int64_t> ParseOptionalInt(const OrderedJson &value) {
  if (value.is_null()) {
    return std::nullopt;
  }
  if (value.is_number_integer()) {
    return value.get<int64_t>();
  }
  return std::nullopt;
}

bool TryGetIntField(const OrderedJson &value, int64_t &out) {
  if (value.is_number_integer()) {
    out = value.get<int64_t>();
    return true;
  }
  if (!value.is_object()) return false;
  if (value.contains("as_int") && value["as_int"].is_number_integer()) {
    out = value["as_int"].get<int64_t>();
    return true;
  }
  if (value.contains("as_expr")) {
    out = -1;
    return true;
  }
  return false;
}

std::vector<int64_t> ParseIntList(const OrderedJson &array, ValidationResult &result, const std::string &code,
                                  const std::string &context) {
  std::vector<int64_t> values;
  if (!array.is_array()) {
    AddError(result, code, "Expected JSON array", context);
    return values;
  }
  for (const auto &item : array) {
    int64_t value = 0;
    if (!TryGetIntField(item, value)) {
      AddError(result, code, "Expected integer element", context);
      return {};
    }
    values.push_back(value);
  }
  return values;
}

ParsedTensorMeta ParseTensorMeta(const OrderedJson &meta, ValidationResult &result, const std::string &context) {
  ParsedTensorMeta parsed;
  if (!meta.is_object()) {
    AddError(result, "INVALID_TENSOR_META", "Tensor meta must be an object", context);
    return parsed;
  }
  if (!meta.contains("dtype") || !meta["dtype"].is_number_integer()) {
    AddError(result, "MISSING_TENSOR_DTYPE", "Tensor meta missing integer dtype", context);
  } else {
    parsed.dtype = meta["dtype"].get<int64_t>();
  }
  if (!meta.contains("sizes")) {
    AddError(result, "MISSING_TENSOR_SIZES", "Tensor meta missing sizes", context);
  } else {
    parsed.sizes = ParseIntList(meta["sizes"], result, "INVALID_TENSOR_SIZES", context);
    if (meta["sizes"].is_array()) {
      parsed.dynamic_dim_symbols.resize(meta["sizes"].size());
      for (size_t i = 0; i < meta["sizes"].size(); ++i) {
        const auto &dim = meta["sizes"][i];
        if (dim.is_object() && dim.contains("as_expr") && dim["as_expr"].is_object()) {
          const auto &expr = dim["as_expr"];
          if (expr.contains("expr_str") && expr["expr_str"].is_string()) {
            const std::string &expr_str = expr["expr_str"].get_ref<const std::string &>();
            size_t start = expr_str.find("Symbol('");
            if (start != std::string::npos) {
              start += 8;
              size_t end = expr_str.find("'", start);
              if (end != std::string::npos) {
                parsed.dynamic_dim_symbols[i] = expr_str.substr(start, end - start);
                TDLOGD("ParseTensorMeta: dim %zu is dynamic, symbol=%s", i, parsed.dynamic_dim_symbols[i].c_str());
              }
            }
          }
        }
      }
    }
  }
  if (meta.contains("requires_grad") && meta["requires_grad"].is_boolean()) {
    parsed.requires_grad = meta["requires_grad"].get<bool>();
  }
  if (meta.contains("device") && meta["device"].is_object()) {
    const auto &device = meta["device"];
    if (device.contains("type") && device["type"].is_string()) {
      parsed.device.type = device["type"].get<std::string>();
    }
    if (device.contains("index")) {
      parsed.device.index = ParseOptionalInt(device["index"]);
    }
  }
  if (meta.contains("strides")) {
    parsed.strides = ParseIntList(meta["strides"], result, "INVALID_TENSOR_STRIDES", context);
  }
  if (meta.contains("storage_offset")) {
    int64_t offset = 0;
    if (!TryGetIntField(meta["storage_offset"], offset)) {
      AddError(result, "INVALID_STORAGE_OFFSET", "storage_offset must be integer", context);
    } else {
      parsed.storage_offset = offset;
    }
  }
  if (meta.contains("layout") && meta["layout"].is_number_integer()) {
    parsed.layout = meta["layout"].get<int64_t>();
  }
  return parsed;
}

std::vector<ParsedValueRef> ParseValueRefs(const OrderedJson &array, ValidationResult &result,
                                           const std::string &code) {
  std::vector<ParsedValueRef> values;
  if (!array.is_array()) {
    AddError(result, code, "Expected JSON array");
    return values;
  }
  for (const auto &item : array) {
    if (!item.is_object()) {
      AddError(result, code, "Value reference must be object");
      continue;
    }
    std::string name;
    if (item.contains("as_tensor") && item["as_tensor"].is_object() &&
        item["as_tensor"].contains("name") && item["as_tensor"]["name"].is_string()) {
      name = item["as_tensor"]["name"].get<std::string>();
    } else if (item.contains("as_sym_int") && item["as_sym_int"].is_object() &&
               item["as_sym_int"].contains("as_name") && item["as_sym_int"]["as_name"].is_string()) {
      name = item["as_sym_int"]["as_name"].get<std::string>();
    } else if (item.contains("as_sym_float") && item["as_sym_float"].is_object() &&
               item["as_sym_float"].contains("as_name") && item["as_sym_float"]["as_name"].is_string()) {
      name = item["as_sym_float"]["as_name"].get<std::string>();
    } else if (item.contains("as_sym_bool") && item["as_sym_bool"].is_object() &&
               item["as_sym_bool"].contains("as_name") && item["as_sym_bool"]["as_name"].is_string()) {
      name = item["as_sym_bool"]["as_name"].get<std::string>();
    } else if (item.contains("as_int") && item["as_int"].is_number_integer()) {
      // For as_int, we don't have a name, use a placeholder
      name = "__int_" + std::to_string(values.size());
    } else if (item.contains("as_float") && item["as_float"].is_number()) {
      name = "__float_" + std::to_string(values.size());
    } else if (item.contains("as_bool") && item["as_bool"].is_boolean()) {
      name = "__bool_" + std::to_string(values.size());
    } else if (item.contains("as_string") && item["as_string"].is_string()) {
      name = "__string_" + std::to_string(values.size());
    } else {
      AddError(result, code, "Value reference must contain as_tensor.name or other supported type");
      continue;
    }
    values.push_back({name});
  }
  return values;
}

PlaceholderKind ParseInputKind(const OrderedJson &spec, std::string &placeholder_name, std::string &logical_name,
                               bool &persistent, NodeInputKind &input_kind, ValidationResult &result) {
  placeholder_name.clear();
  logical_name.clear();
  persistent = false;
  input_kind = NodeInputKind::Tensor;

  auto parse_named_arg = [&](const OrderedJson &obj, const char *arg_key) -> std::string {
    if (!obj.contains(arg_key) || !obj[arg_key].is_object()) {
      AddError(result, "INVALID_INPUT_SPEC", "Input spec missing arg object");
      return "";
    }
    const auto &arg = obj[arg_key];
    if (arg.contains("name") && arg["name"].is_string()) {
      return arg["name"].get<std::string>();
    }
    if (arg.contains("as_tensor") && arg["as_tensor"].is_object() && arg["as_tensor"].contains("name") &&
        arg["as_tensor"]["name"].is_string()) {
      return arg["as_tensor"]["name"].get<std::string>();
    }
    // Handle as_sym_int/as_sym_float/as_sym_bool with as_name
    static const std::vector<const char *> kSymFields = {"as_sym_int", "as_sym_float", "as_sym_bool"};
    for (const auto &field : kSymFields) {
      if (arg.contains(field) && arg[field].is_object() && arg[field].contains("as_name") &&
          arg[field]["as_name"].is_string()) {
        return arg[field]["as_name"].get<std::string>();
      }
    }
    AddError(result, "INVALID_INPUT_SPEC", "Input spec arg missing name");
    return "";
  };

  if (spec.contains("parameter") && spec["parameter"].is_object()) {
    const auto &parameter = spec["parameter"];
    placeholder_name = parse_named_arg(parameter, "arg");
    if (parameter.contains("parameter_name") && parameter["parameter_name"].is_string()) {
      logical_name = parameter["parameter_name"].get<std::string>();
    }
    return PlaceholderKind::Parameter;
  }
  if (spec.contains("buffer") && spec["buffer"].is_object()) {
    const auto &buffer = spec["buffer"];
    placeholder_name = parse_named_arg(buffer, "arg");
    if (buffer.contains("buffer_name") && buffer["buffer_name"].is_string()) {
      logical_name = buffer["buffer_name"].get<std::string>();
    }
    if (buffer.contains("persistent") && buffer["persistent"].is_boolean()) {
      persistent = buffer["persistent"].get<bool>();
    }
    return PlaceholderKind::Buffer;
  }
  if (spec.contains("tensor_constant") && spec["tensor_constant"].is_object()) {
    const auto &tensor_constant = spec["tensor_constant"];
    placeholder_name = parse_named_arg(tensor_constant, "arg");
    if (tensor_constant.contains("tensor_constant_name") && tensor_constant["tensor_constant_name"].is_string()) {
      logical_name = tensor_constant["tensor_constant_name"].get<std::string>();
    }
    return PlaceholderKind::TensorConstant;
  }
  if (spec.contains("user_input") && spec["user_input"].is_object()) {
    const auto &user_input = spec["user_input"];
    placeholder_name = parse_named_arg(user_input, "arg");
    logical_name = placeholder_name;
    
    // Identify value type from arg
    if (user_input.contains("arg") && user_input["arg"].is_object()) {
      const auto &arg = user_input["arg"];
      if (arg.contains("as_tensor")) {
        input_kind = NodeInputKind::Tensor;
      } else if (arg.contains("as_sym_int")) {
        input_kind = NodeInputKind::SymInt;
      } else if (arg.contains("as_sym_float")) {
        input_kind = NodeInputKind::SymFloat;
      } else if (arg.contains("as_sym_bool")) {
        input_kind = NodeInputKind::SymBool;
      } else if (arg.contains("as_int")) {
        input_kind = NodeInputKind::ScalarInt;
      } else if (arg.contains("as_float")) {
        input_kind = NodeInputKind::ScalarFloat;
      } else if (arg.contains("as_bool")) {
        input_kind = NodeInputKind::ScalarBool;
      } else if (arg.contains("as_string")) {
        input_kind = NodeInputKind::ScalarString;
      }
    }
    
    return PlaceholderKind::UserInput;
  }
  if (spec.contains("constant_input") && spec["constant_input"].is_object()) {
    const auto &constant_input = spec["constant_input"];
    if (constant_input.contains("name") && constant_input["name"].is_string()) {
      placeholder_name = constant_input["name"].get<std::string>();
    }
    logical_name = placeholder_name;
    
    // Identify value type from value
    if (constant_input.contains("value") && constant_input["value"].is_object()) {
      const auto &value = constant_input["value"];
      if (value.contains("as_int")) {
        input_kind = NodeInputKind::ScalarInt;
      } else if (value.contains("as_float")) {
        input_kind = NodeInputKind::ScalarFloat;
      } else if (value.contains("as_bool")) {
        input_kind = NodeInputKind::ScalarBool;
      } else if (value.contains("as_string")) {
        input_kind = NodeInputKind::ScalarString;
      }
    }
    
    return PlaceholderKind::ConstantInput;
  }
  AddError(result, "UNKNOWN_INPUT_KIND", "Unsupported input spec kind");
  return PlaceholderKind::Unknown;
}

ParsedAscendIrOpDesc ParseOpDesc(const OrderedJson &desc, ValidationResult &result, const std::string &context) {
  ParsedAscendIrOpDesc parsed;
  parsed.raw_json = desc;
  if (!desc.is_object()) {
    AddError(result, "INVALID_ASCEND_IR_DESC", "Desc must be object", context);
    return parsed;
  }
  if (desc.contains("name") && desc["name"].is_string()) {
    parsed.name = desc["name"].get<std::string>();
  }
  if (desc.contains("dtype") && desc["dtype"].is_string()) {
    parsed.dtype = desc["dtype"].get<std::string>();
  }
  if (desc.contains("layout") && desc["layout"].is_string()) {
    parsed.layout = desc["layout"].get<std::string>();
  }
  if (desc.contains("deviceType") && desc["deviceType"].is_string()) {
    parsed.device_type = desc["deviceType"].get<std::string>();
  }
  if (desc.contains("shape") && desc["shape"].is_object() && desc["shape"].contains("dim")) {
    const auto &dim = desc["shape"]["dim"];
    if (dim.is_array()) {
      for (const auto &item : dim) {
        if (item.is_string()) {
          parsed.shape.push_back(std::stoll(item.get<std::string>()));
        } else if (item.is_number_integer()) {
          parsed.shape.push_back(item.get<int64_t>());
        }
      }
    }
  }
  if (desc.contains("attr")) {
    parsed.attr = desc["attr"];
  }
  return parsed;
}

std::optional<ParsedAscendIr> ParseAscendIrImpl(const OrderedJson &ascend_ir, ValidationResult &result,
                                                const std::string &context) {
  if (ascend_ir.is_null()) {
    return std::nullopt;
  }
  if (!ascend_ir.is_object()) {
    TDLOGE("ParseAscendIr failed: ascend_ir must be object, context=%s", context.c_str());
    AddError(result, "INVALID_ASCEND_IR", "ascend_ir must be object", context);
    return std::nullopt;
  }

  ParsedAscendIr parsed;
  parsed.raw_json = ascend_ir;
  if (ascend_ir.contains("source") && ascend_ir["source"].is_string()) {
    parsed.source = ascend_ir["source"].get<std::string>();
    TDLOGD("ParseAscendIr: source=%s, context=%s", parsed.source.c_str(), context.c_str());
  }
  if (!ascend_ir.contains("ops") || !ascend_ir["ops"].is_array()) {
    TDLOGE("ParseAscendIr failed: missing ops array, context=%s", context.c_str());
    AddError(result, "MISSING_ASCEND_IR_OPS", "ascend_ir missing ops array", context);
    return std::nullopt;
  }
  TDLOGD("ParseAscendIr: ops_count=%zu, context=%s", ascend_ir["ops"].size(), context.c_str());
  for (const auto &op : ascend_ir["ops"]) {
    if (!op.is_object()) {
      TDLOGE("ParseAscendIr failed: op must be object, context=%s", context.c_str());
      AddError(result, "INVALID_ASCEND_IR_OP", "ascend_ir op must be object", context);
      continue;
    }
    ParsedAscendIrOp parsed_op;
    parsed_op.raw_json = op;
    if (op.contains("name") && op["name"].is_string()) {
      parsed_op.name = op["name"].get<std::string>();
    }
    if (op.contains("type") && op["type"].is_string()) {
      parsed_op.type = op["type"].get<std::string>();
    }
    const bool has_buildable_ir_fields = op.contains("name") || op.contains("type") || op.contains("input") ||
                                         op.contains("inputDesc") || op.contains("outputDesc");
    if (has_buildable_ir_fields && (parsed_op.name.empty() || parsed_op.type.empty())) {
      TDLOGE("ParseAscendIr failed: op missing name/type, context=%s", context.c_str());
      AddError(result, "INVALID_ASCEND_IR_OP", "ascend_ir op missing name/type", context);
    }
    if (op.contains("input") && op["input"].is_array()) {
      for (const auto &input : op["input"]) {
        if (input.is_string()) {
          parsed_op.inputs.push_back(input.get<std::string>());
        }
      }
    }
    if (op.contains("attr")) {
      parsed_op.attrs = op["attr"];
    }
    if (op.contains("inputDesc") && op["inputDesc"].is_array()) {
      for (const auto &desc : op["inputDesc"]) {
        parsed_op.input_descs.push_back(ParseOpDesc(desc, result, context));
      }
    }
    if (op.contains("outputDesc") && op["outputDesc"].is_array()) {
      for (const auto &desc : op["outputDesc"]) {
        parsed_op.output_descs.push_back(ParseOpDesc(desc, result, context));
      }
    }
    parsed.ops.push_back(std::move(parsed_op));
  }
  if (ascend_ir.contains("mapping") && ascend_ir["mapping"].is_object()) {
    const auto &mapping = ascend_ir["mapping"];
    if (mapping.contains("inputs") && mapping["inputs"].is_object()) {
      for (const auto &[fx_tensor_name, ge_tensor_name] : mapping["inputs"].items()) {
        if (ge_tensor_name.is_string()) {
          parsed.input_mappings.emplace(fx_tensor_name, ge_tensor_name.get<std::string>());
        }
      }
    }
    if (mapping.contains("outputs") && mapping["outputs"].is_object()) {
      for (const auto &[fx_tensor_name, ge_tensor_name] : mapping["outputs"].items()) {
        if (ge_tensor_name.is_string()) {
          parsed.output_mappings.emplace(fx_tensor_name, ge_tensor_name.get<std::string>());
        }
      }
    } else if (mapping.contains("outputs") && mapping["outputs"].is_array()) {
      size_t index = 0;
      for (const auto &output : mapping["outputs"]) {
        if (output.is_string()) {
          parsed.output_mappings.emplace(std::to_string(index++), output.get<std::string>());
        }
      }
    }
  }
  return parsed;
}

bool IsFunctionalizedTarget(const std::string &target) {
  return target == "torch.ops.higher_order.auto_functionalized_v2";
}

bool IsLegacyFunctionalizedTarget(const std::string &target) {
  return target == "torch.ops.higher_order.auto_functionalized";
}

std::string StripTorchOpsPrefix(const std::string &value) {
  constexpr char kPrefix[] = "torch.ops.";
  return value.rfind(kPrefix, 0) == 0 ? value.substr(sizeof(kPrefix) - 1) : value;
}

std::string CollapseUnderscores(const std::string &value) {
  std::string normalized;
  normalized.reserve(value.size());
  bool last_was_underscore = false;
  for (const char ch : value) {
    if (ch == '_') {
      if (!last_was_underscore) {
        normalized.push_back(ch);
      }
      last_was_underscore = true;
      continue;
    }
    normalized.push_back(ch);
    last_was_underscore = false;
  }
  while (!normalized.empty() && normalized.front() == '_') {
    normalized.erase(normalized.begin());
  }
  while (!normalized.empty() && normalized.back() == '_') {
    normalized.pop_back();
  }
  return normalized;
}

std::string SanitizeNodeStem(const std::string &value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (const unsigned char ch : value) {
    if (std::isalnum(ch)) {
      sanitized.push_back(static_cast<char>(ch));
    } else {
      sanitized.push_back('_');
    }
  }
  sanitized = CollapseUnderscores(sanitized);
  if (sanitized.empty()) {
    return "unknown";
  }
  constexpr size_t kMaxStemLength = 64;
  if (sanitized.size() > kMaxStemLength) {
    sanitized.resize(kMaxStemLength);
    sanitized = CollapseUnderscores(sanitized);
  }
  return sanitized.empty() ? "unknown" : sanitized;
}

std::vector<std::string> InputTensorNames(const ParsedNodeInput &input);

std::vector<std::string> OutputTensorNames(const ParsedNodeOutput &output);

std::string SelectNodeStem(const ParsedNode &parsed) {
  if (!parsed.effective_schema_target.empty()) {
    return SanitizeNodeStem(StripTorchOpsPrefix(parsed.effective_schema_target));
  }
  if (!parsed.target.empty()) {
    return SanitizeNodeStem(StripTorchOpsPrefix(parsed.target));
  }
  if (!parsed.outputs.empty()) {
    const std::vector<std::string> output_names = OutputTensorNames(parsed.outputs.front());
    if (!output_names.empty()) {
      return SanitizeNodeStem(output_names.front());
    }
  }
  return "unknown";
}

std::string BuildNodeName(const ParsedNode &parsed) {
  return "node_" + std::to_string(parsed.json_node_index) + "_" + SelectNodeStem(parsed);
}

std::vector<std::string> ParseTensorListValue(const OrderedJson &value) {
  std::vector<std::string> names;
  if (!value.is_object() || !value.contains("as_tensors") || !value["as_tensors"].is_array()) {
    return names;
  }
  for (const auto &tensor : value["as_tensors"]) {
    if (!tensor.is_object() || !tensor.contains("name") || !tensor["name"].is_string()) {
      return {};
    }
    names.push_back(tensor["name"].get<std::string>());
  }
  return names;
}

std::vector<std::string> InputTensorNames(const ParsedNodeInput &input) {
  if (input.input_kind == NodeInputKind::Tensor) {
    return {input.tensor_name};
  }
  if (input.input_kind == NodeInputKind::TensorList) {
    return input.tensor_names;
  }
  return {};
}

std::vector<std::string> OutputTensorNames(const ParsedNodeOutput &output) {
  if (output.is_tensor) {
    return {output.tensor_name};
  }
  if (output.is_tensor_list) {
    return output.tensor_names;
  }
  return {};
}

std::optional<int64_t> ParseIntArg(const OrderedJson &arg) {
  if (!arg.is_object() || !arg.contains("as_int") || !arg["as_int"].is_number_integer()) {
    return std::nullopt;
  }
  return arg["as_int"].get<int64_t>();
}

std::vector<ParsedNodeInput> RebuildFunctionalizedInputs(const ParsedNode &parsed, ValidationResult &result) {
  std::vector<std::string> all_bases;
  std::vector<int64_t> base_indices;
  std::vector<ParsedNodeInput> tensor_inputs;
  tensor_inputs.reserve(parsed.inputs.size());
  for (size_t index = 1; index < parsed.inputs.size(); ++index) {
    const auto &input = parsed.inputs[index];
    if (input.name == "_all_bases") {
      all_bases = ParseTensorListValue(input.raw_arg);
      continue;
    }
    if (input.name.find("_input_base_index") == 0) {
      const std::optional<int64_t> parsed_index = ParseIntArg(input.raw_arg);
      if (!parsed_index.has_value()) {
        AddError(result, "INVALID_FUNCTIONALIZED_INPUT", "_input_base_index must be integer", parsed.target);
        return {};
      }
      base_indices.push_back(parsed_index.value());
      continue;
    }
    if (input.input_kind == NodeInputKind::Tensor) {
      tensor_inputs.push_back(input);
    }
  }
  std::vector<ParsedNodeInput> normalized_inputs;
  normalized_inputs.reserve(base_indices.size() + tensor_inputs.size());
  for (size_t index = 0; index < base_indices.size(); ++index) {
    const int64_t base_index = base_indices[index];
    if (base_index < 0 || static_cast<size_t>(base_index) >= all_bases.size()) {
      AddError(result, "INVALID_FUNCTIONALIZED_INPUT", "_input_base_index is out of range", parsed.target);
      return {};
    }
    ParsedNodeInput base_input;
    base_input.name = index == 0 ? "self" : "self_" + std::to_string(index);
    base_input.input_kind = NodeInputKind::Tensor;
    base_input.tensor_name = all_bases[static_cast<size_t>(base_index)];
    normalized_inputs.push_back(std::move(base_input));
  }
  normalized_inputs.insert(normalized_inputs.end(), tensor_inputs.begin(), tensor_inputs.end());
  return normalized_inputs;
}

void PopulateEffectiveSchemaInfo(ParsedNode &parsed, ValidationResult &result) {
  parsed.effective_schema_target = parsed.target;
  parsed.schema_metadata_input_count = 0;
  parsed.functionalize.is_functionalized = IsFunctionalizedTarget(parsed.target);
  parsed.functionalize.wrapped_target.clear();
  // todo(@AGENTS): 仅支持 auto_functionalized_v2；若后续确有真实需求，再单独补 v1 兼容。
  if (IsLegacyFunctionalizedTarget(parsed.target)) {
    AddError(result, "UNSUPPORTED_FUNCTIONALIZED_V1",
             "auto_functionalized v1 is not supported; please export auto_functionalized_v2", parsed.target);
    return;
  }
  if (!IsFunctionalizedTarget(parsed.target) || parsed.inputs.empty()) {
    parsed.functionalize.runtime_input_tensor_names = parsed.fx_input_tensor_names;
    parsed.functionalize.runtime_output_tensor_names = parsed.fx_output_tensor_names;
    return;
  }
  const ParsedNodeInput &first_input = parsed.inputs.front();
  if (first_input.input_kind != NodeInputKind::Operator || first_input.operator_name.empty()) {
    AddError(result, "INVALID_FUNCTIONALIZED_INPUT", "functionalized node missing wrapped operator", parsed.target);
    return;
  }
  parsed.effective_schema_target = first_input.operator_name;
  parsed.functionalize.wrapped_target = first_input.operator_name;
  const std::vector<ParsedNodeInput> normalized_inputs = RebuildFunctionalizedInputs(parsed, result);
  if (!result.ok) {
    return;
  }
  if (!normalized_inputs.empty()) {
    parsed.inputs = normalized_inputs;
  }
  parsed.functionalize.metadata_input_count = parsed.schema_metadata_input_count = 1;
  parsed.functionalize.runtime_input_tensor_names.clear();
  for (const auto &input : parsed.inputs) {
    if (input.input_kind == NodeInputKind::Tensor && !input.tensor_name.empty()) {
      parsed.functionalize.runtime_input_tensor_names.push_back(input.tensor_name);
    }
  }
  parsed.functionalize.runtime_output_tensor_names = parsed.fx_output_tensor_names;
}

std::string BuildNodeJsonContext(const ParsedNode &node) {
  return "graph_module.graph.nodes[" + std::to_string(node.json_node_index) + "]";
}

bool IsFallbackNode(const ParsedNode &node) {
  return node.ascend_ir.has_value() && node.ascend_ir->source != "converter" && !node.effective_schema_target.empty();
}

std::string ToCamelCase(const std::string &value) {
  std::string result;
  bool capitalize = true;
  for (const char c : value) {
    if (c == '_' || c == '.') {
      capitalize = true;
      continue;
    }
    if (capitalize) {
      result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      capitalize = false;
    } else {
      result += c;
    }
  }
  return result;
}

std::optional<std::string> BuildFallbackOpType(const ParsedNode &node, ValidationResult &result) {
  if (!IsFallbackNode(node)) {
    return std::nullopt;
  }

  std::string op_str = node.effective_schema_target;
  constexpr char kPrefix[] = "torch.ops.";
  if (op_str.rfind(kPrefix, 0) == 0) {
    op_str = op_str.substr(sizeof(kPrefix) - 1);
  }

  const size_t overload_separator = op_str.find_last_of('.');
  if (overload_separator == std::string::npos) {
    AddError(result, "INVALID_ATEN_TARGET", "Target missing overload separator", BuildNodeJsonContext(node));
    return std::nullopt;
  }

  const std::string overload = op_str.substr(overload_separator + 1);
  const std::string ns_and_op = op_str.substr(0, overload_separator);
  const size_t namespace_separator = ns_and_op.find_last_of('.');
  if (namespace_separator == std::string::npos) {
    AddError(result, "INVALID_ATEN_TARGET", "Target missing namespace separator", BuildNodeJsonContext(node));
    return std::nullopt;
  }

  const std::string op_name = ns_and_op.substr(namespace_separator + 1);
  std::string op_type = ToCamelCase(op_name);
  if (overload == "out") {
    op_type += "Out";
  } else if (!op_name.empty() && op_name.back() == '_') {
    op_type += "Inplace";
  }
  if (op_type.empty()) {
    AddError(result, "INVALID_ATEN_TARGET", "Target op_name is empty", BuildNodeJsonContext(node));
    return std::nullopt;
  }
  return op_type;
}

std::string BuildUniqueFallbackNodeName(const std::string &op_type, std::unordered_map<std::string, size_t> &counters) {
  const size_t next_index = counters[op_type]++;
  if (next_index == 0) {
    return op_type;
  }
  return op_type + "_" + std::to_string(next_index);
}

}  // namespace

EpairModelParser EpairModelParser::Parse(const std::string &model_json, ValidationResult &result) {
  EpairModelParser parser;
  const OrderedJson root = ParseRootJson(model_json, result);
  if (!result.ok) {
    return parser;
  }
  parser.model_ = ParseModelJson(root, result);
  for (size_t node_index = 0; node_index < parser.model_.nodes.size(); ++node_index) {
    const ParsedNode &node = parser.model_.nodes[node_index];
    parser.node_name_to_index_[node.name] = node_index;
    for (const auto &output : node.outputs) {
      for (const auto &tensor_name : OutputTensorNames(output)) {
        parser.output_name_to_index_[tensor_name] = node_index;
      }
    }
  }
  for (size_t input_index = 0; input_index < parser.model_.input_specs.size(); ++input_index) {
    const ParsedInputSpec &spec = parser.model_.input_specs[input_index];
    if (!spec.placeholder_name.empty()) {
      parser.input_spec_to_index_[spec.placeholder_name] = input_index;
    }
  }
  for (size_t output_index = 0; output_index < parser.model_.output_specs.size(); ++output_index) {
    const ParsedOutputSpec &spec = parser.model_.output_specs[output_index];
    if (!spec.name.empty()) {
      parser.output_spec_to_index_[spec.name] = output_index;
    }
  }

  return parser;
}

OrderedJson EpairModelParser::ParseRootJson(const std::string &model_json, ValidationResult &result) {
  TDLOGI("ParseRootJson enter: json_size=%zu", model_json.size());
  try {
    OrderedJson root = OrderedJson::parse(model_json);
    TDLOGI("ParseRootJson exit: success");
    return root;
  } catch (const OrderedJson::parse_error &error) {
    TDLOGE("ParseRootJson failed: %s", error.what());
    AddError(result, "MODEL_JSON_PARSE_ERROR", error.what());
    return OrderedJson::object();
  }
}

ParsedModel EpairModelParser::ParseModelJson(const OrderedJson &root, ValidationResult &result) {
  TDLOGI("ParseModelJson enter");
  ParsedModel model;
  if (root.contains("schema_version") && root["schema_version"].is_object()) {
    const auto &schema = root["schema_version"];
    if (schema.contains("major") && schema["major"].is_number_integer()) {
      model.schema_version.major = schema["major"].get<int64_t>();
    }
    if (schema.contains("minor") && schema["minor"].is_number_integer()) {
      model.schema_version.minor = schema["minor"].get<int64_t>();
    }
    TDLOGD("ParseModelJson: schema_version major=%ld, minor=%ld",
           static_cast<long>(model.schema_version.major),
           static_cast<long>(model.schema_version.minor));
  }
  if (root.contains("opset_version")) {
    model.opset_version = root["opset_version"];
  }
  if (root.contains("torch_version") && root["torch_version"].is_string()) {
    model.torch_version = root["torch_version"].get<std::string>();
    TDLOGD("ParseModelJson: torch_version=%s", model.torch_version.c_str());
  }
  if (root.contains("range_constraints") && root["range_constraints"].is_object()) {
    for (const auto &[name, constraint] : root["range_constraints"].items()) {
      if (constraint.is_object() && constraint.contains("min_val") && constraint.contains("max_val") &&
          constraint["min_val"].is_number_integer() && constraint["max_val"].is_number_integer()) {
        int64_t min_val = constraint["min_val"].get<int64_t>();
        int64_t max_val = constraint["max_val"].get<int64_t>();
        model.range_constraints[name] = {min_val, max_val};
        TDLOGD("ParseModelJson: range_constraint %s=[%ld, %ld]", name.c_str(),
               static_cast<long>(min_val), static_cast<long>(max_val));
      }
    }
    TDLOGD("ParseModelJson: range_constraints=%zu", model.range_constraints.size());
  }
  if (!root.contains("graph_module") || !root["graph_module"].is_object()) {
    TDLOGE("ParseModelJson failed: graph_module missing or invalid");
    AddError(result, "MISSING_GRAPH_MODULE", "graph_module missing or invalid");
    return model;
  }
  const auto &graph_module = root["graph_module"];
  if (!graph_module.contains("graph") || !graph_module["graph"].is_object()) {
    TDLOGE("ParseModelJson failed: graph missing or invalid");
    AddError(result, "MISSING_GRAPH", "graph_module.graph missing or invalid");
    return model;
  }
  TDLOGI("ParseModelJson step: parsing graph");
  const auto &graph = graph_module["graph"];
  if (graph.contains("inputs")) {
    model.graph_inputs = ParseValueRefs(graph["inputs"], result, "INVALID_GRAPH_INPUTS");
    TDLOGD("ParseModelJson: graph_inputs=%zu", model.graph_inputs.size());
  }
  if (graph.contains("outputs")) {
    model.graph_outputs = ParseValueRefs(graph["outputs"], result, "INVALID_GRAPH_OUTPUTS");
    TDLOGD("ParseModelJson: graph_outputs=%zu", model.graph_outputs.size());
  }
  if (graph.contains("tensor_values") && graph["tensor_values"].is_object()) {
    for (const auto &[name, meta] : graph["tensor_values"].items()) {
      model.tensor_values[name] = ParseTensorMeta(meta, result, name);
    }
    TDLOGD("ParseModelJson: tensor_values=%zu", model.tensor_values.size());
  }
  
  // Parse sym_int_values
  if (graph.contains("sym_int_values") && graph["sym_int_values"].is_object()) {
    for (const auto &[name, value] : graph["sym_int_values"].items()) {
      if (value.is_number_integer()) {
        model.sym_int_values[name] = value.get<int64_t>();
      } else if (value.is_object() && value.contains("as_int")) {
        model.sym_int_values[name] = value["as_int"].get<int64_t>();
      } else if (value.is_object() && value.contains("as_expr") && value["as_expr"].is_object()) {
        // For symbolic expressions, use the hint value if available
        const auto &expr = value["as_expr"];
        if (expr.contains("hint") && expr["hint"].is_object() && expr["hint"].contains("as_int")) {
          model.sym_int_values[name] = expr["hint"]["as_int"].get<int64_t>();
        }
      }
    }
    TDLOGD("ParseModelJson: sym_int_values=%zu", model.sym_int_values.size());
  }
  
  // Parse sym_float_values
  if (graph.contains("sym_float_values") && graph["sym_float_values"].is_object()) {
    for (const auto &[name, value] : graph["sym_float_values"].items()) {
      if (value.is_number()) {
        model.sym_float_values[name] = value.get<double>();
      } else if (value.is_object() && value.contains("as_float")) {
        model.sym_float_values[name] = value["as_float"].get<double>();
      } else if (value.is_object() && value.contains("as_expr") && value["as_expr"].is_object()) {
        const auto &expr = value["as_expr"];
        if (expr.contains("hint") && expr["hint"].is_object() && expr["hint"].contains("as_float")) {
          model.sym_float_values[name] = expr["hint"]["as_float"].get<double>();
        }
      }
    }
    TDLOGD("ParseModelJson: sym_float_values=%zu", model.sym_float_values.size());
  }
  
  // Parse sym_bool_values
  if (graph.contains("sym_bool_values") && graph["sym_bool_values"].is_object()) {
    for (const auto &[name, value] : graph["sym_bool_values"].items()) {
      if (value.is_boolean()) {
        model.sym_bool_values[name] = value.get<bool>();
      } else if (value.is_object() && value.contains("as_bool")) {
        model.sym_bool_values[name] = value["as_bool"].get<bool>();
      } else if (value.is_object() && value.contains("as_expr") && value["as_expr"].is_object()) {
        const auto &expr = value["as_expr"];
        if (expr.contains("hint") && expr["hint"].is_object() && expr["hint"].contains("as_bool")) {
          model.sym_bool_values[name] = expr["hint"]["as_bool"].get<bool>();
        }
      }
    }
    TDLOGD("ParseModelJson: sym_bool_values=%zu", model.sym_bool_values.size());
  }
  
  // Parse int_values
  if (graph.contains("int_values") && graph["int_values"].is_object()) {
    for (const auto &[name, value] : graph["int_values"].items()) {
      if (value.is_number_integer()) {
        model.int_values[name] = value.get<int64_t>();
      } else if (value.is_object() && value.contains("as_int")) {
        model.int_values[name] = value["as_int"].get<int64_t>();
      }
    }
    TDLOGD("ParseModelJson: int_values=%zu", model.int_values.size());
  }
  
  // Parse float_values
  if (graph.contains("float_values") && graph["float_values"].is_object()) {
    for (const auto &[name, value] : graph["float_values"].items()) {
      if (value.is_number()) {
        model.float_values[name] = value.get<double>();
      } else if (value.is_object() && value.contains("as_float")) {
        model.float_values[name] = value["as_float"].get<double>();
      }
    }
    TDLOGD("ParseModelJson: float_values=%zu", model.float_values.size());
  }
  
  // Parse bool_values
  if (graph.contains("bool_values") && graph["bool_values"].is_object()) {
    for (const auto &[name, value] : graph["bool_values"].items()) {
      if (value.is_boolean()) {
        model.bool_values[name] = value.get<bool>();
      } else if (value.is_object() && value.contains("as_bool")) {
        model.bool_values[name] = value["as_bool"].get<bool>();
      }
    }
    TDLOGD("ParseModelJson: bool_values=%zu", model.bool_values.size());
  }
  
  // Parse string_values
  if (graph.contains("string_values") && graph["string_values"].is_object()) {
    for (const auto &[name, value] : graph["string_values"].items()) {
      if (value.is_string()) {
        model.string_values[name] = value.get<std::string>();
      } else if (value.is_object() && value.contains("as_string")) {
        model.string_values[name] = value["as_string"].get<std::string>();
      }
    }
    TDLOGD("ParseModelJson: string_values=%zu", model.string_values.size());
  }
  if (graph.contains("nodes") && graph["nodes"].is_array()) {
    TDLOGI("ParseModelJson step: parsing nodes, count=%zu", graph["nodes"].size());
    std::unordered_map<std::string, size_t> fallback_node_name_counters;
    size_t error_count_before = result.issues.size();
    for (const auto &node : graph["nodes"]) {
      if (!node.is_object()) {
        TDLOGE("ParseModelJson failed: node[%zu] is not object", model.nodes.size());
        AddError(result, "INVALID_NODE", "Node must be object");
        continue;
      }
      model.nodes.push_back(ParseNodeJson(node, model.nodes.size(), fallback_node_name_counters, result));
    }
    size_t new_errors = result.issues.size() - error_count_before;
    if (new_errors > 0) {
      TDLOGW("ParseModelJson: %zu node parse errors", new_errors);
    }
    TDLOGI("ParseModelJson: nodes parsed=%zu", model.nodes.size());
  }
  if (graph_module.contains("signature") && graph_module["signature"].is_object()) {
    TDLOGD("ParseModelJson step: parsing signature");
    const auto &signature = graph_module["signature"];
    if (signature.contains("input_specs") && signature["input_specs"].is_array()) {
      for (const auto &spec : signature["input_specs"]) {
        ParsedInputSpec parsed;
        parsed.raw_json = spec;
        parsed.kind = ParseInputKind(spec, parsed.placeholder_name, parsed.logical_name, parsed.persistent, parsed.input_kind, result);
        model.input_specs.push_back(std::move(parsed));
      }
      TDLOGD("ParseModelJson: input_specs=%zu", model.input_specs.size());
    }
    if (signature.contains("output_specs") && signature["output_specs"].is_array()) {
      for (const auto &spec : signature["output_specs"]) {
        ParsedOutputSpec parsed;
        parsed.raw_json = spec;
        if (spec.contains("user_output") && spec["user_output"].is_object() && spec["user_output"].contains("arg") &&
            spec["user_output"]["arg"].is_object() && spec["user_output"]["arg"].contains("as_tensor") &&
            spec["user_output"]["arg"]["as_tensor"].is_object() &&
            spec["user_output"]["arg"]["as_tensor"].contains("name") &&
            spec["user_output"]["arg"]["as_tensor"]["name"].is_string()) {
          parsed.name = spec["user_output"]["arg"]["as_tensor"]["name"].get<std::string>();
        }
        model.output_specs.push_back(std::move(parsed));
      }
      TDLOGD("ParseModelJson: output_specs=%zu", model.output_specs.size());
    }
  }
  if (!result.ok) {
    TDLOGE("ParseModelJson exit: failed, issues=%zu", result.issues.size());
  } else {
    TDLOGI("ParseModelJson exit: success");
  }
  return model;
}

namespace {

bool ParseSymOrScalarInputArg(const OrderedJson &arg, ParsedNodeInput &parsed_input, ParsedNode &parsed) {
  if (!arg.is_object()) return false;
  // Helper to attempt extracting as_name from a single sym argument object
  auto try_sym_name = [](const OrderedJson &obj, std::string &out_name) -> bool {
    if (!obj.is_object()) return false;
    if (obj.contains("as_name") && obj["as_name"].is_string()) {
      out_name = obj["as_name"].get<std::string>();
      return true;
    }
    return false;
  };
  // Check single sym types with as_name
  static const std::vector<const char *> kSingleSymFields = {
      "as_sym_int", "as_sym_float", "as_sym_bool"};
  for (const auto &field : kSingleSymFields) {
    if (arg.contains(field) && arg[field].is_object()) {
      std::string name;
      if (try_sym_name(arg[field], name)) {
        parsed.fx_input_tensor_names.push_back(name);
        parsed_input.tensor_name = name;
        if (std::string(field) == "as_sym_int") {
          parsed_input.input_kind = NodeInputKind::SymInt;
        } else if (std::string(field) == "as_sym_float") {
          parsed_input.input_kind = NodeInputKind::SymFloat;
        } else if (std::string(field) == "as_sym_bool") {
          parsed_input.input_kind = NodeInputKind::SymBool;
        }
      }
      return true;
    }
  }
  // Check list sym types with as_name
  static const std::vector<const char *> kListSymFields = {
      "as_sym_ints", "as_sym_floats", "as_sym_bools"};
  for (const auto &field : kListSymFields) {
    if (arg.contains(field) && arg[field].is_array()) {
      for (const auto &item : arg[field]) {
        std::string name;
        if (try_sym_name(item, name)) {
          parsed.fx_input_tensor_names.push_back(name);
          parsed_input.tensor_names.push_back(name);
        }
      }
      if (std::string(field) == "as_sym_ints") {
        parsed_input.input_kind = NodeInputKind::SymListInt;
      } else if (std::string(field) == "as_sym_floats") {
        parsed_input.input_kind = NodeInputKind::SymListFloat;
      } else if (std::string(field) == "as_sym_bools") {
        parsed_input.input_kind = NodeInputKind::SymListBool;
      }
      return true;
    }
  }
  // Check scalar types (no edge, inline value)
  if (arg.contains("as_int")) {
    parsed_input.input_kind = NodeInputKind::ScalarInt;
    return true;
  }
  if (arg.contains("as_ints")) {
    parsed_input.input_kind = NodeInputKind::ScalarListInt;
    return true;
  }
  if (arg.contains("as_float")) {
    parsed_input.input_kind = NodeInputKind::ScalarFloat;
    return true;
  }
  if (arg.contains("as_floats")) {
    parsed_input.input_kind = NodeInputKind::ScalarListFloat;
    return true;
  }
  if (arg.contains("as_bool")) {
    parsed_input.input_kind = NodeInputKind::ScalarBool;
    return true;
  }
  if (arg.contains("as_bools")) {
    parsed_input.input_kind = NodeInputKind::ScalarListBool;
    return true;
  }
  if (arg.contains("as_string") || arg.contains("as_strings")) {
    parsed_input.input_kind = NodeInputKind::ScalarString;
    return true;
  }
  if (arg.contains("as_scalar_type")) {
    parsed_input.input_kind = NodeInputKind::Enum;
    return true;
  }
  // Other scalar types that don't need special handling
  static const std::vector<const char *> kOtherScalarFields = {
      "as_memory_format", "as_layout", "as_device", "as_complex",
      "as_optional_tensor", "as_optional_tensors"};
  for (const auto &field : kOtherScalarFields) {
    if (arg.contains(field)) return true;
  }
  // Check as_none
  if (arg.contains("as_none") && arg["as_none"].is_boolean() && arg["as_none"].get<bool>()) {
    parsed_input.input_kind = NodeInputKind::None;
    return true;
  }
  return false;
}

}

ParsedNode EpairModelParser::ParseNodeJson(const OrderedJson &node, size_t json_node_index,
                                           std::unordered_map<std::string, size_t> &fallback_name_counters,
                                           ValidationResult &result) {
  ParsedNode parsed;
  parsed.json_node_index = json_node_index;
  if (!node.is_object()) {
    TDLOGE("ParseNode failed: node[%zu] is not object", json_node_index);
    AddError(result, "INVALID_NODE", "Node must be object");
    return parsed;
  }
  if (node.contains("target") && node["target"].is_string()) {
    parsed.target = node["target"].get<std::string>();
  }
  TDLOGD("ParseNode: index=%zu, target=%s", json_node_index, parsed.target.c_str());
  if (node.contains("metadata")) {
    parsed.metadata = node["metadata"];
  }
  if (node.contains("inputs") && node["inputs"].is_array()) {
    for (const auto &input : node["inputs"]) {
      ParsedNodeInput parsed_input;
      if (input.contains("name") && input["name"].is_string()) {
        parsed_input.name = input["name"].get<std::string>();
      }
      if (input.contains("kind")) {
        if (input["kind"].is_number_integer()) {
          parsed_input.kind = input["kind"].get<int64_t>();
        } else if (input["kind"].is_string()) {
          parsed_input.kind = -1;
        }
      }
      if (input.contains("arg")) {
        parsed_input.raw_arg = input["arg"];
        if (input["arg"].is_object() && input["arg"].contains("as_tensor") && input["arg"]["as_tensor"].is_object() &&
            input["arg"]["as_tensor"].contains("name") && input["arg"]["as_tensor"]["name"].is_string()) {
          parsed_input.input_kind = NodeInputKind::Tensor;
          parsed_input.tensor_name = input["arg"]["as_tensor"]["name"].get<std::string>();
          parsed.fx_input_tensor_names.push_back(parsed_input.tensor_name);
        } else if (input["arg"].is_object() && input["arg"].contains("as_tensors") &&
                   input["arg"]["as_tensors"].is_array()) {
          parsed_input.input_kind = NodeInputKind::TensorList;
          parsed_input.tensor_names = ParseTensorListValue(input["arg"]);
          if (parsed_input.tensor_names.empty()) {
            TDLOGE("ParseNode failed: tensor-list input missing names, node=%zu, target=%s",
                   json_node_index, parsed.target.c_str());
            AddError(result, "INVALID_NODE_INPUT", "Node tensor-list input must contain as_tensors[].name",
                     parsed.target);
            continue;
          }
          parsed.fx_input_tensor_names.insert(parsed.fx_input_tensor_names.end(), parsed_input.tensor_names.begin(),
                                              parsed_input.tensor_names.end());
        } else if (input["arg"].is_object() && input["arg"].contains("as_operator") &&
                   input["arg"]["as_operator"].is_string()) {
          parsed_input.input_kind = NodeInputKind::Operator;
          parsed_input.operator_name = input["arg"]["as_operator"].get<std::string>();
        } else if (ParseSymOrScalarInputArg(input["arg"], parsed_input, parsed)) {
          // sym/scalar input handled
        }
      }
      parsed.inputs.push_back(std::move(parsed_input));
    }
  }
  if (node.contains("outputs") && node["outputs"].is_array()) {
    for (const auto &output : node["outputs"]) {
      if (output.is_object() && output.contains("as_none") && output["as_none"].is_boolean() &&
          output["as_none"].get<bool>()) {
        continue;
      }
      ParsedNodeOutput parsed_output;
      if (output.is_object() && output.contains("as_tensor") && output["as_tensor"].is_object() &&
          output["as_tensor"].contains("name") && output["as_tensor"]["name"].is_string()) {
        const std::string output_name = output["as_tensor"]["name"].get<std::string>();
        parsed_output.is_tensor = true;
        parsed_output.tensor_name = output_name;
        parsed.outputs.push_back(parsed_output);
        parsed.fx_output_tensor_names.push_back(output_name);
        if (parsed.name.empty()) {
          parsed.name = output_name;
        }
        continue;
      }
      if (output.is_object() && output.contains("as_tensors") && output["as_tensors"].is_array()) {
        parsed_output.is_tensor_list = true;
        parsed_output.tensor_names = ParseTensorListValue(output);
        if (parsed_output.tensor_names.empty()) {
          TDLOGE("ParseNode failed: tensor-list output missing names, node=%zu, target=%s",
                 json_node_index, parsed.target.c_str());
          AddError(result, "INVALID_NODE_OUTPUT", "Node tensor-list output must contain as_tensors[].name",
                   parsed.target);
          continue;
        }
        parsed.outputs.push_back(parsed_output);
        parsed.fx_output_tensor_names.insert(parsed.fx_output_tensor_names.end(), parsed_output.tensor_names.begin(),
                                             parsed_output.tensor_names.end());
        if (parsed.name.empty()) {
          parsed.name = parsed_output.tensor_names.front();
        }
        continue;
      }
      // Handle sym types with as_name (dataflow edges)
      if (output.is_object()) {
        bool handled = false;
        static const std::vector<std::pair<const char *, const char *>> kSymFields = {
            {"as_sym_int", "as_name"}, {"as_sym_float", "as_name"}, {"as_sym_bool", "as_name"}};
        for (const auto &[sym_field, name_field] : kSymFields) {
          if (output.contains(sym_field) && output[sym_field].is_object() &&
              output[sym_field].contains(name_field) && output[sym_field][name_field].is_string()) {
            const std::string sym_name = output[sym_field][name_field].get<std::string>();
            parsed_output.is_tensor = true;
            parsed_output.tensor_name = sym_name;
            parsed.outputs.push_back(parsed_output);
            parsed.fx_output_tensor_names.push_back(sym_name);
            if (parsed.name.empty()) {
              parsed.name = sym_name;
            }
            handled = true;
            break;
          }
        }
        if (handled) continue;
        // Handle list sym types with as_name (dataflow edges)
        static const std::vector<const char *> kListSymFields = {
            "as_sym_ints", "as_sym_floats", "as_sym_bools"};
        for (const auto &field : kListSymFields) {
          if (output.contains(field) && output[field].is_array()) {
            for (const auto &item : output[field]) {
              if (item.is_object() && item.contains("as_name") && item["as_name"].is_string()) {
                parsed.fx_output_tensor_names.push_back(item["as_name"].get<std::string>());
              }
            }
            handled = true;
            break;
          }
        }
        if (handled) continue;
        // Handle sym types with concrete values (no edge, skip)
        static const std::vector<const char *> kSymConcreteFields = {
            "as_sym_int", "as_sym_float", "as_sym_bool"};
        for (const auto &field : kSymConcreteFields) {
          if (output.contains(field) && output[field].is_object() &&
              !output[field].contains("as_name")) {
            handled = true;
            break;
          }
        }
        // Handle concrete scalar outputs (no edge, skip)
        static const std::vector<const char *> kScalarFields = {
            "as_int", "as_ints", "as_float", "as_floats", "as_bool", "as_bools",
            "as_string", "as_strings"};
        if (!handled) {
          for (const auto &field : kScalarFields) {
            if (output.contains(field)) {
              handled = true;
              break;
            }
          }
        }
        if (handled) continue;
      }
      TDLOGE("ParseNode failed: unsupported output format, node=%zu, target=%s", json_node_index, parsed.target.c_str());
      AddError(result, "INVALID_NODE_OUTPUT", "Node output currently only supports as_tensor or as_tensors",
               parsed.target);
    }
  }
  if (node.contains("ascend_ir")) {
    parsed.ascend_ir = ParseAscendIrJson(node["ascend_ir"], parsed, result);
    if (parsed.ascend_ir.has_value()) {
      TDLOGD("ParseNode: ascend_ir parsed, source=%s, ops=%zu", parsed.ascend_ir->source.c_str(),
             parsed.ascend_ir->ops.size());
    }
  }
  PopulateEffectiveSchemaInfo(parsed, result);
  parsed.source_info.source = parsed.ascend_ir.has_value() ? parsed.ascend_ir->source : std::string();
  parsed.source_info.is_converter = parsed.ascend_ir.has_value() && parsed.ascend_ir->source == "converter";
  parsed.source_info.is_fallback = IsFallbackNode(parsed);
  const std::optional<std::string> fallback_op_type = BuildFallbackOpType(parsed, result);
  if (fallback_op_type.has_value()) {
    parsed.name = BuildUniqueFallbackNodeName(fallback_op_type.value(), fallback_name_counters);
    TDLOGD("ParseNode: fallback node name=%s, type=%s", parsed.name.c_str(), fallback_op_type.value().c_str());
  } else {
    parsed.name = BuildNodeName(parsed);
  }
  if (parsed.functionalize.runtime_input_tensor_names.empty()) {
    parsed.functionalize.runtime_input_tensor_names = parsed.fx_input_tensor_names;
  }
  if (parsed.functionalize.runtime_output_tensor_names.empty()) {
    parsed.functionalize.runtime_output_tensor_names = parsed.fx_output_tensor_names;
  }
  if (parsed.ascend_ir.has_value()) {
    parsed.ascend_ir->boundary_outputs.clear();
    for (size_t index = 0; index < parsed.fx_output_tensor_names.size(); ++index) {
      const std::string &fx_output_name = parsed.fx_output_tensor_names[index];
      auto output_it = parsed.ascend_ir->output_mappings.find(fx_output_name);
      if (output_it == parsed.ascend_ir->output_mappings.end()) {
        output_it = parsed.ascend_ir->output_mappings.find(std::to_string(index));
      }
      if (output_it != parsed.ascend_ir->output_mappings.end()) {
        parsed.ascend_ir->boundary_outputs.push_back({fx_output_name, output_it->second});
      }
    }
    for (const auto &input : parsed.inputs) {
      const std::vector<std::string> tensor_names = InputTensorNames(input);
      if (tensor_names.empty()) {
        continue;
      }
      for (const auto &tensor_name : tensor_names) {
        ParsedBoundaryInput boundary_input;
        boundary_input.fx_tensor_name = tensor_name;
        boundary_input.schema_input_name = input.name;
        const auto input_it = parsed.ascend_ir->input_mappings.find(tensor_name);
        boundary_input.ge_tensor_name =
            input_it == parsed.ascend_ir->input_mappings.end() ? tensor_name : input_it->second;
        parsed.ascend_ir->boundary_inputs.push_back(std::move(boundary_input));
      }
    }
  }
  return parsed;
}

std::optional<ParsedAscendIr> EpairModelParser::ParseAscendIrJson(const OrderedJson &ascend_ir,
                                                                  const ParsedNode &parsed_node,
                                                                  ValidationResult &result) {
  return ParseAscendIrImpl(ascend_ir, result, BuildNodeJsonContext(parsed_node));
}

const ParsedModel &EpairModelParser::Model() const {
  return model_;
}

const SchemaVersion &EpairModelParser::SchemaVersionValue() const {
  return model_.schema_version;
}

const std::vector<ParsedValueRef> &EpairModelParser::GraphInputs() const {
  return model_.graph_inputs;
}

const std::vector<ParsedValueRef> &EpairModelParser::GraphOutputs() const {
  return model_.graph_outputs;
}

const std::vector<ParsedNode> &EpairModelParser::Nodes() const {
  return model_.nodes;
}

const ParsedNode *EpairModelParser::FindNodeByName(const std::string &name) const {
  const auto it = node_name_to_index_.find(name);
  if (it == node_name_to_index_.end()) {
    return nullptr;
  }
  return &model_.nodes[it->second];
}

const ParsedNode *EpairModelParser::FindNodeByOutput(const std::string &output_name) const {
  const auto it = output_name_to_index_.find(output_name);
  if (it == output_name_to_index_.end()) {
    return nullptr;
  }
  return &model_.nodes[it->second];
}

const ParsedTensorMeta *EpairModelParser::FindTensorValue(const std::string &name) const {
  const auto it = model_.tensor_values.find(name);
  if (it == model_.tensor_values.end()) {
    return nullptr;
  }
  return &it->second;
}

const ParsedInputSpec *EpairModelParser::FindInputSpec(const std::string &placeholder_name) const {
  const auto it = input_spec_to_index_.find(placeholder_name);
  if (it == input_spec_to_index_.end()) {
    return nullptr;
  }
  return &model_.input_specs[it->second];
}

const ParsedOutputSpec *EpairModelParser::FindOutputSpec(const std::string &output_name) const {
  const auto it = output_spec_to_index_.find(output_name);
  if (it == output_spec_to_index_.end()) {
    return nullptr;
  }
  return &model_.output_specs[it->second];
}

const ParsedAscendIr *EpairModelParser::FindAscendIr(const std::string &node_name) const {
  const ParsedNode *node = FindNodeByName(node_name);
  if (node == nullptr || !node->ascend_ir.has_value()) {
    return nullptr;
  }
  return &node->ascend_ir.value();
}

std::optional<int64_t> EpairModelParser::FindSymIntValue(const std::string &name) const {
  const auto it = model_.sym_int_values.find(name);
  if (it == model_.sym_int_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<double> EpairModelParser::FindSymFloatValue(const std::string &name) const {
  const auto it = model_.sym_float_values.find(name);
  if (it == model_.sym_float_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<bool> EpairModelParser::FindSymBoolValue(const std::string &name) const {
  const auto it = model_.sym_bool_values.find(name);
  if (it == model_.sym_bool_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<int64_t> EpairModelParser::FindIntValue(const std::string &name) const {
  const auto it = model_.int_values.find(name);
  if (it == model_.int_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<double> EpairModelParser::FindFloatValue(const std::string &name) const {
  const auto it = model_.float_values.find(name);
  if (it == model_.float_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<bool> EpairModelParser::FindBoolValue(const std::string &name) const {
  const auto it = model_.bool_values.find(name);
  if (it == model_.bool_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> EpairModelParser::FindStringValue(const std::string &name) const {
  const auto it = model_.string_values.find(name);
  if (it == model_.string_values.end()) {
    return std::nullopt;
  }
  return it->second;
}

PlaceholderKind EpairModelParser::PlaceholderKindOf(const std::string &name) const {
  const ParsedInputSpec *spec = FindInputSpec(name);
  return spec == nullptr ? PlaceholderKind::Unknown : spec->kind;
}

std::string EpairModelParser::LogicalNameOfPlaceholder(const std::string &name) const {
  const ParsedInputSpec *spec = FindInputSpec(name);
  return spec == nullptr ? std::string() : spec->logical_name;
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
