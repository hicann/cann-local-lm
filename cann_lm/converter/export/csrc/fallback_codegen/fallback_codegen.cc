#include "fallback_codegen/fallback_codegen.h"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <array>
#include <set>
#include <tuple>
#include <dirent.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include "common/logging/logging.h"

namespace torch_delegate::fallback {

using json = nlohmann::json;

namespace {

bool IsTensorType(c10::TypePtr type) {
  if (type->isSubtypeOf(c10::TensorType::get())) return true;
  auto opt_type = type->castRaw<c10::OptionalType>();
  if (opt_type && opt_type->getElementType()->isSubtypeOf(c10::TensorType::get())) return true;
  return false;
}

bool IsScalarType(c10::TypePtr type) {
  if (type->str() == "Scalar") return true;
  if (type->isSubtypeOf(c10::IntType::get())) return true;
  if (type->isSubtypeOf(c10::FloatType::get())) return true;
  if (type->isSubtypeOf(c10::BoolType::get())) return true;
  if (type->isSubtypeOf(c10::StringType::get())) return true;
  auto opt_type = type->castRaw<c10::OptionalType>();
  if (opt_type) return IsScalarType(opt_type->getElementType());
  return false;
}

bool IsOptionalType(c10::TypePtr type) {
  return type->castRaw<c10::OptionalType>() != nullptr;
}

c10::TypePtr UnwrapOptional(c10::TypePtr type) {
  if (auto opt = type->castRaw<c10::OptionalType>()) {
    return opt->getElementType();
  }
  return type;
}

bool IsIntType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::IntType::get());
}

bool IsFloatType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::FloatType::get());
}

bool IsBoolType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::BoolType::get());
}

bool IsStringType(c10::TypePtr type) {
  return UnwrapOptional(type)->isSubtypeOf(c10::StringType::get());
}

bool IsScalarPolyType(c10::TypePtr type) {
  return UnwrapOptional(type)->str() == "Scalar";
}

bool IsIntListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->isSubtypeOf(c10::IntType::get());
  }
  return false;
}

bool IsFloatListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->isSubtypeOf(c10::FloatType::get());
  }
  return false;
}

bool IsBoolListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->isSubtypeOf(c10::BoolType::get());
  }
  return false;
}

bool IsScalarListType(c10::TypePtr type) {
  auto base = UnwrapOptional(type);
  if (auto lt = base->cast<c10::ListType>()) {
    return lt->getElementType()->str() == "Scalar";
  }
  return false;
}

bool IsEnumType(c10::TypePtr type) {
  const std::string &s = UnwrapOptional(type)->str();
  return s == "ScalarType" || s == "Layout" || s == "Device" || s == "MemoryFormat";
}

bool IsTensorListType(c10::TypePtr type) {
  if (type->isSubtypeOf(c10::ListType::ofTensors())) return true;
  auto opt_type = type->castRaw<c10::OptionalType>();
  if (opt_type) return IsTensorListType(opt_type->getElementType());
  return false;
}

bool IsOptional(InputKind kind) {
    return kind == InputKind::OPTIONAL_TENSOR ||
           kind == InputKind::INT_OPT ||
           kind == InputKind::FLOAT_OPT ||
           kind == InputKind::BOOL_OPT ||
           kind == InputKind::STRING_OPT ||
           kind == InputKind::INT_LIST_OPT ||
           kind == InputKind::FLOAT_LIST_OPT ||
           kind == InputKind::BOOL_LIST_OPT ||
           kind == InputKind::SCALAR_OPT ||
           kind == InputKind::SCALAR_TYPE_OPT;
}

bool IsValueDependency(InputKind kind) {
    return kind == InputKind::INT || 
           kind == InputKind::INT_OPT ||
           kind == InputKind::FLOAT || 
           kind == InputKind::FLOAT_OPT ||
           kind == InputKind::BOOL ||
           kind == InputKind::BOOL_OPT ||
           kind == InputKind::STRING ||
           kind == InputKind::STRING_OPT ||
           kind == InputKind::INT_LIST ||
           kind == InputKind::INT_LIST_OPT ||
           kind == InputKind::FLOAT_LIST ||
           kind == InputKind::FLOAT_LIST_OPT ||
           kind == InputKind::BOOL_LIST ||
           kind == InputKind::BOOL_LIST_OPT ||
           kind == InputKind::SCALAR ||
           kind == InputKind::SCALAR_OPT ||
           kind == InputKind::SCALAR_LIST ||
           kind == InputKind::SCALAR_TYPE ||
           kind == InputKind::SCALAR_TYPE_OPT;
}

bool HasDefaultValue(const c10::Argument &arg) {
  return arg.default_value().has_value();
}

std::string ToCamelCase(const std::string &snake_case) {
  std::string result;
  bool capitalize_next = true;
  for (char c : snake_case) {
    if (c == '_') {
      capitalize_next = true;
    } else {
      if (capitalize_next) {
        result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        capitalize_next = false;
      } else {
        result += c;
      }
    }
  }
  return result;
}

class CustomKernelLoader {
 public:
  static CustomKernelLoader &Instance() {
    static CustomKernelLoader loader;
    return loader;
  }

  void LoadFromEnv() {
    if (loaded_) return;

    const char *stub_path_env = getenv("TORCH_DELEGATE_CUSTOM_OP_STUB_PATH");
    if (!stub_path_env || strlen(stub_path_env) == 0) {
      TDLOGD("TORCH_DELEGATE_CUSTOM_OP_STUB_PATH not set, skipping custom kernel load");
      loaded_ = true;
      return;
    }

    std::string stub_path(stub_path_env);
    TDLOGI("Loading custom kernels from: %s", stub_path.c_str());

    struct stat path_stat;
    if (stat(stub_path.c_str(), &path_stat) != 0) {
      TDLOGW("Custom kernel path does not exist: %s", stub_path.c_str());
      loaded_ = true;
      return;
    }

    if (S_ISREG(path_stat.st_mode)) {
      LoadSingleSo(stub_path);
    } else if (S_ISDIR(path_stat.st_mode)) {
      LoadAllSoFromDir(stub_path);
    } else {
      TDLOGW("Custom kernel path is neither file nor directory: %s", stub_path.c_str());
    }

    loaded_ = true;
  }

  ~CustomKernelLoader() {
    for (void *handle : handles_) {
      if (handle) dlclose(handle);
    }
  }

 private:
  CustomKernelLoader() = default;

  void LoadSingleSo(const std::string &so_path) {
    TDLOGD("Attempting to load custom kernel: %s", so_path.c_str());

    struct stat file_stat;
    if (stat(so_path.c_str(), &file_stat) != 0) {
      TDLOGE("File does not exist or inaccessible: %s (errno: %d - %s)", so_path.c_str(), errno, strerror(errno));
      return;
    }

    if (!S_ISREG(file_stat.st_mode)) {
      TDLOGE("Path is not a regular file: %s", so_path.c_str());
      return;
    }

    TDLOGD("File exists, size: %ld bytes, mode: 0%lo, attempting dlopen with RTLD_NOW|RTLD_GLOBAL",
           static_cast<long>(file_stat.st_size), static_cast<unsigned long>(file_stat.st_mode));

    void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      const char *dl_error = dlerror();
      TDLOGE("dlopen FAILED for %s: %s", so_path.c_str(), dl_error ? dl_error : "unknown error");
      TDLOGD("Common causes: ABI mismatch (_GLIBCXX_USE_CXX11_ABI), missing dependencies, or permission denied");
      return;
    }

    handles_.push_back(handle);
    TDLOGI("Successfully loaded custom kernel: %s (handle: %p)", so_path.c_str(), handle);
  }

  void LoadAllSoFromDir(const std::string &dir_path) {
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) {
      TDLOGE("Cannot open directory: %s", dir_path.c_str());
      return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string filename = entry->d_name;
      if (filename.size() > 3 && filename.substr(filename.size() - 3) == ".so") {
        std::string full_path = dir_path + "/" + filename;
        LoadSingleSo(full_path);
      }
    }
    closedir(dir);
  }

  bool loaded_ = false;
  std::vector<void *> handles_;
};

void EnsureCustomKernelsLoaded() {
  CustomKernelLoader::Instance().LoadFromEnv();
}

}  // namespace

std::string FallbackCodegen::GenerateGeOpName(const std::string &op_name, const std::string &overload) {
  bool has_trailing_underscore = !op_name.empty() && op_name.back() == '_';
  std::string base_op_name = has_trailing_underscore ? op_name.substr(0, op_name.size() - 1) : op_name;

  std::string ge_op_name = ToCamelCase(base_op_name);

  if (!overload.empty()) {
    ge_op_name += ToCamelCase(overload);
  } else if (has_trailing_underscore) {
    ge_op_name += "Inplace";
  }

  return ge_op_name;
}

std::string FallbackCodegen::GenerateGeOpIdentifier(const std::string &full_op_name, const std::string &overload) {
  std::string id = full_op_name;
  size_t pos = id.find("::");
  if (pos != std::string::npos) {
    id.replace(pos, 2, "__");
  }
  if (!overload.empty()) {
    id += "_" + overload;
  }
  return id;
}

std::pair<std::string, std::string> FallbackCodegen::SplitNamespaceOp(const std::string &full_op) {
  // Dispatcher format: namespace::op_name (e.g., "aten::matmul")
  size_t colon_pos = full_op.find("::");
  if (colon_pos == std::string::npos) {
    return {"", full_op};
  }
  return {full_op.substr(0, colon_pos), full_op.substr(colon_pos + 2)};
}

OpTarget FallbackCodegen::ParseTargetFromJson(const json &node) {
  OpTarget target;
  target.target = node["target"].get<std::string>();

  // Handle Higher-Order Operators (HOPs)
  // auto_functionalized and auto_functionalized_v2 wrap mutable operators
  // The real operator name is in inputs[0].arg.as_operator
  // Example JSON structure:
  //   {
  //     "target": "torch.ops.higher_order.auto_functionalized_v2",
  //     "inputs": [
  //       {"name": "", "arg": {"as_operator": "torch.ops.xllm_ascend.rope_in_place.default"}, "kind": 1},
  //       ...
  //     ]
  //   }
  if (target.target == "torch.ops.higher_order.auto_functionalized" ||
      target.target == "torch.ops.higher_order.auto_functionalized_v2") {
    TDLOGD("Detected HOP: %s, extracting real operator", target.target.c_str());

    if (node.contains("inputs") && node["inputs"].is_array() && node["inputs"].size() > 0) {
      const auto &first_input = node["inputs"][0];
      if (first_input.contains("arg") && first_input["arg"].is_object() && first_input["arg"].contains("as_operator") &&
          first_input["arg"]["as_operator"].is_string()) {
        // Replace target with real operator name from as_operator
        target.target = first_input["arg"]["as_operator"].get<std::string>();
        TDLOGD("HOP unwrapped: real operator = %s", target.target.c_str());
      } else {
        TDLOGE("auto_functionalized node missing 'as_operator' in first input");
        throw FallbackCodegenError(
            "auto_functionalized node missing 'as_operator' in first input. "
            "Expected: inputs[0].arg.as_operator = \"torch.ops.xxx.op_name.default\"");
      }
    } else {
      TDLOGE("auto_functionalized node has no inputs or invalid inputs structure");
      throw FallbackCodegenError("auto_functionalized node has no inputs or invalid inputs structure");
    }
  }

  // FX graph format: torch.ops.<namespace>.<op_name>.<overload>
  // Examples:
  //   torch.ops.aten.topk.default    -> namespace=aten, op_name=topk, overload=""
  //   torch.ops.aten.mul.Tensor      -> namespace=aten, op_name=mul, overload="Tensor"
  //   torch.ops.my_triton_ns.custom_add.default -> namespace=my_triton_ns, op_name=custom_add, overload=""

  // Remove "torch.ops." prefix if present
  std::string op_str = target.target;
  const std::string prefix = "torch.ops.";
  if (op_str.find(prefix) == 0) {
    op_str = op_str.substr(prefix.size());
  }

  // Find last dot to split overload
  size_t last_dot = op_str.find_last_of('.');
  if (last_dot == std::string::npos) {
    // No dots: just op_name
    target.namespace_name = "";
    target.op_name = op_str;
    target.overload = "";
    return target;
  }

  std::string overload_str = op_str.substr(last_dot + 1);
  std::string ns_op = op_str.substr(0, last_dot);

  // Find second-to-last dot to split namespace and op_name
  size_t second_dot = ns_op.find_last_of('.');
  if (second_dot == std::string::npos) {
    // No namespace: just op_name.overload
    target.namespace_name = "";
    target.op_name = ns_op;
  } else {
    target.namespace_name = ns_op.substr(0, second_dot);
    target.op_name = ns_op.substr(second_dot + 1);
  }

  // Convert "default" to empty string (Dispatcher uses empty for default overload)
  target.overload = (overload_str == "default") ? "" : overload_str;

  return target;
}

std::vector<OpTarget> FallbackCodegen::ParseTargetsFromJson(const json &model) {
  std::vector<OpTarget> targets;

  // Try two possible paths for nodes:
  // 1. Direct "nodes" field (simplified format for test fixtures)
  // 2. ExportedProgram format: "graph_module.graph.nodes" (standard EP JSON)
  json nodes;
  if (model.contains("nodes") && model["nodes"].is_array()) {
    nodes = model["nodes"];
    TDLOGD("Found nodes in direct 'nodes' field (simplified format)");
  } else if (model.contains("graph_module") && model["graph_module"].is_object() &&
             model["graph_module"].contains("graph") && model["graph_module"]["graph"].is_object() &&
             model["graph_module"]["graph"].contains("nodes") && model["graph_module"]["graph"]["nodes"].is_array()) {
    nodes = model["graph_module"]["graph"]["nodes"];
    TDLOGD("Found nodes in graph_module.graph.nodes (ExportedProgram format)");
  } else {
    TDLOGW("No nodes found in JSON, neither 'nodes' nor 'graph_module.graph.nodes' exists");
    return targets;
  }

  TDLOGD("Processing %zu nodes from JSON", nodes.size());
  size_t fallback_count = 0;

  for (const auto &node : nodes) {
    bool needs_fallback = false;
    if (node.contains("ascend_ir") && node["ascend_ir"].is_object()) {
      const auto &ascend_ir = node["ascend_ir"];
      if (!ascend_ir.contains("source") || (ascend_ir.contains("source") && ascend_ir["source"].is_string() &&
                                            ascend_ir["source"].get<std::string>() == "")) {
        needs_fallback = true;
        fallback_count++;
      }
    }

    if (needs_fallback) {
      targets.push_back(ParseTargetFromJson(node));
    }
  }

  TDLOGI("ParseTargetsFromJson: total_nodes=%zu, needs_fallback=%zu", nodes.size(), fallback_count);

  std::set<std::tuple<std::string, std::string, std::string>> seen_ops;
  std::vector<OpTarget> unique_targets;
  unique_targets.reserve(targets.size());

  for (const auto &target : targets) {
    auto key = std::make_tuple(target.namespace_name, target.op_name, target.overload);
    if (seen_ops.insert(key).second) {
      unique_targets.push_back(target);
    } else {
      TDLOGD("Skipping duplicate op: namespace=%s, op_name=%s, overload=%s", target.namespace_name.c_str(),
             target.op_name.c_str(), target.overload.empty() ? "(default)" : target.overload.c_str());
    }
  }

  TDLOGI("ParseTargetsFromJson: deduplicated %zu -> %zu unique ops", targets.size(), unique_targets.size());

  return unique_targets;
}

std::vector<OpTarget> FallbackCodegen::ParseModelJson(const std::string &json_path) {
  std::ifstream f(json_path);
  if (!f) {
    throw FallbackCodegenError("cannot read model.json: " + json_path);
  }

  json model = json::parse(f);
  return ParseTargetsFromJson(model);
}

std::vector<OpTarget> FallbackCodegen::ParseModelJsonFromString(const std::string &model_json) {
  try {
    json model = json::parse(model_json);
    return ParseTargetsFromJson(model);
  } catch (const json::parse_error &e) {
    throw FallbackCodegenError("cannot parse model.json content: " + std::string(e.what()));
  }
}

CodegenResult FallbackCodegen::GenerateFromModelJson(const std::string &model_json) {
  TDLOGI("FallbackCodegen::GenerateFromModelJson() enter");

  EnsureCustomKernelsLoaded();

  std::vector<OpTarget> targets;
  try {
    targets = ParseModelJsonFromString(model_json);
    TDLOGD("Parsed %zu targets from model_json", targets.size());
  } catch (const FallbackCodegenError &e) {
    const char *what = e.what();
    TDLOGE("FallbackCodegen::GenerateFromModelJson() parse failed: %s", what ? what : "(null)");
    CodegenResult result;
    result.success = false;
    result.skipped = false;
    result.error_message = what ? std::string(what) : "Unknown error (e.what() returned null)";
    return result;
  }

  if (targets.empty()) {
    TDLOGI("FallbackCodegen::GenerateFromModelJson() skipped: no nodes need fallback");
    CodegenResult result;
    result.success = true;
    result.skipped = true;
    result.error_message = "Skipped: no nodes need fallback codegen (all nodes have valid ascend_ir.source)";
    return result;
  }

  TDLOGI("FallbackCodegen::GenerateFromModelJson() %zu nodes need fallback codegen", targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    TDLOGD("  target[%zu]: namespace=%s, op_name=%s, overload=%s", i, targets[i].namespace_name.c_str(),
           targets[i].op_name.c_str(), targets[i].overload.empty() ? "(default)" : targets[i].overload.c_str());
  }

  const char *torch_pkg_dir_env = std::getenv("TORCH_PKG_DIR");
  const char *torch_npu_dir_env = std::getenv("TORCH_NPU_DIR");
  const char *output_dir_env = std::getenv("TORCH_DELEGATE_OUTPUT_DIR");
  const char *offline_env = std::getenv("TORCH_DELEGATE_OFFLINE");

  std::string torch_pkg_dir = torch_pkg_dir_env ? torch_pkg_dir_env : "";
  std::string torch_npu_dir = torch_npu_dir_env ? torch_npu_dir_env : "";
  std::string output_dir = output_dir_env ? output_dir_env : "./output_dir";

  bool offline = offline_env ? ([](const char *val) {
    std::string str(val);
    return str == "1" || str == "true" || str == "TRUE";
  })(offline_env)
                             : false;

  TDLOGI("FallbackCodegen config: offline=%s, output_dir=%s, torch_pkg_dir=%s, torch_npu_dir=%s",
         offline ? "true" : "false", output_dir.c_str(), torch_pkg_dir.empty() ? "(not set)" : torch_pkg_dir.c_str(),
         torch_npu_dir.empty() ? "(not set)" : torch_npu_dir.c_str());

  if (torch_pkg_dir.empty() || torch_npu_dir.empty()) {
    TDLOGE("FallbackCodegen::GenerateFromModelJson() failed: TORCH_PKG_DIR or TORCH_NPU_DIR not set");
    CodegenResult result;
    result.success = false;
    result.skipped = false;
    result.error_message = "TORCH_PKG_DIR and TORCH_NPU_DIR must be set when fallback codegen is needed";
    return result;
  }

  TDLOGI("FallbackCodegen::GenerateFromModelJson() starting codegen, mode: %s", offline ? "offline" : "online");

  CodegenResult result = GenerateFromTargets(targets, torch_pkg_dir, torch_npu_dir, output_dir, offline);
  result.skipped = false;

  if (result.success) {
    TDLOGI("FallbackCodegen::GenerateFromModelJson() exit: success, generated %zu ops",
           result.generated_op_names.size());
  } else {
    TDLOGE("FallbackCodegen::GenerateFromModelJson() exit: failed, error=%s", result.error_message.c_str());
  }

  return result;
}

SchemaInfo FallbackCodegen::GetSchemaInfo(const std::string &full_op_name, const std::string &overload) {
  SchemaInfo info;
  info.full_op_name = full_op_name;
  info.overload = overload;

  auto [ns, op] = SplitNamespaceOp(full_op_name);
  info.namespace_name = ns;
  info.op_name = op;

  c10::OperatorName op_name(full_op_name, overload);
  auto handle = torch::Dispatcher::singleton().findSchema(op_name);
  if (!handle.has_value()) {
    TDLOGE("Schema not found for '%s' with overload '%s'", full_op_name.c_str(),
           overload.empty() ? "(default)" : overload.c_str());
    throw FallbackCodegenError("schema not found for '" + full_op_name + "' with overload '" + overload + "'");
  }

  TDLOGD("Schema found for '%s', args=%zu, returns=%zu", full_op_name.c_str(), handle->schema().arguments().size(),
         handle->schema().returns().size());

  const auto &schema = handle->schema();

  // Determine inplace/out semantics from schema alias_info
  // inplace: first argument has alias_info()->isWrite() (e.g., relu_, copy_)
  // out: overload == "out" or argument.is_out() == true
  info.is_inplace = false;
  info.is_out = (overload == "out");

  if (!schema.arguments().empty()) {
    const auto &first_arg = schema.arguments().at(0);
    if (first_arg.alias_info() != nullptr && first_arg.alias_info()->isWrite()) {
      // First argument is written to -> inplace semantics
      info.is_inplace = true;
    }
  }

  int fixed_count = 0;
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto &arg = schema.arguments().at(i);
    SchemaArgInfo arg_info;
    arg_info.name = arg.name();
    arg_info.is_kwarg_only = arg.kwarg_only();
    arg_info.has_default = HasDefaultValue(arg);
    arg_info.is_out = arg.is_out();

    // Extract default value if present
    if (arg_info.has_default) {
      try {
        const auto &default_val = arg.default_value();
        if (!default_val.has_value()) {
          arg_info.default_value = nullptr;
        } else if (default_val->isNone()) {
          arg_info.default_value = nullptr;
        } else if (default_val->isInt()) {
          arg_info.default_value = default_val->toInt();
        } else if (default_val->isDouble()) {
          arg_info.default_value = default_val->toDouble();
        } else if (default_val->isBool()) {
          arg_info.default_value = default_val->toBool();
        } else if (default_val->isString()) {
          arg_info.default_value = default_val->toStringRef();
        } else if (default_val->isIntList()) {
          arg_info.default_value = default_val->toIntVector();
        } else if (default_val->isDoubleList()) {
          arg_info.default_value = default_val->toDoubleVector();
        } else {
          // For other types (Tensor, bool list, etc.), we can't easily extract the default value
          // Mark as having default but leave default_value as null
          arg_info.default_value = nullptr;
        }
      } catch (const std::exception &e) {
        // If we can't extract the default value, just mark it as having one
        arg_info.default_value = nullptr;
      }
    }

    // Parse alias_info for arguments
    auto alias_info = arg.alias_info();
    if (alias_info) {
      arg_info.has_alias = true;
      arg_info.is_alias_write = alias_info->isWrite();
      if (alias_info->isWrite()) {
        info.is_inplace = true;  // Mark as inplace if any argument has write alias
      }
    }

    auto type = arg.type();

    bool is_opt = IsOptionalType(type);
    if (IsTensorType(type)) {
      arg_info.type_category = "tensor";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::OPTIONAL_TENSOR : InputKind::TENSOR;
      arg_info.ge_tensor_type = "TensorType::ALL()";
      if (!arg_info.has_default && !arg_info.is_out) {
        fixed_count++;
      }
    } else if (IsTensorListType(type)) {
      arg_info.type_category = "tensor_list";
      arg_info.input_kind = InputKind::TENSOR_LIST;
      arg_info.is_variadic = true;
      arg_info.fixed_index = fixed_count;
      arg_info.ge_tensor_type = "TensorType::ALL()";
    } else if (IsScalarPolyType(type)) {
      arg_info.type_category = "scalar";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::SCALAR_OPT : InputKind::SCALAR;
      arg_info.ge_tensor_type = "TensorType::ALL()";
    } else if (IsIntType(type)) {
      arg_info.type_category = "int";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::INT_OPT : InputKind::INT;
      arg_info.ge_tensor_type = "IntTensorType()";
    } else if (IsFloatType(type)) {
      arg_info.type_category = "float";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::FLOAT_OPT : InputKind::FLOAT;
      arg_info.ge_tensor_type = "FloatTensorType()";
    } else if (IsBoolType(type)) {
      arg_info.type_category = "bool";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::BOOL_OPT : InputKind::BOOL;
      arg_info.ge_tensor_type = "BoolTensorType()";
    } else if (IsIntListType(type)) {
      arg_info.type_category = "int_list";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::INT_LIST_OPT : InputKind::INT_LIST;
      arg_info.ge_tensor_type = "IntTensorType()";
    } else if (IsFloatListType(type)) {
      arg_info.type_category = "float_list";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::FLOAT_LIST_OPT : InputKind::FLOAT_LIST;
      arg_info.ge_tensor_type = "FloatTensorType()";
    } else if (IsBoolListType(type)) {
      arg_info.type_category = "bool_list";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::BOOL_LIST_OPT : InputKind::BOOL_LIST;
      arg_info.ge_tensor_type = "BoolTensorType()";
    } else if (IsScalarListType(type)) {
      arg_info.type_category = "scalar_list";
      arg_info.input_kind = InputKind::SCALAR_LIST;
      arg_info.ge_tensor_type = "TensorType::ALL()";
    } else if (IsStringType(type)) {
      arg_info.type_category = "string";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::STRING_OPT : InputKind::STRING;
      arg_info.ge_tensor_type = "StringTensorType()";
    } else if (IsEnumType(type)) {
      arg_info.type_category = "enum";
      arg_info.input_kind = (is_opt || arg_info.has_default) ? InputKind::SCALAR_TYPE_OPT : InputKind::SCALAR_TYPE;
      arg_info.ge_tensor_type = "IntTensorType()";
    } else {
      throw FallbackCodegenError("unsupported argument type '" + type->str() + "' for argument '" + arg_info.name +
                                 "' in schema");
    }

    info.arguments.push_back(arg_info);
  }

  info.num_fixed_inputs = fixed_count;

  for (const auto &arg : info.arguments) {
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      info.has_variadic_input = true;
      break;
    }
  }

  if (schema.is_vararg()) {
    info.has_variadic_input = true;
  }

  for (size_t i = 0; i < schema.returns().size(); ++i) {
    const auto &ret = schema.returns().at(i);
    SchemaArgInfo ret_info;
    ret_info.name = ret.name().empty() ? ("output_" + std::to_string(i)) : std::string(ret.name());

    // Parse alias_info for returns
    auto alias_info = ret.alias_info();
    if (alias_info) {
      ret_info.has_alias = true;
      ret_info.is_alias_write = alias_info->isWrite();
    }

    auto type = ret.type();
    if (IsTensorType(type)) {
      ret_info.type_category = "tensor";
    } else if (IsTensorListType(type)) {
      ret_info.type_category = "tensor_list";
      ret_info.is_variadic = true;
    } else {
      throw FallbackCodegenError("unsupported return type '" + type->str() + "'");
    }

    info.returns.push_back(ret_info);
  }

  return info;
}

std::string FallbackCodegen::GenerateHelperFunctions() {
  std::ostringstream os;

  os << "at::ScalarType MapGeDtypeToTorch(ge::DataType dtype) {\n";
  os << "    switch (dtype) {\n";
  os << "        case ge::DT_FLOAT: return at::kFloat;\n";
  os << "        case ge::DT_FLOAT16: return at::kHalf;\n";
  os << "        case ge::DT_BF16: return at::kBFloat16;\n";
  os << "        case ge::DT_DOUBLE: return at::kDouble;\n";
  os << "        case ge::DT_INT8: return at::kChar;\n";
  os << "        case ge::DT_INT16: return at::kShort;\n";
  os << "        case ge::DT_INT32: return at::kInt;\n";
  os << "        case ge::DT_INT64: return at::kLong;\n";
  os << "        case ge::DT_UINT8: return at::kByte;\n";
  os << "        case ge::DT_UINT16: return at::kUInt16;\n";
  os << "        case ge::DT_UINT32: return at::kUInt32;\n";
  os << "        case ge::DT_UINT64: return at::kUInt64;\n";
  os << "        case ge::DT_BOOL: return at::kBool;\n";
  os << "        case ge::DT_COMPLEX64: return at::kComplexFloat;\n";
  os << "        case ge::DT_COMPLEX128: return at::kComplexDouble;\n";
  os << "        case ge::DT_QINT8: return at::kQInt8;\n";
  os << "        case ge::DT_QUINT8: return at::kQUInt8;\n";
  os << "        case ge::DT_QINT32: return at::kQInt32;\n";
  os << "        case ge::DT_HIFLOAT8: return at::kFloat8_e5m2;\n";
  os << "        case ge::DT_FLOAT8_E5M2: return at::kFloat8_e5m2;\n";
  os << "        case ge::DT_FLOAT8_E4M3FN: return at::kFloat8_e4m3fn;\n";
  os << "        case ge::DT_FLOAT8_E8M0: return at::kFloat8_e8m0fnu;\n";
  os << "        default:\n";
  os << "            TORCH_CHECK(false, \"Unsupported GE DataType: \", static_cast<int>(dtype));\n";
  os << "            return at::kFloat; // unreachable\n";
  os << "    }\n";
  os << "}\n\n";

  os << "ge::DataType MapTorchDtypeToGe(at::ScalarType dtype) {\n";
  os << "    switch (dtype) {\n";
  os << "        case at::kFloat: return ge::DT_FLOAT;\n";
  os << "        case at::kHalf: return ge::DT_FLOAT16;\n";
  os << "        case at::kBFloat16: return ge::DT_BF16;\n";
  os << "        case at::kDouble: return ge::DT_DOUBLE;\n";
  os << "        case at::kChar: return ge::DT_INT8;\n";
  os << "        case at::kShort: return ge::DT_INT16;\n";
  os << "        case at::kInt: return ge::DT_INT32;\n";
  os << "        case at::kLong: return ge::DT_INT64;\n";
  os << "        case at::kByte: return ge::DT_UINT8;\n";
  os << "        case at::kUInt16: return ge::DT_UINT16;\n";
  os << "        case at::kUInt32: return ge::DT_UINT32;\n";
  os << "        case at::kUInt64: return ge::DT_UINT64;\n";
  os << "        case at::kBool: return ge::DT_BOOL;\n";
  os << "        case at::kComplexFloat: return ge::DT_COMPLEX64;\n";
  os << "        case at::kComplexDouble: return ge::DT_COMPLEX128;\n";
  os << "        case at::kQInt8: return ge::DT_QINT8;\n";
  os << "        case at::kQUInt8: return ge::DT_QUINT8;\n";
  os << "        case at::kQInt32: return ge::DT_QINT32;\n";
  os << "        case at::kFloat8_e5m2: return ge::DT_FLOAT8_E5M2;\n";
  os << "        case at::kFloat8_e4m3fn: return ge::DT_FLOAT8_E4M3FN;\n";
  os << "        case at::kFloat8_e8m0fnu: return ge::DT_FLOAT8_E8M0;\n";
  os << "        default:\n";
  os << "            TORCH_CHECK(false, \"Unsupported PyTorch ScalarType: \", static_cast<int>(dtype));\n";
  os << "            return ge::DT_FLOAT; // unreachable\n";
  os << "    }\n";
  os << "}\n\n";

  os << "at::Tensor CreateNpuTensor(const gert::Tensor* ge_tensor) {\n";
  os << "    std::vector<int64_t> sizes;\n";
  os << "    for (size_t i = 0; i < ge_tensor->GetShape().GetShape().GetDimNum(); i++) {\n";
  os << "        sizes.push_back(ge_tensor->GetShape().GetShape().GetDim(i));\n";
  os << "    }\n";
  os << "    at::ScalarType torch_dtype = MapGeDtypeToTorch(ge_tensor->GetDataType());\n";
  os << "    auto options = at::TensorOptions().dtype(torch_dtype).device(at::kPrivateUse1);\n";
  os << "    return at_npu::native::from_blob(const_cast<void*>(ge_tensor->GetAddr()), sizes, options);\n";
  os << "}\n\n";

  // TODO: ReadScalarInput uses D2H memcpy because GE places Const node data on device memory
  // for custom ops (REG_AUTO_MAPPING_OP path). The kPlacementHost setting on Const nodes and
  // HostInputs registration on IMPL_OP are both ineffective for custom ops because
  // custom_node_converter.cc hardcodes kOnDeviceHbm placement.
  // If GE fixes this in the future, ReadScalarInput can be replaced with direct GetAddr() reads.
  os << "// ReadScalarInput: D2H copy for scalar inputs (GE places data on device for custom ops)\n";
  os << "std::vector<uint8_t> ReadScalarInput(const gert::Tensor* tensor) {\n";
  os << "    size_t byte_size = tensor->GetSize();\n";
  os << "    std::vector<uint8_t> buf(byte_size);\n";
  os << "    aclrtMemcpy(buf.data(), byte_size, tensor->GetAddr(), byte_size, ACL_MEMCPY_DEVICE_TO_HOST);\n";
  os << "    return buf;\n";
  os << "}\n\n";

  return os.str();
}

std::string FallbackCodegen::GenerateGeOpReg(const SchemaInfo &schema) {
  std::ostringstream os;

  std::string ge_op_name = GenerateGeOpIdentifier(schema.full_op_name, schema.overload);

  os << "REG_OP(" << ge_op_name << ")\n";

  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      os << "    .DYNAMIC_INPUT(" << arg.name << ", TensorType::ALL())\n";
    } else if (arg.type_category == "tensor") {
      if (arg.input_kind == InputKind::OPTIONAL_TENSOR) {
        os << "    .OPTIONAL_INPUT(" << arg.name << ", TensorType::ALL())\n";
      } else {
        os << "    .INPUT(" << arg.name << ", TensorType::ALL())\n";
      }
    } else if (IsValueDependency(arg.input_kind)) {
      if (IsOptional(arg.input_kind)) {
        os << "    .OPTIONAL_INPUT(" << arg.name << ", " << arg.ge_tensor_type << ")\n";
      } else {
        os << "    .INPUT(" << arg.name << ", " << arg.ge_tensor_type << ")\n";
      }
    }
  }

  if (schema.has_variadic_input) {
    os << "    .REQUIRED_ATTR(N, Int)\n";
  }

  // Generate outputs based on schema semantics:
  // 1. First, outputs from returns (non-inplace outputs)
  // 2. Then, inplace outputs (from arguments with write alias)

  // Step 1: Process returns - outputs from schema returns
  for (size_t output_idx = 0; output_idx < schema.returns.size(); ++output_idx) {
    const auto &ret = schema.returns[output_idx];
    if (ret.is_variadic) {
      os << "    .DYNAMIC_OUTPUT(" << ret.name << ", TensorType::ALL())\n";
    } else {
      // For inplace ops, returns without alias are new tensors
      // For non-inplace ops, all returns are outputs
      if (schema.is_inplace && ret.has_alias && ret.is_alias_write) {
        // Return has write alias -> this is the inplace output
        // Skip here, will be handled in Step 2
        continue;
      }
      os << "    .OUTPUT(" << ret.name << ", TensorType::ALL())\n";
    }
  }

  // Step 2: For inplace ops, add outputs for arguments with write alias
  if (schema.is_inplace) {
    // Find all arguments with write alias (inplace arguments)
    for (size_t arg_idx = 0; arg_idx < schema.arguments.size(); ++arg_idx) {
      const auto &arg = schema.arguments[arg_idx];
      if (arg.has_alias && arg.is_alias_write) {
        // This argument is written inplace -> need an output
        os << "    .OUTPUT(" << arg.name << ", TensorType::ALL())\n";
      }
    }

    // Special case: if schema returns void but has inplace arg
    if (schema.returns.empty()) {
      // Already handled above (all inplace args get outputs)
    }
  }

  os << "    .ATTR(op_name, String, \"" << schema.full_op_name << "\")\n";
  os << "    .ATTR(overload, String, \"" << schema.overload << "\")\n";
  os << "    .ATTR(input_kinds, ListInt, {";
  bool first_kind = true;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (!first_kind) os << ", ";
    first_kind = false;
    os << static_cast<int>(arg.input_kind);
  }
  os << "})\n";
  os << "    .ATTR(num_outputs, Int, " << schema.returns.size() << ")\n";
  os << "    .ATTR(is_inplace, Bool, " << (schema.is_inplace ? "true" : "false") << ")\n";
  os << "    .OP_END_FACTORY_REG(" << ge_op_name << ");\n";

  std::vector<int> dep_indices;
  int input_idx = 0;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (IsValueDependency(arg.input_kind)) {
      dep_indices.push_back(input_idx);
    }
    input_idx++;
  }
  os << "IMPL_OP(" << ge_op_name << ")\n";
  if (!dep_indices.empty()) {
    os << "    .InputsDataDependency({";
    for (size_t i = 0; i < dep_indices.size(); ++i) {
      if (i > 0) os << ", ";
      os << dep_indices[i];
    }
    os << "});\n";
  } else {
    os << "    ;\n";
  }

  return os.str();
}

// Helper function to generate code for adding a default value or None
static void EmitDefaultOrNone(std::ostringstream &os, const SchemaArgInfo &arg) {
  if (arg.has_default && !arg.default_value.is_null()) {
    // Use the actual default value
    if (arg.input_kind == InputKind::INT || arg.input_kind == InputKind::INT_OPT) {
      os << "            args.emplace_back(static_cast<int64_t>(" << arg.default_value.get<int64_t>() << "));\n";
    } else if (arg.input_kind == InputKind::FLOAT || arg.input_kind == InputKind::FLOAT_OPT) {
      os << "            args.emplace_back(static_cast<double>(" << arg.default_value.get<double>() << "));\n";
    } else if (arg.input_kind == InputKind::BOOL || arg.input_kind == InputKind::BOOL_OPT) {
      os << "            args.emplace_back(" << (arg.default_value.get<bool>() ? "true" : "false") << ");\n";
    } else if (arg.input_kind == InputKind::INT_LIST || arg.input_kind == InputKind::INT_LIST_OPT) {
      os << "            args.emplace_back(std::vector<int64_t>{";
      const auto &list = arg.default_value.get<std::vector<int64_t>>();
      for (size_t i = 0; i < list.size(); ++i) {
        if (i > 0) os << ", ";
        os << list[i];
      }
      os << "});\n";
    } else if (arg.input_kind == InputKind::STRING || arg.input_kind == InputKind::STRING_OPT) {
      os << "            args.emplace_back(std::string(\"" << arg.default_value.get<std::string>() << "\"));\n";
    } else {
      // For other types, use None
      os << "            args.emplace_back(c10::IValue());\n";
    }
  } else {
    // No default value or couldn't extract it, use None
    os << "            args.emplace_back(c10::IValue());\n";
  }
}

std::string FallbackCodegen::GenerateExecuteMethod(const SchemaInfo &schema) {
  std::ostringstream os;

  os << "    ge::graphStatus Execute(gert::EagerOpExecutionContext* ctx) override {\n";

  int input_idx = 0;
  int ir_index = 0;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) continue;
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      continue;
    }
    os << "        const gert::Tensor* ge_input_" << input_idx << " = ctx->GetInputTensor(" << input_idx << ");\n";
    input_idx++;
    ir_index++;
  }

  os << "        aclrtStream ge_stream = static_cast<aclrtStream>(ctx->GetStream());\n";

  // Debug: print first input tensor info

  os << "\n";

  if (schema.is_out) {
    int num_outputs = static_cast<int>(schema.returns.size());
    os << "        // out version: get output tensor descriptions from GE InferShape\n";
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
      if (schema.returns[output_idx].is_variadic) {
        os << "        // output " << output_idx << " is variadic, skip pre-allocation\n";
        continue;
      }
      os << "        const gert::Tensor* output_desc_" << output_idx << " = ctx->GetOutputTensor(" << output_idx
         << ");\n";
      os << "        gert::StorageShape out_shape_" << output_idx << " = output_desc_" << output_idx
         << "->GetShape();\n";
      os << "        ge::DataType out_dtype_" << output_idx << " = output_desc_" << output_idx << "->GetDataType();\n";
      os << "        gert::StorageFormat out_format_" << output_idx << " = output_desc_" << output_idx
         << "->GetFormat();\n";
      os << "        gert::Tensor* ge_output_" << output_idx << " = ctx->MallocOutputTensor(" << output_idx
         << ", out_shape_" << output_idx << ", out_format_" << output_idx << ", out_dtype_" << output_idx << ");\n";
    }
    os << "\n";
  }

  os << "        std::vector<c10::IValue> args;\n";

  int tensor_idx = 0;
  int out_arg_output_idx = 0;
  for (const auto &arg : schema.arguments) {
    if (arg.is_out) {
      int output_idx = out_arg_output_idx;
      if (output_idx < static_cast<int>(schema.returns.size()) && !schema.returns[output_idx].is_variadic) {
        os << "        args.emplace_back(CreateNpuTensor(ge_output_" << output_idx << "));\n";
      } else {
        os << "        args.emplace_back(c10::IValue()); // variadic output placeholder\n";
      }
      out_arg_output_idx++;
      continue;
    }
    
    if (arg.is_variadic && arg.type_category == "tensor_list") {
      os << "        for (size_t i = 0; i < static_cast<size_t>(N); ++i) {\n";
      os << "            args.emplace_back(CreateNpuTensor(ctx->GetInputTensor(" << tensor_idx << " + i)));\n";
      os << "        }\n";
      tensor_idx++;
    } else if (arg.type_category == "tensor") {
      if (IsOptional(arg.input_kind)) {
        os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
        os << "            args.emplace_back(CreateNpuTensor(ge_input_" << tensor_idx << "));\n";
        os << "        } else {\n";
        os << "            args.emplace_back(c10::IValue());\n";
        os << "        }\n";
      } else {
        os << "        args.emplace_back(CreateNpuTensor(ge_input_" << tensor_idx << "));\n";
      }
      tensor_idx++;
    } else if (arg.input_kind == InputKind::INT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(static_cast<int64_t>(*reinterpret_cast<const int64_t*>(buf_" << tensor_idx << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::INT_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(c10::optional<int64_t>(static_cast<int64_t>(*reinterpret_cast<const int64_t*>(buf_" << tensor_idx << ".data()))));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::FLOAT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(static_cast<double>(*reinterpret_cast<const float*>(buf_" << tensor_idx << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::FLOAT_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(c10::optional<double>(static_cast<double>(*reinterpret_cast<const float*>(buf_" << tensor_idx << ".data()))));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::BOOL) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(static_cast<bool>(*reinterpret_cast<const uint8_t*>(buf_" << tensor_idx << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::BOOL_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(c10::optional<bool>(static_cast<bool>(*reinterpret_cast<const uint8_t*>(buf_" << tensor_idx << ".data()))));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::INT_LIST || arg.input_kind == InputKind::INT_LIST_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            std::vector<int64_t> list_" << tensor_idx << ";\n";
      os << "            size_t count_" << tensor_idx << " = buf_" << tensor_idx << ".size() / sizeof(int64_t);\n";
      os << "            const int64_t* data_" << tensor_idx << " = reinterpret_cast<const int64_t*>(buf_" << tensor_idx << ".data());\n";
      os << "            for (size_t i = 0; i < count_" << tensor_idx << "; ++i) list_" << tensor_idx << ".push_back(data_" << tensor_idx << "[i]);\n";
      os << "            args.emplace_back(std::move(list_" << tensor_idx << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::FLOAT_LIST || arg.input_kind == InputKind::FLOAT_LIST_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            std::vector<double> list_" << tensor_idx << ";\n";
      os << "            size_t count_" << tensor_idx << " = buf_" << tensor_idx << ".size() / sizeof(float);\n";
      os << "            const float* data_" << tensor_idx << " = reinterpret_cast<const float*>(buf_" << tensor_idx << ".data());\n";
      os << "            for (size_t i = 0; i < count_" << tensor_idx << "; ++i) list_" << tensor_idx << ".push_back(static_cast<double>(data_" << tensor_idx << "[i]));\n";
      os << "            args.emplace_back(std::move(list_" << tensor_idx << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::BOOL_LIST || arg.input_kind == InputKind::BOOL_LIST_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            std::vector<bool> list_" << tensor_idx << ";\n";
      os << "            size_t count_" << tensor_idx << " = buf_" << tensor_idx << ".size() / sizeof(uint8_t);\n";
      os << "            const uint8_t* data_" << tensor_idx << " = reinterpret_cast<const uint8_t*>(buf_" << tensor_idx << ".data());\n";
      os << "            for (size_t i = 0; i < count_" << tensor_idx << "; ++i) list_" << tensor_idx << ".push_back(static_cast<bool>(data_" << tensor_idx << "[i]));\n";
      os << "            args.emplace_back(std::move(list_" << tensor_idx << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::SCALAR || arg.input_kind == InputKind::SCALAR_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            if (ge_input_" << tensor_idx << "->GetDataType() == ge::DT_INT64) {\n";
      os << "                args.emplace_back(c10::Scalar(*reinterpret_cast<const int64_t*>(buf_" << tensor_idx << ".data())));\n";
      os << "            } else {\n";
      os << "                args.emplace_back(c10::Scalar(static_cast<double>(*reinterpret_cast<const float*>(buf_" << tensor_idx << ".data()))));\n";
      os << "            }\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::SCALAR_LIST) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            std::vector<c10::Scalar> scalar_list_" << tensor_idx << ";\n";
      os << "            if (ge_input_" << tensor_idx << "->GetDataType() == ge::DT_INT64) {\n";
      os << "                size_t count_" << tensor_idx << " = buf_" << tensor_idx << ".size() / sizeof(int64_t);\n";
      os << "                const int64_t* data_" << tensor_idx << " = reinterpret_cast<const int64_t*>(buf_" << tensor_idx << ".data());\n";
      os << "                for (size_t i = 0; i < count_" << tensor_idx << "; ++i) scalar_list_" << tensor_idx << ".emplace_back(data_" << tensor_idx << "[i]);\n";
      os << "            } else {\n";
      os << "                size_t count_" << tensor_idx << " = buf_" << tensor_idx << ".size() / sizeof(float);\n";
      os << "                const float* data_" << tensor_idx << " = reinterpret_cast<const float*>(buf_" << tensor_idx << ".data());\n";
      os << "                for (size_t i = 0; i < count_" << tensor_idx << "; ++i) scalar_list_" << tensor_idx << ".emplace_back(static_cast<double>(data_" << tensor_idx << "[i]));\n";
      os << "            }\n";
      os << "            args.emplace_back(std::move(scalar_list_" << tensor_idx << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::STRING || arg.input_kind == InputKind::STRING_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            std::string str_" << tensor_idx << "(reinterpret_cast<const char*>(buf_" << tensor_idx << ".data()), buf_" << tensor_idx << ".size());\n";
      os << "            args.emplace_back(std::move(str_" << tensor_idx << "));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else if (arg.input_kind == InputKind::SCALAR_TYPE || arg.input_kind == InputKind::SCALAR_TYPE_OPT) {
      os << "        if (ge_input_" << tensor_idx << " != nullptr) {\n";
      os << "            auto buf_" << tensor_idx << " = ReadScalarInput(ge_input_" << tensor_idx << ");\n";
      os << "            args.emplace_back(static_cast<at::ScalarType>(*reinterpret_cast<const int64_t*>(buf_" << tensor_idx << ".data())));\n";
      os << "        } else {\n";
      EmitDefaultOrNone(os, arg);
      os << "        }\n";
      tensor_idx++;
    } else {
      os << "        args.emplace_back(c10::IValue()); // unsupported kind " << static_cast<int>(arg.input_kind) << "\n";
      tensor_idx++;
    }
  }
  os << "\n";

  os << "        int32_t device_id = 0;\n";
  os << "        aclrtGetDevice(&device_id);\n";
  os << "        c10_npu::NPUStream npu_stream = c10_npu::getStreamFromExternal(ge_stream, device_id);\n";
  os << "        c10_npu::NPUStreamGuard guard(npu_stream);\n\n";

  std::string dispatcher_overload = schema.overload.empty() ? "" : schema.overload;
  os << "        static const auto op_handle = torch::Dispatcher::singleton()\n";
  os << "            .findSchemaOrThrow(\"" << schema.full_op_name << "\", \"" << dispatcher_overload << "\");\n\n";
  os << "        op_handle.callBoxed(args);\n";
  os << "        const size_t num_outputs = " << schema.returns.size() << ";\n";
  os << "        TORCH_CHECK(args.size() >= num_outputs, \"Expected at least num_outputs results\");\n";
  os << "        for (size_t out_i = 0; out_i < num_outputs; ++out_i) {\n";
  os << "            size_t result_idx = args.size() - num_outputs + out_i;\n";
  os << "            at::Tensor result = args[result_idx].toTensor();\n";
  // os << "            if (!result.is_contiguous()) result = result.contiguous();\n";
  os << "            gert::StorageShape storage_shape;\n";
  os << "            gert::Shape& origin_shape = storage_shape.MutableOriginShape();\n";
  os << "            origin_shape.SetDimNum(result.dim());\n";
  os << "            for (int64_t d = 0; d < result.dim(); ++d) origin_shape.SetDim(d, result.size(d));\n";
  os << "            gert::Shape& storage_shape_ref = storage_shape.MutableStorageShape();\n";
  os << "            storage_shape_ref.SetDimNum(result.dim());\n";
  os << "            for (int64_t d = 0; d < result.dim(); ++d) storage_shape_ref.SetDim(d, result.size(d));\n";
  os << "            ge::DataType data_type = MapTorchDtypeToGe(result.scalar_type());\n";
  os << "            gert::StorageFormat format;\n";
  os << "            size_t tensor_bytes = result.numel() * result.element_size();\n";

  if (schema.is_inplace) {
    os << "            if (out_i == 0) {\n";
    os << "                const gert::Tensor* input_0 = ctx->GetInputTensor(0);\n";
    os << "                aclrtMemcpyAsync(const_cast<void*>(input_0->GetAddr()), input_0->GetSize(),\n";
    os << "                    result.data_ptr(), tensor_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, ge_stream);\n";
    os << "                ctx->MakeOutputRefInput(0, out_i);\n";
    os << "            } else {\n";
    os << "                gert::Tensor* ge_out = ctx->MallocOutputTensor(out_i, storage_shape, format, data_type);\n";
    os << "                aclrtMemcpyAsync(const_cast<void*>(ge_out->GetAddr()), tensor_bytes,\n";
    os << "                    result.data_ptr(), tensor_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, ge_stream);\n";
    os << "            }\n";
  } else {
    os << "            gert::Tensor* ge_out = ctx->MallocOutputTensor(out_i, storage_shape, format, data_type);\n";
    os << "            aclrtMemcpyAsync(const_cast<void*>(ge_out->GetAddr()), tensor_bytes,\n";
    os << "                result.data_ptr(), tensor_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, ge_stream);\n";
  }

  os << "        }\n";

  if (schema.is_inplace) {
    int inplace_output_start_idx = 0;
    for (const auto &ret : schema.returns) {
      if (!ret.has_alias || !ret.is_alias_write) {
        inplace_output_start_idx++;
      }
    }
    int inplace_output_idx = inplace_output_start_idx;
    for (size_t arg_idx = 0; arg_idx < schema.arguments.size(); ++arg_idx) {
      const auto &arg = schema.arguments[arg_idx];
      if (arg.has_alias && arg.is_alias_write) {
        os << "        ctx->MakeOutputRefInput(" << inplace_output_idx << ", " << arg_idx << ");\n";
        inplace_output_idx++;
      }
    }
  }

  os << "\n        return ge::GRAPH_SUCCESS;\n";
  os << "    }\n";

  return os.str();
}

std::string FallbackCodegen::GenerateGeOpClass(const SchemaInfo &schema) {
  std::ostringstream os;

  std::string ge_op_name = GenerateGeOpIdentifier(schema.full_op_name, schema.overload);

  os << "namespace ge {\n\n";
  os << "class " << ge_op_name << " : public EagerExecuteOp {\n";
  os << "public:\n";
  os << GenerateExecuteMethod(schema);
  os << "};\n\n";

  os << GenerateGeOpReg(schema) << "\n";
  os << "REG_AUTO_MAPPING_OP(" << ge_op_name << ");\n";
  os << "} // namespace ge\n\n";

  return os.str();
}

std::string FallbackCodegen::EmitCustomOpCpp(const std::vector<SchemaInfo> &schemas, const std::string &gcc_cmd) {
  std::ostringstream os;

  os << "// AUTO-GENERATED by FallbackCodegen\n";

  if (!gcc_cmd.empty()) {
    os << "//\n";
    os << "// Online compilation command (g++):\n";
    os << "//   " << gcc_cmd << "\n";
    os << "//\n";
    os << "// If g++ compilation fails, try CMake method:\n";
    os << "//   cmake -B build && cmake --build build -j4\n";
    os << "//\n";
  }

  os << "#include <torch/torch.h>\n";
  os << "#include <torch_npu/csrc/aten/common/from_blob.h>\n";
  os << "#include <torch_npu/csrc/core/npu/NPUStream.h>\n";
  os << "#include <torch_npu/csrc/core/npu/NPUGuard.h>\n";
  os << "#include <acl/acl.h>\n";
  os << "#include <graph/custom_op.h>\n";
  os << "#include <graph/types.h>\n";
  os << "#include <graph/operator_reg.h>\n";
  os << "#include <register/op_impl_registry.h>\n\n";

  os << "using namespace ge;\n\n";
  os << "namespace {\n\n";
  os << GenerateHelperFunctions();
  os << "}\n\n";

  for (const auto &schema : schemas) {
    os << GenerateGeOpClass(schema);
  }

  return os.str();
}

std::string FallbackCodegen::EmitCMakeLists(const std::string &output_dir, const std::string &torch_pkg_dir,
                                            const std::string &torch_npu_dir) {
  (void)output_dir;  // Reserved for future use
  std::ostringstream os;

  os << "# AUTO-GENERATED by FallbackCodegen\n";
  os << "#\n";
  os << "# 编译方法：\n";
  os << "#   1. 设置环境变量（推荐）：\n";
  os << "#      source ~/Ascend/ascend-toolkit/set_env.sh  # 设置 ASCEND_HOME_PATH\n";
  os << "#      export TORCH_PKG_DIR=/path/to/site-packages/torch\n";
  os << "#      export TORCH_NPU_DIR=/path/to/libtorch_npu\n";
  os << "#      cmake -B build && cmake --build build -j4\n";
  os << "#\n";
  os << "#   2. 直接指定 cmake 变量：\n";
  os << "#      cmake -B build -DTORCH_PKG_DIR=/path/to/site-packages/torch\n";
  os << "#                 -DASCEND_HOME_PATH=/path/to/cann\n";
  os << "#                 -DTORCH_NPU_DIR=/path/to/libtorch_npu\n";
  os << "#      cmake --build build -j4\n";
  os << "#\n";
  os << "#   3. 离线编译（拷贝到另一环境）：\n";
  os << "#      拷贝 custom_op.cpp 和 CMakeLists.txt 到目标机器\n";
  os << "#      按上述方法设置环境变量或 cmake 参数后编译\n";
  os << "#\n\n";

  os << "cmake_minimum_required(VERSION 3.16)\n";
  os << "project(cust_opapi LANGUAGES CXX)\n\n";

  os << "set(CMAKE_CXX_STANDARD 17)\n";
  os << "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -fPIC\")\n\n";

  os << "# ---------------------------------------------------------------------------\n";
  os << "# 依赖路径配置（支持环境变量或 cmake 变量）\n";
  os << "# ---------------------------------------------------------------------------\n";

  os << "# Torch: TORCH_PKG_DIR 指向 Python torch 包目录，内部拼接 TorchConfig.cmake 路径\n";
  if (!torch_pkg_dir.empty() && torch_pkg_dir != "<placeholder>") {
    os << "set(TORCH_PKG_DIR \"\" CACHE PATH \"Path to Python torch package directory\")\n";
    os << "if(TORCH_PKG_DIR STREQUAL \"\")\n";
    os << "    if(DEFINED ENV{TORCH_PKG_DIR})\n";
    os << "        set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})\n";
    os << "    endif()\n";
    os << "endif()\n";
    os << "if(TORCH_PKG_DIR STREQUAL \"\")\n";
    os << "    set(TORCH_PKG_DIR \"" << torch_pkg_dir << "\")  # fallback: codegen 时传入的路径\n";
    os << "endif()\n";
  } else {
    os << "set(TORCH_PKG_DIR \"\" CACHE PATH \"Path to Python torch package directory\")\n";
    os << "if(TORCH_PKG_DIR STREQUAL \"\" AND DEFINED ENV{TORCH_PKG_DIR})\n";
    os << "    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})\n";
    os << "endif()\n";
  }
  os << "if(TORCH_PKG_DIR STREQUAL \"\")\n";
  os << "    message(FATAL_ERROR \"TORCH_PKG_DIR not set. Please set: export TORCH_PKG_DIR=/path/to/site-packages/torch\")\n";
  os << "endif()\n";
  os << "set(TORCH_CMAKE_DIR \"${TORCH_PKG_DIR}/share/cmake/Torch\")\n";
  os << "find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_DIR})\n\n";

  os << "# TORCH_NPU_DIR: torch_npu 库目录（包含 lib/libtorch_npu.so）\n";
  os << "set(TORCH_NPU_DIR \"\" CACHE PATH \"torch_npu library directory\")\n";
  if (!torch_npu_dir.empty()) {
    os << "if(TORCH_NPU_DIR STREQUAL \"\")\n";
    os << "    if(DEFINED ENV{TORCH_NPU_DIR})\n";
    os << "        set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})\n";
    os << "    else()\n";
    os << "        set(TORCH_NPU_DIR \"" << torch_npu_dir << "\")  # fallback\n";
    os << "    endif()\n";
    os << "endif()\n";
  } else {
    os << "if(TORCH_NPU_DIR STREQUAL \"\" AND DEFINED ENV{TORCH_NPU_DIR})\n";
    os << "    set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})\n";
    os << "endif()\n";
  }
  os << "\n";

  os << "# ASCEND_HOME_PATH: CANN 安装路径（从环境变量获取）\n";
  os << "set(ASCEND_HOME_PATH \"\" CACHE PATH \"CANN installation root\")\n";
  os << "if(ASCEND_HOME_PATH STREQUAL \"\" AND DEFINED ENV{ASCEND_HOME_PATH})\n";
  os << "    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})\n";
  os << "endif()\n";
  os << "\n";

  os << "# ---------------------------------------------------------------------------\n";
  os << "# 编译配置\n";
  os << "# ---------------------------------------------------------------------------\n";

  os << "include_directories(${ASCEND_HOME_PATH}/include)\n";
  os << "include_directories(${ASCEND_HOME_PATH}/include/graph)\n";
  os << "include_directories(${ASCEND_HOME_PATH}/include/register)\n\n";

  os << "add_library(cust_opapi SHARED custom_op.cpp)\n\n";

  os << "# Derive pytorch source root from TORCH_NPU_DIR (parent of libtorch_npu/)\n";
  os << "# Required for NPUQueue.h -> torch_npu/csrc/logging/LogContext.h\n";
  os << "get_filename_component(PYTORCH_SRC_DIR \"${TORCH_NPU_DIR}/..\" ABSOLUTE)\n\n";
  os << "target_include_directories(cust_opapi PRIVATE\n";
  os << "    ${TORCH_INCLUDE_DIRS}\n";
  os << "    ${TORCH_NPU_DIR}/include\n";
  os << "    ${ASCEND_HOME_PATH}/include\n";
  os << "    ${ASCEND_HOME_PATH}/include/graph\n";
  os << "    ${PYTORCH_SRC_DIR}\n";
  os << ")\n\n";

  os << "target_link_libraries(cust_opapi PRIVATE\n";
  os << "    ${TORCH_LIBRARIES}\n";
  os << "    -Wl,--no-as-needed\n";
  os << "    torch_npu\n";
  os << "    ascendcl\n";
  os << "    ge_runner\n";
  os << "    opapi\n";
  os << "    -Wl,--as-needed\n";
  os << ")\n\n";

  os << "target_link_directories(cust_opapi PRIVATE\n";
  os << "    ${TORCH_NPU_DIR}/lib\n";
  os << "    ${ASCEND_HOME_PATH}/lib64\n";
  os << ")\n\n";

  os << "set_target_properties(cust_opapi PROPERTIES\n";
  os << "    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}\n";
  os << ")\n";

  return os.str();
}

void FallbackCodegen::Compile(const std::string &output_dir, const std::string &cmake_cmd) {
  std::array<char, 256> buf{};
  std::string output;

  FILE *pipe = popen(cmake_cmd.c_str(), "r");
  if (!pipe) {
    throw FallbackCodegenError("popen failed for cmake command");
  }
  while (fgets(buf.data(), buf.size(), pipe) != nullptr) {
    output += buf.data();
  }
  int rc = pclose(pipe);
  if (rc != 0) {
    throw FallbackCodegenError("cmake failed: " + output);
  }

  std::string make_cmd = "cd " + output_dir + "/build && make -j4 2>&1";
  pipe = popen(make_cmd.c_str(), "r");
  if (!pipe) {
    throw FallbackCodegenError("popen failed for make command");
  }
  output.clear();
  while (fgets(buf.data(), buf.size(), pipe) != nullptr) {
    output += buf.data();
  }
  rc = pclose(pipe);
  if (rc != 0) {
    throw FallbackCodegenError("make failed: " + output);
  }
}

std::string FallbackCodegen::BuildGccCommandFromPaths(const std::string &torch_pkg_dir, const std::string &torch_npu_dir,
                                                      const std::string &cpp_path, const std::string &so_path) {
  std::ostringstream cmd;

  cmd << "g++ -shared -fPIC -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -g";

  std::string torch_base = torch_pkg_dir;

  std::string cann_root;
  const char *env = std::getenv("ASCEND_HOME_PATH");
  if (env && strlen(env) > 0) {
    cann_root = env;
  } else {
    TDLOGE("ASCEND_HOME_PATH not set or empty, cannot build g++ command");
    throw FallbackCodegenError("ASCEND_HOME_PATH must be set for online compilation");
  }

  cmd << " -I" << torch_base << "/include";
  cmd << " -I" << torch_base << "/include/torch/csrc/api/include";
  cmd << " -I" << cann_root << "/include";
  cmd << " -I" << cann_root << "/include/graph";
  cmd << " -I" << cann_root << "/include/register";
  cmd << " -I" << torch_npu_dir << "/include";
  cmd << " -I" << (std::filesystem::path(torch_npu_dir) / "..").lexically_normal().string();

  const std::filesystem::path torch_libs = std::filesystem::path(torch_base).parent_path() / "torch.libs";
  cmd << " -L" << torch_base << "/lib";
  if (std::filesystem::exists(torch_libs)) {
    cmd << " -L" << torch_libs.string();
  }
  cmd << " -L" << torch_npu_dir << "/lib";
  cmd << " -L" << cann_root << "/lib64";

  cmd << " -Wl,-rpath," << torch_base << "/lib";
  if (std::filesystem::exists(torch_libs)) {
    cmd << " -Wl,-rpath," << torch_libs.string();
    cmd << " -Wl,-rpath-link," << torch_libs.string();
  }
  cmd << " -Wl,-rpath," << torch_npu_dir << "/lib";
  cmd << " -Wl,-rpath," << cann_root << "/lib64";

  cmd << " -Wl,--no-as-needed";
  cmd << " -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -lge_runner -llowering -lopapi";
  cmd << " -Wl,--as-needed";

  cmd << " -o " << so_path << " " << cpp_path;

  return cmd.str();
}

void FallbackCodegen::CompileWithGcc(const std::string &gcc_cmd) {
  std::array<char, 4096> buf{};
  std::string output;

  std::string cmd_with_stderr = gcc_cmd + " 2>&1";
  FILE *pipe = popen(cmd_with_stderr.c_str(), "r");
  if (!pipe) {
    throw FallbackCodegenError("popen failed for g++ command");
  }
  while (fgets(buf.data(), buf.size(), pipe) != nullptr) {
    output += buf.data();
  }
  int rc = pclose(pipe);
  if (rc != 0) {
    throw FallbackCodegenError("g++ compilation failed: " + output);
  }
}

CodegenResult FallbackCodegen::GenerateFromTargets(const std::vector<OpTarget> &targets, const std::string &torch_pkg_dir,
                                                   const std::string &torch_npu_dir, const std::string &output_dir,
                                                   bool offline) {
  TDLOGI("FallbackCodegen::GenerateFromTargets() enter, targets=%zu, offline=%s", targets.size(),
         offline ? "true" : "false");

  EnsureCustomKernelsLoaded();

  CodegenResult result;

  try {
    TDLOGD("Fetching schemas for %zu targets", targets.size());
    std::vector<SchemaInfo> schemas;
    for (const auto &target : targets) {
      std::string full_op = target.namespace_name + "::" + target.op_name;
      TDLOGD("  Getting schema: %s, overload=%s", full_op.c_str(),
             target.overload.empty() ? "(default)" : target.overload.c_str());
      schemas.push_back(GetSchemaInfo(full_op, target.overload));
    }

    TDLOGD("Generating custom_op.cpp content");
    std::string cpp_content = EmitCustomOpCpp(schemas);
    TDLOGD("Generating CMakeLists.txt content");
    std::string cmake_content = EmitCMakeLists(output_dir, torch_pkg_dir, torch_npu_dir);

    for (const auto &schema : schemas) {
      std::string ge_op_name = GenerateGeOpIdentifier(schema.full_op_name, schema.overload);
      result.generated_op_names.push_back(ge_op_name);
    }

    if (offline) {
      TDLOGI("Offline mode: generating files to %s", output_dir.c_str());
      std::string work_dir = output_dir;
      if (!std::filesystem::exists(work_dir)) {
        TDLOGI("Creating output directory: %s", work_dir.c_str());
        std::filesystem::create_directories(work_dir);
      }

      std::string cpp_path = work_dir + "/custom_op.cpp";
      std::string cmake_path = work_dir + "/CMakeLists.txt";

      TDLOGI("Generating custom_op.cpp...");
      {
        std::ofstream f(cpp_path);
        if (!f) {
          TDLOGE("Failed to write custom_op.cpp: %s (errno=%d)", cpp_path.c_str(), errno);
          throw FallbackCodegenError("cannot write custom_op.cpp");
        }
        f << cpp_content;
      }
      TDLOGI("Written custom_op.cpp: %s (%zu bytes)", cpp_path.c_str(), cpp_content.size());

      TDLOGI("Generating CMakeLists.txt...");
      {
        std::ofstream f(cmake_path);
        if (!f) {
          TDLOGE("Failed to write CMakeLists.txt: %s (errno=%d)", cmake_path.c_str(), errno);
          throw FallbackCodegenError("cannot write CMakeLists.txt");
        }
        f << cmake_content;
      }
      TDLOGI("Written CMakeLists.txt: %s (%zu bytes)", cmake_path.c_str(), cmake_content.size());

      TDLOGI("Offline mode: files generated successfully");
      TDLOGI("  To compile manually:");
      TDLOGI("    cd %s", work_dir.c_str());
      TDLOGI("    cmake -B build -DTORCH_PKG_DIR=%s -DTORCH_NPU_DIR=%s -DASCEND_HOME_PATH=$ASCEND_HOME_PATH",
             torch_pkg_dir.c_str(), torch_npu_dir.c_str());
      TDLOGI("    cmake --build build -j4");

      result.generated_cpp_path = cpp_path;
      result.generated_cmake_path = cmake_path;
    } else {
      TDLOGI("Online mode: starting compilation");

      const char *keep_env = std::getenv("TORCH_DELEGATE_KEEP_ARTIFACTS");
      bool keep_artifacts = keep_env && ([](const char *val) {
                              std::string str(val);
                              return str == "1" || str == "true" || str == "TRUE";
                            })(keep_env);

      std::string work_dir;
      bool cleanup_after_load = true;

      {
        char temp_dir[] = "/tmp/fbcg_XXXXXX";
        if (mkdtemp(temp_dir) == nullptr) {
          TDLOGE("mkdtemp failed: errno=%d", errno);
          throw FallbackCodegenError("mkdtemp failed");
        }
        work_dir = temp_dir;
        cleanup_after_load = !keep_artifacts;
        TDLOGI("Online mode: using temp_dir=%s, keep=%s", work_dir.c_str(), keep_artifacts ? "true" : "false");
      }

      if (!std::filesystem::exists(work_dir)) {
        TDLOGI("Creating working directory: %s", work_dir.c_str());
        std::filesystem::create_directories(work_dir);
      }

      const char *ascend_home = std::getenv("ASCEND_HOME_PATH");
      TDLOGI("Environment check: ASCEND_HOME_PATH=%s", ascend_home ? ascend_home : "(not set)");

      if (ascend_home && strlen(ascend_home) > 0) {
        std::string acl_header = std::string(ascend_home) + "/include/acl/acl.h";
        std::string acl_lib = std::string(ascend_home) + "/lib64/libascendcl.so";
        bool acl_header_exists = std::filesystem::exists(acl_header);
        bool acl_lib_exists = std::filesystem::exists(acl_lib);
        TDLOGI("CANN paths: acl.h=%s (%s), libascendcl.so=%s (%s)", acl_header.c_str(),
               acl_header_exists ? "exists" : "missing", acl_lib.c_str(), acl_lib_exists ? "exists" : "missing");
      }

      std::string cpp_path = work_dir + "/custom_op.cpp";
      std::string cmake_path = work_dir + "/CMakeLists.txt";
      std::string so_path = work_dir + "/libcust_opapi.so";

      std::string gcc_cmd = BuildGccCommandFromPaths(torch_pkg_dir, torch_npu_dir, cpp_path, so_path);
      TDLOGI("Generated g++ command (%zu chars)", gcc_cmd.size());

      std::string cpp_content_with_cmd = EmitCustomOpCpp(schemas, gcc_cmd);
      TDLOGI("Generated custom_op.cpp size: %zu bytes", cpp_content_with_cmd.size());

      {
        std::ofstream f(cpp_path);
        if (!f) {
          TDLOGE("Failed to write custom_op.cpp: %s (errno=%d)", cpp_path.c_str(), errno);
          throw FallbackCodegenError("cannot write custom_op.cpp");
        }
        f << cpp_content_with_cmd;
      }
      TDLOGI("Written custom_op.cpp to: %s", cpp_path.c_str());

      if (keep_artifacts) {
        {
          std::ofstream f(cmake_path);
          if (!f) {
            TDLOGE("Failed to write CMakeLists.txt: %s (errno=%d)", cmake_path.c_str(), errno);
            throw FallbackCodegenError("cannot write CMakeLists.txt");
          }
          f << cmake_content;
        }
        TDLOGI("Written CMakeLists.txt to: %s", cmake_path.c_str());
      }

      TDLOGI("Starting g++ compilation...");
      try {
        CompileWithGcc(gcc_cmd);
        TDLOGI("Compilation succeeded");
      } catch (const FallbackCodegenError &e) {
        TDLOGE("Compilation FAILED, command was: %s", gcc_cmd.c_str());
        TDLOGE("Compilation error: %s", e.what());
        TDLOGE("Generated file location: %s", cpp_path.c_str());
        TDLOGE("To manually compile, copy the g++ command above and fix the error");
        throw;
      }

      TDLOGI("Loading shared library: %s", so_path.c_str());
      result.so_handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!result.so_handle) {
        const char *dl_error = dlerror();
        TDLOGE("dlopen FAILED: %s", dl_error ? dl_error : "(unknown error)");
        TDLOGE("Shared library path: %s", so_path.c_str());
        TDLOGE("Library file exists: %s", std::filesystem::exists(so_path) ? "yes" : "no");
        if (std::filesystem::exists(so_path)) {
          TDLOGE("Library file size: %zu bytes", std::filesystem::file_size(so_path));
        }
        throw FallbackCodegenError("dlopen failed: " + std::string(dl_error ? dl_error : "unknown error"));
      }
      TDLOGI("Loaded shared library: %s, handle=%p", so_path.c_str(), result.so_handle);

      if (cleanup_after_load) {
        TDLOGI("Cleaning up temporary files: %s", work_dir.c_str());
        unlink(cpp_path.c_str());
        unlink(so_path.c_str());
        rmdir(work_dir.c_str());
        TDLOGI("Cleanup completed");
      } else {
        const char *output_env = std::getenv("TORCH_DELEGATE_OUTPUT_DIR");
        std::string base_dir = output_env ? output_env : "./output_dir";
        std::string model_name = schemas.empty() ? "unknown" : GenerateGeOpIdentifier(schemas[0].full_op_name, schemas[0].overload);
        std::string preserve_dir = base_dir + "/" + model_name;
        if (!std::filesystem::exists(preserve_dir)) {
          std::filesystem::create_directories(preserve_dir);
        }
        std::error_code ec;
        std::filesystem::copy_file(cpp_path, preserve_dir + "/custom_op.cpp", std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::copy_file(cmake_path, preserve_dir + "/CMakeLists.txt", std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::copy_file(so_path, preserve_dir + "/libcust_opapi.so", std::filesystem::copy_options::overwrite_existing, ec);
        TDLOGI("Artifacts copied to: %s", preserve_dir.c_str());
        unlink(cpp_path.c_str());
        unlink(cmake_path.c_str());
        unlink(so_path.c_str());
        rmdir(work_dir.c_str());
        TDLOGI("Temporary files cleaned up: %s", work_dir.c_str());
      }
    }

    result.success = true;
    TDLOGI("FallbackCodegen::GenerateFromTargets() exit: success, generated %zu ops", result.generated_op_names.size());

  } catch (const FallbackCodegenError &e) {
    const char *what = e.what();
    TDLOGE("FallbackCodegen::GenerateFromTargets() failed: %s", what ? what : "(null)");
    result.success = false;
    result.error_message = what ? std::string(what) : "Unknown error (e.what() returned null)";
  } catch (const std::exception &e) {
    const char *what = e.what();
    TDLOGE("FallbackCodegen::GenerateFromTargets() failed: %s", what ? what : "(null)");
    result.success = false;
    result.error_message = what ? std::string(what) : "Unknown error (e.what() returned null)";
  }

  return result;
}

}  // namespace torch_delegate::fallback
