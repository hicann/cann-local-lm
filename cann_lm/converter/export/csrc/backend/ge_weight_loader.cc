#include "backend/ge_weight_loader.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>

#include <nlohmann/json.hpp>

#include "common/abi_compat_utils.h"
#include "common/base64.h"
#include "common/logging/logging.h"
#include "common/pt2_type_constants.h"

namespace td {
namespace {

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

std::string ResolveLogicalName(const std::string &name,
                               const std::unordered_map<std::string, std::string> &placeholder_to_logical_name,
                               const std::unordered_map<std::string, TensorRecordInfo> &logical_name_to_record) {
  if (logical_name_to_record.find(name) != logical_name_to_record.end()) {
    return name;
  }
  const auto it = placeholder_to_logical_name.find(name);
  return it == placeholder_to_logical_name.end() ? std::string() : it->second;
}

size_t Product(const std::vector<int64_t> &dims) {
  if (dims.empty()) {
    return 1;
  }
  return static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>()));
}

size_t ElementSizeForPt2Dtype(int64_t dtype) {
  switch (dtype) {
    case pt2::kDtypeByte:
    case pt2::kDtypeChar:
    case pt2::kDtypeBool:
      return sizeof(uint8_t);
    case pt2::kDtypeShort:
    case pt2::kDtypeHalf:
    case pt2::kDtypeBFloat16:
    case pt2::kDtypeUInt16:
      return sizeof(uint16_t);
    case pt2::kDtypeInt:
      return sizeof(int32_t);
    case pt2::kDtypeLong:
      return sizeof(int64_t);
    case pt2::kDtypeFloat:
      return sizeof(float);
    case pt2::kDtypeDouble:
      return sizeof(double);
    default:
      return 0;
  }
}

int64_t Pt2DtypeFromGeString(const std::string &dtype, ValidationResult &result, const std::string &context) {
  if (dtype == "DT_UINT8") {
    return pt2::kDtypeByte;
  }
  if (dtype == "DT_INT8") {
    return pt2::kDtypeChar;
  }
  if (dtype == "DT_INT16") {
    return pt2::kDtypeShort;
  }
  if (dtype == "DT_INT32") {
    return pt2::kDtypeInt;
  }
  if (dtype == "DT_INT64") {
    return pt2::kDtypeLong;
  }
  if (dtype == "DT_FLOAT16") {
    return pt2::kDtypeHalf;
  }
  if (dtype == "DT_FLOAT") {
    return pt2::kDtypeFloat;
  }
  if (dtype == "DT_DOUBLE") {
    return pt2::kDtypeDouble;
  }
  if (dtype == "DT_BOOL") {
    return pt2::kDtypeBool;
  }
  if (dtype == "DT_BF16") {
    return pt2::kDtypeBFloat16;
  }
  if (dtype == "DT_UINT16") {
    return pt2::kDtypeUInt16;
  }
  AddError(result, "UNSUPPORTED_CONST_DTYPE", "Unsupported inline const dtype", context);
  return 0;
}

int64_t Pt2LayoutFromGeString(const std::string &layout, ValidationResult &result, const std::string &context) {
  if (layout.empty() || layout == "ND") {
    return pt2::kLayoutStrided;
  }
  AddError(result, "UNSUPPORTED_CONST_LAYOUT", "Unsupported inline const layout", context);
  return 0;
}

// todo 这里还存在权重的拷贝，考虑优化掉
template <typename T>
TensorData DecodeBytesAs(const uint8_t *bytes, size_t byte_size) {
  std::vector<T> values(byte_size / sizeof(T));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes, byte_size);
  }
  return values;
}

TensorData DecodeTensorData(const uint8_t *bytes, size_t byte_size, int64_t pt2_dtype, ValidationResult &result,
                            const std::string &context) {
  switch (pt2_dtype) {
    case pt2::kDtypeByte:
    case pt2::kDtypeBool:
      return DecodeBytesAs<uint8_t>(bytes, byte_size);
    case pt2::kDtypeChar:
      return DecodeBytesAs<int8_t>(bytes, byte_size);
    case pt2::kDtypeShort:
      return DecodeBytesAs<int16_t>(bytes, byte_size);
    case pt2::kDtypeInt:
      return DecodeBytesAs<int32_t>(bytes, byte_size);
    case pt2::kDtypeLong:
      return DecodeBytesAs<int64_t>(bytes, byte_size);
    case pt2::kDtypeHalf:
    case pt2::kDtypeBFloat16:
    case pt2::kDtypeUInt16:
      return DecodeBytesAs<uint16_t>(bytes, byte_size);
    case pt2::kDtypeFloat:
      return DecodeBytesAs<float>(bytes, byte_size);
    case pt2::kDtypeDouble:
      return DecodeBytesAs<double>(bytes, byte_size);
    default:
      AddError(result, "UNSUPPORTED_WEIGHT_DTYPE", "Unsupported PT2 dtype", context);
      return std::monostate{};
  }
}

TensorHolder BuildTensorHolder(const std::vector<int64_t> &dims, int64_t dtype, int64_t layout,
                               const uint8_t *bytes, size_t byte_size, ValidationResult &result,
                               const std::string &context,
                               const std::string &size_error_code) {
  TensorHolder holder;
  holder.dims = dims;
  holder.datatype = dtype;
  holder.format = layout;
  const size_t element_size = ElementSizeForPt2Dtype(holder.datatype);
  if (element_size == 0) {
    AddError(result, "UNSUPPORTED_WEIGHT_DTYPE", "Unsupported PT2 dtype", context);
    return {};
  }
  const size_t expected_bytes = Product(holder.dims) * element_size;
  if (byte_size != expected_bytes) {
    AddError(result, size_error_code, "Tensor payload size does not match tensor metadata", context);
    return {};
  }
  holder.value = DecodeTensorData(bytes, byte_size, holder.datatype, result, context);
  if (!result.ok || !holder.HasValue()) {
    return {};
  }
  return holder;
}

std::vector<int64_t> ParseInlineConstDims(const OrderedJson &desc, ValidationResult &result,
                                          const std::string &context) {
  if (!desc.contains("shape") || !desc["shape"].is_object()) {
    AddError(result, "MISSING_CONST_SHAPE", "Inline const tensor desc missing shape object", context);
    return {};
  }
  const auto &shape = desc["shape"];
  if (!shape.contains("dim")) {
    return {};
  }
  const auto &dim = shape["dim"];
  if (!dim.is_array()) {
    AddError(result, "INVALID_CONST_SHAPE", "Inline const shape.dim must be array", context);
    return {};
  }
  std::vector<int64_t> dims;
  dims.reserve(dim.size());
  for (const auto &item : dim) {
    if (item.is_string()) {
      dims.push_back(std::stoll(item.get<std::string>()));
      continue;
    }
    int64_t value = 0;
    if (!TryGetIntField(item, value)) {
      AddError(result, "INVALID_CONST_SHAPE", "Expected integer element", context);
      return {};
    }
    dims.push_back(value);
  }
  return dims;
}
}  // namespace

void BuildConsolidatedLookup(const ParsedModel &model,
                             std::unordered_map<std::string, std::string> &out) {
  for (const auto &spec : model.input_specs) {
    if (spec.kind == PlaceholderKind::Parameter || spec.kind == PlaceholderKind::Buffer ||
        spec.kind == PlaceholderKind::TensorConstant) {
      out[spec.placeholder_name] = spec.logical_name;
    }
  }
}


GeWeightLoader GeWeightLoader::Create(const std::string &weights_index_path, const ParsedModel &model,
                                                ValidationResult &result) {
  GeWeightLoader loader;
  BuildConsolidatedLookup(model, loader.placeholder_to_logical_name_);
  loader.LoadConsolidated(weights_index_path, model, result);

  for (const auto &[placeholder_name, logical_name] : loader.placeholder_to_logical_name_) {
    if (loader.logical_name_to_record_.find(logical_name) == loader.logical_name_to_record_.end()) {
      AddError(result, "MISSING_PLACEHOLDER_RECORD", "No weight/constant record for placeholder logical name",
               placeholder_name);
    }
  }
  return loader;
}


void GeWeightLoader::LoadConsolidated(const std::string &weights_index_path, const ParsedModel &model,
                                       ValidationResult &result) {
  // Parse weight_info.json
  std::ifstream idx_file(weights_index_path);
  if (!idx_file) {
    AddError(result, "MISSING_WEIGHTS_INDEX", "weight_info.json not found", weights_index_path);
    return;
  }
  OrderedJson root;
  try {
    idx_file >> root;
  } catch (const OrderedJson::parse_error &error) {
    AddError(result, "CONSOLIDATED_INDEX_PARSE_ERROR", error.what(), weights_index_path);
    return;
  }

  if (!root.is_object()) {
    AddError(result, "INVALID_CONSOLIDATED_INDEX", "weight_info.json must be a JSON object", weights_index_path);
    return;
  }

  // Derive bin file directory from the index file path
  const std::filesystem::path index_dir = std::filesystem::path(weights_index_path).parent_path();

  // Build logical_name -> is_parameter lookup from model input_specs
  std::unordered_map<std::string, bool> logical_is_parameter;
  for (const auto &spec : model.input_specs) {
    if (spec.kind == PlaceholderKind::Parameter) {
      logical_is_parameter[spec.logical_name] = true;
    } else if (spec.kind == PlaceholderKind::Buffer || spec.kind == PlaceholderKind::TensorConstant) {
      logical_is_parameter[spec.logical_name] = false;
    }
  }

  // Build reverse map: logical_name -> placeholder_name
  std::unordered_map<std::string, std::string> logical_to_placeholder;
  for (const auto &[placeholder, logical] : placeholder_to_logical_name_) {
    logical_to_placeholder[logical] = placeholder;
  }

  for (auto it = root.begin(); it != root.end(); ++it) {
    // Key is the bin file name (e.g. "weights_0.bin")
    const std::string bin_filename = it.key();
    const std::string bin_path = (index_dir / bin_filename).string();

    if (!it.value().is_array()) continue;

    for (const auto &entry : it.value()) {
      if (!entry.is_object()) continue;

      TensorRecordInfo info;
      info.logical_name = entry.value("name", std::string());
      if (info.logical_name.empty()) continue;

      info.record_path = bin_path;
      info.offset = static_cast<size_t>(entry.value("offset", int64_t{0}));
      info.data_size = static_cast<size_t>(entry.value("size", int64_t{0}));
      info.use_pickle = false;

    const auto param_it = logical_is_parameter.find(info.logical_name);
    if (param_it != logical_is_parameter.end()) {
      info.is_parameter = param_it->second;
    }

    // tensor_values may be keyed by placeholder name (e.g. "p_w1");
    // try logical name first, then fall back to placeholder name.
    const auto meta_it = model.tensor_values.find(info.logical_name);
    if (meta_it != model.tensor_values.end()) {
      info.tensor_meta = meta_it->second;
    } else {
      const auto ph_it = logical_to_placeholder.find(info.logical_name);
      if (ph_it != logical_to_placeholder.end()) {
        const auto ph_meta_it = model.tensor_values.find(ph_it->second);
        if (ph_meta_it != model.tensor_values.end()) {
          info.tensor_meta = ph_meta_it->second;
        } else {
          AddError(result, "MISSING_TENSOR_META",
                   "No tensor metadata found for consolidated weight", info.logical_name);
        }
      } else {
        AddError(result, "MISSING_TENSOR_META",
                 "No tensor metadata found for consolidated weight", info.logical_name);
      }
    }

    logical_name_to_record_[info.logical_name] = std::move(info);
    }
  }
}


bool GeWeightLoader::HasTensor(const std::string &name) const noexcept {
  const std::string logical_name = ResolveLogicalName(name, placeholder_to_logical_name_, logical_name_to_record_);
  return !logical_name.empty() && logical_name_to_record_.find(logical_name) != logical_name_to_record_.end();
}

const TensorRecordInfo *GeWeightLoader::GetTensorRecordInfo(const std::string &name) const noexcept {
  const std::string logical_name = ResolveLogicalName(name, placeholder_to_logical_name_, logical_name_to_record_);
  const auto it = logical_name_to_record_.find(logical_name);
  if (it == logical_name_to_record_.end()) {
    return nullptr;
  }
  return &it->second;
}

const ParsedTensorMeta *GeWeightLoader::GetTensorMeta(const std::string &name) const noexcept {
  const TensorRecordInfo *info = GetTensorRecordInfo(name);
  return info == nullptr ? nullptr : &info->tensor_meta;
}

TensorHolder GeWeightLoader::ReadTensorData(const std::string &name, ValidationResult &result) const {
  const TensorRecordInfo *info = GetTensorRecordInfo(name);
  if (info == nullptr) {
    AddError(result, "UNKNOWN_TENSOR_NAME", "Tensor name not found", name);
    return {};
  }
  if (info->use_pickle) {
    AddError(result, "UNSUPPORTED_PICKLE_WEIGHT", "use_pickle=true is not supported", info->logical_name);
    return {};
  }

  // Lazily load the bin file if not yet cached
  auto cache_it = bin_files_cache_.find(info->record_path);
  if (cache_it == bin_files_cache_.end()) {
    std::ifstream bin_file(info->record_path, std::ios::binary | std::ios::ate);
    if (!bin_file) {
      AddError(result, "MISSING_WEIGHTS_BIN", "Consolidated weights bin not found", info->record_path);
      return {};
    }
    const size_t bin_size = static_cast<size_t>(bin_file.tellg());
    bin_file.seekg(0);
    std::vector<uint8_t> data(bin_size);
    bin_file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bin_size));
    cache_it = bin_files_cache_.emplace(info->record_path, std::move(data)).first;
  }

  const auto &bin_data = cache_it->second;
  if (info->offset + info->data_size > bin_data.size()) {
    AddError(result, "CONSOLIDATED_WEIGHT_RANGE_ERROR",
             "Tensor range exceeds bin file size", info->logical_name);
    return {};
  }
  TensorHolder holder = BuildTensorHolder(
      info->tensor_meta.sizes, info->tensor_meta.dtype, info->tensor_meta.layout,
      bin_data.data() + info->offset, info->data_size,
      result, info->record_path, "WEIGHT_SIZE_MISMATCH");
  holder.requires_grad = info->tensor_meta.requires_grad;
  return holder;
}

TensorHolder GeWeightLoader::RestoreInlineConstTensor(const OrderedJson &value_attr, ValidationResult &result,
                                                       const std::string &context) const {
  if (!value_attr.is_object()) {
    AddError(result, "INVALID_CONST_VALUE", "Inline const value attr must be object", context);
    return {};
  }
  if (!value_attr.contains("t") || !value_attr["t"].is_object()) {
    AddError(result, "UNSUPPORTED_CONST_VALUE", "Inline const value attr must contain a tensor payload 't'", context);
    return {};
  }
  const auto &tensor = value_attr["t"];
  if (tensor.contains("packed_data")) {
    AddError(result, "UNSUPPORTED_CONST_PACKED_DATA", "Inline const packed_data payload is not supported", context);
    return {};
  }
  if (!tensor.contains("desc") || !tensor["desc"].is_object()) {
    AddError(result, "MISSING_CONST_DESC", "Inline const tensor payload missing desc", context);
    return {};
  }
  if (!tensor.contains("data") || !tensor["data"].is_string()) {
    AddError(result, "MISSING_CONST_DATA", "Inline const tensor payload missing base64 data", context);
    return {};
  }

  const auto &desc = tensor["desc"];
  if (!desc.contains("dtype") || !desc["dtype"].is_string()) {
    AddError(result, "MISSING_CONST_DTYPE", "Inline const tensor desc missing dtype", context);
    return {};
  }
  if (!desc.contains("layout") || !desc["layout"].is_string()) {
    AddError(result, "MISSING_CONST_LAYOUT", "Inline const tensor desc missing layout", context);
    return {};
  }
  const std::vector<int64_t> dims = ParseInlineConstDims(desc, result, context);
  if (!result.ok) {
    return {};
  }

  const int64_t dtype = Pt2DtypeFromGeString(desc["dtype"].get<std::string>(), result, context);
  const int64_t layout = Pt2LayoutFromGeString(desc["layout"].get<std::string>(), result, context);
  if (!result.ok) {
    return {};
  }
  std::vector<uint8_t> bytes;
  if (!internal::DecodeBase64(tensor["data"].get<std::string>(), bytes)) {
    AddError(result, "INVALID_CONST_DATA", "Inline const data is not valid base64", context);
    return {};
  }
  if (!result.ok) {
    return {};
  }
  return BuildTensorHolder(dims, dtype, layout, bytes.data(), bytes.size(), result, context,
                           "CONST_SIZE_MISMATCH");
}

}  // namespace td
