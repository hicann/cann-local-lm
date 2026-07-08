#include "backend/ge_weight_loader.h"

#include <cstring>
#include <numeric>
#include <string>

#include <nlohmann/json.hpp>

#include "common/abi_compat_utils.h"
#include "common/base64.h"
#include "common/epair_archive_constants.h"
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
    AddError(result, "INVALID_WEIGHT_TENSOR_META", "Tensor meta must be object", context);
    return parsed;
  }
  if (meta.contains("dtype") && meta["dtype"].is_number_integer()) {
    parsed.dtype = meta["dtype"].get<int64_t>();
  } else {
    AddError(result, "MISSING_WEIGHT_DTYPE", "Missing integer dtype", context);
  }
  if (meta.contains("sizes")) {
    parsed.sizes = ParseIntList(meta["sizes"], result, "INVALID_WEIGHT_SIZES", context);
  } else {
    AddError(result, "MISSING_WEIGHT_SIZES", "Missing sizes array", context);
  }
  if (meta.contains("requires_grad") && meta["requires_grad"].is_boolean()) {
    parsed.requires_grad = meta["requires_grad"].get<bool>();
  }
  if (meta.contains("device") && meta["device"].is_object()) {
    const auto &device = meta["device"];
    if (device.contains("type") && device["type"].is_string()) {
      parsed.device.type = device["type"].get<std::string>();
    }
    if (device.contains("index") && device["index"].is_number_integer()) {
      parsed.device.index = device["index"].get<int64_t>();
    }
  }
  if (meta.contains("strides")) {
    parsed.strides = ParseIntList(meta["strides"], result, "INVALID_WEIGHT_STRIDES", context);
  }
  if (meta.contains("storage_offset")) {
    int64_t value = 0;
    if (!TryGetIntField(meta["storage_offset"], value)) {
      AddError(result, "INVALID_WEIGHT_STORAGE_OFFSET", "storage_offset must be integer", context);
    } else {
      parsed.storage_offset = value;
    }
  }
  if (meta.contains("layout") && meta["layout"].is_number_integer()) {
    parsed.layout = meta["layout"].get<int64_t>();
  }
  return parsed;
}

void ParseConfigFile(const EpairArchive &archive, const std::string &config_record, const std::string &base_dir,
                     bool mark_parameter_from_flag, std::unordered_map<std::string, TensorRecordInfo> &out_records,
                     ValidationResult &result) {
  if (!archive.HasRecord(config_record.c_str())) {
    return;
  }
  OrderedJson root;
  try {
    const RecordData record = archive.ReadRecord(config_record.c_str());
    root = OrderedJson::parse(record.AsString());
  } catch (const OrderedJson::parse_error &error) {
    AddError(result, "CONFIG_JSON_PARSE_ERROR", error.what(), config_record);
    return;
  }
  if (!root.contains("config") || !root["config"].is_object()) {
    AddError(result, "INVALID_WEIGHT_CONFIG", "Config file missing 'config' object", config_record);
    return;
  }
  for (const auto &[logical_name, entry] : root["config"].items()) {
    if (!entry.is_object()) {
      AddError(result, "INVALID_WEIGHT_ENTRY", "Config entry must be object", logical_name);
      continue;
    }
    TensorRecordInfo info;
    info.logical_name = logical_name;
    if (!entry.contains("path_name") || !entry["path_name"].is_string()) {
      AddError(result, "MISSING_WEIGHT_PATH", "Config entry missing path_name", logical_name);
      continue;
    }
    info.record_path = base_dir + entry["path_name"].get<std::string>();
    if (entry.contains("is_param") && entry["is_param"].is_boolean()) {
      info.is_parameter = mark_parameter_from_flag ? entry["is_param"].get<bool>() : false;
    }
    if (entry.contains("use_pickle") && entry["use_pickle"].is_boolean()) {
      info.use_pickle = entry["use_pickle"].get<bool>();
    }
    if (!entry.contains("tensor_meta")) {
      AddError(result, "MISSING_WEIGHT_TENSOR_META", "Config entry missing tensor_meta", logical_name);
      continue;
    }
    info.tensor_meta = ParseTensorMeta(entry["tensor_meta"], result, logical_name);
    out_records[logical_name] = std::move(info);
  }
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


GeWeightLoader GeWeightLoader::Create(const EpairArchive &archive, const ParsedModel &model, ValidationResult &result) {
  GeWeightLoader loader;
  loader.archive_ = &archive;
  BuildConsolidatedLookup(model, loader.placeholder_to_logical_name_);

  // Try consolidated mode first, fall back to legacy
  if (archive.HasRecord(epair_spec::kConsolidatedWeightsIndex)) {
    loader.consolidated_ = true;
    loader.LoadConsolidated(archive, model, result);
  } else {
    loader.consolidated_ = false;
    loader.LoadLegacy(archive, model, result);
  }

  for (const auto &[placeholder_name, logical_name] : loader.placeholder_to_logical_name_) {
    if (loader.logical_name_to_record_.find(logical_name) == loader.logical_name_to_record_.end()) {
      AddError(result, "MISSING_PLACEHOLDER_RECORD", "No weight/constant record for placeholder logical name",
               placeholder_name);
    }
  }

  return loader;
}

void GeWeightLoader::LoadLegacy(const EpairArchive &archive, const ParsedModel &model, ValidationResult &result) {
  (void)model;
  ParseConfigFile(archive, epair_spec::kWeightsConfigJson, epair_spec::kWeightsDirectory, true,
                  logical_name_to_record_, result);
  ParseConfigFile(archive, epair_spec::kConstantsConfigJson, epair_spec::kConstantsDirectory, false,
                  logical_name_to_record_, result);
}

void GeWeightLoader::LoadConsolidated(const EpairArchive &archive, const ParsedModel &model, ValidationResult &result) {
  if (!archive.HasRecord(epair_spec::kConsolidatedWeightsBin)) {
    AddError(result, "MISSING_WEIGHTS_BIN", "Consolidated weights.bin not found",
             epair_spec::kConsolidatedWeightsBin);
    return;
  }

  // Read and cache the entire weights.bin once
  const RecordData bin_record = archive.ReadRecord(epair_spec::kConsolidatedWeightsBin);
  if (bin_record.Empty()) {
    AddError(result, "EMPTY_WEIGHTS_BIN", "Consolidated weights.bin is empty",
             epair_spec::kConsolidatedWeightsBin);
    return;
  }
  consolidated_data_.assign(bin_record.data.get(), bin_record.data.get() + bin_record.size);

  // Build logical_name -> is_parameter lookup from model input_specs
  std::unordered_map<std::string, bool> logical_is_parameter;
  for (const auto &spec : model.input_specs) {
    if (spec.kind == PlaceholderKind::Parameter) {
      logical_is_parameter[spec.logical_name] = true;
    } else if (spec.kind == PlaceholderKind::Buffer || spec.kind == PlaceholderKind::TensorConstant) {
      logical_is_parameter[spec.logical_name] = false;
    }
  }

  // Parse the consolidated index JSON
  const RecordData idx_record = archive.ReadRecord(epair_spec::kConsolidatedWeightsIndex);
  OrderedJson root;
  try {
    root = OrderedJson::parse(idx_record.AsString());
  } catch (const OrderedJson::parse_error &error) {
    AddError(result, "CONSOLIDATED_INDEX_PARSE_ERROR", error.what(), epair_spec::kConsolidatedWeightsIndex);
    return;
  }

  if (!root.is_array()) {
    AddError(result, "INVALID_CONSOLIDATED_INDEX", "Consolidated index must be a JSON array",
             epair_spec::kConsolidatedWeightsIndex);
    return;
  }

  for (const auto &entry : root) {
    if (!entry.is_object()) continue;

    TensorRecordInfo info;
    info.logical_name = entry.value("name", std::string());
    if (info.logical_name.empty()) continue;

    info.record_path = epair_spec::kConsolidatedWeightsBin;
    info.offset = static_cast<size_t>(entry.value("offset", int64_t{0}));
    info.data_size = static_cast<size_t>(entry.value("size", int64_t{0}));
    info.use_pickle = false;

    // Populate is_parameter from model input_specs
    const auto param_it = logical_is_parameter.find(info.logical_name);
    if (param_it != logical_is_parameter.end()) {
      info.is_parameter = param_it->second;
    }

    // Populate tensor_meta from model.tensor_values (populated during
    // model JSON parsing). The keys in tensor_values use the same FQN
    // as logical_name in input_specs / consolidated index.
    const auto meta_it = model.tensor_values.find(info.logical_name);
    if (meta_it != model.tensor_values.end()) {
      info.tensor_meta = meta_it->second;
    } else {
      AddError(result, "MISSING_TENSOR_META",
               "No tensor metadata found for consolidated weight", info.logical_name);
    }

    logical_name_to_record_[info.logical_name] = std::move(info);
  }
}

bool GeWeightLoader::HasTensor(const std::string &name) const {
  const std::string logical_name = ResolveLogicalName(name, placeholder_to_logical_name_, logical_name_to_record_);
  return !logical_name.empty() && logical_name_to_record_.find(logical_name) != logical_name_to_record_.end();
}

const TensorRecordInfo *GeWeightLoader::GetTensorRecordInfo(const std::string &name) const {
  const std::string logical_name = ResolveLogicalName(name, placeholder_to_logical_name_, logical_name_to_record_);
  const auto it = logical_name_to_record_.find(logical_name);
  if (it == logical_name_to_record_.end()) {
    return nullptr;
  }
  return &it->second;
}

const ParsedTensorMeta *GeWeightLoader::GetTensorMeta(const std::string &name) const {
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

  if (consolidated_) {
    // Consolidated mode: read from cached weights.bin at offset+size
    if (info->offset + info->data_size > consolidated_data_.size()) {
      AddError(result, "CONSOLIDATED_WEIGHT_RANGE_ERROR",
               "Tensor range exceeds weights.bin size", info->logical_name);
      return {};
    }
    TensorHolder holder = BuildTensorHolder(
        info->tensor_meta.sizes, info->tensor_meta.dtype, info->tensor_meta.layout,
        consolidated_data_.data() + info->offset, info->data_size,
        result, info->record_path, "WEIGHT_SIZE_MISMATCH");
    holder.requires_grad = info->tensor_meta.requires_grad;
    return holder;
  }

  // Legacy mode: read individual record file
  if (!archive_->HasRecord(info->record_path.c_str())) {
    AddError(result, "MISSING_WEIGHT_RECORD", "Archive missing tensor record", info->record_path);
    return {};
  }
  if (!archive_->HasRecord(epair_spec::kByteorder)) {
    AddError(result, "UNSUPPORTED_BYTEORDER", "PT2 archive byteorder must exist and be little-endian",
             info->logical_name);
    return {};
  }
  const RecordData byteorder_record = archive_->ReadRecord(epair_spec::kByteorder);
  const std::string byteorder = byteorder_record.AsString();
  if (byteorder != "little") {
    AddError(result, "UNSUPPORTED_BYTEORDER", "PT2 archive byteorder must exist and be little-endian",
             info->logical_name);
    return {};
  }
  const RecordData record = archive_->ReadRecord(info->record_path.c_str());
  TensorHolder holder = BuildTensorHolder(info->tensor_meta.sizes, info->tensor_meta.dtype, info->tensor_meta.layout,
                                          record.data.get(), record.size, result, info->record_path,
                                          "WEIGHT_SIZE_MISMATCH");
  holder.requires_grad = info->tensor_meta.requires_grad;
  return holder;
}

TensorHolder GeWeightLoader::RestoreInlineConstTensor(const OrderedJson &value_attr, ValidationResult &result,
                                                      const std::string &context) {
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
