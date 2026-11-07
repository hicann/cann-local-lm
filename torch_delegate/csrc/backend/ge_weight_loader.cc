/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/ge_weight_loader.h"

#include <cstring>
#include <numeric>
#include <string>

#include <nlohmann/json.hpp>

#include "backend/parsed_model.h"
#include "common/abi_compat_utils.h"
#include "common/base64.h"
#include "common/epair_archive_constants.h"
#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "common/pt2_type_constants.h"

#include "common/json_parse_utils.h"

namespace td {
namespace {

Status ParseConfigFile(const EpairArchive &archive, const std::string &config_record, const std::string &base_dir,
                       bool mark_parameter_from_flag, std::unordered_map<std::string, TensorRecordInfo> &out_records) {
  if (!archive.HasRecord(config_record.c_str())) {
    return SUCCESS;
  }
  OrderedJson root;
  try {
    const RecordData record = archive.ReadRecord(config_record.c_str());
    root = OrderedJson::parse(record.AsString());
  } catch (const OrderedJson::parse_error &error) {
    TDLOGE("Config JSON parse error, record=%s, error=%s", config_record.c_str(), error.what());
    return INVALID_PARAM;
  }
  if (!root.contains("config") || !root["config"].is_object()) {
    TDLOGE("Config file missing 'config' object, record=%s", config_record.c_str());
    return INVALID_PARAM;
  }
  for (const auto &[logical_name, entry] : root["config"].items()) {
    if (!entry.is_object()) {
      TDLOGE("Config entry must be object, logical_name=%s", logical_name.c_str());
      return INVALID_PARAM;
    }
    TensorRecordInfo info;
    info.logical_name = logical_name;
    if (!entry.contains("path_name") || !entry["path_name"].is_string()) {
      TDLOGE("Config entry missing path_name, logical_name=%s", logical_name.c_str());
      return INVALID_PARAM;
    }
    info.record_path = base_dir + entry["path_name"].get<std::string>();
    if (entry.contains("is_param") && entry["is_param"].is_boolean()) {
      info.is_parameter = mark_parameter_from_flag ? entry["is_param"].get<bool>() : false;
    }
    if (entry.contains("use_pickle") && entry["use_pickle"].is_boolean()) {
      info.use_pickle = entry["use_pickle"].get<bool>();
    }
    if (!entry.contains("tensor_meta")) {
      TDLOGE("Config entry missing tensor_meta, logical_name=%s", logical_name.c_str());
      return INVALID_PARAM;
    }
    TD_CHK_STATUS(ParseTensorMeta(entry["tensor_meta"], logical_name, info.tensor_meta, "WEIGHT"));
    out_records[logical_name] = std::move(info);
  }
  return SUCCESS;
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

// todo 这里还存在权重的拷贝，考虑优化掉
template <typename T>
TensorData DecodeBytesAs(const uint8_t *bytes, size_t byte_size) {
  std::vector<T> values(byte_size / sizeof(T));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes, byte_size);
  }
  return values;
}

Status DecodeTensorData(const uint8_t *bytes, size_t byte_size, int64_t pt2_dtype, const std::string &context,
                        TensorData &out) {
  switch (pt2_dtype) {
    case pt2::kDtypeByte:
    case pt2::kDtypeBool:
      out = DecodeBytesAs<uint8_t>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeChar:
      out = DecodeBytesAs<int8_t>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeShort:
      out = DecodeBytesAs<int16_t>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeInt:
      out = DecodeBytesAs<int32_t>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeLong:
      out = DecodeBytesAs<int64_t>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeHalf:
    case pt2::kDtypeBFloat16:
    case pt2::kDtypeUInt16:
      out = DecodeBytesAs<uint16_t>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeFloat:
      out = DecodeBytesAs<float>(bytes, byte_size);
      return SUCCESS;
    case pt2::kDtypeDouble:
      out = DecodeBytesAs<double>(bytes, byte_size);
      return SUCCESS;
    default:
      TDLOGE("Unsupported PT2 dtype, context=%s, dtype=%ld", context.c_str(), static_cast<long>(pt2_dtype));
      return INVALID_PARAM;
  }
}

Status BuildTensorHolder(const std::vector<int64_t> &dims, int64_t dtype, int64_t layout, const uint8_t *bytes,
                         size_t byte_size, const std::string &context, TensorHolder &out) {
  out = TensorHolder{};
  out.dims = dims;
  out.datatype = dtype;
  out.format = layout;
  const size_t element_size = ElementSizeForPt2Dtype(out.datatype);
  if (element_size == 0) {
    TDLOGE("Unsupported PT2 dtype for element size check, context=%s, dtype=%ld", context.c_str(),
           static_cast<long>(dtype));
    return INVALID_PARAM;
  }
  const size_t expected_bytes = Product(out.dims) * element_size;
  if (byte_size != expected_bytes) {
    TDLOGE("Tensor payload size does not match tensor metadata, context=%s, expected=%zu, actual=%zu", context.c_str(),
           expected_bytes, byte_size);
    return INVALID_PARAM;
  }
  TD_CHK_STATUS(DecodeTensorData(bytes, byte_size, out.datatype, context, out.value));
  return out.HasValue() ? SUCCESS : INTERNAL_ERROR;
}

Status ParseInlineConstDims(const OrderedJson &desc, const std::string &context, std::vector<int64_t> &out) {
  out.clear();
  if (!desc.contains("shape") || !desc["shape"].is_object()) {
    TDLOGE("Inline const tensor desc missing shape object, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  const auto &shape = desc["shape"];
  if (!shape.contains("dim")) {
    return SUCCESS;
  }
  const auto &dim = shape["dim"];
  if (!dim.is_array()) {
    TDLOGE("Inline const shape.dim must be array, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  out.reserve(dim.size());
  for (const auto &item : dim) {
    if (item.is_string()) {
      out.push_back(std::stoll(item.get<std::string>()));
      continue;
    }
    int64_t value = 0;
    if (!TryGetIntField(item, value)) {
      TDLOGE("Expected integer element in inline const shape, context=%s", context.c_str());
      return INVALID_PARAM;
    }
    out.push_back(value);
  }
  return SUCCESS;
}

}  // namespace

Status GeWeightLoader::Create(const EpairArchive &archive, const ParsedModel &model, GeWeightLoader &out) {
  out = GeWeightLoader{};
  out.archive_ = &archive;

  for (const auto &spec : model.InputSpecs()) {
    if (spec.kind == PlaceholderKind::Parameter || spec.kind == PlaceholderKind::Buffer ||
        spec.kind == PlaceholderKind::TensorConstant) {
      out.placeholder_to_logical_name_[spec.placeholder_name] = spec.logical_name;
    }
  }

  TD_CHK_STATUS(ParseConfigFile(archive, epair_spec::kWeightsConfigJson, epair_spec::kWeightsDirectory, true,
                                out.logical_name_to_record_));
  TD_CHK_STATUS(ParseConfigFile(archive, epair_spec::kConstantsConfigJson, epair_spec::kConstantsDirectory, false,
                                out.logical_name_to_record_));

  for (const auto &[placeholder_name, logical_name] : out.placeholder_to_logical_name_) {
    if (out.logical_name_to_record_.find(logical_name) == out.logical_name_to_record_.end()) {
      TDLOGE("No weight/constant record for placeholder logical name, placeholder=%s, logical_name=%s",
             placeholder_name.c_str(), logical_name.c_str());
      return INVALID_PARAM;
    }
  }

  return SUCCESS;
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

Status GeWeightLoader::ReadTensorData(const std::string &name, TensorHolder &out) const {
  out = TensorHolder{};
  const TensorRecordInfo *info = GetTensorRecordInfo(name);
  if (info == nullptr) {
    TDLOGE("Tensor name not found, name=%s", name.c_str());
    return INVALID_PARAM;
  }
  if (info->use_pickle) {
    TDLOGE("use_pickle=true is not supported, logical_name=%s", info->logical_name.c_str());
    return UNIMPLEMENTED;
  }
  if (!archive_->HasRecord(info->record_path.c_str())) {
    TDLOGE("Archive missing tensor record, record_path=%s", info->record_path.c_str());
    return INVALID_PARAM;
  }
  if (!archive_->HasRecord(epair_spec::kByteorder)) {
    TDLOGE("PT2 archive byteorder must exist, logical_name=%s", info->logical_name.c_str());
    return INVALID_PARAM;
  }
  const RecordData byteorder_record = archive_->ReadRecord(epair_spec::kByteorder);
  const std::string byteorder = byteorder_record.AsString();
  if (byteorder != "little") {
    TDLOGE("PT2 archive byteorder must be little-endian, logical_name=%s, byteorder=%s", info->logical_name.c_str(),
           byteorder.c_str());
    return INVALID_PARAM;
  }
  const RecordData record = archive_->ReadRecord(info->record_path.c_str());
  TD_CHK_STATUS(BuildTensorHolder(info->tensor_meta.sizes, info->tensor_meta.dtype, info->tensor_meta.layout,
                                  record.data.get(), record.size, info->record_path, out));
  out.requires_grad = info->tensor_meta.requires_grad;
  return SUCCESS;
}

Status GeWeightLoader::RestoreInlineConstTensor(const OrderedJson &value_attr, const AtenTensorDescMapper &mapper,
                                                const std::string &context, TensorHolder &out) {
  out = TensorHolder{};
  if (!value_attr.is_object()) {
    TDLOGE("Inline const value attr must be object, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  if (!value_attr.contains("t") || !value_attr["t"].is_object()) {
    TDLOGE("Inline const value attr must contain tensor payload 't', context=%s", context.c_str());
    return INVALID_PARAM;
  }
  const auto &tensor = value_attr["t"];
  if (tensor.contains("packed_data")) {
    TDLOGE("Inline const packed_data payload is not supported, context=%s", context.c_str());
    return UNIMPLEMENTED;
  }
  if (!tensor.contains("desc") || !tensor["desc"].is_object()) {
    TDLOGE("Inline const tensor payload missing desc, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  const auto &desc = tensor["desc"];
  if (!desc.contains("dtype") || !desc["dtype"].is_string()) {
    TDLOGE("Inline const tensor desc missing dtype, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  if (!desc.contains("layout") || !desc["layout"].is_string()) {
    TDLOGE("Inline const tensor desc missing layout, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  std::vector<int64_t> dims;
  TD_CHK_STATUS(ParseInlineConstDims(desc, context, dims));

  int64_t dtype = 0;
  int64_t layout = 0;
  TD_CHK_STATUS(mapper.MapGeStringToPt2Dtype(desc["dtype"].get<std::string>(), context, dtype));
  TD_CHK_STATUS(mapper.MapGeStringToPt2Layout(desc["layout"].get<std::string>(), context, layout));
  if (!tensor.contains("data")) {
    TDLOGI("Inline const tensor has no data field, treating as empty tensor, context=%s", context.c_str());
    return BuildTensorHolder(dims, dtype, layout, nullptr, 0, context, out);
  }
  if (!tensor["data"].is_string()) {
    TDLOGE("Inline const tensor data must be string, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  std::vector<uint8_t> bytes;
  if (!internal::DecodeBase64(tensor["data"].get<std::string>(), bytes)) {
    TDLOGE("Inline const data is not valid base64, context=%s", context.c_str());
    return INVALID_PARAM;
  }
  return BuildTensorHolder(dims, dtype, layout, bytes.data(), bytes.size(), context, out);
}

}  // namespace td
