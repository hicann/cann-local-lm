/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/attr_parser.h"

#include <array>
#include <initializer_list>
#include <string>

#include <graph/types.h>

#include "backend/aten_tensor_desc_mapper.h"
#include "common/base64.h"
#include "common/json_parse_utils.h"
#include "common/logging/logging.h"

namespace td {
namespace {

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

std::optional<ge::DataType> ParseJsonDataTypeLike(const OrderedJson &value, const std::string &context) {
  if (const std::optional<int64_t> dtype = ParseJsonIntLike(value); dtype.has_value()) {
    if (dtype.value() < 0 || dtype.value() > static_cast<int64_t>(ge::DT_HIFLOAT4)) {
      TDLOGE("Unsupported ge::DataType value, context=%s", context.c_str());
      return std::nullopt;
    }
    return static_cast<ge::DataType>(dtype.value());
  }
  if (value.is_string()) {
    const td::AtenTensorDescMapper mapper;
    ge::DataType dtype = ge::DT_UNDEFINED;
    if (mapper.MapGeDtypeString(value.get<std::string>(), context, dtype) != SUCCESS) {
      TDLOGE("Unsupported ge::DataType value, context=%s", context.c_str());
      return std::nullopt;
    }
    return dtype;
  }
  TDLOGE("DataType attribute must be int-like or string, context=%s", context.c_str());
  return std::nullopt;
}

ge::AttrValue BuildEmptyListAttr(ListAttrKind kind) {
  ge::AttrValue attr;
  switch (kind) {
    case ListAttrKind::kInt:
      attr.SetAttrValue(std::vector<int64_t>{});
      break;
    case ListAttrKind::kFloat:
      attr.SetAttrValue(std::vector<float>{});
      break;
    case ListAttrKind::kBool:
      attr.SetAttrValue(std::vector<bool>{});
      break;
    case ListAttrKind::kString:
      attr.SetAttrValue(std::vector<ge::AscendString>{});
      break;
    case ListAttrKind::kDataType:
      attr.SetAttrValue(std::vector<ge::DataType>{});
      break;
  }
  return attr;
}

std::optional<ge::AttrValue> ParseListIntAttr(const OrderedJson *payload, const std::string &context, bool &ok) {
  std::vector<int64_t> values;
  values.reserve(payload->size());
  for (const auto &entry : *payload) {
    const std::optional<int64_t> item = ParseJsonIntLike(entry);
    if (!item.has_value()) {
      TDLOGE("ParseListAttrValue failed: ListInt item not int-like, context=%s", context.c_str());
      ok = false;
      return std::nullopt;
    }
    values.push_back(item.value());
  }
  TDLOGD("ParseListAttrValue: int list size=%zu, context=%s", values.size(), context.c_str());
  ge::AttrValue attr;
  attr.SetAttrValue(values);
  return attr;
}

std::optional<ge::AttrValue> ParseListFloatAttr(const OrderedJson *payload, const std::string &context, bool &ok) {
  std::vector<float> values;
  values.reserve(payload->size());
  for (const auto &entry : *payload) {
    const std::optional<double> item = ParseJsonFloatLike(entry);
    if (!item.has_value()) {
      TDLOGE("ParseListAttrValue failed: ListFloat item not number-like, context=%s", context.c_str());
      ok = false;
      return std::nullopt;
    }
    values.push_back(static_cast<float>(item.value()));
  }
  TDLOGD("ParseListAttrValue: float list size=%zu, context=%s", values.size(), context.c_str());
  ge::AttrValue attr;
  attr.SetAttrValue(values);
  return attr;
}

std::optional<ge::AttrValue> ParseListBoolAttr(const OrderedJson *payload, const std::string &context, bool &ok) {
  std::vector<bool> values;
  values.reserve(payload->size());
  for (const auto &entry : *payload) {
    if (!entry.is_boolean()) {
      TDLOGE("ParseListAttrValue failed: ListBool item not boolean, context=%s", context.c_str());
      ok = false;
      return std::nullopt;
    }
    values.push_back(entry.get<bool>());
  }
  TDLOGD("ParseListAttrValue: bool list size=%zu, context=%s", values.size(), context.c_str());
  ge::AttrValue attr;
  attr.SetAttrValue(values);
  return attr;
}

std::optional<ge::AttrValue> ParseListStringAttr(const OrderedJson *payload, const std::string &context, bool &ok) {
  std::vector<ge::AscendString> values;
  values.reserve(payload->size());
  for (size_t index = 0; index < payload->size(); ++index) {
    const auto &entry = (*payload)[index];
    if (!entry.is_string()) {
      TDLOGE("ParseListAttrValue failed: ListString item not string, context=%s", context.c_str());
      ok = false;
      return std::nullopt;
    }
    std::string decoded;
    if (!td::internal::DecodeBase64ToString(entry.get<std::string>(), decoded)) {
      TDLOGE("ParseListAttrValue failed: ListString base64 decode error, context=%s, index=%zu", context.c_str(),
             index);
      ok = false;
      return std::nullopt;
    }
    values.emplace_back(decoded.c_str());
  }
  TDLOGD("ParseListAttrValue: string list size=%zu, context=%s", values.size(), context.c_str());
  ge::AttrValue attr;
  attr.SetAttrValue(values);
  return attr;
}

std::optional<ge::AttrValue> ParseListDataTypeAttr(const OrderedJson *payload, const std::string &context, bool &ok) {
  std::vector<ge::DataType> values;
  values.reserve(payload->size());
  for (size_t index = 0; index < payload->size(); ++index) {
    const std::optional<ge::DataType> item =
        ParseJsonDataTypeLike((*payload)[index], context + ":dt[" + std::to_string(index) + "]");
    if (!item.has_value()) {
      TDLOGE("ParseListAttrValue failed: DataType parse error, context=%s, index=%zu", context.c_str(), index);
      ok = false;
      return std::nullopt;
    }
    values.push_back(item.value());
  }
  TDLOGD("ParseListAttrValue: datatype list size=%zu, context=%s", values.size(), context.c_str());
  ge::AttrValue attr;
  attr.SetAttrValue(values);
  return attr;
}

std::optional<ge::AttrValue> ParseListAttrValue(const OrderedJson &value, const std::string &context, bool &ok) {
  const OrderedJson *val_type = FindField(value, {"valType", "val_type"});
  const ListAttrTypeInfo *info =
      val_type != nullptr && val_type->is_string() ? FindListAttrTypeInfo(val_type->get<std::string>()) : nullptr;
  if (info == nullptr) {
    TDLOGW("ParseListAttrValue: unsupported valType, skipping attr, context=%s, json=%s", context.c_str(),
           value.dump().c_str());
    return std::nullopt;
  }
  const OrderedJson *payload = FindField(value, {info->payload_key});
  if (payload == nullptr) {
    TDLOGD("ParseListAttrValue: empty list, kind=%d, context=%s", static_cast<int>(info->kind), context.c_str());
    return BuildEmptyListAttr(info->kind);
  }
  if (!payload->is_array()) {
    TDLOGE("ParseListAttrValue failed: payload is not array, context=%s", context.c_str());
    ok = false;
    return std::nullopt;
  }
  switch (info->kind) {
    case ListAttrKind::kInt:
      return ParseListIntAttr(payload, context, ok);
    case ListAttrKind::kFloat:
      return ParseListFloatAttr(payload, context, ok);
    case ListAttrKind::kBool:
      return ParseListBoolAttr(payload, context, ok);
    case ListAttrKind::kString:
      return ParseListStringAttr(payload, context, ok);
    case ListAttrKind::kDataType:
      return ParseListDataTypeAttr(payload, context, ok);
  }
  return std::nullopt;
}

}  // namespace

bool IsOptionalUnfed(const ParsedAscendIrOpDesc &desc) {
  if (!desc.attr.is_object() || !desc.attr.contains("_is_unfed_optional")) {
    return false;
  }
  const auto &flag = desc.attr["_is_unfed_optional"];
  return flag.is_object() && flag.contains("i") && ParseJsonIntLike(flag["i"]) == 1;
}

std::optional<ge::AttrValue> ParseAttrValue(const OrderedJson &value, const std::string &context, bool &ok) {
  ge::AttrValue attr_value;
  if (!value.is_object()) {
    TDLOGW("ParseAttrValue: payload is not object, skipping, context=%s, json=%s", context.c_str(),
           value.dump().c_str());
    return std::nullopt;
  }
  if (value.contains("list")) {
    auto result_value = ParseListAttrValue(value["list"], context, ok);
    if (result_value.has_value()) {
      TDLOGD("ParseAttrValue: list attribute parsed, context=%s", context.c_str());
    }
    return result_value;
  }
  if (value.contains("b") && value["b"].is_boolean()) {
    attr_value.SetAttrValue(value["b"].get<bool>());
    TDLOGD("ParseAttrValue: bool attribute, context=%s, value=%s", context.c_str(),
           value["b"].get<bool>() ? "true" : "false");
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
      TDLOGE("ParseAttrValue failed: base64 decode error, context=%s", context.c_str());
      ok = false;
      return std::nullopt;
    }
    attr_value.SetAttrValue(ge::AscendString(decoded.c_str()));
    TDLOGD("ParseAttrValue: string attribute, context=%s, value=%s", context.c_str(), decoded.c_str());
    return attr_value;
  }
  if (value.contains("dt")) {
    const std::optional<ge::DataType> dtype = ParseJsonDataTypeLike(value["dt"], context);
    if (dtype.has_value()) {
      attr_value.SetAttrValue(dtype.value());
      TDLOGD("ParseAttrValue: datatype attribute, context=%s, value=%d", context.c_str(),
             static_cast<int>(dtype.value()));
    } else {
      TDLOGE("ParseAttrValue failed: datatype parse error, context=%s", context.c_str());
      ok = false;
      return std::nullopt;
    }
    return attr_value;
  }
  TDLOGW("ParseAttrValue: unsupported attribute payload, skipping, context=%s, json=%s", context.c_str(),
         value.dump().c_str());
  return std::nullopt;
}

}  // namespace td
