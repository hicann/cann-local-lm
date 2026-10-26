/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "backend/aten_tensor_desc_mapper.h"

#include "common/logging/assert_macros.h"
#include "common/logging/logging.h"
#include "common/pt2_type_constants.h"
#include "torch_delegate/td_types.h"

namespace td {

Status AtenTensorDescMapper::MapGeStringToPt2Dtype(const std::string &dtype, const std::string &context,
                                                   int64_t &out) const {
  if (dtype == "DT_UINT8") {
    out = pt2::kDtypeByte;
    return SUCCESS;
  }
  if (dtype == "DT_INT8") {
    out = pt2::kDtypeChar;
    return SUCCESS;
  }
  if (dtype == "DT_INT16") {
    out = pt2::kDtypeShort;
    return SUCCESS;
  }
  if (dtype == "DT_INT32") {
    out = pt2::kDtypeInt;
    return SUCCESS;
  }
  if (dtype == "DT_INT64") {
    out = pt2::kDtypeLong;
    return SUCCESS;
  }
  if (dtype == "DT_FLOAT16") {
    out = pt2::kDtypeHalf;
    return SUCCESS;
  }
  if (dtype == "DT_FLOAT") {
    out = pt2::kDtypeFloat;
    return SUCCESS;
  }
  if (dtype == "DT_DOUBLE") {
    out = pt2::kDtypeDouble;
    return SUCCESS;
  }
  if (dtype == "DT_BOOL") {
    out = pt2::kDtypeBool;
    return SUCCESS;
  }
  if (dtype == "DT_BF16") {
    out = pt2::kDtypeBFloat16;
    return SUCCESS;
  }
  if (dtype == "DT_UINT16") {
    out = pt2::kDtypeUInt16;
    return SUCCESS;
  }
  TDLOGE("Unsupported inline const dtype, context=%s, dtype=%s", context.c_str(), dtype.c_str());
  return INVALID_PARAM;
}

Status AtenTensorDescMapper::MapGeStringToPt2Layout(const std::string &layout, const std::string &context,
                                                    int64_t &out) const {
  if (layout.empty() || layout == "ND") {
    out = pt2::kLayoutStrided;
    return SUCCESS;
  }
  TDLOGE("Unsupported inline const layout, context=%s, layout=%s", context.c_str(), layout.c_str());
  return INVALID_PARAM;
}

Status AtenTensorDescMapper::MapPt2Dtype(int64_t dtype, const std::string &context, ge::DataType &out) const {
  switch (dtype) {
    case pt2::kDtypeByte:
      out = ge::DT_UINT8;
      return SUCCESS;
    case pt2::kDtypeChar:
      out = ge::DT_INT8;
      return SUCCESS;
    case pt2::kDtypeShort:
      out = ge::DT_INT16;
      return SUCCESS;
    case pt2::kDtypeInt:
      out = ge::DT_INT32;
      return SUCCESS;
    case pt2::kDtypeLong:
      out = ge::DT_INT64;
      return SUCCESS;
    case pt2::kDtypeHalf:
      out = ge::DT_FLOAT16;
      return SUCCESS;
    case pt2::kDtypeFloat:
      out = ge::DT_FLOAT;
      return SUCCESS;
    case pt2::kDtypeDouble:
      out = ge::DT_DOUBLE;
      return SUCCESS;
    case pt2::kDtypeBool:
      out = ge::DT_BOOL;
      return SUCCESS;
    case pt2::kDtypeBFloat16:
      out = ge::DT_BF16;
      return SUCCESS;
    case pt2::kDtypeUInt16:
      out = ge::DT_UINT16;
      return SUCCESS;
    default:
      TDLOGE("Unsupported PT2 dtype: %ld, context=%s", static_cast<long>(dtype), context.c_str());
      return INVALID_PARAM;
  }
}

Status AtenTensorDescMapper::MapPt2Layout(int64_t layout, const std::string &context, ge::Format &out) const {
  if (layout == pt2::kLayoutStrided) {
    out = ge::FORMAT_ND;
    return SUCCESS;
  }
  TDLOGE("Unsupported PT2 layout: %ld, context=%s", static_cast<long>(layout), context.c_str());
  return INVALID_PARAM;
}

Status AtenTensorDescMapper::MapGeDtypeString(const std::string &dtype, const std::string &context,
                                              ge::DataType &out) const {
  if (dtype.empty()) {
    out = ge::DT_UNDEFINED;
    return SUCCESS;
  }
  if (dtype == "DT_FLOAT") {
    out = ge::DT_FLOAT;
    return SUCCESS;
  }
  if (dtype == "DT_FLOAT16") {
    out = ge::DT_FLOAT16;
    return SUCCESS;
  }
  if (dtype == "DT_BF16") {
    out = ge::DT_BF16;
    return SUCCESS;
  }
  if (dtype == "DT_DOUBLE") {
    out = ge::DT_DOUBLE;
    return SUCCESS;
  }
  if (dtype == "DT_INT64") {
    out = ge::DT_INT64;
    return SUCCESS;
  }
  if (dtype == "DT_INT32") {
    out = ge::DT_INT32;
    return SUCCESS;
  }
  if (dtype == "DT_INT16") {
    out = ge::DT_INT16;
    return SUCCESS;
  }
  if (dtype == "DT_INT8") {
    out = ge::DT_INT8;
    return SUCCESS;
  }
  if (dtype == "DT_UINT8") {
    out = ge::DT_UINT8;
    return SUCCESS;
  }
  if (dtype == "DT_UINT16") {
    out = ge::DT_UINT16;
    return SUCCESS;
  }
  if (dtype == "DT_BOOL") {
    out = ge::DT_BOOL;
    return SUCCESS;
  }
  TDLOGE("Unsupported GE dtype string, context=%s, dtype=%s", context.c_str(), dtype.c_str());
  return INVALID_PARAM;
}

Status AtenTensorDescMapper::MapGeLayoutString(const std::string &layout, const std::string &context,
                                               ge::Format &out) const {
  if (layout.empty() || layout == "ND") {
    out = ge::FORMAT_ND;
    return SUCCESS;
  }
  TDLOGE("Unsupported GE layout string, context=%s, layout=%s", context.c_str(), layout.c_str());
  return INVALID_PARAM;
}

Status AtenTensorDescMapper::MapPt2TensorMeta(const ParsedTensorMeta &meta, const std::string &context,
                                              GeTensorDescInfo &out) const {
  out = GeTensorDescInfo{};
  out.shape = meta.sizes;
  out.has_shape = true;
  TD_CHK_STATUS(MapPt2Dtype(meta.dtype, context, out.dtype));
  out.has_dtype = out.dtype != ge::DT_UNDEFINED;
  TD_CHK_STATUS(MapPt2Layout(meta.layout, context, out.format));
  out.has_format = true;
  return SUCCESS;
}

Status AtenTensorDescMapper::MapAscendIrDesc(const ParsedAscendIrOpDesc *desc, const ParsedTensorMeta *fallback_meta,
                                             const std::string &context, GeTensorDescInfo &out) const {
  out = GeTensorDescInfo{};
  if (desc != nullptr && desc->raw_json.contains("shape") && desc->raw_json["shape"].is_object()) {
    out.shape = desc->shape;
    out.has_shape = true;
  } else if (fallback_meta != nullptr) {
    out.shape = fallback_meta->sizes;
    out.has_shape = true;
  }

  if (desc != nullptr && !desc->dtype.empty()) {
    TD_CHK_STATUS(MapGeDtypeString(desc->dtype, context, out.dtype));
    out.has_dtype = out.dtype != ge::DT_UNDEFINED;
  } else if (fallback_meta != nullptr) {
    TD_CHK_STATUS(MapPt2Dtype(fallback_meta->dtype, context, out.dtype));
    out.has_dtype = out.dtype != ge::DT_UNDEFINED;
  }

  if (desc != nullptr && !desc->layout.empty()) {
    TD_CHK_STATUS(MapGeLayoutString(desc->layout, context, out.format));
    out.has_format = true;
  } else if (fallback_meta != nullptr) {
    TD_CHK_STATUS(MapPt2Layout(fallback_meta->layout, context, out.format));
    out.has_format = true;
  }
  return SUCCESS;
}

}  // namespace td
