#include "backend/aten_tensor_desc_mapper.h"

#include "common/logging/logging.h"
#include "common/pt2_type_constants.h"

namespace td {
namespace {}  // namespace

ge::DataType AtenTensorDescMapper::MapPt2Dtype(int64_t dtype, ValidationResult &result,
                                               const std::string &context) const {
  switch (dtype) {
    case pt2::kDtypeByte:
      return ge::DT_UINT8;
    case pt2::kDtypeChar:
      return ge::DT_INT8;
    case pt2::kDtypeShort:
      return ge::DT_INT16;
    case pt2::kDtypeInt:
      return ge::DT_INT32;
    case pt2::kDtypeLong:
      return ge::DT_INT64;
    case pt2::kDtypeHalf:
      return ge::DT_FLOAT16;
    case pt2::kDtypeFloat:
      return ge::DT_FLOAT;
    case pt2::kDtypeDouble:
      return ge::DT_DOUBLE;
    case pt2::kDtypeBool:
      return ge::DT_BOOL;
    case pt2::kDtypeBFloat16:
      return ge::DT_BF16;
    case pt2::kDtypeUInt16:
      return ge::DT_UINT16;
    default:
      AddError(result, "UNSUPPORTED_PT2_DTYPE", "Unsupported PT2 dtype: " + std::to_string(dtype), context);
      return ge::DT_UNDEFINED;
  }
}

ge::Format AtenTensorDescMapper::MapPt2Layout(int64_t layout, ValidationResult &result,
                                              const std::string &context) const {
  if (layout == pt2::kLayoutStrided) {
    return ge::FORMAT_ND;
  }
  AddError(result, "UNSUPPORTED_PT2_LAYOUT", "Unsupported PT2 layout: " + std::to_string(layout), context);
  return ge::FORMAT_ND;
}

ge::DataType AtenTensorDescMapper::MapGeDtypeString(const std::string &dtype, ValidationResult &result,
                                                    const std::string &context) const {
  if (dtype.empty()) {
    return ge::DT_UNDEFINED;
  }
  if (dtype == "DT_FLOAT") {
    return ge::DT_FLOAT;
  }
  if (dtype == "DT_FLOAT16") {
    return ge::DT_FLOAT16;
  }
  if (dtype == "DT_BF16") {
    return ge::DT_BF16;
  }
  if (dtype == "DT_DOUBLE") {
    return ge::DT_DOUBLE;
  }
  if (dtype == "DT_INT64") {
    return ge::DT_INT64;
  }
  if (dtype == "DT_INT32") {
    return ge::DT_INT32;
  }
  if (dtype == "DT_INT16") {
    return ge::DT_INT16;
  }
  if (dtype == "DT_INT8") {
    return ge::DT_INT8;
  }
  if (dtype == "DT_UINT8") {
    return ge::DT_UINT8;
  }
  if (dtype == "DT_UINT16") {
    return ge::DT_UINT16;
  }
  if (dtype == "DT_BOOL") {
    return ge::DT_BOOL;
  }
  AddError(result, "UNSUPPORTED_GE_DTYPE", "Unsupported GE dtype string", context + ":" + dtype);
  return ge::DT_UNDEFINED;
}

ge::Format AtenTensorDescMapper::MapGeLayoutString(const std::string &layout, ValidationResult &result,
                                                   const std::string &context) const {
  if (layout.empty() || layout == "ND") {
    return ge::FORMAT_ND;
  }
  AddError(result, "UNSUPPORTED_GE_LAYOUT", "Unsupported GE layout string", context + ":" + layout);
  return ge::FORMAT_ND;
}

GeTensorDescInfo AtenTensorDescMapper::MapPt2TensorMeta(const ParsedTensorMeta &meta, ValidationResult &result,
                                                         const std::string &context) const {
  GeTensorDescInfo info;
  info.shape = meta.sizes;
  info.has_shape = true;
  info.dtype = MapPt2Dtype(meta.dtype, result, context);
  info.has_dtype = info.dtype != ge::DT_UNDEFINED;
  info.format = MapPt2Layout(meta.layout, result, context);
  info.has_format = true;
  return info;
}

GeTensorDescInfo AtenTensorDescMapper::MapAscendIrDesc(const ParsedAscendIrOpDesc *desc,
                                                       const ParsedTensorMeta *fallback_meta, ValidationResult &result,
                                                       const std::string &context) const {
  GeTensorDescInfo info;
  if (desc != nullptr && desc->raw_json.contains("shape") && desc->raw_json["shape"].is_object()) {
    info.shape = desc->shape;
    info.has_shape = true;
  } else if (fallback_meta != nullptr) {
    info.shape = fallback_meta->sizes;
    info.has_shape = true;
  }

  if (desc != nullptr && !desc->dtype.empty()) {
    info.dtype = MapGeDtypeString(desc->dtype, result, context);
    info.has_dtype = info.dtype != ge::DT_UNDEFINED;
  } else if (fallback_meta != nullptr) {
    info.dtype = MapPt2Dtype(fallback_meta->dtype, result, context);
    info.has_dtype = info.dtype != ge::DT_UNDEFINED;
  }

  if (desc != nullptr && !desc->layout.empty()) {
    info.format = MapGeLayoutString(desc->layout, result, context);
    info.has_format = true;
  } else if (fallback_meta != nullptr) {
    info.format = MapPt2Layout(fallback_meta->layout, result, context);
    info.has_format = true;
  }
  return info;
}

}  // namespace td
