#pragma once

#include <optional>
#include <string>
#include <vector>

#include <graph/types.h>

#include "backend/model_types.h"
#include "common/logging/logging.h"

namespace td {

struct GeTensorDescInfo {
    std::vector<int64_t> shape;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
    bool has_shape = false;
    bool has_dtype = false;
    bool has_format = false;
};

class AtenTensorDescMapper {
public:
    ge::DataType MapPt2Dtype(int64_t dtype, ValidationResult& result, const std::string& context) const;

    ge::Format MapPt2Layout(int64_t layout, ValidationResult& result, const std::string& context) const;

    ge::DataType MapGeDtypeString(const std::string& dtype, ValidationResult& result, const std::string& context) const;

    ge::Format MapGeLayoutString(const std::string& layout, ValidationResult& result, const std::string& context) const;

    GeTensorDescInfo MapPt2TensorMeta(const ParsedTensorMeta& meta, ValidationResult& result,
                                      const std::string& context) const;

    GeTensorDescInfo MapAscendIrDesc(const ParsedAscendIrOpDesc* desc, const ParsedTensorMeta* fallback_meta,
                                     ValidationResult& result, const std::string& context) const;
};

} // namespace td
