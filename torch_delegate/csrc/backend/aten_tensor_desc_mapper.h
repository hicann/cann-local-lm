/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <cstdint>

#include <graph/types.h>

#include "backend/model_types.h"
#include "torch_delegate/td_types.h"

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
  Status MapPt2Dtype(int64_t dtype, const std::string &context, ge::DataType &out) const;

  Status MapPt2Layout(int64_t layout, const std::string &context, ge::Format &out) const;

  Status MapGeDtypeString(const std::string &dtype, const std::string &context, ge::DataType &out) const;

  Status MapGeLayoutString(const std::string &layout, const std::string &context, ge::Format &out) const;

  Status MapGeStringToPt2Dtype(const std::string &dtype, const std::string &context, int64_t &out) const;

  Status MapGeStringToPt2Layout(const std::string &layout, const std::string &context, int64_t &out) const;

  Status MapPt2TensorMeta(const ParsedTensorMeta &meta, const std::string &context, GeTensorDescInfo &out) const;

  Status MapAscendIrDesc(const ParsedAscendIrOpDesc *desc, const ParsedTensorMeta *fallback_meta,
                         const std::string &context, GeTensorDescInfo &out) const;
};

}  // namespace td
