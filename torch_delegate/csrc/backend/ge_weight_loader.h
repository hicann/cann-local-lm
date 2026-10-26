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

#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "backend/aten_tensor_desc_mapper.h"
#include "backend/model_types.h"
#include "epair_archive.h"
#include "torch_delegate/td_types.h"

namespace td {

class ParsedModel;

class GeWeightLoader {
 public:
  static Status Create(const EpairArchive &archive, const ParsedModel &model, GeWeightLoader &out);

  bool HasTensor(const std::string &name) const;
  const TensorRecordInfo *GetTensorRecordInfo(const std::string &name) const;
  const ParsedTensorMeta *GetTensorMeta(const std::string &name) const;

  Status ReadTensorData(const std::string &name, TensorHolder &out) const;

  static Status RestoreInlineConstTensor(const OrderedJson &value_attr, const AtenTensorDescMapper &mapper,
                                         const std::string &context, TensorHolder &out);

 private:
  const EpairArchive *archive_ = nullptr;
  std::unordered_map<std::string, std::string> placeholder_to_logical_name_;
  std::unordered_map<std::string, TensorRecordInfo> logical_name_to_record_;
};

}  // namespace td
