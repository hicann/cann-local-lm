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

#if __has_include(<pt2_archive_constants.h>)
#include <pt2_archive_constants.h>
#elif __has_include(<torch/csrc/export/pt2_archive_constants.h>)
#include <torch/csrc/export/pt2_archive_constants.h>
#elif __has_include(<torch/export/pt2_archive_constants.h>)
#include <torch/export/pt2_archive_constants.h>
#endif

namespace td {
namespace epair_spec {

inline constexpr char kArchiveFormat[] = "archive_format";
inline constexpr char kArchiveVersion[] = "archive_version";
inline constexpr char kByteorder[] = "byteorder";
inline constexpr char kModelsDirectory[] = "models/";
inline constexpr char kModelJson[] = "models/model.json";
inline constexpr char kWeightsDirectory[] = "data/weights/";
inline constexpr char kWeightsConfigJson[] = "data/weights/model_weights_config.json";
inline constexpr char kConstantsDirectory[] = "data/constants/";
inline constexpr char kConstantsConfigJson[] = "data/constants/model_constants_config.json";

inline constexpr char kAnnotatedModelJson[] = "extra/ep_json_with_ascend_ir.json";
inline constexpr char kAnnotatedManifestJson[] = "extra/epair_manifest.json";

}  // namespace epair_spec
}  // namespace td
