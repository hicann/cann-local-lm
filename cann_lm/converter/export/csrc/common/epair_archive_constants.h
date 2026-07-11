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

// Consolidated weights format (single weights.bin + JSON index)
inline constexpr char kConsolidatedWeightsBin[] = "data/weights.bin";
inline constexpr char kConsolidatedWeightsIndex[] = "data/weights_index.json";

inline constexpr char kAnnotatedModelJson[] = "data/ep_json_with_ascend_ir.json";
inline constexpr char kAnnotatedManifestJson[] = "data/epair_manifest.json";

}  // namespace epair_spec
}  // namespace td
