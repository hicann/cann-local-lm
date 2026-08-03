#pragma once

namespace td {
namespace epair_spec {

// Consolidated weights format (single weights.bin + JSON index)
inline constexpr char kConsolidatedWeightsBin[] = "data/weights.bin";
inline constexpr char kConsolidatedWeightsIndex[] = "data/weights_index.json";

inline constexpr char kAnnotatedModelJson[] = "data/ep_json_with_ascend_ir.json";

}  // namespace epair_spec
}  // namespace td
