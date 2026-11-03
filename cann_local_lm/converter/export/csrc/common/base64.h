#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace td::internal {

bool DecodeBase64(const std::string& encoded, std::vector<uint8_t>& decoded);

bool DecodeBase64ToString(const std::string& encoded, std::string& decoded);

} // namespace td::internal
