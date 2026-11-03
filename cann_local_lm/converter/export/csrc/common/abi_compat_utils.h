#pragma once

#include <filesystem>
#include <string>

#include <graph/ascend_string.h>

namespace td::internal {

inline std::string CharToString(const char* value) { return value == nullptr ? std::string() : std::string(value); }

inline std::string AscendStringToString(const ge::AscendString& value)
{
    const char* raw = value.GetString();
    return raw == nullptr ? std::string() : std::string(raw);
}

inline ge::AscendString StringToAscendString(const std::string& value) { return ge::AscendString(value.c_str()); }

inline std::filesystem::path CharToPath(const char* value) { return std::filesystem::path(CharToString(value)); }

inline const char* PathToChar(const std::filesystem::path& value) { return value.c_str(); }

} // namespace td::internal
