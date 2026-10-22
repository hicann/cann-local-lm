/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/utils/string_utils.h
 * @brief String utility functions — inline implementations.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/utils/string_utils.h
 */
#ifndef UTILS_STRING_UTILS_H
#define UTILS_STRING_UTILS_H

#include <string>
#include <vector>
#include <cctype>

class StringUtils {
public:
    static std::string Trim(const std::string& str)
    {
        if (str.empty()) {
            return str;
        }
        auto start = str.find_first_not_of(" \t\n\v\f\r");
        if (start == std::string::npos) {
            return "";
        }
        auto end = str.find_last_not_of(" \t\n\v\f\r");
        return str.substr(start, end - start + 1);
    }

    static std::vector<std::string> Split(const std::string& str, const char delimiter)
    {
        std::vector<std::string> results;
        if (str.empty()) {
            return results;
        }
        size_t start = 0;
        size_t end = str.find(delimiter);
        while (end != std::string::npos) {
            results.push_back(str.substr(start, end - start));
            start = end + 1;
            end = str.find(delimiter, start);
        }
        results.push_back(str.substr(start));
        return results;
    }
};

#endif  // UTILS_STRING_UTILS_H
