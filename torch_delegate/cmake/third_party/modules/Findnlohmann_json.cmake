# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Findnlohmann_json.cmake - Find prebuilt nlohmann_json library
#
# This module finds nlohmann_json from CANN_3RD_LIB_PATH or CMAKE_PREFIX_PATH.
# It does NOT download or build - use build_third_party.sh for that.
#
# Usage:
#   find_package(nlohmann_json REQUIRED)
#
# Variables:
#   CANN_3RD_LIB_PATH      - Third-party libraries installation directory
#   nlohmann_json_FOUND   - True if found
# -----------------------------------------------------------------------------------------------------------

if(TARGET nlohmann_json::nlohmann_json)
    set(nlohmann_json_FOUND TRUE)
    return()
endif()

# Add CANN_3RD_LIB_PATH to CMAKE_PREFIX_PATH for searching
if(CANN_3RD_LIB_PATH AND NOT "${CANN_3RD_LIB_PATH}" IN_LIST CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${CANN_3RD_LIB_PATH}/json")
endif()

# Try to find nlohmann_json using CONFIG mode first
find_package(nlohmann_json CONFIG
    NAMES nlohmann_json
    PATHS
        "${CANN_3RD_LIB_PATH}/json"
        "${CMAKE_SOURCE_DIR}/output/third_party/json"
    NO_DEFAULT_PATH
    QUIET
)

if(nlohmann_json_FOUND)
    message(STATUS "[nlohmann_json] Found via CONFIG: ${nlohmann_json_DIR}")
    return()
endif()

# Try to find header file directly
find_path(_NLOHMANN_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS
        "${CANN_3RD_LIB_PATH}/json/include"
        "${CMAKE_SOURCE_DIR}/output/third_party/json/include"
    NO_DEFAULT_PATH
)

if(_NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "[nlohmann_json] Found header at: ${_NLOHMANN_JSON_INCLUDE_DIR}")
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_NLOHMANN_JSON_INCLUDE_DIR}"
    )
    set(nlohmann_json_FOUND TRUE)
    return()
endif()

# Not found
set(nlohmann_json_FOUND FALSE)

if(nlohmann_json_FIND_REQUIRED)
    message(FATAL_ERROR
        "[nlohmann_json] nlohmann_json not found.\n"
        "[nlohmann_json] Options:\n"
        "[nlohmann_json]   1. Build third-party libraries: ./build_third_party.sh\n"
        "[nlohmann_json]   2. Specify CANN_3RD_LIB_PATH: -DCANN_3RD_LIB_PATH=/path/to/third_party\n"
        "[nlohmann_json]   3. Specify nlohmann_json_DIR: -Dnlohmann_json_DIR=/path/to/json/share/cmake/nlohmann_json"
    )
endif()
