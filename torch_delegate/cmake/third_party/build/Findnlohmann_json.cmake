# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Findnlohmann_json.cmake - Find or build nlohmann_json library
#
# This module provides nlohmann_json::nlohmann_json target.
# Priority: gitcode mirror > PyTorch official
# -----------------------------------------------------------------------------------------------------------

if(TARGET nlohmann_json_build)
    return()
endif()

include(ExternalProject)

find_path(NLOHMANN_JSON_INCLUDE
    NAMES nlohmann/json.hpp
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}/json/include
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(NLOHMANN_JSON_INCLUDE)
    message(STATUS "[nlohmann_json] nlohmann_json found at ${NLOHMANN_JSON_INCLUDE}")
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE}"
    )
    ExternalProject_Add(nlohmann_json_build
        SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/json
        CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    return()
endif()

message(STATUS "[nlohmann_json] nlohmann_json not found, finding binary file.")

set(NLOHMANN_JSON_VERSION "3.11.3")
set(NLOHMANN_JSON_DIR "${CMAKE_THIRD_PARTY_LIB_DIR}/json")
set(NLOHMANN_JSON_ARCHIVE "${NLOHMANN_JSON_DIR}/json-${NLOHMANN_JSON_VERSION}.tar.gz")
set(NLOHMANN_JSON_URL "")
set(NLOHMANN_JSON_EXTRA_ARGS "")

if(EXISTS "${NLOHMANN_JSON_ARCHIVE}")
    message(STATUS "[nlohmann_json] ${NLOHMANN_JSON_ARCHIVE} found.")
    set(NLOHMANN_JSON_URL "${NLOHMANN_JSON_ARCHIVE}")
elseif(EXISTS "${NLOHMANN_JSON_DIR}/include/nlohmann/json.hpp")
    message(STATUS "[nlohmann_json] Already installed at ${NLOHMANN_JSON_DIR}")
    ExternalProject_Add(nlohmann_json_build
        SOURCE_DIR ${NLOHMANN_JSON_DIR}
        CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    return()
else()
    message(STATUS "[nlohmann_json] ${NLOHMANN_JSON_ARCHIVE} not found, will download from gitcode.")
    set(NLOHMANN_JSON_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v${NLOHMANN_JSON_VERSION}/json-${NLOHMANN_JSON_VERSION}.tar.gz")
    list(APPEND NLOHMANN_JSON_EXTRA_ARGS
        DOWNLOAD_DIR ${NLOHMANN_JSON_DIR}
    )
endif()

ExternalProject_Add(nlohmann_json_build
    URL ${NLOHMANN_JSON_URL}
    TLS_VERIFY OFF
    ${NLOHMANN_JSON_EXTRA_ARGS}
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DJSON_MultipleHeaders=ON
        -DJSON_BuildTests=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${NLOHMANN_JSON_DIR}
        <SOURCE_DIR>
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    EXCLUDE_FROM_ALL TRUE
)
