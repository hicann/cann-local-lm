# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# FindGTest.cmake - Find prebuilt GoogleTest library
#
# This module finds GoogleTest from CANN_3RD_LIB_PATH or CMAKE_PREFIX_PATH.
# It does NOT download or build - use build_third_party.sh for that.
#
# Usage:
#   find_package(GTest REQUIRED)
#
# Variables:
#   CANN_3RD_LIB_PATH  - Third-party libraries installation directory
#   GTest_FOUND       - True if found
# -----------------------------------------------------------------------------------------------------------

if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
    set(GTest_FOUND TRUE)
    return()
endif()

# Add CANN_3RD_LIB_PATH to search paths
if(CANN_3RD_LIB_PATH AND NOT "${CANN_3RD_LIB_PATH}" IN_LIST CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${CANN_3RD_LIB_PATH}/gtest")
endif()

# Try CONFIG mode first
find_package(GTest CONFIG
    NAMES GTest gtest
    PATHS
        "${CANN_3RD_LIB_PATH}/gtest"
        "${CMAKE_SOURCE_DIR}/output/third_party/gtest"
    NO_DEFAULT_PATH
    QUIET
)

if(GTest_FOUND)
    message(STATUS "[GTest] Found via CONFIG: ${GTest_DIR}")
    return()
endif()

# Try to find static libraries directly
set(_GTEST_SEARCH_PATHS
    "${CANN_3RD_LIB_PATH}/gtest"
    "${CMAKE_SOURCE_DIR}/output/third_party/gtest"
)

find_path(_GTEST_INCLUDE_DIR
    NAMES gtest/gtest.h
    PATHS ${_GTEST_SEARCH_PATHS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

find_library(_GTEST_LIBRARY
    NAMES gtest
    PATHS ${_GTEST_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

find_library(_GTEST_MAIN_LIBRARY
    NAMES gtest_main
    PATHS ${_GTEST_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if(_GTEST_INCLUDE_DIR AND _GTEST_LIBRARY AND _GTEST_MAIN_LIBRARY)
    message(STATUS "[GTest] Found static libs at: ${_GTEST_INCLUDE_DIR}")

    add_library(GTest::gtest STATIC IMPORTED)
    set_target_properties(GTest::gtest PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_GTEST_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_GTEST_LIBRARY}"
    )

    add_library(GTest::gtest_main STATIC IMPORTED)
    set_target_properties(GTest::gtest_main PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_GTEST_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES GTest::gtest
        IMPORTED_LOCATION "${_GTEST_MAIN_LIBRARY}"
    )

    set(GTest_FOUND TRUE)
    return()
endif()

# Not found
set(GTest_FOUND FALSE)

if(GTest_FIND_REQUIRED)
    message(FATAL_ERROR
        "[GTest] GoogleTest not found.\n"
        "[GTest] Options:\n"
        "[GTest]   1. Build third-party libraries: ./build_third_party.sh\n"
        "[GTest]   2. Specify CANN_3RD_LIB_PATH: -DCANN_3RD_LIB_PATH=/path/to/third_party\n"
        "[GTest]   3. Specify GTest_DIR: -DGTest_DIR=/path/to/gtest/lib/cmake/GTest"
    )
endif()
