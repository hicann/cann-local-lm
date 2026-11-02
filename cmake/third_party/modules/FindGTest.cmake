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
    list(APPEND CMAKE_PREFIX_PATH "${CANN_3RD_LIB_PATH}")
endif()

# 镜像 (lib_cache 扁平布局) 与非镜像 (output/third_party CONFIG 包) 搜索根
set(_GTEST_SEARCH_PATHS
    "${CANN_3RD_LIB_PATH}/lib_cache/gtest"
    "${CANN_3RD_LIB_PATH}/gtest"
    "${CMAKE_SOURCE_DIR}/output/third_party/gtest"
)
if(MIRROR_THIRD_PARTY)
    # 镜像 gtest 变体兜底: compile.sh 按 gcc 版本软链 lib_cache, 若当前变体 gtest 未就位,
    # 回退其它镜像变体 (gcc15/gcc15x86/ubuntu20), 保证任一变体有完整 gtest 即可命中.
    list(APPEND _GTEST_SEARCH_PATHS
        "${CANN_3RD_LIB_PATH}/gcc15/lib_cache/gtest"
        "${CANN_3RD_LIB_PATH}/gcc15x86/lib_cache/gtest"
        "${CANN_3RD_LIB_PATH}/ubuntu20/lib_cache/gtest"
    )
endif()

# Try CONFIG mode first
find_package(GTest CONFIG
    NAMES GTest gtest
    PATHS ${_GTEST_SEARCH_PATHS}
    NO_DEFAULT_PATH
    QUIET
)

if(GTest_FOUND)
    message(STATUS "[GTest] Found via CONFIG: ${GTest_DIR}")
    return()
endif()

# Try to find static libraries directly

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
        "[GTest] search roots: ${_GTEST_SEARCH_PATHS}\n"
        "[GTest] find_path(gtest/gtest.h) -> '${_GTEST_INCLUDE_DIR}'\n"
        "[GTest] find_library(gtest) -> '${_GTEST_LIBRARY}'\n"
        "[GTest] find_library(gtest_main) -> '${_GTEST_MAIN_LIBRARY}'\n"
        "[GTest] Options:\n"
        "[GTest]   1. Build third-party libraries: bash build.sh third_party\n"
        "[GTest]   2. Specify CANN_3RD_LIB_PATH: -DCANN_3RD_LIB_PATH=/path/to/third_party\n"
        "[GTest]   3. Specify GTest_DIR: -DGTest_DIR=/path/to/gtest/lib/cmake/GTest"
    )
endif()
