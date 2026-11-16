# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(torch_delegate CXX)

include(CMakePackageConfigHelpers)

set(TORCH_DELEGATE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(TORCH_DELEGATE_PACKAGE_VERSION "0.1.0" CACHE STRING "TorchDelegate package version")

# ---------------------------------------------------------------------------
# CANN_3RD_LIB_PATH — optional; points to third-party libraries installation directory
#
# If specified, libtorch, nlohmann_json, and gtest will be searched from this path.
# Default: ${CMAKE_SOURCE_DIR}/output/third_party
# ---------------------------------------------------------------------------
set(CANN_3RD_LIB_PATH "" CACHE PATH "Third-party libraries installation directory (libtorch, json, gtest)")

if(CANN_3RD_LIB_PATH STREQUAL "")
    set(CANN_3RD_LIB_PATH "${TORCH_DELEGATE_SOURCE_DIR}/output/third_party")
    message(STATUS "CANN_3RD_LIB_PATH not specified, using default: ${CANN_3RD_LIB_PATH}")
else()
    message(STATUS "CANN_3RD_LIB_PATH: ${CANN_3RD_LIB_PATH}")
endif()

# ---------------------------------------------------------------------------
# Torch_DIR — optional; explicit libtorch CMake config directory (highest priority)
#
# Priority for libtorch lookup is:
#   1. -DTorch_DIR=<...>
#   2. TORCH_PKG_DIR environment variable (Python torch package dir)
#   3. Auto-detected Python torch package
#   4. CANN_3RD_LIB_PATH/libtorch
# ---------------------------------------------------------------------------
set(Torch_DIR "" CACHE PATH "libtorch CMake config directory (TorchConfig.cmake)")

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(BUILD_DELEGATE "Build delegate library (torch_delegate_backend)" ON)
option(BUILD_TESTS "Build tests and test fixtures" OFF)

# ---------------------------------------------------------------------------
# Common compile/link flags (required for ALL targets)
# ---------------------------------------------------------------------------
option(ENABLE_ASAN "Build with AddressSanitizer" OFF)

# 生成 compile_commands.json，供 clangd / IDE 代码导航使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 基础告警选项：不依赖 ASAN 等运行时配置，供必须规避 ASAN 的特殊 target
# （如会被 Python 进程 dlopen 的 stub）单独引用。
set(DELEGATE_COMMON_WARNING_FLAGS
    -Wall
    -Wextra
    -Werror
)

set(DELEGATE_COMMON_FLAGS ${DELEGATE_COMMON_WARNING_FLAGS})

if(ENABLE_ASAN)
    list(APPEND DELEGATE_COMMON_FLAGS
        -fsanitize=address
        -fno-omit-frame-pointer
    )
endif()

list(APPEND CMAKE_MODULE_PATH
    "${TORCH_DELEGATE_SOURCE_DIR}/cmake/modules"
    "${TORCH_DELEGATE_SOURCE_DIR}/cmake/third_party/modules"
)

include(TorchDelegateCannCommon)

torch_delegate_cann_include_search_paths(TORCH_DELEGATE_CANN_INCLUDE_SEARCH_PATHS)
find_path(TORCH_DELEGATE_ACL_RT_HEADER_DIR
    NAMES acl/acl_rt.h
    PATHS ${TORCH_DELEGATE_CANN_INCLUDE_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

if(NOT TORCH_DELEGATE_ACL_RT_HEADER_DIR)
    message(FATAL_ERROR
        "CANN header not found: acl/acl_rt.h\n"
        "ASCEND_INSTALL_PATH: ${ASCEND_INSTALL_PATH}\n"
        "Please check the path or source CANN environment."
    )
endif()

find_package(libtorch_cpu REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

find_package(ascendcl REQUIRED)
find_package(acl_rt REQUIRED)
find_package(graph REQUIRED)
find_package(graph_base REQUIRED)
find_package(metadef REQUIRED)
find_package(runtime REQUIRED)
find_package(securec REQUIRED)
find_package(ascend_hal REQUIRED)

# 测试打桩, 确保测试场景链接的是 ge_stub
set(TORCH_DELEGATE_GE_RUNNER_V2_LINK_TARGET ge_runner_v2)
if(BUILD_TESTS)
    include(TorchDelegateTestStubs)
    torch_delegate_add_ge_runner_v2_stub()
    torch_delegate_add_epair_loader_c_stub()
    set(TORCH_DELEGATE_GE_RUNNER_V2_LINK_TARGET ge_runner_v2_stub)
else()
    find_package(ge_runner_v2 REQUIRED)
endif()


enable_testing()

if(BUILD_DELEGATE)
    add_subdirectory(csrc)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NOT BUILD_TESTS)
    set(TORCH_DELEGATE_CMAKE_INSTALL_DIR "lib/cmake/TorchDelegate")

    configure_package_config_file(
        "${TORCH_DELEGATE_SOURCE_DIR}/cmake/TorchDelegateConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/TorchDelegateConfig.cmake"
        INSTALL_DESTINATION "${TORCH_DELEGATE_CMAKE_INSTALL_DIR}"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/TorchDelegateConfigVersion.cmake"
        VERSION "${TORCH_DELEGATE_PACKAGE_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/TorchDelegateConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TorchDelegateConfigVersion.cmake"
        DESTINATION "${TORCH_DELEGATE_CMAKE_INSTALL_DIR}"
    )

    # Only install Find modules that are still referenced by Config.cmake:
    #   - TorchDelegateCannCommon.cmake (included by graph/graph_base/metadef)
    #   - Findgraph / Findgraph_base / Findmetadef (public deps in exported target)
    # All other CANN libs (acl_rt, ge_runner_v2, runtime, securec, ascend_hal)
    # and libtorch_cpu are private to libtorch_delegate_backend.so — they are
    # resolved at runtime via DT_NEEDED + LD_LIBRARY_PATH, not needed at
    # consumer compile/link time.
    install(FILES
        "${TORCH_DELEGATE_SOURCE_DIR}/cmake/modules/TorchDelegateCannCommon.cmake"
        "${TORCH_DELEGATE_SOURCE_DIR}/cmake/modules/Findgraph.cmake"
        "${TORCH_DELEGATE_SOURCE_DIR}/cmake/modules/Findgraph_base.cmake"
        "${TORCH_DELEGATE_SOURCE_DIR}/cmake/modules/Findmetadef.cmake"
        DESTINATION "${TORCH_DELEGATE_CMAKE_INSTALL_DIR}/modules"
    )
endif()
