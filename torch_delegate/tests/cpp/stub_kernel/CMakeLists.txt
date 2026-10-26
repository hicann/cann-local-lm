# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(stub_kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find libtorch (user must specify TORCH_PKG_DIR as Python torch package directory)
# Example: cmake -DTORCH_PKG_DIR=/path/to/site-packages/torch ..
set(TORCH_PKG_DIR "" CACHE PATH "Path to Python torch package directory")
if(TORCH_PKG_DIR STREQUAL "" AND DEFINED ENV{TORCH_PKG_DIR})
    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})
endif()
if(TORCH_PKG_DIR STREQUAL "")
    message(FATAL_ERROR "TORCH_PKG_DIR must be specified (cmake -DTORCH_PKG_DIR=/path/to/site-packages/torch or set env TORCH_PKG_DIR)")
endif()
set(TORCH_CMAKE_DIR "${TORCH_PKG_DIR}/share/cmake/Torch")
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)

# Find torch_npu (user must specify TORCH_NPU_DIR or it can be auto-detected)
if(NOT TORCH_NPU_DIR)
    if(DEFINED ENV{TORCH_NPU_DIR})
        set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})
    endif()
endif()
if(NOT TORCH_NPU_DIR)
    message(FATAL_ERROR "TORCH_NPU_DIR must be specified (cmake -DTORCH_NPU_DIR=... or set env TORCH_NPU_DIR)")
endif()

message(STATUS "Torch found: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch NPU dir: ${TORCH_NPU_DIR}")

# Find CANN SDK (for ACL headers)
if(NOT CANN_ROOT)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    endif()
endif()
if(NOT CANN_ROOT)
    message(FATAL_ERROR "CANN_ROOT must be specified (cmake -DCANN_ROOT=... or set env ASCEND_HOME_PATH)")
endif()
message(STATUS "CANN root: ${CANN_ROOT}")

# Derive pytorch source root from TORCH_NPU_DIR (parent of libtorch_npu/)
# Required for NPUQueue.h -> "torch_npu/csrc/logging/LogContext.h"
get_filename_component(PYTORCH_SRC_DIR "${TORCH_NPU_DIR}" DIRECTORY)
message(STATUS "PyTorch source dir: ${PYTORCH_SRC_DIR}")

# Find libtorch_npu (provides NPUStream symbols at runtime)
find_library(TORCH_NPU_LIB
    NAMES torch_npu
    PATHS ${TORCH_NPU_DIR}/lib
    NO_DEFAULT_PATH
)
if(NOT TORCH_NPU_LIB)
    message(FATAL_ERROR "libtorch_npu.so not found in ${TORCH_NPU_DIR}/lib")
endif()
message(STATUS "Torch NPU library: ${TORCH_NPU_LIB}")

add_library(stub_kernel SHARED stub_kernel.cc)

target_include_directories(stub_kernel PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(stub_kernel PRIVATE ${TORCH_NPU_DIR}/include)
target_include_directories(stub_kernel PRIVATE ${CANN_ROOT}/include)
target_include_directories(stub_kernel PRIVATE ${PYTORCH_SRC_DIR})

target_link_libraries(stub_kernel PRIVATE ${TORCH_LIBRARIES} ${TORCH_NPU_LIB})

# ABI=1 (match libtorch)
target_compile_definitions(stub_kernel PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

# Basic warnings
target_compile_options(stub_kernel PRIVATE -Wall -Wextra)

# Output to build directory
set_target_properties(stub_kernel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_RPATH "${TORCH_NPU_DIR}/lib"
    BUILD_RPATH "${TORCH_NPU_DIR}/lib"
)

message(STATUS "Output: ${CMAKE_CURRENT_BINARY_DIR}/libstub_kernel.so")
