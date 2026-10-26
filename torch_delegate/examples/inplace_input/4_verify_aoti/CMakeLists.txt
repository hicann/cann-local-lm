# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(verify_aoti CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_PKG_DIR "" CACHE PATH "Path to Python torch package directory")
if(TORCH_PKG_DIR STREQUAL "" AND DEFINED ENV{TORCH_PKG_DIR})
    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})
endif()
if(TORCH_PKG_DIR STREQUAL "")
    message(FATAL_ERROR "TORCH_PKG_DIR not specified. Set it to /path/to/site-packages/torch")
endif()
set(TORCH_CMAKE_DIR "${TORCH_PKG_DIR}/share/cmake/Torch")
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)

# Resolve torch runtime library paths at configure time.
# Python wheel 布局因架构而异：x86_64 全在 torch/lib/,
# aarch64 可能将部分 .so 拆分到 torch.libs/。
# 仅在路径存在时加入 RPATH，避免不存在的目录导致问题。
set(TORCH_RUNTIME_DIRS "")
list(APPEND TORCH_RUNTIME_DIRS "${TORCH_PKG_DIR}/lib")
get_filename_component(_TORCH_PARENT "${TORCH_PKG_DIR}" DIRECTORY)
set(_TORCH_EXTERNAL_LIBS "${_TORCH_PARENT}/torch.libs")
if(EXISTS "${_TORCH_EXTERNAL_LIBS}")
    list(APPEND TORCH_RUNTIME_DIRS "${_TORCH_EXTERNAL_LIBS}")
endif()
set(CMAKE_BUILD_RPATH ${TORCH_RUNTIME_DIRS})
set(CMAKE_INSTALL_RPATH ${TORCH_RUNTIME_DIRS})

add_executable(verify_aoti main.cpp)
target_link_libraries(verify_aoti PRIVATE ${TORCH_LIBRARIES})
target_include_directories(verify_aoti PRIVATE ${TORCH_INCLUDE_DIRS})
foreach(_torch_runtime_dir IN LISTS TORCH_RUNTIME_DIRS)
    target_link_options(verify_aoti PRIVATE
        "LINKER:-rpath-link,${_torch_runtime_dir}"
    )
endforeach()
unset(_torch_runtime_dir)
