# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.14)
project(inplace_verify CXX)

# ============================================
# Ascend CANN 配置
# ============================================
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/cann)
endif ()

set(common_compile_options -std=c++17 -Wall)

message(STATUS "Ascend Toolkit path: ${ASCEND_PATH}")

set(ASCEND_INCLUDE_DIRS ${CMAKE_SOURCE_DIR} ${ASCEND_PATH}/include)
set(ASCEND_LIB_DIRS ${ASCEND_PATH}/lib64)

set(ASCEND_LIBS ge_runner_v2 graph graph_base acl_rt)

foreach (LIB ${ASCEND_LIBS})
    find_library(${LIB}_PATH ${LIB} PATHS ${ASCEND_LIB_DIRS})
    if (NOT ${LIB}_PATH)
        message(FATAL_ERROR "Required library not found: ${LIB}")
    else ()
        message(STATUS "Found library: ${LIB} at ${${LIB}_PATH}")
    endif ()
endforeach ()

# ============================================
# TorchDelegate 配置
# ============================================
set(TorchDelegate_ROOT_DIR "" CACHE PATH "Path to the TorchDelegate install directory")
set(CANN_ROOT "${ASCEND_PATH}" CACHE PATH "Path to CANN root")

if(DEFINED ENV{TORCH_DELEGATE_ROOT})
    set(TorchDelegate_ROOT_DIR $ENV{TORCH_DELEGATE_ROOT})
    message(STATUS "TorchDelegate_ROOT_DIR from TORCH_DELEGATE_ROOT: ${TorchDelegate_ROOT_DIR}")
endif()

if(TorchDelegate_ROOT_DIR STREQUAL "")
    message(FATAL_ERROR
            "TorchDelegate_ROOT_DIR not specified.\n"
            "  export TORCH_DELEGATE_ROOT=/path/to/libtorch_delegate\n"
            "  cmake -B build -DTorchDelegate_ROOT_DIR=/path/to/libtorch_delegate"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${TorchDelegate_ROOT_DIR}")
find_package(TorchDelegate REQUIRED)

# ============================================
# libtorch — 仅将 torch 库目录加入 target_link_directories，
# 用于链接器解析 torch_delegate_backend.so 的 DT_NEEDED。
# 不链接 libtorch，不包含其头文件。
#
# Python wheel 布局因架构而异:
#   x86_64:  所有 .so 在 torch/lib/
#   aarch64: 部分 .so 拆分到 torch.libs/ 目录
# 通过 target_link_directories 覆盖两种布局，链接器可自动解析传递 DT_NEEDED。
# ============================================
set(TORCH_PKG_DIR "" CACHE PATH "Path to Python torch package directory")
if(TORCH_PKG_DIR STREQUAL "" AND DEFINED ENV{TORCH_PKG_DIR})
    set(TORCH_PKG_DIR $ENV{TORCH_PKG_DIR})
    message(STATUS "[Torch] found TORCH_PKG_DIR env: ${TORCH_PKG_DIR}")
elseif(NOT TORCH_PKG_DIR STREQUAL "")
    message(STATUS "[Torch] found TORCH_PKG_DIR from -D: ${TORCH_PKG_DIR}")
endif()
if(TORCH_PKG_DIR STREQUAL "")
    message(FATAL_ERROR
        "TORCH_PKG_DIR not specified.\n"
        "  cmake -B build -DTORCH_PKG_DIR=/path/to/site-packages/torch\n"
        "  or: export TORCH_PKG_DIR=/path/to/site-packages/torch"
    )
endif()

set(TORCH_LIB_PATH "${TORCH_PKG_DIR}/lib")
get_filename_component(_TORCH_PARENT_DIR "${TORCH_PKG_DIR}" DIRECTORY)

set(TORCH_LINK_DIRS "${TORCH_LIB_PATH}")
set(_TORCH_EXTERNAL_LIBS "${_TORCH_PARENT_DIR}/torch.libs")
if(EXISTS "${_TORCH_EXTERNAL_LIBS}")
    list(APPEND TORCH_LINK_DIRS "${_TORCH_EXTERNAL_LIBS}")
    message(STATUS "[Torch] also using torch.libs: ${_TORCH_EXTERNAL_LIBS}")
endif()
message(STATUS "[Torch] TORCH_LINK_DIRS: ${TORCH_LINK_DIRS}")

# ============================================
# 目标构建
# ============================================
function(setup_target target_name)
    target_include_directories(${target_name} PRIVATE ${ASCEND_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE
        ${ASCEND_LIB_DIRS}
        ${TORCH_LINK_DIRS}
    )
    foreach(_torch_link_dir IN LISTS TORCH_LINK_DIRS)
        target_link_options(${target_name} PRIVATE
            "LINKER:-rpath-link,${_torch_link_dir}"
        )
    endforeach()
    unset(_torch_link_dir)

    target_link_libraries(${target_name} PRIVATE
        ${ASCEND_LIBS}
        TorchDelegate::torch_delegate_backend
    )
    target_compile_options(${target_name} PRIVATE ${common_compile_options})
endfunction()

add_executable(inplace_verify
        main.cpp
        reference_data.cpp
)

setup_target(inplace_verify)
