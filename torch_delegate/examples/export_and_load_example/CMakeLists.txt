# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(export_and_load_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# TorchDelegate 安装目录
# ---------------------------------------------------------------------------
set(TorchDelegate_ROOT_DIR "" CACHE PATH "Path to the TorchDelegate install directory")

if(DEFINED ENV{TORCH_DELEGATE_ROOT})
    set(TorchDelegate_ROOT_DIR $ENV{TORCH_DELEGATE_ROOT})
endif()

if(TorchDelegate_ROOT_DIR STREQUAL "")
    message(FATAL_ERROR
        "TorchDelegate_ROOT_DIR not specified.\n"
        "  cmake -B build -DTorchDelegate_ROOT_DIR=/path/to/libtorch_delegate\n"
        "  or: export TORCH_DELEGATE_ROOT=/path/to/libtorch_delegate"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${TorchDelegate_ROOT_DIR}")
find_package(TorchDelegate REQUIRED)

# ---------------------------------------------------------------------------
# CANN 安装目录 (ASCEND_HOME_PATH)
# ---------------------------------------------------------------------------
set(CANN_ROOT "" CACHE PATH "Path to CANN installation (ASCEND_HOME_PATH)")

if(CANN_ROOT STREQUAL "" AND DEFINED ENV{ASCEND_HOME_PATH})
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
endif()

if(CANN_ROOT STREQUAL "")
    message(FATAL_ERROR
        "CANN_ROOT not specified.\n"
        "  cmake -B build -DCANN_ROOT=/path/to/cann\n"
        "  or: export ASCEND_HOME_PATH=/path/to/cann"
    )
endif()

# libtorch — 仅将 torch 库目录加入 target_link_directories，
# 用于链接器解析 torch_delegate_backend.so 的 DT_NEEDED。
# 不链接 libtorch，不包含其头文件。
#
# Python wheel 布局因架构而异:
#   x86_64:  所有 .so 在 torch/lib/
#   aarch64: 部分 .so 拆分到 torch.libs/ 目录
# 通过 target_link_directories 覆盖两种布局，链接器可自动解析传递 DT_NEEDED。
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

add_executable(export_and_load_example main.cpp)

target_include_directories(export_and_load_example PRIVATE
    ${CANN_ROOT}/include
)

target_link_directories(export_and_load_example PRIVATE
    ${CANN_ROOT}/lib64
    ${TORCH_LINK_DIRS}
)

foreach(_torch_link_dir IN LISTS TORCH_LINK_DIRS)
    target_link_options(export_and_load_example PRIVATE
        "LINKER:-rpath-link,${_torch_link_dir}"
    )
endforeach()
unset(_torch_link_dir)

target_link_libraries(export_and_load_example PRIVATE
    TorchDelegate::torch_delegate_backend
    ge_runner_v2
    acl_rt
)
