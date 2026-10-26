# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(epair_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TorchDelegate_ROOT "" CACHE PATH "Path to torch_delegate package root")
set(CANN_ROOT "" CACHE PATH "Path to CANN installation")
set(TORCH_NPU_DIR "" CACHE PATH "Path to libtorch_npu")

if(DEFINED ENV{TORCH_DELEGATE_ROOT})
    set(TorchDelegate_ROOT $ENV{TORCH_DELEGATE_ROOT})
    message(STATUS "TorchDelegate_ROOT from TORCH_DELEGATE_ROOT: ${TorchDelegate_ROOT}")
endif()
if(TORCH_NPU_DIR STREQUAL "" AND DEFINED ENV{TORCH_NPU_DIR})
    set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})
    message(STATUS "TORCH_NPU_DIR from env: ${TORCH_NPU_DIR}")
endif()

if(TorchDelegate_ROOT STREQUAL "")
    message(FATAL_ERROR
        "TorchDelegate_ROOT not specified.\n"
        "  export TORCH_DELEGATE_ROOT=/path/to/libtorch_delegate\n"
        "  cmake -B build -DTorchDelegate_ROOT=${TORCH_DELEGATE_ROOT}"
    )
endif()
if(CANN_ROOT STREQUAL "")
    message(FATAL_ERROR "CANN_ROOT not specified")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TorchDelegate_ROOT}")
find_package(TorchDelegate REQUIRED)

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

set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
set(CANN_LIB_DIRS ${CANN_ROOT}/lib64)

add_executable(epair_runner
    epair_runner.cpp
    tensor_io.cpp
)

target_include_directories(epair_runner PRIVATE
    ${CANN_INCLUDE_DIRS}
    ${TorchDelegate_ROOT}/inc
)

target_link_directories(epair_runner PRIVATE
    ${CANN_LIB_DIRS}
    ${TORCH_LINK_DIRS}
)

foreach(_torch_link_dir IN LISTS TORCH_LINK_DIRS)
    target_link_options(epair_runner PRIVATE
        "LINKER:-rpath-link,${_torch_link_dir}"
    )
endforeach()
unset(_torch_link_dir)

set(CUST_OPAPI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../custom_op/build" CACHE PATH "Path to custom op library")
if(EXISTS "${CUST_OPAPI_DIR}/libcust_opapi.so")
    target_link_directories(epair_runner PRIVATE ${CUST_OPAPI_DIR})
    target_link_libraries(epair_runner PRIVATE cust_opapi)
else()
    message(WARNING "libcust_opapi.so not found at ${CUST_OPAPI_DIR}, custom ops may not be registered")
endif()

target_link_libraries(epair_runner PRIVATE
    TorchDelegate::torch_delegate_backend
    acl_rt
    ge_runner_v2
)

target_compile_options(epair_runner PRIVATE -Wall -Wextra)
