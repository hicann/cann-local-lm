# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.18)
project(host_cpu_mixed_case CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# TorchDelegate 库查找（支持两种模式）
# ============================================================
set(TorchDelegate_ROOT_DIR "" CACHE PATH "Path to the TorchDelegate install directory")
set(TORCH_DELEGATE_BUILD_DIR "" CACHE PATH "Path to TorchDelegate source build directory")

if(TorchDelegate_ROOT_DIR STREQUAL "" AND DEFINED ENV{TORCH_DELEGATE_ROOT})
    set(TorchDelegate_ROOT_DIR $ENV{TORCH_DELEGATE_ROOT})
endif()
if(TORCH_DELEGATE_BUILD_DIR STREQUAL "" AND DEFINED ENV{TORCH_DELEGATE_BUILD_DIR})
    set(TORCH_DELEGATE_BUILD_DIR $ENV{TORCH_DELEGATE_BUILD_DIR})
endif()

if(NOT TorchDelegate_ROOT_DIR STREQUAL "")
    message(STATUS "TorchDelegate: using installed package at ${TorchDelegate_ROOT_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${TorchDelegate_ROOT_DIR}")
    find_package(TorchDelegate REQUIRED)
    set(DELEGATE_LINK_TARGET TorchDelegate::torch_delegate_backend)
    get_target_property(_delegate_lib_location ${DELEGATE_LINK_TARGET} LOCATION)
    get_filename_component(DELEGATE_LIB_DIR "${_delegate_lib_location}" DIRECTORY)
    set(DELEGATE_INCLUDE_DIR "${TorchDelegate_ROOT_DIR}/inc/torch_delegate")
elseif(NOT TORCH_DELEGATE_BUILD_DIR STREQUAL "")
    message(STATUS "TorchDelegate: using source build at ${TORCH_DELEGATE_BUILD_DIR}")
    set(DELEGATE_LINK_TARGET torch_delegate_backend)
    set(DELEGATE_LIB_DIR "${TORCH_DELEGATE_BUILD_DIR}/csrc")
    get_filename_component(DELEGATE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(DELEGATE_INCLUDE_DIR "${DELEGATE_SOURCE_DIR}/csrc/inc/external")
else()
    message(FATAL_ERROR
        "TorchDelegate not found. Configure with one of:\n"
        "  1. Installed package: -DTorchDelegate_ROOT_DIR=/path/to/libtorch_delegate\n"
        "  2. Source build: -DTORCH_DELEGATE_BUILD_DIR=/path/to/build/release"
    )
endif()

# CANN 路径
set(CANN_ROOT "" CACHE PATH "Path to CANN installation")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    message(STATUS "CANN_ROOT from ASCEND_HOME_PATH: ${CANN_ROOT}")
endif()

if(CANN_ROOT STREQUAL "")
    message(FATAL_ERROR "CANN_ROOT not specified")
endif()

set(CANN_INCLUDE_DIRS
    "${CANN_ROOT}/include"
    "${CANN_ROOT}/include/graph"
    "${CANN_ROOT}/include/register"
)

# PyTorch 路径
execute_process(
    COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(TORCH_LIBS_DIR "${TORCH_PKG_DIR}/../torch.libs")
get_filename_component(TORCH_LIBS_DIR "${TORCH_LIBS_DIR}" ABSOLUTE)

# torch_npu 路径
set(TORCH_NPU_DIR "" CACHE PATH "Path to libtorch_npu")
if(TORCH_NPU_DIR STREQUAL "" AND DEFINED ENV{TORCH_NPU_DIR})
    set(TORCH_NPU_DIR $ENV{TORCH_NPU_DIR})
endif()
get_filename_component(TORCH_NPU_DIR "${TORCH_NPU_DIR}" ABSOLUTE)

# 构建 HostCpu runner（HostCpu 自定义算子由加载期 codegen 生成注册）
add_executable(host_cpu_runner host_cpu_runner.cpp)

# 注意：host_cpu_runner 仅依赖 CANN（GE/ACL）与 TorchDelegate 头文件，
# 不引入 torch/torch_npu 头目录——torch_npu 自带的旧版 GE 头（include/ge、
# include/graph）与 CANN 新版头文件同名冲突，混入搜索路径会导致编译失败。
target_include_directories(host_cpu_runner PRIVATE
    ${CANN_INCLUDE_DIRS}
    "${DELEGATE_INCLUDE_DIR}"
    "${TORCH_PKG_DIR}/include"
    "${TORCH_PKG_DIR}/include/torch/csrc/api/include"
)

target_link_directories(host_cpu_runner PRIVATE
    "${DELEGATE_LIB_DIR}"
    "${CANN_ROOT}/lib64"
    "${TORCH_PKG_DIR}/lib"
    "${TORCH_LIBS_DIR}"
    "${TORCH_NPU_DIR}/lib"
)

target_link_libraries(host_cpu_runner PRIVATE
    ${DELEGATE_LINK_TARGET}
    ge_runner_v2
    ascendcl
    graph
    graph_base
    metadef
    runtime
    dl
)

set_target_properties(host_cpu_runner PROPERTIES
    BUILD_RPATH "${CANN_ROOT}/lib64;${DELEGATE_LIB_DIR};${TORCH_PKG_DIR}/lib;${TORCH_LIBS_DIR}"
    INSTALL_RPATH "${CANN_ROOT}/lib64;${DELEGATE_LIB_DIR};${TORCH_PKG_DIR}/lib;${TORCH_LIBS_DIR}"
    LINK_FLAGS "-Wl,-rpath-link,${TORCH_LIBS_DIR} -Wl,--allow-shlib-undefined"
)

# ============================================================
# 构建 libcustom_ops.so（C++ 侧 Dispatcher 注册，供 runner dlopen）
# ============================================================
add_library(custom_ops SHARED custom_ops_reg.cpp)

target_include_directories(custom_ops PRIVATE
    "${TORCH_PKG_DIR}/include"
    "${TORCH_PKG_DIR}/include/torch/csrc/api/include"
)

target_link_directories(custom_ops PRIVATE
    "${TORCH_PKG_DIR}/lib"
    "${TORCH_LIBS_DIR}"
)

target_link_libraries(custom_ops PRIVATE torch_cpu c10 dl)

set_target_properties(custom_ops PROPERTIES
    BUILD_RPATH "${TORCH_PKG_DIR}/lib;${TORCH_LIBS_DIR}"
    INSTALL_RPATH "${TORCH_PKG_DIR}/lib;${TORCH_LIBS_DIR}"
    LINK_FLAGS "-Wl,-rpath-link,${TORCH_LIBS_DIR} -Wl,--allow-shlib-undefined"
)
