# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Findlibtorch_cpu.cmake - Download and install libtorch_cpu (cxx11-abi)
#
# This module downloads libtorch CPU version with cxx11 ABI and provides Torch target.
# libtorch is downloaded directly from PyTorch official site (no gitcode mirror).
# -----------------------------------------------------------------------------------------------------------

if(TARGET libtorch_cpu_build)
    return()
endif()

include(ExternalProject)

set(LIBTORCH_CPU_VERSION "2.9.0")
set(LIBTORCH_CPU_DIR "${CMAKE_THIRD_PARTY_LIB_DIR}/libtorch")
set(LIBTORCH_CPU_CONFIG "${LIBTORCH_CPU_DIR}/share/cmake/Torch/TorchConfig.cmake")

if(EXISTS "${LIBTORCH_CPU_CONFIG}")
    message(STATUS "[libtorch_cpu] libtorch already installed at ${LIBTORCH_CPU_DIR}")
    set(Torch_DIR "${LIBTORCH_CPU_DIR}/share/cmake/Torch")
    find_package(Torch CONFIG PATHS "${Torch_DIR}" NO_DEFAULT_PATH)
    ExternalProject_Add(libtorch_cpu_build
        SOURCE_DIR ${LIBTORCH_CPU_DIR}
        CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    return()
endif()

message(STATUS "[libtorch_cpu] libtorch not found, finding binary file.")

find_program(UNZIP_PROGRAM unzip)
if(NOT UNZIP_PROGRAM)
    message(FATAL_ERROR "[libtorch_cpu] unzip not found, required to extract libtorch archive")
endif()

set(LIBTORCH_CPU_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_CPU_VERSION}%2Bcpu.zip")
set(LIBTORCH_CPU_ARCHIVE "${CMAKE_THIRD_PARTY_LIB_DIR}/libtorch/libtorch-${LIBTORCH_CPU_VERSION}.zip")
set(LIBTORCH_CPU_EXTRA_ARGS "")

if(EXISTS "${LIBTORCH_CPU_ARCHIVE}")
    message(STATUS "[libtorch_cpu] ${LIBTORCH_CPU_ARCHIVE} found.")
    set(LIBTORCH_CPU_URL "${LIBTORCH_CPU_ARCHIVE}")
else()
    message(STATUS "[libtorch_cpu] ${LIBTORCH_CPU_ARCHIVE} not found, will download from PyTorch.")
    list(APPEND LIBTORCH_CPU_EXTRA_ARGS
        DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/libtorch
    )
endif()

ExternalProject_Add(libtorch_cpu_build
    URL ${LIBTORCH_CPU_URL}
    TLS_VERIFY OFF
    ${LIBTORCH_CPU_EXTRA_ARGS}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E make_directory ${CMAKE_THIRD_PARTY_LIB_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> ${CMAKE_THIRD_PARTY_LIB_DIR}/libtorch
    EXCLUDE_FROM_ALL TRUE
)
