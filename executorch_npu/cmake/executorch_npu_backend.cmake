# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/executorch_npu_executorch.cmake")
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH
    "CANN installation root")
set(EXECUTORCH_NPU_HOME "${ASCEND_HOME_PATH}" CACHE PATH
    "CANN installation root used by ExecuTorch-NPU")
if(NOT EXISTS "${EXECUTORCH_NPU_HOME}/include/acl/acl.h")
  message(FATAL_ERROR
    "ASCEND_HOME_PATH must point to a CANN installation containing include/acl/acl.h")
endif()
find_library(EXECUTORCH_NPU_ASCENDCL_LIBRARY ascendcl
  HINTS
  "${EXECUTORCH_NPU_HOME}/lib64"
  "${EXECUTORCH_NPU_HOME}/aarch64-linux/lib64"
  "${EXECUTORCH_NPU_HOME}/x86_64-linux/lib64"
  "${EXECUTORCH_NPU_HOME}/aarch64-linux/devlib"
  "${EXECUTORCH_NPU_HOME}/x86_64-linux/devlib"
  REQUIRED NO_DEFAULT_PATH)

get_filename_component(EXECUTORCH_NPU_SDK_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(EXECUTORCH_NPU_SDK_INCLUDE_DIR
  "${EXECUTORCH_NPU_SDK_ROOT}/include")
set(EXECUTORCH_NPU_SDK_LIB_DIR
  "${EXECUTORCH_NPU_SDK_ROOT}/lib")

add_library(executorch_npu_manifest_archive STATIC IMPORTED GLOBAL)
set_target_properties(executorch_npu_manifest_archive PROPERTIES
  IMPORTED_LOCATION
    "${EXECUTORCH_NPU_SDK_LIB_DIR}/libexecutorch_npu_manifest.a"
  INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_NPU_SDK_INCLUDE_DIR}")

add_library(executorch_npu_backend_archive STATIC IMPORTED GLOBAL)
set_target_properties(executorch_npu_backend_archive PROPERTIES
  IMPORTED_LOCATION
    "${EXECUTORCH_NPU_SDK_LIB_DIR}/libexecutorch_npu_backend.a"
  INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_NPU_SDK_INCLUDE_DIR}")

add_library(executorch_npu_backend INTERFACE)
target_link_libraries(executorch_npu_backend INTERFACE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,executorch_npu_backend_archive>"
  executorch_npu_manifest_archive
  executorch_npu::executorch_runtime
  OpenSSL::Crypto
  "${EXECUTORCH_NPU_ASCENDCL_LIBRARY}")

add_library(executorch_npu::backend ALIAS executorch_npu_backend)
