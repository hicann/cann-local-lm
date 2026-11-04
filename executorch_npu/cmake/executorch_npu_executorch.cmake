# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
if(TARGET executorch_npu_executorch_runtime)
  return()
endif()

find_package(executorch CONFIG REQUIRED)

if(NOT EXECUTORCH_BUILD_VERSION)
  message(FATAL_ERROR
    "ExecuTorch 1.5.0 or newer full wheel with the C++ SDK is required")
endif()
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+"
  EXECUTORCH_NPU_EXECUTORCH_VERSION "${EXECUTORCH_BUILD_VERSION}")
if(EXECUTORCH_NPU_EXECUTORCH_VERSION VERSION_LESS 1.5.0)
  message(FATAL_ERROR
    "ExecuTorch ${EXECUTORCH_BUILD_VERSION} is unsupported; 1.5.0 is the minimum")
endif()
if(NOT EXECUTORCH_NPU_EXECUTORCH_VERSION VERSION_LESS 1.6.0)
  message(WARNING
    "ExecuTorch ${EXECUTORCH_BUILD_VERSION} is newer than the supported 1.5 release line")
endif()

add_library(executorch_npu_executorch_runtime INTERFACE)
if(TARGET executorch::runtime)
  target_link_libraries(
    executorch_npu_executorch_runtime INTERFACE executorch::runtime
  )
elseif(EXECUTORCH_RUNTIME_LIBRARY_DIR AND EXECUTORCH_INCLUDE_DIRS)
  # ExecuTorch 1.5 exposes imported targets with CMake >= 3.28. Its package
  # also provides this variable-based compatibility path for older CMake.
  find_library(
    EXECUTORCH_NPU_EXECUTORCH_LIBRARY executorch
    HINTS "${EXECUTORCH_RUNTIME_LIBRARY_DIR}"
    REQUIRED
    NO_DEFAULT_PATH
  )
  target_include_directories(
    executorch_npu_executorch_runtime SYSTEM
    INTERFACE ${EXECUTORCH_INCLUDE_DIRS}
  )
  if(EXECUTORCH_COMPILE_DEFINITIONS)
    target_compile_definitions(
      executorch_npu_executorch_runtime
      INTERFACE ${EXECUTORCH_COMPILE_DEFINITIONS}
    )
  endif()
  if(EXECUTORCH_CXX_STANDARD)
    target_compile_features(
      executorch_npu_executorch_runtime
      INTERFACE cxx_std_${EXECUTORCH_CXX_STANDARD}
    )
  endif()
  target_link_libraries(
    executorch_npu_executorch_runtime
    INTERFACE "${EXECUTORCH_NPU_EXECUTORCH_LIBRARY}"
  )
  if(EXECUTORCH_RUNTIME_LIBRARY_DIR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
      executorch_npu_executorch_runtime
      INTERFACE
      "LINKER:--enable-new-dtags"
      "LINKER:-rpath,${EXECUTORCH_RUNTIME_LIBRARY_DIR}"
    )
  endif()
else()
  message(FATAL_ERROR
    "The ExecuTorch package does not expose its 1.5 C++ runtime")
endif()

add_library(
  executorch_npu::executorch_runtime
  ALIAS executorch_npu_executorch_runtime
)
