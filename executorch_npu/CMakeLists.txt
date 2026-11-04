# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.24)
project(executorch_npu LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTORCH_NPU_NLOHMANN_JSON_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json/include"
)
if(NOT EXISTS
   "${EXECUTORCH_NPU_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR "Vendored nlohmann/json.hpp is missing")
endif()
add_library(executorch_npu_nlohmann_json INTERFACE)
target_include_directories(
  executorch_npu_nlohmann_json INTERFACE
  "${EXECUTORCH_NPU_NLOHMANN_JSON_INCLUDE_DIR}"
)
add_library(
  nlohmann_json::nlohmann_json ALIAS executorch_npu_nlohmann_json
)

add_library(
  executorch_npu_manifest STATIC
  runtime/manifest.cpp
)
target_include_directories(
  executorch_npu_manifest PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  executorch_npu_manifest PUBLIC nlohmann_json::nlohmann_json
)
target_compile_options(
  executorch_npu_manifest PRIVATE -Wall -Wextra -Werror -fexceptions
)

option(EXECUTORCH_NPU_BUILD_RUNTIME "Build the Ascend ExecuTorch runtime" ON)
option(EXECUTORCH_NPU_BUILD_RUNNER "Build the Ascend validation runner" OFF)
set(EXECUTORCH_INCLUDE_DIR
    ""
    CACHE PATH "ExecuTorch wheel include directory"
)
if(EXECUTORCH_NPU_BUILD_RUNTIME)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)

  if(NOT EXECUTORCH_INCLUDE_DIR)
    find_package(executorch CONFIG REQUIRED)
    if(NOT EXECUTORCH_INCLUDE_DIRS)
      message(FATAL_ERROR
        "ExecuTorch package did not provide EXECUTORCH_INCLUDE_DIRS")
    endif()
    list(GET EXECUTORCH_INCLUDE_DIRS 0 EXECUTORCH_INCLUDE_DIR)
  endif()
  if(NOT EXISTS
     "${EXECUTORCH_INCLUDE_DIR}/executorch/runtime/backend/interface.h")
    message(FATAL_ERROR
      "EXECUTORCH_INCLUDE_DIR must contain executorch/runtime/backend/interface.h")
  endif()
  set(EXECUTORCH_NPU_EXECUTORCH_INCLUDE_DIRS
      "${EXECUTORCH_INCLUDE_DIR}"
      "${EXECUTORCH_INCLUDE_DIR}/executorch/runtime/core/portable_type/c10"
  )

  set(ASCEND_HOME_PATH
      "$ENV{ASCEND_HOME_PATH}"
      CACHE PATH "CANN installation root"
  )
  if(NOT EXISTS "${ASCEND_HOME_PATH}/include/acl/acl.h")
    message(
      FATAL_ERROR
        "ASCEND_HOME_PATH must point to a CANN installation containing include/acl/acl.h"
    )
  endif()
  add_library(
    executorch_npu_backend STATIC
    runtime/ascend_backend.cpp
    runtime/sha256.cpp
  )
  target_include_directories(
    executorch_npu_backend
    SYSTEM PRIVATE
    "${ASCEND_HOME_PATH}/include"
    ${EXECUTORCH_NPU_EXECUTORCH_INCLUDE_DIRS}
  )
  target_compile_definitions(
    executorch_npu_backend PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
  )
  target_link_libraries(
    executorch_npu_backend
    PUBLIC executorch_npu_manifest
    PRIVATE OpenSSL::Crypto
  )
  include(cmake/executorch_npu_executorch.cmake)
  find_library(
    ASCENDCL_LIBRARY ascendcl
    HINTS
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/aarch64-linux/lib64"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64"
    "${ASCEND_HOME_PATH}/aarch64-linux/devlib"
    "${ASCEND_HOME_PATH}/x86_64-linux/devlib"
    REQUIRED
    NO_DEFAULT_PATH
  )
  target_link_libraries(
    executorch_npu_backend
    PRIVATE executorch_npu::executorch_runtime "${ASCENDCL_LIBRARY}"
  )
  target_compile_options(
    executorch_npu_backend PRIVATE
    -Wall -Wextra -Werror -fexceptions
  )

  # The backend registers through static initialization and must not be
  # discarded when linked from this archive.
  if(APPLE)
    target_link_options(
      executorch_npu_backend
      INTERFACE
      "SHELL:LINKER:-force_load,$<TARGET_FILE:executorch_npu_backend>"
    )
  else()
    target_link_options(
      executorch_npu_backend
      INTERFACE
      "SHELL:LINKER:--whole-archive $<TARGET_FILE:executorch_npu_backend> LINKER:--no-whole-archive"
    )
  endif()

  if(EXECUTORCH_NPU_BUILD_RUNNER)
    add_executable(
      executorch_npu_runner tools/ascend_runner.cpp
    )
    target_link_libraries(
      executorch_npu_runner
      PRIVATE
      executorch_npu_backend
      executorch_npu::executorch_runtime
    )
    target_compile_options(
      executorch_npu_runner PRIVATE -Wall -Wextra -Werror
    )
    add_executable(
      executorch_npu_stateful_runner tools/ascend_stateful_runner.cpp
    )
    target_link_libraries(
      executorch_npu_stateful_runner
      PRIVATE
      executorch_npu_backend
      executorch_npu::executorch_runtime
    )
    target_compile_options(
      executorch_npu_stateful_runner PRIVATE -Wall -Wextra -Werror
    )
  endif()
endif()

if(BUILD_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_session_sampling.cpp")
    add_executable(
      executorch_npu_session_sampling_test tests/cpp/test_session_sampling.cpp
    )
    target_include_directories(
      executorch_npu_session_sampling_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools"
    )
    target_compile_options(
      executorch_npu_session_sampling_test PRIVATE -Wall -Wextra -Werror -UNDEBUG
    )
    add_test(
      NAME executorch_npu_session_sampling_test
      COMMAND executorch_npu_session_sampling_test
    )
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_manifest.cpp")
    add_executable(
      executorch_npu_manifest_test tests/cpp/test_manifest.cpp
    )
    target_link_libraries(
      executorch_npu_manifest_test PRIVATE executorch_npu_manifest
    )
    target_compile_options(
      executorch_npu_manifest_test PRIVATE -Wall -Wextra -Werror -UNDEBUG
    )
    add_test(
      NAME executorch_npu_manifest_test
      COMMAND executorch_npu_manifest_test
    )
  endif()
  if(EXECUTORCH_NPU_BUILD_RUNTIME AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_state_api.cpp")
    add_executable(
      executorch_npu_state_api_test tests/cpp/test_state_api.cpp
    )
    target_link_libraries(
      executorch_npu_state_api_test PRIVATE executorch_npu_backend
    )
    target_compile_options(
      executorch_npu_state_api_test PRIVATE -Wall -Wextra -Werror -UNDEBUG
    )
    add_test(
      NAME executorch_npu_state_api_test
      COMMAND executorch_npu_state_api_test
    )
  endif()
endif()

install(DIRECTORY runtime/include/ DESTINATION include)
install(TARGETS executorch_npu_manifest ARCHIVE DESTINATION lib)
if(TARGET executorch_npu_backend)
  install(TARGETS executorch_npu_backend ARCHIVE DESTINATION lib)
endif()

if(EXECUTORCH_NPU_BUILD_RUNNER)
  install(
    TARGETS executorch_npu_runner executorch_npu_stateful_runner
    RUNTIME DESTINATION bin
  )
endif()
