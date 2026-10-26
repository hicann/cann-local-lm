# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
file(GLOB TORCH_DELEGATE_COMMON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc"
)
file(GLOB TORCH_DELEGATE_BACKEND_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cc"
)
file(GLOB TORCH_DELEGATE_FALLBACK_CODEGEN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/fallback_codegen/*.cc"
)

set(TORCH_DELEGATE_LIBRARY_SOURCES
    ${TORCH_DELEGATE_COMMON_SOURCES}
    ${TORCH_DELEGATE_BACKEND_SOURCES}
    ${TORCH_DELEGATE_FALLBACK_CODEGEN_SOURCES}
)

set(TORCH_DELEGATE_GE_RUNNER_V2_BACKEND_LINK_TARGET
    "${TORCH_DELEGATE_GE_RUNNER_V2_LINK_TARGET}"
)

if(TORCH_DELEGATE_GE_RUNNER_V2_BACKEND_LINK_TARGET STREQUAL "ge_runner_v2_stub")
    set(TORCH_DELEGATE_GE_RUNNER_V2_BACKEND_LINK_TARGET
        "$<BUILD_INTERFACE:ge_runner_v2_stub>"
    )
endif()

add_library(torch_delegate_backend SHARED ${TORCH_DELEGATE_LIBRARY_SOURCES})

target_include_directories(torch_delegate_backend SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/external>
    $<INSTALL_INTERFACE:inc>
)

target_include_directories(torch_delegate_backend SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(torch_delegate_backend PUBLIC
    graph
    graph_base
    metadef
)

target_link_libraries(torch_delegate_backend PRIVATE
    acl_rt
    libtorch_cpu::libtorch_cpu
    nlohmann_json::nlohmann_json
    ascendcl_headers
    ${TORCH_DELEGATE_GE_RUNNER_V2_BACKEND_LINK_TARGET}
)

target_compile_definitions(torch_delegate_backend PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(torch_delegate_backend PRIVATE ${DELEGATE_COMMON_FLAGS})
target_link_options(torch_delegate_backend PRIVATE ${DELEGATE_COMMON_FLAGS})

if(NOT BUILD_TESTS)
    # Install rules for packaging
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/external/torch_delegate"
        DESTINATION "inc"
        FILES_MATCHING PATTERN "*.h"
    )

    install(TARGETS torch_delegate_backend
        EXPORT TorchDelegateTargets
        LIBRARY DESTINATION "lib"
    )

    install(EXPORT TorchDelegateTargets
        FILE TorchDelegateTargets.cmake
        NAMESPACE TorchDelegate::
        DESTINATION "lib/cmake/TorchDelegate"
    )
endif()
