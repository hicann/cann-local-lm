cmake_minimum_required(VERSION 3.18)
project(cann_lm_backend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TORCH_DELEGATE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# ── CANN ──
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/modules")
include(TorchDelegateCannCommon)
include(DelegateCommonFlags)
torch_delegate_cann_include_search_paths(CANN_INCLUDE_DIRS)
torch_delegate_cann_library_search_paths(CANN_LIB_DIRS)

# ── nlohmann_json ──
find_package(nlohmann_json REQUIRED)

# ── CANN libs ──
find_package(ascendcl REQUIRED)
find_package(acl_rt REQUIRED)
find_package(graph REQUIRED)
find_package(graph_base REQUIRED)
find_package(metadef REQUIRED)
find_package(runtime REQUIRED)
find_package(securec REQUIRED)
find_package(ascend_hal REQUIRED)

# ── ge_runner_v2 (stub from CANN if available) ──
find_package(ge_runner_v2 QUIET)
if(NOT ge_runner_v2_FOUND)
    add_library(ge_runner_v2 INTERFACE IMPORTED)
    message(STATUS "[ge_runner_v2] not found, using INTERFACE stub")
endif()

# ── sources ──
file(GLOB TORCH_DELEGATE_COMMON_SOURCES CONFIGURE_DEPENDS
    "${CSRC_DIR}/common/*.cc"
)
file(GLOB TORCH_DELEGATE_BACKEND_SOURCES CONFIGURE_DEPENDS
    "${CSRC_DIR}/backend/*.cc"
)

add_library(torch_delegate_backend SHARED
    ${TORCH_DELEGATE_COMMON_SOURCES}
    ${TORCH_DELEGATE_BACKEND_SOURCES}
)

# .so 固定落 build 根 (不按源码树嵌套): build_cpp.sh → build/ (统一入口)
# (保 example CMakeLists fallback 契约: ${CANN_LM_CSRC_DIR}/build/libtorch_delegate_backend.so)
set_target_properties(torch_delegate_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_include_directories(torch_delegate_backend SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CSRC_DIR}/inc/external>
)

target_include_directories(torch_delegate_backend SYSTEM PRIVATE
    ${CSRC_DIR}
)

target_link_libraries(torch_delegate_backend PUBLIC
    graph
    graph_base
    metadef
)

target_link_libraries(torch_delegate_backend PRIVATE
    ascendcl
    acl_rt
    nlohmann_json::nlohmann_json
    runtime
    securec
    ascend_hal
    ge_runner_v2
)

target_compile_definitions(torch_delegate_backend PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
delegate_apply_common_flags(torch_delegate_backend)

# ── cann_lm_compile (epair→OM 编译工具) ──
# main: tools/compile_main.cc; reg_register.h 在 inc/external (经 backend PUBLIC include 可达)
# 链 torch_delegate_backend (EpairModelLoader) + ge_runner_v2/ge_compiler/acl_rt (aclgrphBuildModel)
# backend 不链 torch (csrc 不用 torch) → 工具无 torch 依赖, 运行只需 CANN
option(BUILD_COMPILE_TOOL "Build cann_lm_compile (epair→OM) tool" ON)
if(BUILD_COMPILE_TOOL)
    add_executable(cann_lm_compile tools/compile_main.cc)
    delegate_apply_common_flags(cann_lm_compile)
    target_include_directories(cann_lm_compile PRIVATE ${CANN_INCLUDE_DIRS})
    target_link_directories(cann_lm_compile PRIVATE ${CANN_LIB_DIRS})
    target_link_libraries(cann_lm_compile PRIVATE
        torch_delegate_backend
        ge_runner_v2
        ge_compiler
        acl_rt
    )
    set_target_properties(cann_lm_compile PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
