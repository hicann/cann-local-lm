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
torch_delegate_cann_include_search_paths(CANN_INCLUDE_DIRS)
torch_delegate_cann_library_search_paths(CANN_LIB_DIRS)

# ── PyTorch ──
find_package(Torch REQUIRED)

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
file(GLOB TORCH_DELEGATE_FALLBACK_CODEGEN_SOURCES CONFIGURE_DEPENDS
    "${CSRC_DIR}/fallback_codegen/*.cc"
)

add_library(torch_delegate_backend SHARED
    ${TORCH_DELEGATE_COMMON_SOURCES}
    ${TORCH_DELEGATE_BACKEND_SOURCES}
    ${TORCH_DELEGATE_FALLBACK_CODEGEN_SOURCES}
)

target_include_directories(torch_delegate_backend SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CSRC_DIR}/inc/external>
)

target_include_directories(torch_delegate_backend SYSTEM PRIVATE
    ${CSRC_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(torch_delegate_backend PUBLIC
    graph
    graph_base
    metadef
)

target_link_libraries(torch_delegate_backend PRIVATE
    ascendcl
    acl_rt
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
    runtime
    securec
    ascend_hal
    ge_runner_v2
)

target_compile_definitions(torch_delegate_backend PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(torch_delegate_backend PRIVATE -Wall -Wextra)
target_link_options(torch_delegate_backend PRIVATE -Wall -Wextra)
