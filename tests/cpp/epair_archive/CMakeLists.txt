cmake_minimum_required(VERSION 3.18)
project(cann_lm_tests CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── paths ──
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(CSRC_DIR "${PROJECT_ROOT}/cann_local_lm/converter/export/csrc")
# 自带三方库 (bash scripts/build.sh third_party 产出), 不再依赖 sibling torch-delegate-harmony
# 镜像内 （MIRROR_THIRD_PARYT=ON）: 三方库直连CANN_3RD_LIB_PATH/lib_cache，PATHS 仅作非镜像兜底
if(MIRROR_THIRD_PARTY AND CANN_3RD_LIB_PATH)
    set(THIRD_PARTY_ROOT "${CANN_3RD_LIB_PATH}/lib_cache")
else()
    set(THIRD_PARTY_ROOT "${PROJECT_ROOT}/output/third_party")
endif()
set(FIXTURES_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
set(CANN_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT CANN_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH not set. Source CANN env first (source scripts/set_env_compile.sh).")
endif()

# ── CANN ── (本仓 cmake/modules 自包含: Find*.cmake + DelegateCommonFlags + TorchDelegateCannCommon)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT}/cmake/modules")
include(TorchDelegateCannCommon)
include(DelegateCommonFlags)
torch_delegate_cann_include_search_paths(CANN_INCLUDE_DIRS)
torch_delegate_cann_library_search_paths(CANN_LIB_DIRS)

# ── deps ── (third_party 提示: 顶层 build_cpp.sh 传 -D..._DIR; 独立构建时回退自带路径)
# 注意：find_package 带 PATHS 会跳过 module模式（镜像下顶层 PREPEND 的
# cmake/third_party/modules 需纯 module 模式命中 lib_cache），故镜像下不带PATHS.
if(MIRROR_THIRD_PARTY)
    find_package(nlohmann_json REQUIRED)
    find_package(GTest REQUIRED)
else()
    find_package(nlohmann_json REQUIRED PATHS "${THIRD_PARTY_ROOT}/json/share/cmake/nlohmann_json" NO_DEFAULT_PATH)
    find_package(GTest REQUIRED PATHS "${THIRD_PARTY_ROOT}/gtest/lib/cmake/GTest" NO_DEFAULT_PATH)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(ascendcl REQUIRED)
find_package(acl_rt REQUIRED)
find_package(graph REQUIRED)
find_package(graph_base REQUIRED)
find_package(metadef REQUIRED)
find_package(runtime REQUIRED)
find_package(securec REQUIRED)
find_package(ascend_hal REQUIRED)
find_package(ge_runner_v2 QUIET)
if(NOT ge_runner_v2_FOUND)
    add_library(ge_runner_v2 INTERFACE IMPORTED)
endif()

# ── torch_delegate_backend: in-tree target (顶层 add_subdirectory) 或 fallback 预编译 .so ──
set(TD_BACKEND_LIB "${PROJECT_ROOT}/build/libtorch_delegate_backend.so")
if(NOT TARGET torch_delegate_backend)
    if(NOT EXISTS "${TD_BACKEND_LIB}")
        message(FATAL_ERROR
            "torch_delegate_backend not found at ${TD_BACKEND_LIB}\n"
            "  顶层统一构建: bash scripts/cann_lm_compile/build_cpp.sh\n"
            "  bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend")
    endif()
    add_library(torch_delegate_backend SHARED IMPORTED GLOBAL)
    set_target_properties(torch_delegate_backend PROPERTIES
        IMPORTED_LOCATION "${TD_BACKEND_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CSRC_DIR}/inc/external"
    )
endif()

# CANN 依赖 (test exe 直接 link, 不依赖 backend target 的 INTERFACE 传播)
set(CANN_LM_TEST_DEPS
    graph graph_base metadef ascendcl acl_rt
    nlohmann_json::nlohmann_json
    runtime securec ascend_hal ge_runner_v2
)

# Build rpath-link flags for link-time resolution of transitive deps
set(RPATH_LINK_FLAGS)
foreach(_dir ${CANN_LIB_DIRS})
    list(APPEND RPATH_LINK_FLAGS "-Wl,-rpath-link,${_dir}")
endforeach()
list(APPEND RPATH_LINK_FLAGS "-Wl,-rpath-link,${TORCH_INSTALL_PREFIX}/lib")
list(APPEND RPATH_LINK_FLAGS "-Wl,-rpath-link,${TORCH_INSTALL_PREFIX}/../torch.libs")
list(APPEND RPATH_LINK_FLAGS "-Wl,-rpath-link,/usr/lib/aarch64-linux-gnu")

# ── generate fixtures (directory-based, use stamp file) ──
set(FIXTURE_STAMP "${FIXTURES_DIR}/.fixtures_generated.stamp")
add_custom_command(
    OUTPUT ${FIXTURE_STAMP}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/gen_fixtures.py"
            "--output-dir" "${FIXTURES_DIR}"
    COMMAND ${CMAKE_COMMAND} -E touch ${FIXTURE_STAMP}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_fixtures.py"
    COMMENT "Generating test fixtures → ${FIXTURES_DIR}"
)
add_custom_target(generate_fixtures ALL DEPENDS ${FIXTURE_STAMP})

# ── helper function to create a test executable ──
function(add_cann_lm_test TEST_NAME SOURCE_FILE)
    set(oneValueArgs NEEDS_FIXTURES)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "" ${ARGN})

    add_executable(${TEST_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
    if(ARG_NEEDS_FIXTURES)
        add_dependencies(${TEST_NAME} generate_fixtures)
    endif()

    target_include_directories(${TEST_NAME} PRIVATE
        "${CSRC_DIR}"
        "${CANN_INCLUDE_DIRS}"
    )
    target_compile_definitions(${TEST_NAME} PRIVATE
        FIXTURES_DIR="${FIXTURES_DIR}"
    )
    target_link_directories(${TEST_NAME} PRIVATE
        ${CANN_LIB_DIRS}
    )
    target_link_libraries(${TEST_NAME} PRIVATE
        torch_delegate_backend
        GTest::gtest_main
        ${CANN_LM_TEST_DEPS}
    )
    delegate_apply_common_flags(${TEST_NAME})
    target_link_options(${TEST_NAME} PRIVATE
        -Wl,--disable-new-dtags
        ${RPATH_LINK_FLAGS}
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CANN_LIB_DIRS}:$ENV{LD_LIBRARY_PATH}"
    )
endfunction()

# ── test targets ──
# Tests that need .epair fixtures
add_cann_lm_test(test_consolidated_weights test_consolidated_weights.cc NEEDS_FIXTURES)
add_cann_lm_test(test_epair_archive test_epair_archive.cc NEEDS_FIXTURES)
add_cann_lm_test(test_epair_model_parser test_epair_model_parser.cc NEEDS_FIXTURES)

# Tests that don't need fixtures (pure unit tests)
add_cann_lm_test(test_base64 test_base64.cc)
add_cann_lm_test(test_value_registry test_value_registry.cc)
add_cann_lm_test(test_aten_schema_resolver test_aten_schema_resolver.cc)
add_cann_lm_test(test_aten_tensor_desc_mapper test_aten_tensor_desc_mapper.cc)
