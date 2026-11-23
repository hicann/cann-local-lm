# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
find_package(ascendcl REQUIRED)
find_package(acl_rt REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FIXTURES_DIR "${CMAKE_BINARY_DIR}/tests/cpp/fixtures")
set(CANN_STUB_LIB_DIR "${CMAKE_BINARY_DIR}/tests/cpp/cann_stub/lib64")
set(GE_STUB_LIB_DIR "${CMAKE_BINARY_DIR}/tests/cpp/ge_stub/lib64")

set(TORCH_DELEGATE_TEST_RPATH_OPTIONS
    -Wl,--disable-new-dtags
    "-Wl,-rpath,${GE_STUB_LIB_DIR}"
    "-Wl,-rpath,${CANN_STUB_LIB_DIR}"
    "-Wl,-rpath,${CMAKE_BINARY_DIR}/stub_lib"
)

foreach(_cann_dir IN LISTS TORCH_DELEGATE_CANN_RUNTIME_DIRS)
    list(APPEND TORCH_DELEGATE_TEST_RPATH_OPTIONS "-Wl,-rpath,${_cann_dir}")
endforeach()
unset(_cann_dir)

set(TORCH_DELEGATE_TEST_LD_LIBRARY_PATH
    "${GE_STUB_LIB_DIR}:${CANN_STUB_LIB_DIR}:${TORCH_DELEGATE_CANN_RUNTIME_PATH}"
)

if(LIBTORCH_CPU_RUNTIME_DIRS)
    string(JOIN ":" TORCH_DELEGATE_LIBTORCH_RUNTIME_PATH ${LIBTORCH_CPU_RUNTIME_DIRS})
    message(STATUS
        "[torch_delegate_tests] Extending LD_LIBRARY_PATH with libtorch runtime dirs: "
        "${TORCH_DELEGATE_LIBTORCH_RUNTIME_PATH}"
    )
    set(TORCH_DELEGATE_TEST_LD_LIBRARY_PATH
        "${TORCH_DELEGATE_TEST_LD_LIBRARY_PATH}:${TORCH_DELEGATE_LIBTORCH_RUNTIME_PATH}"
    )
endif()

set(TORCH_DELEGATE_FIXTURE_OUTPUTS
    "${FIXTURES_DIR}/all_npu.epair"
    "${FIXTURES_DIR}/mixed.epair"
    "${FIXTURES_DIR}/all_cpu.epair"
    "${FIXTURES_DIR}/empty_compute.epair"
    "${FIXTURES_DIR}/npu_cpu_npu.epair"
    "${FIXTURES_DIR}/shared_input.epair"
    "${FIXTURES_DIR}/non_adjacent.epair"
    "${FIXTURES_DIR}/no_model_json.epair"
    "${FIXTURES_DIR}/missing_op_type.epair"
    "${FIXTURES_DIR}/annotated_mm_add.epair"
    "${FIXTURES_DIR}/annotated_mm_add_bad_shape_dim.epair"
    "${FIXTURES_DIR}/annotated_mixed.epair"
    "${FIXTURES_DIR}/annotated_host_cpu_non4byte_inputs.epair"
    "${FIXTURES_DIR}/annotated_converter_no_ops.epair"
    "${FIXTURES_DIR}/annotated_attr_types.epair"
    "${FIXTURES_DIR}/annotated_fallback_with_attrs.epair"
    "${FIXTURES_DIR}/annotated_dynamic_concat.epair"
    "${FIXTURES_DIR}/annotated_dynamic_concat_bad_prefix.epair"
    "${FIXTURES_DIR}/annotated_auto_functionalized.epair"
    "${FIXTURES_DIR}/annotated_bad_fallback_input.epair"
    "${FIXTURES_DIR}/annotated_missing_weight_record.epair"
    "${FIXTURES_DIR}/annotated_bad_weight_size.epair"
    "${FIXTURES_DIR}/annotated_inline_const.epair"
    "${FIXTURES_DIR}/annotated_inline_scalar_const.epair"
    "${FIXTURES_DIR}/annotated_inline_const_bad_value.epair"
    "${FIXTURES_DIR}/annotated_inline_const_desc_mismatch.epair"
    "${FIXTURES_DIR}/missing_graph_source.epair"
)

add_custom_command(
    OUTPUT ${TORCH_DELEGATE_FIXTURE_OUTPUTS}
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/cpp/gen_fixtures.py"
            "--output-dir" "${FIXTURES_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/gen_fixtures.py"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating .epair test fixtures -> ${FIXTURES_DIR}"
)

add_custom_target(generate_fixtures ALL
    DEPENDS ${TORCH_DELEGATE_FIXTURE_OUTPUTS}
)

add_library(cann_stub SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cann_stub/ascendcl.cc"
)

set_target_properties(cann_stub PROPERTIES
    OUTPUT_NAME "ascendcl"
    LIBRARY_OUTPUT_DIRECTORY "${CANN_STUB_LIB_DIR}"
)

target_include_directories(cann_stub PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cann_stub"
)

target_link_libraries(cann_stub PRIVATE ascendcl_headers)
target_compile_definitions(cann_stub PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(cann_stub PRIVATE ${DELEGATE_COMMON_FLAGS} -g)
target_link_options(cann_stub PRIVATE ${DELEGATE_COMMON_FLAGS})

file(GLOB TORCH_DELEGATE_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_*.cc"
)

add_executable(torch_delegate_tests ${TORCH_DELEGATE_TEST_SOURCES})
add_dependencies(torch_delegate_tests generate_fixtures)
add_dependencies(torch_delegate_tests ge_runner_v2_stub)

set_target_properties(torch_delegate_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(torch_delegate_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
    "${PROJECT_SOURCE_DIR}/csrc/inc/external"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cann_stub"
)

target_compile_definitions(torch_delegate_tests PRIVATE
    FIXTURES_DIR="${FIXTURES_DIR}"
    ASCEND_HOME_PATH="$ENV{ASCEND_HOME_PATH}"
    CANN_STUB_LIB_DIR="${CANN_STUB_LIB_DIR}"
)

target_link_directories(torch_delegate_tests PRIVATE ${TORCH_DELEGATE_CANN_RUNTIME_DIRS})
target_link_libraries(torch_delegate_tests PRIVATE
    ge_runner_v2_stub
    torch_delegate_backend
    # 下面链接libtorch, arm平台链接时找不到torch.libs下的so
    libtorch_cpu::libtorch_cpu
    acl_rt
    cann_stub
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    dl
)

target_compile_options(torch_delegate_tests PRIVATE ${DELEGATE_COMMON_FLAGS})
target_link_options(torch_delegate_tests PRIVATE
    ${DELEGATE_COMMON_FLAGS}
    ${TORCH_DELEGATE_TEST_RPATH_OPTIONS}
)

add_test(NAME torch_delegate_tests COMMAND torch_delegate_tests)
set_tests_properties(torch_delegate_tests PROPERTIES
        ENVIRONMENT
        "LD_LIBRARY_PATH=${TORCH_DELEGATE_TEST_LD_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
)
