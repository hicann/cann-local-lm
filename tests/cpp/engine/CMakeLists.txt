# tests/cpp/engine/CMakeLists.txt — engine UT 定义
#
# 前置：cann_lm_engine target 已定义（host 由 cann_lm/engine/CMakeLists.txt，
#       OHOS 由 cann_lm/engine/ohos/CMakeLists.txt）
# 前置：GTest 已 find_package（host）或 FetchContent（OHOS）
#
# Host: 顶层 CMakeLists.txt 在 BUILD_TESTS=ON 时 add_subdirectory(tests/cpp/engine)
# OHOS: ohos/CMakeLists.txt add_subdirectory(tests/cpp/engine)

if(NOT TARGET cann_lm_engine)
    message(FATAL_ERROR "cann_lm_engine target not found")
endif()

set(_ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../cann_lm/engine)
set(MODEL_EXECUTOR_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model_executor)

enable_testing()

# ── 辅助函数：注册 UT 可执行 + ctest ──
# CANN_LM_CROSS_BUILD=ON 时（OHOS 交叉编译）用 add_test 静态注册（不能运行目标二进制）
if(NOT CANN_LM_CROSS_BUILD)
    find_package(GTest)
    include(GoogleTest)
endif()

function(cannlm_register_tests name)
    if(CANN_LM_CROSS_BUILD)
        add_test(NAME ${name} COMMAND ${name})
    else()
        gtest_discover_tests(${name} ${ARGN})
    endif()
endfunction()

if(CANN_LM_CROSS_BUILD)
    set(_GTEST_LIBS gtest gtest_main)
else()
    set(_GTEST_LIBS GTest::gtest GTest::gtest_main)
endif()

function(cannlm_add_engine_ut name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE cann_lm_engine ${_GTEST_LIBS})
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(${name} PRIVATE ${_ENGINE_DIR})
    cannlm_register_tests(${name})
endfunction()

# ── model_executor UT（多源文件合一）──
add_executable(model_executor_ut
    ${MODEL_EXECUTOR_TEST_DIR}/test_kv_cache_pool.cpp
    ${MODEL_EXECUTOR_TEST_DIR}/test_model_executor.cpp
    ${MODEL_EXECUTOR_TEST_DIR}/test_dfx_collector.cpp
    ${MODEL_EXECUTOR_TEST_DIR}/test_tensor_manager.cpp
)
target_link_libraries(model_executor_ut PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(model_executor_ut PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(model_executor_ut PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(model_executor_ut
    PROPERTIES ENVIRONMENT "CANN_LM_NDK_LIB_DIR=${CMAKE_BINARY_DIR}")

# ── InputAssembler UT ──
add_executable(ut_input_assembler ${MODEL_EXECUTOR_TEST_DIR}/test_input_assembler.cpp)
target_link_libraries(ut_input_assembler PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_input_assembler PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_input_assembler PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_input_assembler
    PROPERTIES ENVIRONMENT "CANN_LM_NDK_LIB_DIR=${CMAKE_BINARY_DIR}")

# ── LinearStatePool UT（文件存在才注册）──
if(EXISTS "${MODEL_EXECUTOR_TEST_DIR}/test_linear_state_pool.cpp")
add_executable(ut_linear_state_pool ${MODEL_EXECUTOR_TEST_DIR}/test_linear_state_pool.cpp)
target_link_libraries(ut_linear_state_pool PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_linear_state_pool PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_linear_state_pool PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_linear_state_pool)
endif()

# ── 独立 UT targets ──
cannlm_add_engine_ut(ut_engine_executor ${CMAKE_CURRENT_SOURCE_DIR}/test_engine_executor.cpp)
cannlm_add_engine_ut(ut_c_api           ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api.cpp)
cannlm_add_engine_ut(ut_scheduler       ${CMAKE_CURRENT_SOURCE_DIR}/test_scheduler.cpp)
cannlm_add_engine_ut(ut_sampler        ${CMAKE_CURRENT_SOURCE_DIR}/test_sampler.cpp)
cannlm_add_engine_ut(ut_request_handle ${CMAKE_CURRENT_SOURCE_DIR}/test_request_handle.cpp)
cannlm_add_engine_ut(ut_share_weights_executor ${CMAKE_CURRENT_SOURCE_DIR}/test_share_weights_executor.cpp)
cannlm_add_engine_ut(ut_data_type_utils ${CMAKE_CURRENT_SOURCE_DIR}/test_data_type_utils.cpp)

# ut_backend_shareweights 需要 ohos_compat.h force-include
add_executable(ut_backend_shareweights ${CMAKE_CURRENT_SOURCE_DIR}/test_backend_shareweights.cpp)
target_link_libraries(ut_backend_shareweights PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_backend_shareweights PRIVATE -Wall -Wextra -Wpedantic
    -include ${_ENGINE_DIR}/hiai_acl/src/ohos_compat.h)
target_include_directories(ut_backend_shareweights PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_backend_shareweights)

# ── ut_tokenizer（需要 fixture dir + nlohmann/json include）──
add_executable(ut_tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/test_tokenizer.cpp)
target_link_libraries(ut_tokenizer PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_tokenizer PRIVATE ${_ENGINE_DIR})
target_compile_definitions(ut_tokenizer PRIVATE
    TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(ut_tokenizer PRIVATE ${_ENGINE_DIR}/../../cmake/third_party)
cannlm_register_tests(ut_tokenizer)

# ── ut_text_embedding ──
add_executable(ut_text_embedding ${CMAKE_CURRENT_SOURCE_DIR}/test_text_embedding.cpp)
target_link_libraries(ut_text_embedding PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_text_embedding PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_text_embedding PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_text_embedding
    PROPERTIES ENVIRONMENT "CANN_LM_NDK_LIB_DIR=${CMAKE_BINARY_DIR}")

# ── tokenizer realdata（检测 tmp/ 实物存在才注册）──
set(TMP_TOKENIZER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../tmp")
if(EXISTS "${TMP_TOKENIZER_DIR}/qwen/tokenizer.json" AND EXISTS "${TMP_TOKENIZER_DIR}/gemma/tokenizer.json")
    add_executable(tokenizer_realdata ${CMAKE_CURRENT_SOURCE_DIR}/test_tokenizer_realdata.cpp)
    target_link_libraries(tokenizer_realdata PRIVATE cann_lm_engine ${_GTEST_LIBS})
    target_compile_options(tokenizer_realdata PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(tokenizer_realdata PRIVATE ${_ENGINE_DIR})
    target_compile_definitions(tokenizer_realdata PRIVATE
        TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        TOKENIZER_REALDATA_DIR="${TMP_TOKENIZER_DIR}")
    target_include_directories(tokenizer_realdata PRIVATE ${_ENGINE_DIR}/../../cmake/third_party)
    cannlm_register_tests(tokenizer_realdata)
else()
    message(STATUS "tokenizer_realdata: tmp/ tokenizer.json 缺失，跳过")
endif()

# ── stub_ndk（host 回归：NDK 打桩 .so）──
if(NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/stub_ndk stub_ndk_build EXCLUDE_FROM_ALL)
    foreach(t model_executor_ut ut_input_assembler ut_text_embedding ut_engine_executor)
        if(TARGET ${t})
            add_dependencies(${t} stub_neural_network_core stub_hiai_foundation)
        endif()
    endforeach()
endif()
