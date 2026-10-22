cmake_minimum_required(VERSION 3.10)
project(cann_lm_engine
    LANGUAGES C CXX
    VERSION 0.1.0
    DESCRIPTION "CANN-LM engine C++ API skeleton (stub, v1.2 architecture)"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ── backend 子项目（独立 cann_backend 库：api/device_base/acl/ndk + hiai_acl shim）──
# EXCLUDE_FROM_ALL：backend 自带的 demo/test 不进默认构建；
# cann_backend 本体经下方 target_link_libraries 的依赖关系驱动构建。
add_subdirectory(backend EXCLUDE_FROM_ALL)

add_library(cann_lm_engine STATIC
    api/cpp/engine_executor.cpp
    api/cpp/request_handle.cpp
    api/c/c_api.cpp

    core/tensor_manager.cpp
    core/context.cpp
    utils/data_type_utils.cpp
    utils/debug_tensor_dump.cpp # 【临时调试】推理前后 dump 输入/输出工位
    core/component/text_embedding/text_embedding.cpp
    core/component/model_executor/model_executor.cpp
    core/component/kv_cache_pool/kv_cache_pool.cpp
    core/component/linear_state_pool/linear_state_pool.cpp   # Qwen3.5 线性状态池（support_qwen3.5.md §4.4）
    core/component/input_assembler/input_assembler.cpp
    core/component/dfx_collector/dfx_collector.cpp
    core/component/sampler/sampler.cpp
    core/component/tokenizer/tokenizer.cpp
    core/component/tokenizer/bpe_tokenizer.cpp
    core/component/tokenizer/qwen_tokenizer.cpp
    core/component/tokenizer/vocab_store.cpp
    core/component/tokenizer/special_token_handler.cpp

    # ── sample 算法内核（cann_lm::sampler::Sampler，由 core/component/sampler.cpp 适配层转调）──
    core/sample/sampler.cc

    pipeline/history_prompt_manager.cpp
    pipeline/pipeline_executor.cpp
    pipeline/preprocessor.cpp
    pipeline/scheduler.cpp
)

target_include_directories(cann_lm_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party   # nlohmann/json（D-09 vendored，仓库根）
)

# backend 通路：工厂/设备模型/张量的具体实现（含 hiai_acl shim 依赖，PUBLIC 传递给测试/冒烟目标）
target_link_libraries(cann_lm_engine PUBLIC cann_backend)
# OHOS 交叉编译时链接 hilog NDK 库(提供 OH_LOG_Print);host 构建不链接
if(OHOS OR OHOS_PLATFORM OR CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    target_link_libraries(cann_lm_engine PUBLIC hilog_ndk.z)
endif()

target_compile_options(cann_lm_engine PRIVATE
    -Wall -Wextra -Wpedantic
)

add_executable(api_smoke main/api_smoke.cpp)
target_link_libraries(api_smoke PRIVATE cann_lm_engine)
target_compile_options(api_smoke PRIVATE
    -Wall -Wextra -Wpedantic
)

add_executable(c_api_smoke main/c_api_smoke.c)
target_link_libraries(c_api_smoke PRIVATE cann_lm_engine)
target_compile_options(c_api_smoke PRIVATE
    -Wall -Wextra -Wpedantic
)

# 【临时调试代码 — 验证完成后删除】golden_verify: 模型精度验证工具
# 从 golden 二进制输入文件灌入模型，推理后 dump 输出为 output_x.bin
add_executable(golden_verify main/golden_verify.cpp)
target_link_libraries(golden_verify PRIVATE cann_lm_engine)
target_compile_options(golden_verify PRIVATE
    -Wall -Wextra -Wpedantic
)
target_include_directories(golden_verify PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)   # nlohmann/json

# ── 单元测试（GoogleTest，TEST-WRITE RED 阶段）──────────────────────────────
set(MODEL_EXECUTOR_TEST_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cann_lm/engine/core/model_executor")

find_package(GTest)
enable_testing()

if(GTest_FOUND)

add_executable(model_executor_ut
    ${MODEL_EXECUTOR_TEST_DIR}/test_kv_cache_pool.cpp
    ${MODEL_EXECUTOR_TEST_DIR}/test_model_executor.cpp
    ${MODEL_EXECUTOR_TEST_DIR}/test_dfx_collector.cpp
    ${MODEL_EXECUTOR_TEST_DIR}/test_tensor_manager.cpp
)
target_link_libraries(model_executor_ut PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(model_executor_ut PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(model_executor_ut PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(model_executor_ut)

# ── InputAssembler UT（含 gemma4 双 bool mask 用例，docs/mask_fill.md）──
# 独立目标：model_executor_ut 预存损坏（test_model_executor.cpp 等旧 API 文件
# host 编不过，见下条 ut_linear_state_pool 注释），本组件用例独立注册保证随
# host UT 回归跑。host 无 NDK/stub so 时触碰张量数据的用例 GTEST_SKIP
#（纯 name/索引防御用例无条件跑），stub .so 编法见 PROJECT.md「host UT 编译」节。
add_executable(ut_input_assembler ${MODEL_EXECUTOR_TEST_DIR}/test_input_assembler.cpp)
target_link_libraries(ut_input_assembler PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_input_assembler PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_input_assembler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_input_assembler)

# ── Qwen3.5 LinearStatePool UT（support_qwen3.5.md §4.4/§7）──
# 独立目标（不并入预存损坏的 model_executor_ut——其旧 API 测试文件 host 编不过，
# 独立注册保证本组件用例随 host UT 回归跑）：纯逻辑用例无后端可跑，创建类用例
# host 无 NDK so 时 GTEST_SKIP，见测试文件头注释。
add_executable(ut_linear_state_pool
    ${MODEL_EXECUTOR_TEST_DIR}/test_linear_state_pool.cpp)
target_link_libraries(ut_linear_state_pool PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_linear_state_pool PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_linear_state_pool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_linear_state_pool)

# ── Additional unit tests (tests/ut/) ─────────────────────────────────────────
set(UT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ut")

add_executable(ut_engine_executor ${UT_DIR}/ut_engine_executor.cpp)
target_link_libraries(ut_engine_executor PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_engine_executor PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_engine_executor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_engine_executor)

add_executable(ut_tokenizer ${UT_DIR}/ut_tokenizer.cpp)
target_link_libraries(ut_tokenizer PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_tokenizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ut_tokenizer PRIVATE
    TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/tokenizer")
target_include_directories(ut_tokenizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
gtest_discover_tests(ut_tokenizer)

add_executable(ut_c_api ${UT_DIR}/ut_c_api.cpp)
target_link_libraries(ut_c_api PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_c_api PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_c_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_c_api)

add_executable(ut_scheduler ${UT_DIR}/ut_scheduler.cpp)
target_link_libraries(ut_scheduler PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_scheduler PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_scheduler)

# ── Additional UT targets (tests/ut/) — added per docs/ut/DESIGN.md §7 ──────────

add_executable(ut_sampler ${UT_DIR}/ut_sampler.cpp)
target_link_libraries(ut_sampler PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_sampler PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_sampler)

add_executable(ut_request_handle ${UT_DIR}/ut_request_handle.cpp)
target_link_libraries(ut_request_handle PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_request_handle PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_request_handle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_request_handle)

# ── 权重共享链路 UT（shareWeightsModelId：shim 快速失败 + 默认0不绑定契约）──
# ohos_compat.h force-include：UT 直接包含 CANNKit 头（经 hiai_ndk_options.h →
# neural_network_core.h），需剥离 Clang __availability__ 属性（同 cann_backend 机制）。
add_executable(ut_backend_shareweights ${UT_DIR}/ut_backend_shareweights.cpp)
target_link_libraries(ut_backend_shareweights PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_backend_shareweights PRIVATE -Wall -Wextra -Wpedantic
    -include ${CMAKE_CURRENT_SOURCE_DIR}/hiai_acl/src/ohos_compat.h)
target_include_directories(ut_backend_shareweights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_backend_shareweights)

# ── 权重共享链路 UT · executor 层（Load share_weights_model_id 透传 + ModelId 暴露）──
# 独立目标（不并入 ut_backend_shareweights——本目标经链接期符号覆盖
# hiai::CreateModelManager/CreateBuiltModel/CreateNDTensorBuffer，会劫持其
# TC002 对真实 NDK impl 守卫的 host 验证；mock 对齐当前 backend API，
# 见测试文件头注释）。
add_executable(ut_share_weights_executor ${UT_DIR}/ut_share_weights_executor.cpp)
target_link_libraries(ut_share_weights_executor PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_share_weights_executor PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_share_weights_executor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_share_weights_executor)

# ── TextEmbedding UT（SR20260828001，DESIGN §8：落 tests/cann_lm/engine/core/）──

add_executable(ut_data_type_utils ${UT_DIR}/ut_data_type_utils.cpp)
target_link_libraries(ut_data_type_utils PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_data_type_utils PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_data_type_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_data_type_utils)

# ── TextEmbedding UT（SR20260828001，DESIGN §8：落 tests/ut/）──
add_executable(ut_text_embedding
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ut/ut_text_embedding.cpp)
target_link_libraries(ut_text_embedding PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
target_compile_options(ut_text_embedding PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_text_embedding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(ut_text_embedding)

# ── Tokenizer realdata UT（B 层，SR20260728001 M-8：检测 tmp/ 实物存在才注册）──
set(TMP_TOKENIZER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tmp")
if(EXISTS "${TMP_TOKENIZER_DIR}/qwen/tokenizer.json" AND EXISTS "${TMP_TOKENIZER_DIR}/gemma/tokenizer.json")
    add_executable(tokenizer_realdata ${UT_DIR}/ut_tokenizer_realdata.cpp)
    target_link_libraries(tokenizer_realdata PRIVATE cann_lm_engine GTest::gtest GTest::gtest_main)
    target_compile_options(tokenizer_realdata PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(tokenizer_realdata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(tokenizer_realdata PRIVATE
        TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/tokenizer"
        TOKENIZER_REALDATA_DIR="${TMP_TOKENIZER_DIR}")
    target_include_directories(tokenizer_realdata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
    gtest_discover_tests(tokenizer_realdata)
else()
    message(STATUS "tokenizer_realdata: tmp/qwen 或 tmp/gemma tokenizer.json 缺失，跳过 B 层测试目标")
endif()

endif()  # GTest_FOUND

# ── bbit: Black-Box Integration Test (tests/bbit/) ────────────────────────────
# bbit is a real user of the engine — links directly, calls all C API functions.
# In stub-era: Generate returns FAILURE (no backend); lifecycle still succeeds.
# In real deployment: full inference flow succeeds end-to-end.
set(BBIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/bbit")

add_executable(bbit ${BBIT_DIR}/bbit.cpp)
target_link_libraries(bbit PRIVATE cann_lm_engine)
target_compile_options(bbit PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(bbit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bbit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)  # nlohmann/json（D-09 vendored，配置解析）

# CTest: check mode (NULL defense) and full flow (config-driven, all C API functions).
add_test(NAME bbit_check COMMAND bbit --check)
add_test(NAME bbit_flow  COMMAND bbit --config ${BBIT_DIR}/testcase.json)

# host-stub 端到端(stub .so 在场 → 真 backend 跑通 SUCCESS;WORKING_DIRECTORY=repo 根
# 让 testcase_host/meta_host 内的相对路径(tests/bbit/meta_host.json、tests/fixtures/...)解析)
add_test(NAME bbit_host_flow
    COMMAND bbit --config ${BBIT_DIR}/testcase_host.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# ── stub_ndk(host 回归:external_ndk 层 OH_NN* 打桩 .so,shim/backend 零改)──────
# 仅 native host 构建接入(非交叉编译);OHOS 交叉编译用 ohos/CMakeLists.txt,不经此。
# EXCLUDE_FROM_ALL:probe 源仍引用已删除的 backend_base 旧接口,待迁移——
# 默认构建不编译,显式 --target 方可触发。
if(NOT CMAKE_CROSSCOMPILING)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../tests/stub_ndk stub_ndk_build EXCLUDE_FROM_ALL)
endif()
