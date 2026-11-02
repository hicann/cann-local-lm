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

# 三方库 (nlohmann_json / GTest): Find 模块 (cmake/third_party/modules → output/third_party)
# 与 examples/test_engine (bbit) 同款机制; 子目录模式 (OHOS) 下父级已建 target 则模块早退复用
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/third_party/modules)
# 尊重外部传入 (CI 镜像: build_test.sh 透传 -DCANN_3RD_LIB_PATH=...; OHOS 父级 :70 设 CACHE)。
# 无条件 set 会把 CI 传入值覆盖成仓内 output/third_party (CI 上不存在), FindGTest 三级
# 搜索全落空 → GTest::gtest target not found; FindGTest 静态库级(第二级)本可直接命中镜像
# 的 gtest.h + libgtest.a + libgtest_main.a (裸静态库布局, 无 GTestConfig.cmake)
if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../output/third_party)
endif()
find_package(nlohmann_json REQUIRED)

enable_testing()

# ── 辅助函数：注册 UT 可执行 + ctest ──
# CANN_LM_CROSS_BUILD=ON 时（OHOS 交叉编译）用 add_test 静态注册（不能运行目标二进制）
if(NOT CANN_LM_CROSS_BUILD)
    # REQUIRED: 缺 gtest 时由 FindGTest.cmake 直接 FATAL_ERROR（带三条修复
    # 选项），否则静默失败后在 target_link_libraries 处爆晦涩的 target not found
    find_package(GTest REQUIRED)
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

# ── model_executor UT（多源文件合一；原 model_executor/ 子目录已上移拍平）──
add_executable(model_executor_ut
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kv_cache_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dfx_collector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_manager.cpp
)
target_link_libraries(model_executor_ut PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(model_executor_ut PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(model_executor_ut PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(model_executor_ut
    PROPERTIES ENVIRONMENT "CANN_LM_NDK_LIB_DIR=${CMAKE_BINARY_DIR}")

# ── InputAssembler UT ──
add_executable(ut_input_assembler ${CMAKE_CURRENT_SOURCE_DIR}/test_input_assembler.cpp)
target_link_libraries(ut_input_assembler PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_input_assembler PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_input_assembler PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_input_assembler
    PROPERTIES ENVIRONMENT "CANN_LM_NDK_LIB_DIR=${CMAKE_BINARY_DIR}")

# ── LinearStatePool UT（文件存在才注册）──
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_linear_state_pool.cpp")
add_executable(ut_linear_state_pool ${CMAKE_CURRENT_SOURCE_DIR}/test_linear_state_pool.cpp)
target_link_libraries(ut_linear_state_pool PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_linear_state_pool PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_linear_state_pool PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_linear_state_pool)
endif()

# ── 独立 UT targets ──
# ut_engine_executor 的 KvPoolEnsureTest 组走真建 KV tensor（NDK 代理路径）：
# 注册期须注入 stub .so 目录（同 model_executor_ut/ut_input_assembler/
# ut_text_embedding）——缺此环境变量时 NDKProxy 找不到桩库，TC608/609/614
# 在 ctest 下环境性误红（直跑带变量则 15/15 全绿）。cannlm_add_engine_ut
# 不透传注册参数（ARGN 全量入 add_executable），故用显式注册
add_executable(ut_engine_executor ${CMAKE_CURRENT_SOURCE_DIR}/test_engine_executor.cpp)
target_link_libraries(ut_engine_executor PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_engine_executor PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_engine_executor PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_engine_executor
    PROPERTIES ENVIRONMENT "CANN_LM_NDK_LIB_DIR=${CMAKE_BINARY_DIR}")
cannlm_add_engine_ut(ut_c_api           ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api.cpp)
cannlm_add_engine_ut(ut_scheduler       ${CMAKE_CURRENT_SOURCE_DIR}/test_scheduler.cpp)
cannlm_add_engine_ut(ut_sampler        ${CMAKE_CURRENT_SOURCE_DIR}/test_sampler.cpp)
cannlm_add_engine_ut(ut_request_handle ${CMAKE_CURRENT_SOURCE_DIR}/test_request_handle.cpp)
cannlm_add_engine_ut(ut_share_weights_executor ${CMAKE_CURRENT_SOURCE_DIR}/test_share_weights_executor.cpp)
cannlm_add_engine_ut(ut_data_type_utils ${CMAKE_CURRENT_SOURCE_DIR}/test_data_type_utils.cpp)

# ── PromptCache UT（环形 KV 池/缓存台账/增量切块，docs/prompt_cache_spec.md）──
cannlm_add_engine_ut(ut_prompt_cache ${CMAKE_CURRENT_SOURCE_DIR}/test_prompt_cache.cpp)

# ── Scheduler 多块 prefill 出口语义 UT（is_prefill_last 旗）──
# 独立目标缘由（同 ut_share_weights_executor）：经链接期符号覆盖 hiai:: 工厂做
# mock；不并入 ut_scheduler——其 12 例契约恰是 "host 真工厂返 nullptr"，
# mock 覆盖会反转其被测前提（见测试文件头注释）。
cannlm_add_engine_ut(ut_scheduler_multichunk ${CMAKE_CURRENT_SOURCE_DIR}/test_scheduler_multichunk.cpp)

# ── eos 停止（meta tokenizer_config.eos_token_id，2026-10-29 裁决）管线级 UT ──
# 独立目标缘由（同 ut_scheduler_multichunk）：链接期符号覆盖 hiai:: 工厂做
# mock（input_embed INT32 TokenId 路径 / 输出 {1,512} logits 行 / Run 记
# new_kv_cache_pos 作台账观测通道）；fixture 路径经宏绝对注入（meta
# tokenizer_file 须可寻，与 ctest CWD 无关）
cannlm_add_engine_ut(ut_eos_stop ${CMAKE_CURRENT_SOURCE_DIR}/test_eos_stop.cpp)
target_compile_definitions(ut_eos_stop PRIVATE
    EOS_TOKENIZER_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer")

# ── VocabStore L4 .bincache sidecar UT（perf_optimize.md §5 L4①-④）──
cannlm_add_engine_ut(ut_vocab_store ${CMAKE_CURRENT_SOURCE_DIR}/test_vocab_store.cpp)

# ── Logger 格式化 UT（stderr_capture.h 与源文件同目录，相对 include 自解析）──
cannlm_add_engine_ut(ut_logger ${CMAKE_CURRENT_SOURCE_DIR}/test_logger.cpp)

# ut_backend_shareweights 需要 ohos_compat.h force-include
add_executable(ut_backend_shareweights ${CMAKE_CURRENT_SOURCE_DIR}/test_backend_shareweights.cpp)
target_link_libraries(ut_backend_shareweights PRIVATE cann_lm_engine ${_GTEST_LIBS})
target_compile_options(ut_backend_shareweights PRIVATE -Wall -Wextra -Wpedantic
    -include ${_ENGINE_DIR}/hiai_acl/src/ohos_compat.h)
target_include_directories(ut_backend_shareweights PRIVATE ${_ENGINE_DIR})
cannlm_register_tests(ut_backend_shareweights)

# ── ut_tokenizer（需要 fixture dir + nlohmann/json include）──
add_executable(ut_tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/test_tokenizer.cpp)
target_link_libraries(ut_tokenizer PRIVATE cann_lm_engine ${_GTEST_LIBS} nlohmann_json::nlohmann_json)
target_compile_options(ut_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ut_tokenizer PRIVATE ${_ENGINE_DIR})
target_compile_definitions(ut_tokenizer PRIVATE
    TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer")
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
    target_link_libraries(tokenizer_realdata PRIVATE cann_lm_engine ${_GTEST_LIBS} nlohmann_json::nlohmann_json)
    target_compile_options(tokenizer_realdata PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(tokenizer_realdata PRIVATE ${_ENGINE_DIR})
    target_compile_definitions(tokenizer_realdata PRIVATE
        TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer"
        TOKENIZER_REALDATA_DIR="${TMP_TOKENIZER_DIR}")
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
