# CANN-LM engine OHOS 独立构建(过渡版,首版交叉编译验证)
# 不改动 ../CMakeLists.txt(保证 Linux 原构建不受影响)。
# 仅在 -DOHOS_PLATFORM=OHOS 经 ohos.toolchain.cmake 交叉编译时使用。
#
# 产物:
#   libcann_lm_engine.so  (SHARED, 鸿蒙 musl 目标)
#   libc++_shared.so      (随 .so 分发)
#   9 个可执行: model_executor_ut / ut_engine_executor / ut_tokenizer /
#              ut_c_api / ut_scheduler / ut_text_embedding / bbit /
#              api_smoke / c_api_smoke
#   (backend 经 add_subdirectory 接入 cann_backend 静态库,其自带 demo/test
#    EXCLUDE_FROM_ALL 不在默认构建产物中)
#   若 -DSIGN_ON=ON(默认):每个产物再产 *_signed 版
# 后续:验证通过后再合入 ../CMakeLists.txt 的 OHOS 分支(共享源码清单)。

cmake_minimum_required(VERSION 3.16)
project(cann_lm_engine_ohos LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(_E ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests)

# ── backend 子项目(独立 cann_backend 库:api/device_base/acl/ndk + hiai_acl shim)──
# 与 ../CMakeLists.txt 相同:EXCLUDE_FROM_ALL 不进默认构建 backend 自带 demo/test;
# OHOS 部署通路 = NDK(CANN_BACKEND_DEFAULT_NDK 默认 ON,定义 CANN_DEVICE_NDK)。
add_subdirectory(${_E}/backend backend EXCLUDE_FROM_ALL)

# ── engine 源码清单(与 ../CMakeLists.txt 保持一致;过渡期复制) ──────────────
set(ENGINE_SRC
    ${_E}/api/cpp/engine_executor.cpp
    ${_E}/api/cpp/request_handle.cpp
    ${_E}/api/c/c_api.cpp
    ${_E}/core/tensor_manager.cpp
    ${_E}/core/context.cpp
    ${_E}/utils/data_type_utils.cpp
    ${_E}/utils/debug_tensor_dump.cpp # 【临时调试】推理前后 dump 输入/输出工位
    ${_E}/core/component/text_embedding/text_embedding.cpp
    ${_E}/core/component/model_executor/model_executor.cpp
    ${_E}/core/component/kv_cache_pool/kv_cache_pool.cpp
    ${_E}/core/component/linear_state_pool/linear_state_pool.cpp   # Qwen3.5 线性状态池(support_qwen3.5.md §4.4)
    ${_E}/core/component/input_assembler/input_assembler.cpp
    ${_E}/core/component/dfx_collector/dfx_collector.cpp
    ${_E}/core/component/sampler/sampler.cpp
    ${_E}/core/component/tokenizer/tokenizer.cpp
    ${_E}/core/component/tokenizer/bpe_tokenizer.cpp
    ${_E}/core/component/tokenizer/qwen_tokenizer.cpp
    ${_E}/core/component/tokenizer/vocab_store.cpp
    ${_E}/core/component/tokenizer/special_token_handler.cpp
    ${_E}/core/sample/sampler.cc
    ${_E}/pipeline/history_prompt_manager.cpp
    ${_E}/pipeline/pipeline_executor.cpp
    ${_E}/pipeline/preprocessor.cpp
    ${_E}/pipeline/scheduler.cpp
)

# ── SHARED 库 ───────────────────────────────────────────────────────────────
add_library(cann_lm_engine SHARED ${ENGINE_SRC})
target_include_directories(cann_lm_engine PUBLIC ${_E})
target_include_directories(cann_lm_engine PRIVATE ${_E}/../../third_party)   # nlohmann/json（D-09 vendored）
# backend 通路:工厂/设备模型/张量的具体实现(含 hiai_acl shim 依赖,PIC 由 toolchain 设定)
# hilog_ndk.z:infra/log.h 的 __OHOS__ 分支(OH_LOG_Print)所需
target_link_libraries(cann_lm_engine PUBLIC cann_backend hilog_ndk.z)
target_compile_options(cann_lm_engine PRIVATE -Wall -Wextra -Wpedantic)

# RPATH 设为 $ORIGIN,便于在鸿蒙设备上就近找到 libc++_shared.so / libcann_lm_engine.so
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "\$ORIGIN")
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# 拷贝 libc++_shared.so 到产物目录(运行时 C++ 运行库)
set(_CXX_SHARED "${OHOS_SDK_NATIVE}/llvm/lib/aarch64-linux-ohos/libc++_shared.so")
if(EXISTS "${_CXX_SHARED}")
    add_custom_command(TARGET cann_lm_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${_CXX_SHARED}" "$<TARGET_FILE_DIR:cann_lm_engine>/libc++_shared.so"
        COMMENT "Copy libc++_shared.so to output dir")
else()
    message(WARNING "libc++_shared.so not found at ${_CXX_SHARED}; copy manually for device run")
endif()

# ── GTest:本地源码 FetchContent(交叉编译不能用主机 find_package) ──────────
# gtest_discover_tests 在交叉编译下失效(它要运行宿主跑不了的目标二进制),
# 故改用静态 add_test(NAME ... COMMAND <exe>),由设备端 ctest 或直接 ./ 运行。
include(FetchContent)
set(GTEST_TAR "/mnt/workspace/gtest-1.14.0.tar.gz" CACHE FILEPATH "Path to googletest v1.14.0 source tarball")
if(NOT EXISTS "${GTEST_TAR}")
    message(FATAL_ERROR "GTest source tarball not found: ${GTEST_TAR} (download from https://github.com/google/googletest/releases/tag/v1.14.0)")
endif()
FetchContent_Declare(googletest URL "${GTEST_TAR}")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# googletest 1.14 无 BUILD_GTEST 选项;其自带测试/样例由 gtest_build_tests /
# gtest_build_samples 控制,二者默认 OFF,故无需显式抑制。
FetchContent_MakeAvailable(googletest)

enable_testing()

# ── 辅助:注册一个 gtest 可执行 + 静态 ctest 项 ──────────────────────────────
function(cannlm_add_gtest_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE cann_lm_engine gtest gtest_main)
    target_include_directories(${name} PRIVATE ${_E})
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    # 静态注册:设备端 ctest 直接执行整个二进制(交叉编译不可 discover)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ── UT 可执行 ─────────────────────────────────────────────────────────────────
# ⚠️ model_executor_ut / ut_text_embedding 暂不注册：源 mock 旧 cann_llm_engine
# backend 接口（GetDesc/.name/.size/DataType::UNDEFINED 等），未随 hiai 迁移改编，
# 编入必编译失败。迁完后恢复（host 侧 ../CMakeLists.txt 同因、无 GTest 跳过）。
# （另 ut_sampler/ut_request_handle/tokenizer_realdata 历史上即未在 OHOS 注册。）
set(ME_TEST_DIR ${TESTS_ROOT}/cann_lm/engine/core/model_executor)
cannlm_add_gtest_test(model_executor_ut
    ${ME_TEST_DIR}/test_kv_cache_pool.cpp
    ${ME_TEST_DIR}/test_input_assembler.cpp
    ${ME_TEST_DIR}/test_model_executor.cpp
    ${ME_TEST_DIR}/test_dfx_collector.cpp
    ${ME_TEST_DIR}/test_tensor_manager.cpp)

set(UT_DIR ${TESTS_ROOT}/ut)
cannlm_add_gtest_test(ut_engine_executor ${UT_DIR}/ut_engine_executor.cpp)
cannlm_add_gtest_test(ut_tokenizer       ${UT_DIR}/ut_tokenizer.cpp)
target_compile_definitions(ut_tokenizer PRIVATE
    TOKENIZER_FIXTURE_DIR="${TESTS_ROOT}/fixtures/tokenizer")
target_include_directories(ut_tokenizer PRIVATE ${_E}/../../third_party)
cannlm_add_gtest_test(ut_c_api           ${UT_DIR}/ut_c_api.cpp)
cannlm_add_gtest_test(ut_scheduler       ${UT_DIR}/ut_scheduler.cpp)

# ── TextEmbedding UT（SR20260828001；与 ../CMakeLists.txt 一一对应，勿漂移）──
# ⚠️ 同上暂不注册：mock 旧 backend 接口未迁移（__fp16 直接引用亦失效，
#    text_embedding.cpp 已改软件 fp16）。
set(TE_TEST_DIR ${TESTS_ROOT}/cann_lm/engine/core)
cannlm_add_gtest_test(ut_text_embedding ${TE_TEST_DIR}/ut_text_embedding.cpp)

# ── bbit:黑盒集成(自带 main,非 gtest;ctest 跑 --check 与配置驱动完整流程) ──
add_executable(bbit ${TESTS_ROOT}/bbit/bbit.cpp)
target_link_libraries(bbit PRIVATE cann_lm_engine)
target_include_directories(bbit PRIVATE ${_E})
target_include_directories(bbit PRIVATE ${_E}/../../third_party)   # nlohmann/json（D-09 vendored，配置解析）
target_compile_options(bbit PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME bbit_check COMMAND bbit --check)
add_test(NAME bbit_flow  COMMAND bbit --config ${TESTS_ROOT}/bbit/testcase.json)

# ── 冒烟程序(C++ / C,自带 main) ───────────────────────────────────────────
add_executable(api_smoke ${_E}/main/api_smoke.cpp)
target_link_libraries(api_smoke PRIVATE cann_lm_engine)
target_include_directories(api_smoke PRIVATE ${_E})
target_compile_options(api_smoke PRIVATE -Wall -Wextra -Wpedantic)

add_executable(c_api_smoke ${_E}/main/c_api_smoke.c)
target_link_libraries(c_api_smoke PRIVATE cann_lm_engine)
target_include_directories(c_api_smoke PRIVATE ${_E})
target_compile_options(c_api_smoke PRIVATE -Wall -Wextra -Wpedantic)

# 【临时调试代码 — 验证完成后删除】golden_verify: 模型精度验证工具
add_executable(golden_verify ${_E}/main/golden_verify.cpp)
target_link_libraries(golden_verify PRIVATE cann_lm_engine)
target_include_directories(golden_verify PRIVATE ${_E})
target_include_directories(golden_verify PRIVATE ${_E}/../../third_party)   # nlohmann/json
target_compile_options(golden_verify PRIVATE -Wall -Wextra -Wpedantic)

# ── 签名:不在 CMake 内做(本机 cmake 3.20 对 OHOS 未知系统的 POST_BUILD
#    生成器表达式求值不稳)。改由仓库根 sign_ohos.sh 在 build 后对 build_ohos/
#    下每个产物跑 binary-sign-tool.jar -selfSign 1。build_ohos.sh 会自动调用。
#    jar 与架构无关,本机 java 21 可跑。

message(STATUS "OHOS cross-build configured: ${_OHOS_TRIPLE}, sysroot=${CMAKE_SYSROOT}")
