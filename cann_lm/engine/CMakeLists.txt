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

# engine 源清单(STATIC 与 host SHARED 共用,避免 whole-archive 包装):
# SHARED 直接从源编 → C API(extern "C", 默认导出)符号自然进 .so,无需 --whole-archive。
set(CANN_LM_ENGINE_SRCS
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

    pipeline/history_prompt_manager.cpp
    pipeline/pipeline_executor.cpp
    pipeline/preprocessor.cpp
    pipeline/scheduler.cpp
)

add_library(cann_lm_engine STATIC ${CANN_LM_ENGINE_SRCS})

target_include_directories(cann_lm_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/third_party   # nlohmann/json（D-09 vendored）
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

# ── host SHARED .so: 供 python ctypes 绑定(FFI)本机加载跑 Tier B 单测 ────────
# 仅 native host(非交叉编译): OHOS 的 libcann_lm_engine.so 由 build_ohos.sh +
# package_ohos.sh 产,不经此。SHARED 复用同源 + 链 cann_backend(引擎源引用其 API,
# 按需拉对象);C API extern "C" 默认导出(无 -fvisibility=hidden)。
# 产物 libcann_lm_engine.so 落 ${CMAKE_BINARY_DIR};运行期 NDK 调用经 NDKProxy
# dlopen——本机由 CANN_LM_NDK_LIB_DIR 指 build 目录 stub .so 顶(见 ndk_proxy.cpp)。
if(NOT CMAKE_CROSSCOMPILING)
    add_library(cann_lm_engine_shared SHARED ${CANN_LM_ENGINE_SRCS})
    target_include_directories(cann_lm_engine_shared
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/third_party)
    target_link_libraries(cann_lm_engine_shared PRIVATE cann_backend)
    target_compile_options(cann_lm_engine_shared PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(cann_lm_engine_shared PROPERTIES
        OUTPUT_NAME cann_lm_engine
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# ── UT 由 tests/cpp/engine/CMakeLists.txt 定义（顶层 BUILD_TESTS=ON 时 add_subdirectory）──
