# CANN-LM engine OHOS 独立构建(过渡版,首版交叉编译验证)
# 不改动 ../CMakeLists.txt(保证 Linux 原构建不受影响)。
# 仅在 -DOHOS_PLATFORM=OHOS 经 ohos.toolchain.cmake 交叉编译时使用。
#
# 产物:
#   libcann_lm_engine.so  (SHARED, 鸿蒙 musl 目标)
#   libc++_shared.so      (随 .so 分发)
#   (backend 经 add_subdirectory 接入 cann_backend 静态库,其自带 demo/test
#    EXCLUDE_FROM_ALL 不在默认构建产物中)
#   签名由仓库根 sign_ohos.sh 在 build 后跑 binary-sign-tool.jar -selfSign 1。
# UT 不交叉编译——UT 是 host 端验证(build.sh -u 跑),不部署到鸿蒙设备。
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
    ${_E}/pipeline/history_prompt_manager.cpp
    ${_E}/pipeline/pipeline_executor.cpp
    ${_E}/pipeline/preprocessor.cpp
    ${_E}/pipeline/scheduler.cpp
)

# ── SHARED 库 ───────────────────────────────────────────────────────────────
add_library(cann_lm_engine SHARED ${ENGINE_SRC})
# nlohmann/json: find_package (Findnlohmann_json.cmake, 含 vendored fallback)
# CMAKE_MODULE_PATH 让 cmake 找到 Findnlohmann_json.cmake;
# CANN_3RD_LIB_PATH 让 Find 模块搜到 output/third_party/json;
# CMAKE_FIND_ROOT_PATH 追加 third_party (toolchain 设了 ONLY, 不加则 find_package 只搜 OHOS SDK)。
list(APPEND CMAKE_MODULE_PATH "${_E}/../../cmake/third_party/modules")
set(CANN_3RD_LIB_PATH "${_E}/../../output/third_party" CACHE PATH "Third-party libraries directory")
list(APPEND CMAKE_FIND_ROOT_PATH "${_E}/../../output/third_party")
find_package(nlohmann_json REQUIRED)
target_include_directories(cann_lm_engine PUBLIC ${_E})
target_link_libraries(cann_lm_engine PUBLIC cann_backend hilog_ndk.z nlohmann_json::nlohmann_json)
target_compile_options(cann_lm_engine PRIVATE -Wall -Wextra -Wpedantic)

# SONAME: 让 bbit/api_smoke 等 NEEDED 记 libcann_lm_engine.so 而非绝对路径
target_link_options(cann_lm_engine PRIVATE "-Wl,-soname,libcann_lm_engine.so")

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

# ── 签名:不在 CMake 内做(本机 cmake 3.20 对 OHOS 未知系统的 POST_BUILD
#    生成器表达式求值不稳)。改由仓库根 sign_ohos.sh 在 build 后对 build_ohos/
#    下每个产物跑 binary-sign-tool.jar -selfSign 1。build_ohos.sh 会自动调用。
#    jar 与架构无关,本机 java 21 可跑。

message(STATUS "OHOS cross-build configured: ${_OHOS_TRIPLE}, sysroot=${CMAKE_SYSROOT}")
