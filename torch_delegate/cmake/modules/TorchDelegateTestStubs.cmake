# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
include_guard(GLOBAL)

function(torch_delegate_add_ge_runner_v2_stub)
    if(TARGET ge_runner_v2_stub)
        return()
    endif()

    set(_td_tests_dir "${TORCH_DELEGATE_SOURCE_DIR}/tests")
    set(_td_ge_stub_lib_dir "${CMAKE_BINARY_DIR}/tests/cpp/ge_stub/lib64")

    add_library(ge_runner_v2_stub SHARED
        "${_td_tests_dir}/cpp/ge_stub/ge_stub.cc"
    )

    set_target_properties(ge_runner_v2_stub PROPERTIES
        OUTPUT_NAME "ge_runner_v2"
        LIBRARY_OUTPUT_DIRECTORY "${_td_ge_stub_lib_dir}"
    )

    target_include_directories(ge_runner_v2_stub PRIVATE
        "${_td_tests_dir}/cpp"
    )

    target_link_libraries(ge_runner_v2_stub PRIVATE
        graph
        graph_base
        metadef
    )
    target_compile_definitions(ge_runner_v2_stub PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_compile_options(ge_runner_v2_stub PRIVATE ${DELEGATE_COMMON_FLAGS} -g)
    target_link_options(ge_runner_v2_stub PRIVATE ${DELEGATE_COMMON_FLAGS})
endfunction()

# td_epair_loader_c_stub 不走 ASAN，原因：
#   1) ASAN 运行时必须先于任何插桩模块完成初始化，要求主程序（首个加载的
#      可执行文件）链接 -fsanitize=address 或通过 LD_PRELOAD 预载 libasan.so，
#      后续 dlopen 的插桩模块才能解析到 __asan_init 等运行时符号；
#   2) 该 stub 供 Python UT 使用，由 pytest 进程通过 ctypes.CDLL 直接 dlopen，
#      而 CPython 解释器既未插桩也未预载 libasan，stub 一旦插桩，dlopen 时会
#      因找不到 __asan_* 符号直接加载失败，Python UT 全部用例无法执行；
#   3) 即使强行 LD_PRELOAD libasan，Python 解释器与 libtorch 均未插桩，ASAN
#      只能覆盖 ctypes 边界内的少量分配，收益有限且容易引入与被测逻辑无关
#      的噪声报错；
#   4) 真实 backend so 的内存安全已由 C++ UT 覆盖（build_test.sh 默认
#      ENABLE_ASAN=ON 并 LD_PRELOAD libasan 后运行 torch_delegate_tests），
#      Python UT 仅验证 ctypes 绑定层的签名/状态码映射/生命周期，不执行
#      真实 .epair 解析等内存敏感路径；
#   5) 故此处只引用不含 ASAN 的 DELEGATE_COMMON_WARNING_FLAGS。
function(torch_delegate_add_epair_loader_c_stub)
    if(TARGET td_epair_loader_c_stub)
        return()
    endif()

    set(_td_stub_lib_dir "${CMAKE_BINARY_DIR}/tests/python/epair_loader_stub/lib64")

    add_library(td_epair_loader_c_stub SHARED
        "${TORCH_DELEGATE_SOURCE_DIR}/tests/python/stub/epair_model_loader_c_stub.cc"
    )

    set_target_properties(td_epair_loader_c_stub PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_td_stub_lib_dir}"
    )

    target_compile_options(td_epair_loader_c_stub PRIVATE ${DELEGATE_COMMON_WARNING_FLAGS} -g)
    target_link_options(td_epair_loader_c_stub PRIVATE ${DELEGATE_COMMON_WARNING_FLAGS})
endfunction()
