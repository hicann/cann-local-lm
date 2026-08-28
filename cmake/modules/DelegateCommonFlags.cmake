include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# 公共编译/链接选项 (AGENTS.md「公共 Flag 原则」+「零警告原则」+「ASAN 强制原则」)
#
# 所有 C++ target 必须通过 delegate_apply_common_flags() 接入，禁止在单个
# target 上直接硬编码 -Wall/-Wextra/-Werror 等公共 flags。
#
# - compile: DELEGATE_COMMON_FLAGS (-Wall -Wextra -Werror, 零警告)
# - link   : 不带 -Wall/-Wextra/-Werror (linker 不认)；仅 ASAN 时追加 -fsanitize
# - ASAN  : option(ENABLE_ASAN)，C++ 测试强制开启 (`build_cpp.sh --asan && ctest --test-dir build`)
#           backend lib / example binary 默认 OFF (避免 .so 注入污染 GE 运行时)
# ---------------------------------------------------------------------------

# 注意: -Werror 放 compile 选项；AGENTS.md「零警告原则」要求 PR 不引入新警告。
set(DELEGATE_COMMON_FLAGS -Wall -Wextra -Werror CACHE STRING
    "Common compile flags for CANN-LM C++ targets (零警告原则)")

option(ENABLE_ASAN "Enable AddressSanitizer (C++ tests 强制)" OFF)

# ASAN 专用 flags (compile + link)，仅在 ENABLE_ASAN=ON 时生效。
# -static-libasan: 静态链入 libasan, 免 LD_PRELOAD (否则 "ASan runtime does not come
#   first in initial library list" → ctest 裸跑失败)。加后 ctest --test-dir build 可直接跑 ASAN 二进制。
set(DELEGATE_ASAN_FLAGS -fsanitize=address -static-libasan -fno-omit-frame-pointer)

# delegate_apply_common_flags(<target>)
#
# 给 target 接入公共编译/链接选项。target 自身的非公共选项仍可用
# target_compile_options/target_link_options 单独追加。
function(delegate_apply_common_flags TARGET)
    target_compile_options(${TARGET} PRIVATE ${DELEGATE_COMMON_FLAGS})
    if(ENABLE_ASAN)
        target_compile_options(${TARGET} PRIVATE ${DELEGATE_ASAN_FLAGS})
        target_link_options(${TARGET} PRIVATE -fsanitize=address -static-libasan)
    endif()
endfunction()
