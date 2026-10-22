/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file tests/stub_ndk/probe_forward.cpp
 * @brief Probe:验证 A2 修复—未加载 ModelExecutor::Forward 返 FAILURE(原 INVALID_STATE)。
 *
 * bbit happy path(Load 成功)不触达 not-ready 分支,故 A2 经 bbit 验证不了。
 * 本探针直构 nullptr pool/asm/dfx 的 ModelExecutor(operator bool=false)→ Forward →
 * 断言返 FAILURE(1),非 INVALID_STATE(3)。
 */
#include "core/component/model_executor.h"

#include <cstdio>

int main() {
    cann_lm::engine::ModelExecutor exec(nullptr, nullptr, nullptr);
    cann_lm::engine::ForwardOptions opts;  // default(空 slots)— not-ready 分支先于 slots 检查
    cann_lm::engine::Status st = exec.Forward(opts);
    fprintf(stderr, "[probe_forward] Forward on not-loaded -> code=%d\n", st.code);
    // A2:未加载应返 FAILURE(1);原 bug 返 INVALID_STATE(3)
    if (st.code == cann_lm::engine::Status::FAILURE) {
        printf("probe_forward: PASS (FAILURE, not INVALID_STATE)\n");
        return 0;
    }
    fprintf(stderr, "FAIL: expected FAILURE(1), got %d (INVALID_STATE=3? A2 未修?)\n", st.code);
    return 1;
}
