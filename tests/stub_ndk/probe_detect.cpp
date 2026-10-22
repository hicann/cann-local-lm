/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file tests/stub_ndk/probe_detect.cpp
 * @brief Probe:验证 A1 修复—DetectAvailableDevice stub .so 在场返 NDK,缺席返 NONE。
 *
 * 运行两次:① build/ 下(stub .so 旁 exe)→ 期望 NDK(exit 1);
 *           ② 拷到无 .so 的干净目录 → 期望 NONE(exit 2,原 bug 返 NDK 假探测)。
 * exit code = (int)DeviceType(1=NDK, 2=NONE),便于脚本判定。
 */
#include "device_base/cann_device_factory.h"

#include <cstdio>

int main() {
    cann_llm_engine::DeviceType t = cann_llm_engine::DetectAvailableDevice();
    fprintf(stderr, "[probe_detect] DetectAvailableDevice -> %d (1=NDK, 2=NONE)\n", static_cast<int>(t));
    printf("probe_detect: type=%d\n", static_cast<int>(t));
    return static_cast<int>(t);  // 1=NDK(stub 在场),2=NONE(无 stub,A1 修复后诚实)
}
