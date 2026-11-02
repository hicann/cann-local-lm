/* Copyright (C) 2020-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * License for more details at:
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// ═══════════════════════════════════════════════════════════════════════════
// 类式 Logger（实例级级别过滤 + printf 风格 + 1KB 截断防线）。
// 与 log.h（全局宏、无级别门控）互补：需要按实例控制级别/复用同一前缀格式的
// 场景（加载器/管线分阶段日志）用本类；散点诊断仍走宏。
// 由来：dcdd77f 合入带入 tests/.../test_logger.cpp（契约即本头的 API 面），
// 其依赖的 infra/logger.h 未随合入落仓——本头按该测试契约补齐
// （Logger(LogLevel)/Logf/Log、级别过滤序 Trace<Info<Warn<Error、
// "[LEVEL] msg" 输出行、>1KB 截断不崩溃）。
// ═══════════════════════════════════════════════════════════════════════════
#ifndef INFRA_LOGGER_H
#define INFRA_LOGGER_H

#include <cstdarg>
#include <cstdio>

namespace cann_lm::engine {

/// 日志级别（序即过滤序：低于实例阈值的级别被丢弃）。
enum class LogLevel { Trace = 0, Info = 1, Warn = 2, Error = 3 };

/// 实例级 Logger：构造定阈值，Logf/Log 输出到 stderr。
/// 输出行格式 "[LEVEL] <message>\n"（无文件/行号——调用方语境前缀自行
/// 进 message，与 CANN_LLM_LOG* 宏的 file/func/line 尾缀格式互不影响）。
/// 非线程安全（单实例单线程持有；跨线程各自持实例或外部加锁）。
class Logger {
public:
    explicit Logger(LogLevel threshold) : threshold_(threshold) {}

    /// printf 风格输出。消息体经 1KB 栈缓冲 vsnprintf——超长截断
    /// （保证 NUL 终止、不崩溃、不二次扩内存；日志路径不引入分配）。
    void Logf(LogLevel level, const char* fmt, ...)
    {
        if (level < threshold_)
            return;
        char buf[kMaxMessageBytes];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        Emit(level, buf);
    }

    /// 整串输出（无格式化；语义同 Logf("%s", msg) 但免一次解析）。
    void Log(LogLevel level, const char* msg)
    {
        if (level < threshold_)
            return;
        Emit(level, msg != nullptr ? msg : "(null)");
    }

    LogLevel threshold() const noexcept { return threshold_; }

private:
    static constexpr std::size_t kMaxMessageBytes = 1024; // 消息体上限（截断防线）

    void Emit(LogLevel level, const char* msg) const { std::fprintf(stderr, "[%s] %s\n", LevelName(level), msg); }

    static const char* LevelName(LogLevel level) noexcept
    {
        switch (level) {
            case LogLevel::Trace:
                return "TRACE";
            case LogLevel::Info:
                return "INFO";
            case LogLevel::Warn:
                return "WARN";
            case LogLevel::Error:
                return "ERROR";
        }
        return "LOG"; // 防御（枚举外值）
    }

    LogLevel threshold_;
};

} // namespace cann_lm::engine

#endif // INFRA_LOGGER_H
