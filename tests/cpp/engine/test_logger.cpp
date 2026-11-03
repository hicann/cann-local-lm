#include "infra/logger.h"

#include "stderr_capture.h"

#include <gtest/gtest.h>

#include <string>

namespace cann_local_lm::engine {
namespace {

TEST(LoggerLogfTest, FormatsIntAndString)
{
    Logger logger(LogLevel::Trace);
    test_util::StderrCapture cap;
    logger.Logf(LogLevel::Error, "loader.load.init: backend Init failed (st=%d, dev=%s)", 42, "npu");
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("[ERROR] loader.load.init: backend Init failed (st=42, dev=npu)"), std::string::npos);
}

TEST(LoggerLogfTest, RespectsLevelFilter)
{
    Logger logger(LogLevel::Warn);
    test_util::StderrCapture cap;
    logger.Logf(LogLevel::Info, "filtered.message: %d", 1);
    logger.Logf(LogLevel::Warn, "kept.message: %d", 2);
    const std::string out = cap.Stop();
    EXPECT_EQ(out.find("filtered.message"), std::string::npos);
    EXPECT_NE(out.find("[WARN] kept.message: 2"), std::string::npos);
}

TEST(LoggerLogfTest, TruncatesOverlongMessageWithoutCrash)
{
    Logger logger(LogLevel::Trace);
    test_util::StderrCapture cap;
    const std::string long_msg(4096, 'x');
    logger.Logf(LogLevel::Error, "%s", long_msg.c_str());
    const std::string out = cap.Stop();
    // 1KB 缓冲截断：不崩溃、有输出、且不短于缓冲上限的一半
    EXPECT_NE(out.find("[ERROR] xxx"), std::string::npos);
    EXPECT_GT(out.size(), 512u);
}

TEST(LoggerLogfTest, EmptyFormatArgs)
{
    Logger logger(LogLevel::Trace);
    test_util::StderrCapture cap;
    logger.Logf(LogLevel::Warn, "plain.message");
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("[WARN] plain.message"), std::string::npos);
}

TEST(LoggerLogTest, ExistingLogBehaviorUnchanged)
{
    Logger logger(LogLevel::Trace);
    test_util::StderrCapture cap;
    logger.Log(LogLevel::Warn, "tokenizer.factory.create: plain log");
    const std::string out = cap.Stop();
    EXPECT_NE(out.find("[WARN] tokenizer.factory.create: plain log"), std::string::npos);
}

} // namespace
} // namespace cann_local_lm::engine
