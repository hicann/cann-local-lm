#pragma once

// UT 专用：RAII 捕获 stderr（dup2 重定向到 tmpfile），供断言 Logger 输出内容。
// 用法：{ StderrCapture cap; <触发日志>; std::string out = cap.Stop(); }
// Stop() 恢复 stderr 并返回捕获内容；析构幂等。

#include <cstdio>
#include <string>
#include <unistd.h>

namespace cann_lm::engine::test_util {

class StderrCapture {
public:
    StderrCapture() : saved_fd_(::dup(STDERR_FILENO)), file_(std::tmpfile())
    {
        if (saved_fd_ >= 0 && file_ != nullptr) {
            ::fflush(stderr);
            ::dup2(::fileno(file_), STDERR_FILENO);
            active_ = true;
        } else if (saved_fd_ >= 0) {
            ::close(saved_fd_); // tmpfile 失败：不重定向，归还 dup 出的 fd
            saved_fd_ = -1;
        }
    }

    ~StderrCapture() { Stop(); }

    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    /// 恢复 stderr 并返回捕获到的全部内容（幂等，只读一次）。
    std::string Stop()
    {
        if (active_ && saved_fd_ >= 0) {
            ::fflush(stderr);
            ::dup2(saved_fd_, STDERR_FILENO);
            ::close(saved_fd_);
            saved_fd_ = -1;
            active_ = false;
        }
        if (file_ != nullptr) {
            if (!read_) {
                char buf[4096];
                size_t n;
                ::rewind(file_);
                while ((n = ::fread(buf, 1, sizeof(buf), file_)) > 0)
                    captured_.append(buf, n);
                read_ = true;
            }
            ::fclose(file_); // tmpfile 已 unlink，读毕关闭归还 fd
            file_ = nullptr;
        }
        return captured_;
    }

private:
    int saved_fd_ = -1;
    std::FILE* file_ = nullptr;
    bool active_ = false;
    bool read_ = false;
    std::string captured_;
};

} // namespace cann_lm::engine::test_util
