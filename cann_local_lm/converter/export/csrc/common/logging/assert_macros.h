#ifndef TORCH_DELEGATE_COMMON_LOGGING_ASSERT_MACROS_H
#define TORCH_DELEGATE_COMMON_LOGGING_ASSERT_MACROS_H

#include <cstdarg>
#include <memory>
#include <string>
#include <vector>
#include "common/logging/logging.h"
#include "torch_delegate/td_types.h"

namespace td {

struct ErrorResult {
    operator bool() const { return false; }

    operator Status() const { return INVALID_PARAM; }

    template <typename T>
    operator std::unique_ptr<T>() const
    {
        return nullptr;
    }

    template <typename T>
    operator std::shared_ptr<T>() const
    {
        return nullptr;
    }

    template <typename T>
    operator T*() const
    {
        return nullptr;
    }

    template <typename T>
    operator std::vector<T>() const
    {
        return {};
    }

    operator std::string() const { return ""; }

    template <typename T>
    operator T() const
    {
        return T();
    }
};

inline std::string CreateErrorMsg(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    const size_t len = static_cast<size_t>(vsnprintf(nullptr, 0, format, args_copy));
    va_end(args_copy);

    std::string msg(len + 1, '\0');
    vsnprintf(msg.data(), len + 1, format, args);
    va_end(args);

    msg.resize(len);
    return msg;
}

inline std::string CreateErrorMsg() { return ""; }

inline const char* StatusName(const Status status) noexcept
{
    switch (status) {
        case SUCCESS:
            return "SUCCESS";
        case FAILED:
            return "FAILED";
        case INVALID_PARAM:
            return "INVALID_PARAM";
        case INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case UNIMPLEMENTED:
            return "UNIMPLEMENTED";
        default:
            return "UNKNOWN_STATUS";
    }
}

} // namespace td

#define TD_ASSERT(expr, ...)                                   \
    do {                                                       \
        if (!(expr)) {                                         \
            std::string msg = td::CreateErrorMsg(__VA_ARGS__); \
            if (msg.empty()) {                                 \
                TDLOGE("Assert failed: %s", #expr);            \
            } else {                                           \
                TDLOGE("%s", msg.c_str());                     \
            }                                                  \
            return td::ErrorResult();                          \
        }                                                      \
    } while (0)

#define TD_ASSERT_NOTNULL(ptr, ...) TD_ASSERT((ptr) != nullptr, __VA_ARGS__)
#define TD_ASSERT_TRUE(expr, ...) TD_ASSERT((expr), __VA_ARGS__)

#define TD_CHK_STATUS(expr)                                                                    \
    do {                                                                                       \
        td::Status status = (expr);                                                            \
        if (status != td::SUCCESS) {                                                           \
            TDLOGE("Check failed: %s, code: %lld (%s)", #expr, static_cast<long long>(status), \
                   td::StatusName(status));                                                    \
            return status;                                                                     \
        }                                                                                      \
    } while (0)

#endif
