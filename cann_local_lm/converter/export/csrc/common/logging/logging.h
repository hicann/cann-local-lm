#ifndef TORCH_DELEGATE_COMMON_LOGGING_LOGGING_H
#define TORCH_DELEGATE_COMMON_LOGGING_LOGGING_H

#include <cstring>
#include <string>
#include <vector>
#include "acl/acl_base_rt.h"

namespace td {
enum class IssueSeverity { Error, Warning };

struct ValidationIssue {
    IssueSeverity severity;
    std::string code;
    std::string message;
    std::string context;
};

struct ValidationResult {
    bool ok;
    std::vector<ValidationIssue> issues;
};

inline const char* TDBaseName(const char* path)
{
    const char* base = std::strrchr(path, '/');
    return base == nullptr ? path : base + 1;
}

inline std::string AddError(ValidationResult& result, const std::string& code, const std::string& message,
                            const std::string& context = "")
{
    result.ok = false;
    result.issues.push_back({IssueSeverity::Error, code, message, context});
    std::string summary = "Validation error: code=" + code + ", message=" + message;
    if (!context.empty()) {
        summary += ", context=" + context;
    }
    return summary;
}
} // namespace td
#define TDLOGI(fmt, ...) aclAppLog(ACL_INFO, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOGE(fmt, ...) aclAppLog(ACL_ERROR, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOGW(fmt, ...) aclAppLog(ACL_WARNING, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#define TDLOGD(fmt, ...) aclAppLog(ACL_DEBUG, __FUNCTION__, td::TDBaseName(__FILE__), __LINE__, fmt, ##__VA_ARGS__)
#endif
