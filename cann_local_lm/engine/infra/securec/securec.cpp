/* cann_llm backend support: securec 最小兼容层实现（host 无系统 securec 库时编入）。 */
#include "securec.h"
#include <string.h>
#include <errno.h>

errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    if (dest == nullptr || src == nullptr) {
        return EINVAL;
    }
    if (count > destMax) {
        return ERANGE;
    }
    if (count > 0U) {
        (void)memcpy(dest, src, count);
    }
    return EOK;
}

errno_t memset_s(void* dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr) {
        return EINVAL;
    }
    if (count > destMax) {
        return ERANGE;
    }
    if (count > 0U) {
        (void)memset(dest, c, count);
    }
    return EOK;
}
