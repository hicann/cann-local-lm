#ifndef CANN_LLM_SUPPORT_SECUREC_H
#define CANN_LLM_SUPPORT_SECUREC_H
/* host 环境无华为 securec 库时的最小兼容层。仅当 CMake 检测不到系统 securec.h 时编入。 */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int errno_t;
#ifndef EOK
#define EOK 0
#endif

/* 仅保留有消费方的函数（base_buffer/customdata_util/nd_tensor_buffer/hiai_ndk_executor：
 * memcpy_s/memset_s）。strcpy_s/strncpy_s/memmove_s 当前全仓零调用，待有消费方再补。 */
errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count);
errno_t memset_s(void* dest, size_t destMax, int c, size_t count);

#ifdef __cplusplus
}
#endif
#endif
