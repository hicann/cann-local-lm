#include "ndk_create_itf.h"
#include "infra/log.h"

namespace hiai {
std::shared_ptr<IModelManager> CreateModelManager()
{
    CANN_LLM_LOGI("DDK pipe is NDK.");
    return CreateModelManagerFromNDK();
}
} // namespace hiai
