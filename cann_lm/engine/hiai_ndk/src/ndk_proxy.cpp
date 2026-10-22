/**
 * cann_llm hiai_ndk: NDK 动态加载代理实现（多路径探测 dlopen，参考 cann_llm hiai_acl/ndk 方式）。
 */

#include "model_manager/general_model_manager/ndk/hiai_ndk/ndk_proxy.h"

#include <dlfcn.h>

#include <vector>

#include "infra/log.h"
#include "ndk_paths.h"

namespace hiai {
namespace {
// 依次探测候选目录，返回第一个 dlopen 成功的句柄；全部失败返回 nullptr。
void* DlopenFirstFound(const std::vector<std::string>& dirs, const std::string& libName)
{
    for (const auto& dir : dirs) {
        const std::string path = hiai_ndk::JoinPath(dir, libName);
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            return handle;
        }
    }
    return nullptr;
}
} // namespace

NDKProxy* NDKProxy::GetInstance()
{
    static NDKProxy instance;
    // 懒加载：首次 GetSymbol 触发；两个 magic static 保证线程安全。
    static const bool handlesInited = []() {
        CANN_LLM_LOGI("NDKProxy: hzy.");
        const std::vector<std::string> dirs = hiai_ndk::GetNdkSearchPaths();
        instance.hiaiHandle_ = DlopenFirstFound(dirs, "libhiai_foundation.so");
        instance.nncoreHandle_ = DlopenFirstFound(dirs, "libneural_network_core.so");
        if (instance.hiaiHandle_ == nullptr && instance.nncoreHandle_ == nullptr) {
            instance.initFailed_ = true;
            CANN_LLM_LOGE("NDKProxy: no NDK library found from all candidate paths.");
        } else if (instance.hiaiHandle_ == nullptr || instance.nncoreHandle_ == nullptr) {
            CANN_LLM_LOGW("NDKProxy: partially initialized, symbols of missing library resolve to nullptr.");
        }
        return true;
    }();
    (void)handlesInited;
    return &instance;
}

void* NDKProxy::GetSymbol(std::string symbol)
{
    NDKProxy* instance = GetInstance();
    if (instance->initFailed_) {
        CANN_LLM_LOGE("NDKProxy: init failed, symbol %s unavailable.", symbol.c_str());
        return nullptr;
    }

    void* primary = nullptr;
    void* secondary = nullptr;
    if (symbol.rfind("OH_NN", 0) == 0) {
        primary = instance->nncoreHandle_;
        secondary = instance->hiaiHandle_;
    } else {
        primary = instance->hiaiHandle_;
        secondary = instance->nncoreHandle_;
    }

    if (primary != nullptr) {
        void* sym = dlsym(primary, symbol.c_str());
        if (sym != nullptr) {
            return sym;
        }
    }
    if (secondary != nullptr) {
        void* sym = dlsym(secondary, symbol.c_str());
        if (sym != nullptr) {
            return sym;
        }
    }
    CANN_LLM_LOGW("NDKProxy: symbol %s not found in loaded NDK libraries.", symbol.c_str());
    return nullptr;
}
} // namespace hiai
