/**
 * cann_llm hiai_ndk: NDK 动态加载代理（多路径探测 dlopen，参考 cann_llm hiai_acl/ndk 方式）。
 */
#ifndef CANN_LLM_HIAI_NDK_PROXY_H
#define CANN_LLM_HIAI_NDK_PROXY_H
#include <map>  // framework 原版头经 dynamic_load_helper.h 透传 <map>，wrapper 零改动源码依赖此传递包含
#include <string>
namespace hiai {
class NDKProxy {
public:
    static void* GetSymbol(std::string symbol);
private:
    NDKProxy() = default;
    ~NDKProxy() = default;
    NDKProxy(const NDKProxy&) = delete;
    NDKProxy& operator=(const NDKProxy&) = delete;
    static NDKProxy* GetInstance();
    void* hiaiHandle_ {nullptr};
    void* nncoreHandle_ {nullptr};
    bool initFailed_ {false};
};
} // namespace hiai
#endif
