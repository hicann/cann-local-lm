/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/ndk/ndk_proxy.h
 * @brief NDKProxy — dynamic loading of NDK shared libraries.
 *
 * Singleton that dlopen's libneural_network_core.so and libhiai_foundation.so,
 * caches dlsym function pointers. All HIAI_NDK_* functions call through
 * NDKProxy::GetSymbol().
 *
 * Evolved from harmony_acl — no API changes, same dynamic loading pattern.
 */

#ifndef HIAI_ACL_NDK_PROXY_H
#define HIAI_ACL_NDK_PROXY_H

#include <string>
#include <mutex>
#include <map>

#include "dynamic_load_helper.h"

namespace hiai {

class NDKProxy {
public:
    static void* GetSymbol(std::string symbol);
    static bool IsInitialized();

private:
    NDKProxy();
    ~NDKProxy();
    NDKProxy(const NDKProxy&) = delete;
    NDKProxy& operator=(const NDKProxy&) = delete;
    static NDKProxy* GetInstance();

private:
    DynamicLoadHelper nncoreHelper_; // libneural_network_core.so
    DynamicLoadHelper hiaiHelper_;   // libhiai_foundation.so
    bool initialized_ = false;
};

} // namespace hiai

#endif // HIAI_ACL_NDK_PROXY_H
