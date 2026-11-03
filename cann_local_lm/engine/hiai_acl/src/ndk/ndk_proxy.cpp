/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/ndk/ndk_proxy.cpp
 * @brief NDKProxy implementation — dynamic loading of NDK .so libraries.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/ndk/ndk_proxy.cpp
 * Key change: uses multi-path search (ndk_paths.h) instead of hardcoded
 * __OHOS__ ifdef paths.  Tries DevEco Studio SDK path, system paths,
 * and bare name in order.
 */
#include "ndk_proxy.h"

#include "ndk_paths.h"
#include "utils/log.h"

namespace hiai {

NDKProxy::NDKProxy()
{
    // auto paths = GetNdkSearchPaths();
    std::string path = "/system/lib64/ndk";
    bool nncoreLoaded = false;
    bool hiaiLoaded = false;

    // Try each candidate path for libhiai_foundation.so
    // for (const auto& dir : paths) {
    std::string hiaiPath = JoinPath(path, "libhiai_foundation.so");
    if (hiaiHelper_.Init(hiaiPath)) {
        hiaiLoaded = true;
        ACL_LOGI("Loaded libhiai_foundation.so from: %s", hiaiPath.c_str());
        //     break;
    }
    // }

    // Try each candidate path for libneural_network_core.so
    // for (const auto& dir : paths) {
    std::string nncorePath = JoinPath(path, "libneural_network_core.so");
    if (nncoreHelper_.Init(nncorePath)) {
        nncoreLoaded = true;
        ACL_LOGI("Loaded libneural_network_core.so from: %s", nncorePath.c_str());
        // break;
    }
    // }

    initialized_ = nncoreLoaded && hiaiLoaded;
    if (!initialized_) {
        ACL_LOGE("NDKProxy init failed: nncore=%d hiai=%d", nncoreLoaded, hiaiLoaded);
    }
}

NDKProxy* NDKProxy::GetInstance()
{
    static NDKProxy instance;
    return &instance;
}

NDKProxy::~NDKProxy()
{
    nncoreHelper_.Deinit();
    hiaiHelper_.Deinit();
}

void* NDKProxy::GetSymbol(std::string symbol)
{
    if (!GetInstance()->initialized_) {
        ACL_LOGE("NDKProxy not initialized");
        return nullptr;
    }
    if (!GetInstance()->nncoreHelper_.IsValid() || !GetInstance()->hiaiHelper_.IsValid()) {
        ACL_LOGE("One or both helpers are invalid");
        return nullptr;
    }
    // Route: OH_NN* symbols → nncore, HMS_* / other symbols → hiai_foundation
    const std::string OH_NN_PREFIX = "OH_NN";
    const std::string NN_PREFIX = "NN_";
    if (symbol.substr(0, OH_NN_PREFIX.size()) == OH_NN_PREFIX || symbol.substr(0, NN_PREFIX.size()) == NN_PREFIX) {
        return GetInstance()->nncoreHelper_.GetSymbol(symbol);
    }
    return GetInstance()->hiaiHelper_.GetSymbol(symbol);
}

bool NDKProxy::IsInitialized() { return GetInstance()->initialized_; }

} // namespace hiai
