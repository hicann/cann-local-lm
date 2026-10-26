/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/ndk/dynamic_load_helper.cpp
 * @brief DynamicLoadHelper implementation — uses DynamicLoader abstraction.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/ndk/dynamic_load_helper.cpp
 * Key change: replaces raw dlopen/dlsym/dlclose with DynamicLoader, removing
 * platform-specific code (realpath, PATH_MAX, linux/limits.h).
 */
#include "dynamic_load_helper.h"

#include "dynamic_loader.h"
#include "utils/log.h"

namespace hiai {

DynamicLoadHelper::DynamicLoadHelper() {}

DynamicLoadHelper::~DynamicLoadHelper() { Deinit(); }

bool DynamicLoadHelper::Init(const std::string& file, bool isNeedClose)
{
    std::lock_guard<std::mutex> lock(funcMapMutex_);
    if (handle_ != nullptr) {
        ACL_LOGE("handle is not null, file: %s", file.c_str());
        return false;
    }

    if (file.empty()) {
        ACL_LOGE("file is empty");
        return false;
    }

    // Use DynamicLoader instead of raw dlopen — handles platform differences
    // (LoadLibraryA on Windows, dlopen on POSIX).  No realpath needed;
    // the platform loader resolves paths itself.
    DynamicLoader loader;
    if (!loader.Open(file)) {
        ACL_LOGE("DynamicLoader::Open failed, file: %s", file.c_str());
        return false;
    }

    // Transfer ownership: store the raw handle for GetSymbol access
    // We keep the DynamicLoader alive by storing it as the handle.
    // Since DynamicLoader is RAII, we heap-allocate it.
    handle_ = new DynamicLoader(std::move(loader));
    isNeedClose_ = isNeedClose;
    return true;
}

void DynamicLoadHelper::Deinit()
{
    std::lock_guard<std::mutex> lock(funcMapMutex_);
    if (handle_ == nullptr) {
        return;
    }
    if (isNeedClose_) {
        auto* loader = static_cast<DynamicLoader*>(handle_);
        loader->Close();
        delete loader;
    }
    handle_ = nullptr;
    funcMap_.clear();
}

bool DynamicLoadHelper::IsValid() const { return handle_ != nullptr; }

void* DynamicLoadHelper::GetSymbol(const std::string& symbol)
{
    std::lock_guard<std::mutex> lock(funcMapMutex_);
    if (handle_ == nullptr) {
        ACL_LOGE("handle is null, symbol: %s", symbol.c_str());
        return nullptr;
    }
    auto it = funcMap_.find(symbol);
    if (it != funcMap_.cend()) {
        return it->second;
    }
    auto* loader = static_cast<DynamicLoader*>(handle_);
    auto funcPtr = loader->GetSymbol(symbol);
    if (funcPtr == nullptr) {
        ACL_LOGE("GetSymbol failed, symbol: %s", symbol.c_str());
    }
    funcMap_[symbol] = funcPtr;
    return funcPtr;
}

} // namespace hiai
