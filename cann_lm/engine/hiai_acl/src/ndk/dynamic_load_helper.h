/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/ndk/dynamic_load_helper.h
 * @brief DynamicLoadHelper — dlopen/dlsym wrapper with symbol caching.
 *
 * Manages a single .so handle and caches resolved symbol addresses.
 * Thread-safe via mutex-protected symbol map.
 */

#ifndef HIAI_ACL_DYNAMIC_LOAD_HELPER_H
#define HIAI_ACL_DYNAMIC_LOAD_HELPER_H

#include <string>
#include <mutex>
#include <map>

namespace hiai {

class DynamicLoadHelper {
public:
    DynamicLoadHelper();
    ~DynamicLoadHelper();

    bool Init(const std::string& file, bool isNeedClose = true);
    void Deinit();
    bool IsValid() const;

    template <typename T>
    T* GetFunction(const std::string& symbol) {
        auto funcPtr = GetSymbol(symbol);
        return reinterpret_cast<T*>(funcPtr);
    }

    void* GetSymbol(const std::string& symbol);

private:
    bool isNeedClose_{true};
    void* handle_{nullptr};
    std::mutex funcMapMutex_;
    std::map<std::string, void*> funcMap_;
};

}  // namespace hiai

#endif  // HIAI_ACL_DYNAMIC_LOAD_HELPER_H
