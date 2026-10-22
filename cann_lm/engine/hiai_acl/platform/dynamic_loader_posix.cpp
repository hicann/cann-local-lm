/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/platform/dynamic_loader_posix.cpp
 * @brief DynamicLoader — POSIX implementation (dlopen/dlsym/dlclose).
 *
 * Compiled on Linux / Android / OpenHarmony.  Excluded on Windows via
 * CMake conditional (only one platform .cpp is compiled per build).
 */
#ifndef _WIN32

#include "dynamic_loader.h"

#include <dlfcn.h>

namespace hiai {

DynamicLoader::~DynamicLoader() {
    Close();
}

DynamicLoader::DynamicLoader(DynamicLoader&& other) noexcept
    : handle_(other.handle_) {
    other.handle_ = nullptr;
}

DynamicLoader& DynamicLoader::operator=(DynamicLoader&& other) noexcept {
    if (this != &other) {
        Close();
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

bool DynamicLoader::Open(const std::string& libPath) {
    Close();  // close any previously opened handle
    // RTLD_LAZY: resolve symbols on first use
    // RTLD_GLOBAL: make symbols available to subsequently loaded libraries
    handle_ = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    return handle_ != nullptr;
}

void* DynamicLoader::GetSymbol(const std::string& symbolName) const {
    if (!handle_) return nullptr;
    return dlsym(handle_, symbolName.c_str());
}

void DynamicLoader::Close() {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

bool DynamicLoader::IsOpen() const {
    return handle_ != nullptr;
}

}  // namespace hiai

#endif  // !_WIN32
