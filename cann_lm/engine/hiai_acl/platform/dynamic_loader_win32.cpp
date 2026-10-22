/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/platform/dynamic_loader_win32.cpp
 * @brief DynamicLoader — Windows implementation (LoadLibraryA/GetProcAddress).
 *
 * Compiled on Windows.  Excluded on POSIX via CMake conditional.
 *
 * Note: ARM OpenHarmony .so files cannot be loaded on Windows x64 (wrong
 * architecture).  The code still compiles and links — runtime loading
 * will fail as expected.  The shim's NDKProxy will log the error and
 * return nullptr for all symbol lookups, causing hardware-dependent calls
 * to return error codes.  This is by design: upper-layer code is fully
 * developed and compilable on Windows; actual NPU execution happens on
 * the target device.
 */
#ifdef _WIN32

#include "dynamic_loader.h"

// NOMINMAX: prevent Windows headers from defining min/max macros that
// conflict with std::min/std::max used elsewhere in the shim.
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

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
    Close();
    // Suppress Windows hard-error dialogs when probing .so paths that may be non-PE
    // (e.g., ARM OpenHarmony ELF in the OHOS SDK sysroot: arm-linux-ohos/libneural_network_core.so).
    // Without this, LoadLibraryA on a non-PE file pops a modal "损坏的镜像/Bad Image"
    // dialog (STATUS_INVALID_IMAGE_NOT_MZ 0xc000012f) that blocks the process — ctest
    // then appears to hang (timeout) until a human dismisses the dialog. With SEM flags,
    // LoadLibraryA returns NULL silently for non-PE/wrong-arch files, the shim logs the
    // failure + continues probing, + the host stub .so (valid PE) is found + used.
    UINT prevMode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    HMODULE hMod = LoadLibraryA(libPath.c_str());
    SetErrorMode(prevMode);
    if (hMod != nullptr) {
        handle_ = static_cast<void*>(hMod);
    }
    return handle_ != nullptr;
}

void* DynamicLoader::GetSymbol(const std::string& symbolName) const {
    if (!handle_) return nullptr;
    HMODULE hMod = static_cast<HMODULE>(handle_);
    return reinterpret_cast<void*>(GetProcAddress(hMod, symbolName.c_str()));
}

void DynamicLoader::Close() {
    if (handle_) {
        FreeLibrary(static_cast<HMODULE>(handle_));
        handle_ = nullptr;
    }
}

bool DynamicLoader::IsOpen() const {
    return handle_ != nullptr;
}

}  // namespace hiai

#endif  // _WIN32
