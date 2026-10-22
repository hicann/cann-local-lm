/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/platform/dynamic_loader.h
 * @brief Abstract interface for dynamic library loading.
 *
 * Platform-specific implementations:
 *   - dynamic_loader_posix.cpp  (dlopen/dlsym/dlclose)
 *   - dynamic_loader_win32.cpp  (LoadLibraryA/GetProcAddress/FreeLibrary)
 *
 * CMake selects the correct implementation at build time. Business logic
 * never calls dlopen/LoadLibrary directly — always goes through this
 * interface, keeping the shim portable across Windows and POSIX targets.
 */
#ifndef HIAI_ACL_DYNAMIC_LOADER_H
#define HIAI_ACL_DYNAMIC_LOADER_H

#include <string>

namespace hiai {

/// RAII wrapper around platform dynamic library loading.
/// On destruction, closes the library handle if one is open.
class DynamicLoader {
public:
    DynamicLoader() = default;
    ~DynamicLoader();

    DynamicLoader(const DynamicLoader&) = delete;
    DynamicLoader& operator=(const DynamicLoader&) = delete;
    DynamicLoader(DynamicLoader&& other) noexcept;
    DynamicLoader& operator=(DynamicLoader&& other) noexcept;

    /// Open a shared library by path or name.
    /// @param libPath Full path or library name (system search path applies
    ///                when no directory separator is present).
    /// @return true if loaded successfully.
    bool Open(const std::string& libPath);

    /// Resolve a symbol from the loaded library.
    /// @param symbolName Function symbol name (C linkage, undecorated).
    /// @return Function pointer, or nullptr if not found / no library loaded.
    void* GetSymbol(const std::string& symbolName) const;

    /// Close the library handle. Safe to call when not open.
    void Close();

    /// Check if a library is currently loaded.
    bool IsOpen() const;

private:
    void* handle_ = nullptr;
};

}  // namespace hiai

#endif  // HIAI_ACL_DYNAMIC_LOADER_H
