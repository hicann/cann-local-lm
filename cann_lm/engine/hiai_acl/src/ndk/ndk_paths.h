/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/src/ndk/ndk_paths.h
 * @brief NDK shared library search paths.
 *
 * NDKProxy uses these candidate paths when calling DynamicLoader::Open()
 * to load libneural_network_core.so and libhiai_foundation.so.
 *
 * On Windows (cross-compile reference): the DevEco Studio SDK ships ARM
 * OpenHarmony .so files.  They cannot actually be loaded on x64 Windows
 * (wrong architecture), but the paths are probed in order.
 *
 * On OpenHarmony/Linux: system library paths are searched.
 */
#ifndef HIAI_ACL_NDK_PATHS_H
#define HIAI_ACL_NDK_PATHS_H

#include <string>
#include <vector>

namespace hiai {

/// Returns candidate search directories for NDK shared libraries.
/// An empty string means "let the system loader search its default paths"
/// (i.e. pass bare library name like "libneural_network_core.so").
inline std::vector<std::string> GetNdkSearchPaths() {
    return {
        // DevEco Studio SDK (Windows cross-compile reference — ARM .so files)
        "C:\\Program Files\\Huawei\\DevEco Studio\\sdk\\default\\openharmony\\native\\sysroot\\usr\\lib\\arm-linux-ohos",
        // OpenHarmony system library paths
        "/system/lib64/ndk",
        "/system/lib/ndk",
        "/usr/lib64",
        "/usr/lib",
        "/system/lib",
        // Empty = bare name, let system loader search default paths
        "",
    };
}

/// Builds a full path: dir + "/" + libName, or just libName if dir is empty.
inline std::string JoinPath(const std::string& dir, const std::string& libName) {
    if (dir.empty()) {
        return libName;
    }
    // Handle both / and \ separators — the path is passed to the platform
    // loader which handles its native separator.
    return dir + "/" + libName;
}

}  // namespace hiai

#endif  // HIAI_ACL_NDK_PATHS_H
