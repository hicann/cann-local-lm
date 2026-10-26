/**
 * @file info/application_target_sdk_version.h
 * @brief Stub for the OpenHarmony NDK API-level header (C3).
 *
 * The real header is provided by the OHOS NDK sysroot and defines API-level
 * macros (e.g. TARGET_SDK_VERSION) used by OHOS NDK headers for deprecation
 * gating. neural_network_core.h includes it by convention but references no
 * symbol from it (verified via workspace search), so an empty stub is
 * sufficient for host (Windows/MinGW) builds of the shim.
 *
 * The shim only dlopen's the real NPU .so at runtime; this compile-time stub
 * does not affect runtime behavior. On a real OHOS target the sysroot version
 * of this file takes precedence (it appears earlier on the include path).
 */
#ifndef INFO_APPLICATION_TARGET_SDK_VERSION_H
#define INFO_APPLICATION_TARGET_SDK_VERSION_H
#endif // INFO_APPLICATION_TARGET_SDK_VERSION_H
