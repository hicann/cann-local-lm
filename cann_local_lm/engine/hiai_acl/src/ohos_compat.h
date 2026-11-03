/**
 * @file hiai_acl/src/ohos_compat.h
 * @brief Host-build compatibility shims for OHOS NDK headers under GCC (C3).
 *
 * The CANNKit headers (neural_network_core.h, hiai_options.h, hiai_aipp_param.h,
 * ...) use the Clang-specific `__attribute__((__availability__(ohos,
 * introduced=X.Y.Z)))` attribute to gate APIs by OHOS SDK version. GCC (MinGW)
 * cannot parse this syntax — it errors with "too many decimal points in number"
 * (on X.Y.Z) and "'introduced' was not declared in this scope".
 *
 * Force-included before every shim TU (via CMake `-include`), this header strips
 * `__availability__(...)` to an empty attribute so GCC parses the CANNKit headers.
 * This is safe for host builds: the shim only dlopen's the real NPU .so at runtime;
 * compile-time API-availability gating does not affect runtime behavior. On a real
 * OHOS/Clang target this file is a no-op (Clang supports the attribute natively,
 * and the macro is a no-op redefinition of a reserved identifier).
 */
#ifndef HIAI_ACL_OHOS_COMPAT_H
#define HIAI_ACL_OHOS_COMPAT_H

// Strip Clang `__availability__(...)` availability attributes (GCC-incompatible).
// VARIADIC: the CANNKit headers call it as `__availability__(ohos, introduced=X.Y.Z)`
// — 2 comma-separated args. A single-param macro would error "passed 2 arguments but
// takes just 1"; `__VA_ARGS__` consumes any arg count. Expands to empty so GCC never
// tokenizes the version literal or the `introduced` keyword.
//
// NOTE: `__availability__` is a reserved identifier (double-underscore); redefining
// it is technically UB but works in practice and is the standard host-compat trick
// for OHOS NDK headers under GCC.
#ifndef __clang__
#define __availability__(...)
#endif

#endif // HIAI_ACL_OHOS_COMPAT_H
