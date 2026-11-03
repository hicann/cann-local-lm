cmake_minimum_required(VERSION 3.16)
project(hiai_acl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Source files =====
set(HIAI_ACL_SOURCES
    src/acl_mdl.cpp
    src/acl_rt.cpp
    src/acl_manager.cpp
    src/acl_data.cpp
    src/ndk/ndk_proxy.cpp
    src/ndk/dynamic_load_helper.cpp
    src/ndk/hiai_ndk_nncore.cpp
    src/ndk/hiai_ndk_executor.cpp
    src/ndk/hiai_ndk_options.cpp  # C2 follow-on: re-included (HMS_HiAIOptions_Set* for ModelOptions tuning/band/...)
    # hiai_ndk_tensor.cpp remains EXCLUDED (cache funcs, dead for LLM; SDD §4.6.2).
    # AIPP sources removed — not in LLM interface.
)

# Platform-specific DynamicLoader — only one compiled per platform
if(WIN32)
    list(APPEND HIAI_ACL_SOURCES platform/dynamic_loader_win32.cpp)
else()
    list(APPEND HIAI_ACL_SOURCES platform/dynamic_loader_posix.cpp)
endif()

# ===== Vendored header directories =====
# ACL SDK headers (acl_base.h, acl_rt.h, acl_mdl.h, acl/error_codes/*.h) — the
# subset actually used by ndk/ + shim src/, relocated from ascendcl/ to external_acl/.
set(ACL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external_acl)
# CANNKit headers (neural_network_core.h, hiai_options.h, ...) — provided by the
# external_cann/ staging area (NOT vendored into the repo; SDD §3 says "vendored
# CANNKit/" but the real location is engine/external_cann/, a hidden external SDK
# drop). Point CANNKIT_INCLUDE_DIR here so `#include "neural_network_runtime/..."`
# resolves.
set(CANNKIT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external_cann)

# ===== Independent library target =====
# STATIC on all platforms for easy linking into cann_backend.
# For target deployment (OpenHarmony), change to SHARED via CMake variable.
if(NOT DEFINED HIAI_ACL_LIBRARY_TYPE)
    set(HIAI_ACL_LIBRARY_TYPE STATIC)
endif()

add_library(hiai_ascendcl ${HIAI_ACL_LIBRARY_TYPE} ${HIAI_ACL_SOURCES})

# PUBLIC includes: ACL + CANNKit headers are needed by consumers (ndk/ backend)
# PRIVATE includes: shim internal headers (src/, utils/, platform/)
target_include_directories(hiai_ascendcl
    PUBLIC
        ${ACL_INCLUDE_DIR}
        ${CANNKIT_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/platform
)

# POSIX needs libdl for dlopen/dlsym (though DynamicLoader isolates it,
# the platform .cpp still links dlopen directly)
# C3: force-include ohos_compat.h before every TU — strips Clang-specific
# `__availability__(ohos, introduced=X.Y.Z)` attributes from the CANNKit headers
# (external_cann/) so GCC can parse them. The shim dlopen's the real .so at
# runtime; compile-time availability gating is irrelevant to host builds.
# On a real OHOS/Clang target this is a no-op (Clang supports the attribute).
target_compile_options(hiai_ascendcl PRIVATE
    -include ${CMAKE_CURRENT_SOURCE_DIR}/src/ohos_compat.h)

if(NOT WIN32)
    target_link_libraries(hiai_ascendcl PRIVATE ${CMAKE_DL_LIBS})
endif()

set_target_properties(hiai_ascendcl PROPERTIES
    OUTPUT_NAME hiai_ascendcl
    POSITION_INDEPENDENT_CODE ON
)
