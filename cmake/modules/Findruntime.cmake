include(TorchDelegateCannCommon)

if(TARGET runtime)
    set(runtime_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_RUNTIME_INCLUDE_DIR
    NAMES runtime/runtime/rt.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_path(_RUNTIME_ACL_INCLUDE_DIR
    NAMES acl/error_codes/rt_error_codes.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

if(NOT _RUNTIME_INCLUDE_DIR AND _RUNTIME_ACL_INCLUDE_DIR)
    # Newer CANN layouts may only ship ACL runtime headers under acl/error_codes
    # while libruntime.so is still present. None of the current targets include
    # runtime/runtime/rt.h directly, so reuse the ACL include root as a minimal
    # compatibility fallback to keep imported target resolution working.
    set(_RUNTIME_INCLUDE_DIR "${_RUNTIME_ACL_INCLUDE_DIR}")
endif()

find_library(_RUNTIME_LIBRARY
    NAMES runtime
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(runtime
    FOUND_VAR runtime_FOUND
    REQUIRED_VARS _RUNTIME_INCLUDE_DIR _RUNTIME_ACL_INCLUDE_DIR _RUNTIME_LIBRARY
)

if(runtime_FOUND)
    message(STATUS "[runtime] library: ${_RUNTIME_LIBRARY}")

    add_library(runtime SHARED IMPORTED)
    set_target_properties(runtime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_RUNTIME_INCLUDE_DIR};${_RUNTIME_ACL_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_RUNTIME_LIBRARY}"
    )
endif()
