include(TorchDelegateCannCommon)

if(TARGET acl_rt)
    set(acl_rt_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_ACL_RT_INCLUDE_DIR
    NAMES acl/acl_rt.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_ACL_RT_LIBRARY
    NAMES acl_rt
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(acl_rt
    FOUND_VAR acl_rt_FOUND
    REQUIRED_VARS _ACL_RT_INCLUDE_DIR _ACL_RT_LIBRARY
)

if(acl_rt_FOUND)
    message(STATUS "[acl_rt] include: ${_ACL_RT_INCLUDE_DIR}")
    message(STATUS "[acl_rt] library: ${_ACL_RT_LIBRARY}")

    add_library(acl_rt SHARED IMPORTED)
    set_target_properties(acl_rt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ACL_RT_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_ACL_RT_LIBRARY}"
    )
endif()