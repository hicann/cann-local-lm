include(TorchDelegateCannCommon)

if(TARGET metadef)
    set(metadef_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_METADEF_INCLUDE_DIR
    NAMES graph/ascend_string.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_METADEF_LIBRARY
    NAMES metadef
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(metadef
    FOUND_VAR metadef_FOUND
    REQUIRED_VARS _METADEF_INCLUDE_DIR _METADEF_LIBRARY
)

if(metadef_FOUND)
    message(STATUS "[metadef] library: ${_METADEF_LIBRARY}")

    add_library(metadef SHARED IMPORTED)
    set_target_properties(metadef PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_METADEF_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_METADEF_LIBRARY}"
    )
endif()
