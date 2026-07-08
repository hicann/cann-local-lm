include(TorchDelegateCannCommon)

if(TARGET graph_base)
    set(graph_base_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_GRAPH_BASE_INCLUDE_DIR
    NAMES graph/types.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_GRAPH_BASE_LIBRARY
    NAMES graph_base
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(graph_base
    FOUND_VAR graph_base_FOUND
    REQUIRED_VARS _GRAPH_BASE_INCLUDE_DIR _GRAPH_BASE_LIBRARY
)

if(graph_base_FOUND)
    message(STATUS "[graph_base] library: ${_GRAPH_BASE_LIBRARY}")

    add_library(graph_base SHARED IMPORTED)
    set_target_properties(graph_base PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_GRAPH_BASE_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_GRAPH_BASE_LIBRARY}"
    )
endif()
