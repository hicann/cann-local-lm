include(TorchDelegateCannCommon)

if(TARGET ge_runner_v2)
    set(ge_runner_v2_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_GE_RUNNER_V2_INCLUDE_DIR
    NAMES ge/ge_api_v2.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_GE_RUNNER_V2_LIBRARY
    NAMES ge_runner_v2
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ge_runner_v2
    FOUND_VAR ge_runner_v2_FOUND
    REQUIRED_VARS _GE_RUNNER_V2_INCLUDE_DIR _GE_RUNNER_V2_LIBRARY
)

if(ge_runner_v2_FOUND)
    message(STATUS "[ge_runner_v2] library: ${_GE_RUNNER_V2_LIBRARY}")

    add_library(ge_runner_v2 SHARED IMPORTED)
    set_target_properties(ge_runner_v2 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_GE_RUNNER_V2_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "graph;graph_base;runtime;securec;ascend_hal"
        IMPORTED_LOCATION "${_GE_RUNNER_V2_LIBRARY}"
    )
endif()
