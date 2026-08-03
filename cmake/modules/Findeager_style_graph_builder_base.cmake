include(TorchDelegateCannCommon)

if(TARGET eager_style_graph_builder_base)
    set(eager_style_graph_builder_base_FOUND TRUE)
    return()
endif()

torch_delegate_cann_include_search_paths(_td_include_paths)
torch_delegate_cann_library_search_paths(_td_library_paths)

find_path(_ES_GRAPH_BUILDER_INCLUDE_DIR
    NAMES ge/es_graph_builder.h
    PATHS ${_td_include_paths}
    NO_DEFAULT_PATH
)

find_library(_ES_GRAPH_BUILDER_LIBRARY
    NAMES eager_style_graph_builder_base
    PATHS ${_td_library_paths}
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(eager_style_graph_builder_base
    FOUND_VAR eager_style_graph_builder_base_FOUND
    REQUIRED_VARS _ES_GRAPH_BUILDER_INCLUDE_DIR _ES_GRAPH_BUILDER_LIBRARY
)

if(eager_style_graph_builder_base_FOUND)
    message(STATUS "[eager_style_graph_builder_base] library: ${_ES_GRAPH_BUILDER_LIBRARY}")

    add_library(eager_style_graph_builder_base SHARED IMPORTED)
    set_target_properties(eager_style_graph_builder_base PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ES_GRAPH_BUILDER_INCLUDE_DIR}"
        IMPORTED_LOCATION "${_ES_GRAPH_BUILDER_LIBRARY}"
    )
endif()
