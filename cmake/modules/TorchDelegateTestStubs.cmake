include_guard(GLOBAL)

function(torch_delegate_add_ge_runner_v2_stub)
    if(TARGET ge_runner_v2_stub)
        return()
    endif()

    set(_td_tests_dir "${TORCH_DELEGATE_SOURCE_DIR}/tests")
    set(_td_ge_stub_lib_dir "${CMAKE_BINARY_DIR}/tests/cpp/ge_stub/lib64")

    add_library(ge_runner_v2_stub SHARED
        "${_td_tests_dir}/cpp/ge_stub/ge_stub.cc"
    )

    set_target_properties(ge_runner_v2_stub PROPERTIES
        OUTPUT_NAME "ge_runner_v2"
        LIBRARY_OUTPUT_DIRECTORY "${_td_ge_stub_lib_dir}"
    )

    target_include_directories(ge_runner_v2_stub PRIVATE
        "${_td_tests_dir}/cpp"
    )

    target_link_libraries(ge_runner_v2_stub PRIVATE
        graph
        graph_base
        metadef
    )
    target_compile_definitions(ge_runner_v2_stub PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_compile_options(ge_runner_v2_stub PRIVATE ${DELEGATE_COMMON_FLAGS} -g)
    target_link_options(ge_runner_v2_stub PRIVATE ${DELEGATE_COMMON_FLAGS})
endfunction()
