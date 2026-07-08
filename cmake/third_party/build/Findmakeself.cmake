# -----------------------------------------------------------------------------------------------------------
# Findmakeself.cmake - Find or download makeself tool
#
# This module provides makeself_build target.
# makeself is a self-extracting archive generator for .run package creation.
# -----------------------------------------------------------------------------------------------------------

if(TARGET makeself_build)
    return()
endif()

include(ExternalProject)

set(MAKESELF_DIR "${CMAKE_THIRD_PARTY_LIB_DIR}/makeself")

if(EXISTS "${MAKESELF_DIR}/makeself.sh" AND EXISTS "${MAKESELF_DIR}/makeself-header.sh")
    message(STATUS "[makeself] makeself found at ${MAKESELF_DIR}")
    add_custom_target(makeself_build)
    return()
endif()

message(STATUS "[makeself] makeself not found, will download from gitcode.")

set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-${MAKESELF_VERSION}.0/makeself-release-${MAKESELF_VERSION}.tar.gz")

ExternalProject_Add(makeself_build
    URL ${MAKESELF_URL}
    TLS_VERIFY OFF
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory "${MAKESELF_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "<SOURCE_DIR>/makeself.sh" "${MAKESELF_DIR}/makeself.sh"
        COMMAND ${CMAKE_COMMAND} -E copy "<SOURCE_DIR>/makeself-header.sh" "${MAKESELF_DIR}/makeself-header.sh"
        COMMAND chmod +x "${MAKESELF_DIR}/makeself.sh" "${MAKESELF_DIR}/makeself-header.sh"
    EXCLUDE_FROM_ALL TRUE
)
