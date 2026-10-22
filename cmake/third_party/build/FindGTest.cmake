# -----------------------------------------------------------------------------------------------------------
# FindGTest.cmake - Find or build GoogleTest library
#
# This module provides GTest::gtest and GTest::gtest_main targets.
# It builds GoogleTest with old libstdc++ ABI so tests can link against the
# main project after the core targets are switched to ABI=0.
# -----------------------------------------------------------------------------------------------------------

if(TARGET gtest_build)
    return()
endif()

include(ExternalProject)

find_package(GTest CONFIG
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}/gtest
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(GTest_FOUND)
    message(STATUS "[GTest] GTest found: ${GTest_DIR}")
    ExternalProject_Add(gtest_build
        SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/gtest
        CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    return()
endif()

message(STATUS "[GTest] GTest not found, finding binary file.")

set(GTEST_VERSION "1.14.0")
set(GTEST_DIR "${CMAKE_THIRD_PARTY_LIB_DIR}/gtest")
set(GTEST_ARCHIVE "${GTEST_DIR}/googletest-${GTEST_VERSION}.tar.gz")
set(GTEST_URL "")
set(GTEST_EXTRA_ARGS "")

if(EXISTS "${GTEST_ARCHIVE}")
    message(STATUS "[GTest] ${GTEST_ARCHIVE} found.")
    set(GTEST_URL "${GTEST_ARCHIVE}")
elseif(EXISTS "${GTEST_DIR}/lib/libgtest.a")
    message(STATUS "[GTest] Already installed at ${GTEST_DIR}")
    ExternalProject_Add(gtest_build
        SOURCE_DIR ${GTEST_DIR}
        CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
        EXCLUDE_FROM_ALL TRUE
    )
    return()
else()
    message(STATUS "[GTest] ${GTEST_ARCHIVE} not found, will download from gitcode.")
    set(GTEST_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v${GTEST_VERSION}/googletest-${GTEST_VERSION}.tar.gz")
    list(APPEND GTEST_EXTRA_ARGS
        DOWNLOAD_DIR ${GTEST_DIR}
    )
endif()

set(GTEST_CXXFLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=1")

ExternalProject_Add(gtest_build
    URL ${GTEST_URL}
    TLS_VERIFY OFF
    ${GTEST_EXTRA_ARGS}
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DCMAKE_CXX_FLAGS=${GTEST_CXXFLAGS}
        -DCMAKE_C_FLAGS=${GTEST_CXXFLAGS}
        -DCMAKE_INSTALL_PREFIX=${GTEST_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DINSTALL_GTEST=ON
        <SOURCE_DIR>
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    EXCLUDE_FROM_ALL TRUE
)
