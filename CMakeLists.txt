cmake_minimum_required(VERSION 3.23)

# Our C++ project ...
project(bit DESCRIPTION "C++ classes for working in GF(2)" LANGUAGES CXX)

# Add a target for the "library" we are building (${PROJECT_NAME} is header only -- hence INTERFACE).
# Also add an alias that prepends a "namespace" -- if clients use that to link to, they get better error messages.
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# We use some C++20 features.
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Where to find the headers (e.g., how to resolve `#include <bit/bit.h>`).
target_sources(${PROJECT_NAME} INTERFACE
    FILE_SET    library_headers
    TYPE        HEADERS
    BASE_DIRS   include/)

# That's it unless we are developing the library instead of just using it.
if (PROJECT_IS_TOP_LEVEL)

    # Append our local directory of CMake modules to the default ones searched by CMake.
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

    # Prevent in-source builds for the example programs.
    include(disable_in_source_builds)

    # Make the compiler issue warnings for "bad" code, etc.
    include(compiler_init)
    compiler_init(${PROJECT_NAME})

    # Debug builds get the BIT_VERIFY flag set.
    target_compile_definitions(${PROJECT_NAME} INTERFACE $<$<CONFIG:Debug>: -DBIT_VERIFY>)

    # For neatness, we put the example executables in build/bin/.
    include(GNUInstallDirs)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

    # Our example programs use the header-only `utilities` library -- grab the current version from GitHub.
    include(fetch_content)
    fetch_content(utilities URL https://github.com/nessan/utilities/releases/download/current/utilities.zip)

    # Walk through the examples/ directory and build a target for each .cpp file with appropriate linkage.
    # We have a CMake module that makes that traversal straightforward.
    include(add_executables)
    add_executables(examples ${PROJECT_NAME}::${PROJECT_NAME} utilities::utilities)

endif()
