cmake_minimum_required(VERSION 3.24)

# Our C++ project ...
project(bit DESCRIPTION "C++ classes for working in GF(2)" LANGUAGES CXX)

# Add a target for the "library" we are building (bit is header only -- hence INTERFACE).
# Also add an alias that prepends a "namespace" -- if clients use that to link to, they get better error messages.
add_library(bit INTERFACE)
add_library(bit::bit ALIAS bit)

# We use some C++20 features.
target_compile_features(bit INTERFACE cxx_std_20)

# Where to find the bit headers (e.g., how to resolve `#include <bit/bit.h>`).
target_include_directories(bit INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/bit>)

# That's it unless we are developing the library instead of just using it.
if (PROJECT_IS_TOP_LEVEL)

    # Append our local directory of CMake modules to the default ones searched by CMake.
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

    # Prevent in-source builds for the example programs.
    include(disable_in_source_builds)
    disable_in_source_builds()

    # Make the compiler issue warnings for "bad" code, etc.
    include(compiler_init)
    compiler_init(bit)

    # Debug builds get the BIT_DEBUG flag set otherwise we set the BIT_NDEBUG flag
    target_compile_definitions(bit INTERFACE $<IF:$<CONFIG:Debug>, BIT_DEBUG=1, BIT_NDEBUG=1>)

    # For neatness, we put the example executables in build/bin/.
    include(GNUInstallDirs)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

    # Our example programs use the header-only `utilities` library.
    include(FetchContent)
    FetchContent_Declare(utilities URL https://github.com/nessan/utilities/releases/download/current/utilities.zip)
    FetchContent_MakeAvailable(utilities)

    # Walk through the examples/ directory and build a target for each .cpp file with appropriate linkage.
    # We have a CMake module that makes that traversal straightforward.
    include(add_executables)
    add_executables(examples bit::bit utilities::utilities)

endif()
