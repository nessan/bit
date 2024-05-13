/// @brief Three replacements for the standard `assert(condition)` macro that add an informational message.
/// @link  https://nessan.github.io/bit
/// SPDX-FileCopyrightText:  2024 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include <exception>
#include <format>
#include <iostream>
#include <string>

/// @brief Exit using the bit::exit(...) method automatically adding location information to the payload.
#define bit_assertion_failed(...) bit::exit(__func__, __FILE__, __LINE__, std::format(__VA_ARGS__))

/// @def The `bit_always_assert` macro cannot be switched off with compiler flags.
#define bit_always_assert(cond, ...) \
    if (!(cond)) bit_assertion_failed("Statement '{}' is NOT true: {}\n", #cond, std::format(__VA_ARGS__))

// If BIT_NDEBUG is set then BIT_DEBUG should *not* be set.
#if defined(BIT_NDEBUG)
    #undef BIT_DEBUG
#endif

/// @def The `bit_debug_assert` macro expands to a no-op *unless* the `BIT_DEBUG` flag is set.
#ifdef BIT_DEBUG
    #define bit_debug_assert(cond, ...) bit_always_assert(cond, __VA_ARGS__)
#else
    #define bit_debug_assert(cond, ...) void(0)
#endif

/// @def The `bit_assert`macro expands to a no-op *only if* the `BIT_NDEBUG` flag is set.
#ifdef BIT_NDEBUG
    #define bit_assert(cond, ...) void(0)
#else
    #define bit_assert(cond, ...) bit_always_assert(cond, __VA_ARGS__)
#endif

namespace bit {

/// @brief Given a path like `/home/jj/dev/project/src/foo.cpp` this returns its basename `foo.cpp`
inline std::string
basename(std::string_view path)
{
    char sep = '/';
#ifdef _WIN32
    sep = '\\';
#endif
    auto i = path.rfind(sep, path.length());
    if (i != std::string::npos) return std::string{path.substr(i + 1, path.length() - i)};
    return "";
}

/// @brief This function prints an error message with source code location information and exits the program.
/// @note  Generally this is only called from the @c bit_assertion_failed macro which adds the needed location info.
inline void
exit(std::string_view func, std::string_view path, std::size_t line, std::string_view payload = "")
{
    std::cerr << std::format("\nBIT ASSERTION FAILURE:\nFunction '{}' ({}, line {})", func, basename(path), line);
    if (!payload.empty()) std::cerr << ":\n" << payload;
    std::cerr << '\n' << std::endl;
    ::exit(1);
}

} // namespace bit
