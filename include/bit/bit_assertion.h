/// @brief Two replacements for the standard @c assert(condition) macro that add an informational message.
/// @link  https://nessan.github.io/bit
/// SPDX-FileCopyrightText:  2024 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include <exception>
#include <format>
#include <iostream>
#include <string>

/// @brief Enforce flag consistency.
#ifdef NDEBUG
    #undef BIT_DEBUG
#endif

/// @brief This is called if an assertion fails -- exits the program using the @c bit::exit(...) method.
/// @note  This is a macro that automatically adds the needed location information to the payload.
#define bit_exit(...) bit::exit(__func__, __FILE__, __LINE__, std::format(__VA_ARGS__))

/// @brief The @c bit_debug_assertion macro expands to a no-op @b unless the @c BIT_DEBUG flag is set.
#ifdef BIT_DEBUG
    #define bit_debug_assertion(cond, ...) bit_exit("Statement '{}' is NOT true: {}\n", #cond, std::format(__VA_ARGS__))
#else
    #define bit_debug_assertion(cond, ...) void(0)
#endif

/// @brief The @c bit_assertion macro expands to a no-op @b if the @c NDEBUG flag is set.
/// @note  This is the same behaviour as the like the standard @c assert macro.
#ifdef NDEBUG
    #define bit_assertion(cond, ...) void(0)
#else
    #define bit_assertion(cond, ...) bit_exit("Statement '{}' is NOT true: {}\n", #cond, std::format(__VA_ARGS__))
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
/// @note  Generally this is only called from the @c bit_exit macro which adds the needed location info.
inline void
exit(std::string_view func, std::string_view path, std::size_t line, std::string_view payload = "")
{
    std::cerr << std::format("\nBIT ASSERTION FAILURE:\nFunction '{}' ({}, line {})", func, basename(path), line);
    if (!payload.empty()) std::cerr << ":\n" << payload;
    std::cerr << '\n' << std::endl;
    ::exit(1);
}

} // namespace bit
