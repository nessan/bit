/// @brief Verification macro that checks a condition and exit with a message if that check fails.
/// @link  https://nessan.github.io/bit
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include <exception>
#include <format>
#include <iostream>
#include <string>

/// @brief Failed verifications call this macro to exit the program using the @c bit::exit(...) function.
/// @note  This macro automatically adds the needed source code location information to the exit call.
#define bit_exit(...) bit::exit(__func__, __FILE__, __LINE__, std::format(__VA_ARGS__))

/// @brief The @c bit_verify macro expands to a no-op @b unless the @c BIT_VERIFY flag is set.
#ifdef BIT_VERIFY
    #define bit_verify(cond, ...) \
        if (!(cond)) bit_exit("Statement '{}' is NOT true: {}\n", #cond, std::format(__VA_ARGS__))
#else
    #define bit_verify(cond, ...) void(0)
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
    return i != std::string::npos ? std::string{path.substr(i + 1, path.length() - i)} : "";
}

/// @brief This function prints an error message with source code location information and exits the program.
/// @note  Generally this is only called from the @c bit_exit macro which adds the needed location info.
inline void
exit(std::string_view func, std::string_view path, std::size_t line, std::string_view payload = "")
{
    std::cerr << std::format("\nBIT VERIFY FAILED:\nFunction '{}' ({}, line {})", func, basename(path), line);
    if (!payload.empty()) std::cerr << ":\n" << payload;
    std::cerr << '\n' << std::endl;
    ::exit(1);
}

} // namespace bit
