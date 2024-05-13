/// @brief Parse a bit-matrix from a string/stream.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using matrix_type = bit::matrix<std::uint64_t>;

    // Read a string and convert to a bit-matrix
    while (true) {
        std::string input;
        std::cout << "String input (q to move on)? ";
        std::getline(std::cin, input);
        if (input == "Q" || input == "q") break;

        auto m = matrix_type::from(input);
        if (m) { std::print("'{}' parsed as:\n{}\n", input, *m); }
        else {
            std::print("Failed to parse '{}' as a bit-matrix!\n", input);
        }
    }

    // Read a bit-matrix directly from a stream
    while (true) {
        matrix_type m;
        std::cout << "Stream input (any invalid bit-matrix to quit)? ";
        try {
            std::cin >> m;
            std::print("Parsed as:\n{}\n", m);
        }
        catch (...) {
            std::print("Couldn't parse that input as a bit-matrix! Quitting ...\n");
            break;
        }
    }

    return 0;
}
