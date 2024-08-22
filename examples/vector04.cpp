/// @brief Parsing a @c bit::vector from a string/stream.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"
int
main()
{
    using vector_type = bit::vector<std::uint64_t>;

    // Read a string and convert to a bit::vector
    while (true) {
        std::string input;
        std::print("String input (q to move on)? ");
        std::getline(std::cin, input);
        if (input == "Q" || input == "q") break;
        auto v = vector_type::from(input);
        if (v)
            std::print("'{}' parsed as: {}\n", input, *v);
        else
            std::print("Failed to parse '{}' as a bit-vector!\n", input);
    }

    // Read a bit::vector directly from a stream
    while (true) {
        vector_type v;
        std::print("Stream input (any invalid bit::vector to quit)? ");
        try {
            std::cin >> v;
            std::print("Parsed as {}\n", v);
        }
        catch (...) {
            std::print("Couldn't parse that input as a bit::vector! Quitting ...\n");
            break;
        }
    }

    return 0;
}
