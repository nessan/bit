/// @brief Basic check on the @c bit::characteristic_polynomial(M) function.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using block_type = std::uint8_t;
    using matrix_type = bit::matrix<block_type>;

    // Read a string and convert to a bit-matrix.
    while (true) {
        std::string input;
        std::cout << "String input (q to quit)? ";
        std::getline(std::cin, input);
        if (input == "Q" || input == "q") break;

        auto mat = matrix_type::from(input);
        if (mat) {
            std::print("'{}' parsed as:\n{}\n", input, *mat);
            auto p = bit::characteristic_polynomial(*mat);
            std::print("Matrix has characteristic polynomial p(x): {}\n", p);

            // Let's see whether the matrix is a zero of characteristic polynomial as it should be?
            auto sum = p(*mat);
            if (sum.none())
                std::print("And happily p(M) yielded the zero bit-matrix as expected.\n");
            else
                std::print("!!!Oops p(M) is not the zero bit-matrix!!!:\n{}\n", sum);
        }
        else {
            std::print("Failed to parse '{}' as a bit-matrix!\n", input);
        }
    }
    return 0;
}
