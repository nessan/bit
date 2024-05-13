/// @brief Polynomial sums of bit-matrices.
/// SPDX-FileCopyrightText:  2024 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include <bit/bit.h>
#include <utilities/utilities.h>

int
main()
{
    using block_type = std::uint8_t;
    using vector_type = bit::vector<block_type>;
    using matrix_type = bit::matrix<block_type>;

    // Strings we will parse into polynomial coefficients and a bit-matrix.
    std::string p_str;
    std::string m_str;

    // And we're off ...
    while (true) {

        // Read the polynomial coefficients as a string and convert it to a bit-vector if possible.
        std::cout << "Polynomial coefficients (eg. '100101') or 'q' to quit? ";
        std::getline(std::cin, p_str);
        if (p_str == "Q" || p_str == "q") break;
        auto p = vector_type::from(p_str);
        if (!p) {
            std::cout << "Failed to parse '" << p_str << "' as a bit-vector!\n";
            continue;
        }
        std::cout << "Polynomial: " << p->to_polynomial() << '\n';

        // Read another string and convert it to a bit-matrix if possible.
        std::cout << "Matrix elements (eg. '100 010 001' for the 3x3 identity) or 'q' to quit? ";
        std::getline(std::cin, m_str);
        if (m_str == "Q" || m_str == "q") break;
        auto M = matrix_type::from(m_str);
        if (!M) {
            std::cout << "Failed to parse '" << m_str << "' as a bit-matrix!\n";
            continue;
        }

        // Print the polynomial sum
        auto sum = bit::polynomial_sum(*p, *M);
        std::cout << "Polynomial p: " << p->to_polynomial() << '\n';
        std::cout << "Matrix M:\n" << *M << '\n';
        std::cout << "p(M):\n" << sum << '\n';
    }
    return 0;
}
