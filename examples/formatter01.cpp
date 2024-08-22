/// @brief Quick exercise of the @c std::formatter class specializations.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include <bit/bit.h>

int
main()
{
    using block_type = std::uint8_t;

    using vector_type = bit::vector<block_type>;
    auto v = vector_type::random(18);
    std::cout << std::format("Vector default specifier:   {}\n", v);
    std::cout << std::format("Vector bit-order specifier: {:b}\n", v);
    std::cout << std::format("Vector pretty specifier:    {:p}\n", v);
    std::cout << std::format("Vector hex specifier:       {:x}\n", v);
    std::cout << std::format("Vector invalid specifier:   {:X}\n", v);

    using polynomial_type = bit::polynomial<block_type>;
    auto p = polynomial_type::random(13);
    std::cout << std::format("Polynomial default specifier: {}\n", p);
    std::cout << std::format("Polynomial 'y' specifier:     {:y}\n", p);
    std::cout << std::format("Polynomial 'M' specifier:     {:M}\n", p);
    std::cout << std::format("Polynomial 'mat' specifier:   {:mat}\n", p);

    using matrix_type = bit::matrix<block_type>;
    auto m = matrix_type::random(4, 4);
    std::cout << std::format("Matrix default specifier: \n{}\n", m);
    std::cout << std::format("Matrix pretty specifier:  \n{:p}\n", m);
    std::cout << std::format("Matrix hex specifier:     \n{:x}\n", m);
    std::cout << std::format("Matrix invalid specifier: \n{:X}\n", m);
}