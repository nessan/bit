/// @brief Checks on Gaussian Elimination for bit-matrices.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using block_type = std::uint64_t;
    using vector_type = bit::vector<block_type>;
    using matrix_type = bit::matrix<block_type>;

    std::size_t n = 10;

    auto A = matrix_type::random(n);
    auto b = vector_type::random(n);
    auto x = bit::solve(A, b);

    if (x) {
        std::print("bit-matrix A, solution x, right hand side b, and as a check A.x:\n");
        bit::print(A, *x, b, bit::dot(A, *x));
    }
    else {
        std::print("bit-matrix A with right hand side b has NO solution!:\n");
        bit::print(A, b);
    }
}