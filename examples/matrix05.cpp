/// @brief Basic checks on echelon forms for a bit-matrix.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using matrix_type = bit::matrix<std::uint64_t>;

    // Create a matrix and get its echelon & reduced echelon forms
    auto          A = matrix_type::random(12);
    bit::vector<> pivots;
    std::print("Original, Row-Echelon, Reduced-Row-Echelon versions of a bit-matrix:\n");
    bit::print(A, echelon_form(A), reduced_echelon_form(A, &pivots));

    // Analyze the rank of the matrix etc.
    auto n = A.rows();
    auto r = pivots.count();
    auto f = n - r;
    std::print("bit-matrix size:           {} x {}\n", n, n);
    std::print("bit-matrix rank:           {}\n", r);
    std::print("number of free variables:  {}\n", f);
    if (f > 0) {
        std::print("indices of free variables: ");
        pivots.flip().if_set_call([](std::size_t k) { std::cout << k << ' '; });
    }
    std::print("\n");
    return 0;
}
