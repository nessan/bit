/// @brief Basic check on the invert method for bit-matrices.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using matrix_type = bit::matrix<std::uint64_t>;

    std::size_t N = 100;
    std::size_t trials = 1000;
    std::size_t fails = 0;
    for (std::size_t trial = 0; trial < trials; ++trial) {
        auto A = matrix_type::random(N, N);
        auto B = bit::invert(A);
        if (!B) ++fails;
    }

    // Stats ...
    auto p = matrix_type::probability_singular(N);
    std::print("bit-matrix size: {} x {}\n", N, N);
    std::print("prob[singular]:  {:.2f}%\n", 100 * p);
    std::print("trials:          {}\n", trials);
    std::print("inverse failed:  {} times\n", fails);
    std::print("expected:        {} times\n", int(p * double(trials)));

    return 0;
}
