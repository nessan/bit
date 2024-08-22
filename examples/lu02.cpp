/// @brief Some checks on LU decomposition of bit-matrices.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using block_type = std::uint64_t;
    using vector_type = bit::vector<block_type>;
    using matrix_type = bit::matrix<block_type>;

    // Number of trials
    std::size_t trials = 32;

    // Each trial will run on a bit-matrix of this size
    std::size_t N = 16;

    // Number of non-singular bit-matrices
    std::size_t singular = 0;

    // Start the trials ...
    for (std::size_t n = 0; n < trials; ++n) {

        // Create a random bit-matrix & bit-vector
        auto A = matrix_type::random(N);
        auto b = vector_type::random(N);

        // LU decompose the bit-matrix & solve A.x = b
        auto LU = bit::lu(A);
        if (auto x = LU(b); x) {
            auto Ax = bit::dot(A, *x);
            std::print("x: {}; A.x: {}; b: {}; A.x == b? {}\n", *x, Ax, b, Ax == b);
        }

        // Count the number of singular bit-matrices we come across
        if (LU.singular()) singular++;
    }

    // Stats ...
    auto p = matrix_type::probability_singular(N);
    std::print("\nSingularity stats ...\n");
    std::print("bit-matrix size: {} x {}\n", N, N);
    std::print("prob[singular]:  {:.2f}%\n", 100 * p);
    std::print("trials:          {}\n", trials);
    std::print("singular:        {} times\n", singular);
    std::print("expected:        {} times\n", int(p * double(trials)));

    return 0;
}