/// @brief Some checks on LU decomposition of bit-matrices.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using matrix_type = bit::matrix<std::uint64_t>;

    // Number of trials
    std::size_t trials = 100;

    // Each trial will run on a bit-matrix of this size
    std::size_t N = 100;

    // Number of non-singular bit-matrices
    std::size_t singular = 0;

    // Start the trials ...
    for (std::size_t n = 0; n < trials; ++n) {

        // Create a random bit-matrix & decompose it
        auto A = matrix_type::random(N);
        auto LU = bit::lu(A);

        // Extract L & U as their own bit-matrices & compute their product
        auto U = LU.U();
        auto L = LU.L();
        auto lu = bit::dot(L, U);

        // Check whether P.A = L.U as it should. Exit early if not!
        LU.permute(A);
        if (A != lu) {
            std::print("P.A, L.U:\n");
            print(A, lu);
            std::print("THEY DON'T MATCH -- CODING ERROR. EXITING ...\n");
            exit(1);
        }

        // Count the number of singular bit-matrices we come across
        if (LU.singular()) singular++;
    }

    // Stats ...
    auto p = matrix_type::probability_singular(N);
    std::print("bit-matrix size: {} x {}\n", N, N);
    std::print("prob[singular]:  {:.2f}%\n", 100 * p);
    std::print("trials:          {}\n", trials);
    std::print("singular:        {} times\n", singular);
    std::print("expected:        {} times\n", int(p * double(trials)));
    std::print("In ALL cases P.A == L.U!\n");

    return 0;
}