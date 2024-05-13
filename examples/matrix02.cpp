/// @brief Bit-matrix sub-matrices.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    std::size_t M = 16;
    std::size_t N = 16;

    // Create a matrix of all ones
    bit::matrix A(M, N);
    A.set();

    std::print("Matrix, lower triangular sub-matrix, and the strictly lower triangular sub-matrix:\n");
    print(A, A.lower(), A.strictly_lower());

    std::print("Matrix, upper triangular sub-matrix, and the strictly upper triangular sub-matrix:\n");
    print(A, A.upper(), A.strictly_upper());

    return 0;
}
