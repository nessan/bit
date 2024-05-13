/// @brief This little sample program creates a randomly filled 6 x 5 Matrix, adds a column,
/// sets all the bits in that new column and then extracts the coefficients for the characteristic
/// polynomial of the resulting 6 x 6 matrix. After each step it prints the result to `std::cout`.
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT

// tag::doc[]
#include <bit/bit.h>
int
main()
{
    auto mat = bit::matrix<>::random(6);
    auto poly = bit::characteristic_polynomial(mat);
    std::cout << "The bit-matrix A:\n" << mat << "\n";
    std::cout << "Characteristic polynomial p(x): " << poly.to_polynomial() << ".\n";
    std::cout << "p(A) the characteristic polynomial sum of the matrix (should be all zeros):\n";
    std::cout << bit::polynomial_sum(poly, mat) << "\n";
    return 0;
}
// end::doc[]
