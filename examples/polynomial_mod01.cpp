/// @brief Basic check on the polynomial_mod function.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    std::size_t N = 447'124'345;
    auto        p = bit::vector<>::from(1234019u);
    auto        r = bit::polynomial_mod(N, p);
    std::print("r(x) = x^{} mod p(x)\n", N);
    std::print("p(x) = {}\n", p.to_polynomial());
    std::print("r(x) = {}\n", r.to_polynomial());
    return 0;
}