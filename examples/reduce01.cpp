/// @brief Basic check on the @c reduce method in the bit-polynomial class.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    utilities::pretty_print_thousands();
    std::size_t     N = 447'124'345;
    bit::polynomial p{bit::vector<>::from(1234019u)};
    auto            r = p.reduce(N);
    std::print("x^({:L}) mod ({}) = {}\n", N, p, r);
    return 0;
}