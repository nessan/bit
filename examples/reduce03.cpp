/// @brief Set up a more challenging x^N mod P(x) problem for profiling.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    utilities::pretty_print_thousands();

    // Our polynomial
    bit::polynomial p{bit::vector<>::from(12340771959861u)};
    std::print("Polynomial p(x) = {}\n", p);

    // We are going to compute x^e mod p(x) where e is either N or 2^N
    // std::size_t N = 447'124'345;
    std::size_t N = 44712434;
    bool        N_is_pow2 = true;

    utilities::stopwatch sw;
    std::print("Calling `p.reduce({:L}, {})` ...\n", N, N_is_pow2);
    std::cout << std::flush;
    sw.click();
    auto r = p.reduce(N, N_is_pow2);
    sw.click();
    std::print("To get r(x) = {} in {:.6f} seconds.\n", r, sw.lap());

    return 0;
}
