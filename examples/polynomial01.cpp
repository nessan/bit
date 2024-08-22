/// @brief Basic check on the @c bit::polynomial class.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using poly_type = bit::polynomial<std::uint8_t>;

    poly_type p0;
    std::print("p0 = {} has coefficients: {:p}.\n", p0, p0.coefficients());
    std::print("Number of 0 & 1 coefficients: {} & {} respectively.\n", p0.count0(), p0.count1());

    std::size_t n = 11;
    auto        p1 = poly_type::random(n);
    std::print("p1 = {} has coefficients: {:p}.\n", p1, p1.coefficients());
    std::print("Number of 0 & 1 coefficients: {} & {} respectively.\n", p1.count0(), p1.count1());

    poly_type p2{n};
    p2.set();
    std::print("p2 = {} has coefficients: {:p}.\n", p2, p2.coefficients());
    std::print("Number of 0 & 1 coefficients: {} & {} respectively.\n", p2.count0(), p2.count1());

    auto p3 = p2;
    p3[n - 1] = 0;
    p3[n - 2] = 0;
    p3[1] = 0;
    p3[0] = 0;
    std::print("p3 = {} has coefficients: {:p}\n", p3, p3.coefficients());
    std::print("Number of 0 & 1 coefficients: {} & {} respectively.\n", p3.count0(), p3.count1());

    p3.make_monic();
    std::print("p3.make_monic() = {} has coefficients: {:p}\n", p3, p3.coefficients());
    std::print("Number of 0 & 1 coefficients: {} & {} respectively.\n", p3.count0(), p3.count1());
}