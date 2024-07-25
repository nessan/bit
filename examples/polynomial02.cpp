/// @brief More basic checks on the @c bit::polynomial class.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using poly_type = bit::polynomial<std::uint8_t>;

    poly_type p0;
    std::print("p0          : {}\n", p0);
    std::print("p0^2        : {}\n", p0.squared());

    auto p1 = poly_type::random(11);
    std::print("p1          : {}\n", p1);

    p0 += p1;
    std::print("p0 += p1    : {}\n", p0);

    p0 += p0;
    std::print("p0 += p0    : {}\n", p0);

    auto p2 = poly_type::random(17);
    std::print("p1          : {}\n", p1);
    std::print("p2          : {}\n", p2);
    std::print("p1 + p2     : {}\n", p1 + p2);

    std::print("p1^2        : {}\n", p1.squared());
    std::print("p1*p1       : {}\n", p1*p1);
    std::print("p2^2        : {}\n", p2.squared());
    std::print("(p1 + p2)^2 : {}\n", (p1 + p2).squared());

    auto p3 = p1*p2;
    p1 *= p2;
    std::print("p1*p2       : {}\n", p3);
    std::print("p1 *= p2    : {}\n", p1);
}