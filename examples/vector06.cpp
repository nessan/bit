/// @brief Creating a @c bit::vector with a checker-board pattern of 0/1 elements.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;
    std::size_t n = 123;
    auto        u = vector_type::checker_board(n, 1);
    auto        v = vector_type::checker_board(n, 0);
    std::print("Starting with 1: {}\n", u);
    std::print("Starting with 0: {}\n", v);
}