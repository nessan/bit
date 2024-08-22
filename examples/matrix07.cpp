/// @brief Checker boards
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    std::size_t r = 4;
    std::size_t c = 12;
    auto u = bit::matrix<>::checker_board(r, c, 1);
    auto v = bit::matrix<>::checker_board(r, c, 0);
    std::print("Starting with 1:\n{}\n", u);
    std::print("Starting with 0:\n{}\n", v);
}