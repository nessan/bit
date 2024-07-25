/// @brief quick check on Danilevsky's method.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    auto M = bit::matrix<>::identity(7);
    auto p = bit::characteristic_polynomial(M);
    std::print("Characteristic polynomial: {}\n", p);
    return 0;
}
