/// @brief Check on Danilevsky's method.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    auto M1 = bit::matrix<>::identity(7);
    auto poly = bit::characteristic_polynomial(M1);
    poly.description("characteristic_polynomial() method returned");
    std::print("Characteristic polynomial: {}\n", poly.to_polynomial());
    return 0;
}
