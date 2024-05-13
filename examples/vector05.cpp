/// @brief Joining bit-vectors.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    std::size_t nu = 1;
    std::size_t nv = 7;

    vector_type u(nu);
    vector_type v(nv);
    v.flip();

    u.description("U");
    v.description("V");

    u.append(v);
    u.description("U|V");
    return 0;
}