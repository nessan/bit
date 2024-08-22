/// @brief Basic check on the bit-vector  @c split(...) method and the @c bit::join(...) function.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using block_type = std::uint8_t;
    using vector_type = bit::vector<block_type>;
    using poly_type = bit::polynomial<block_type>;

    std::size_t n = 17;
    auto        u = vector_type::random(n);

    vector_type v, w;
    for (std::size_t i = 0; i < n; ++i) {
        u.split(i, v, w);
        auto        z = bit::join(v, w);
        std::string msg = (u == z ? "match." : "MATCH FAILED!");
        std::print("u: {} -> join({}, {}) = {} {}\n", u, v, w, z, msg);
    }

    poly_type p{u};
    poly_type f, g;
    for (std::size_t i = 0; i < n; ++i) {
        p.split(i, f, g);
        auto        z = bit::times_x(g, i) + f;
        std::string msg = (p == z ? "match." : "MATCH FAILED!");
        std::print("p: {} -> [{}] + [{}]*x^{} = {} {}\n", p, f, g, i, z, msg);
    }
}