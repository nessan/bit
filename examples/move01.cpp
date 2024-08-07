/// @brief Create a bit::vector from a pre-filled store of bits either as a copy or as a move.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT

#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;
    using block_store_type = vector_type::block_store_type;

    std::size_t      n = 22;
    block_store_type blocks(vector_type::blocks_needed(n));
    std::fill(blocks.begin(), blocks.end(), std::numeric_limits<vector_type::block_type>::max());

    // Construct a bit-vector from an l-value reference to `blocks` which stays unchanged after the call.
    vector_type u{n, blocks};
    std::cout << "bit::vector(" << n << ", blocks)            = " << u << '\n';
    std::cout << "post-construction blocks size      = " << blocks.size() << '\n';

    // Construct a bit-vector from an r-value reference to `blocks` which is emptied by the call.
    vector_type v{n, std::move(blocks)};
    std::cout << "bit::vector(" << n << ", std::move(blocks)) = " << u << '\n';
    std::cout << "post-construction blocks size      = " << blocks.size() << '\n';
}