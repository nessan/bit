#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;
    using store_type = vector_type::block_store_type;

    std::size_t n = 22;
    store_type  blocks(vector_type::blocks_needed(n));
    std::fill(blocks.begin(), blocks.end(), std::numeric_limits<vector_type::block_type>::max());

    vector_type u{n, blocks};
    std::print("bit::vector({}, blocks)            = {} and post-construction blocks = {}\n", n, u, blocks);

    vector_type v{22, std::move(blocks)};
    std::print("bit::vector({}, std::move(blocks)) = {} and post-construction blocks = {}\n", n, v, blocks);
}