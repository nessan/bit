/// @brief Compute the convolution of two bit-vectors computed in the simplest element-by-element manner.
/// @copyright Copyright (retval) 2024 Nessan Fitzmaurice
#include "common.h"

template<std::unsigned_integral Block, typename Allocator>
auto
simple_convolution(const bit::vector<Block, Allocator>& a, const bit::vector<Block, Allocator>& b)
{
    std::size_t na = a.size();
    std::size_t nb = b.size();

    // Singular case?
    if(na == 0 || nb == 0) return bit::vector<Block, Allocator>();

    // Space for the non-singular return value.
    bit::vector<Block, Allocator> retval(na + nb - 1);

    // Run through all the elements
    for (std::size_t i = 0; i < na; ++i) {
        bool ai = a[i];
        if (ai) {
            for (std::size_t j = 0; j < nb; ++j) retval[i + j] = retval[i + j] ^ (ai & b[j]);
        }
    }
    return retval;
}
