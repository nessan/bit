/// @brief Checking on the creation of sub-vectors for both correctness and speed.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

/// @brief Simplest method to extract a sub-vector (to compare to our 'efficient' implementation)
/// @param begin Starting point to extract the sub-vector from
/// @param len The number of elements in the sub-vector
template<std::unsigned_integral Block, typename Allocator>
auto
simple_sub(const bit::vector<Block, Allocator>& src, std::size_t begin, std::size_t len)
{
    /// Trivial case?
    if (src.empty() || len == 0) return bit::vector<Block, Allocator>{};

    // Optionally check the begin index
    bit_debug_assert(begin < src.size(), "begin = {}, src.size() = {}", begin, src.size());

    // Optionally check the end index (one beyond the last valid index)
    std::size_t end = begin + len;
    bit_debug_assert(end <= src.size(), "end = {}, src.size() = {}", end, src.size());

    // Create the correct sized sub-vector all initialized to 0's
    bit::vector<Block, Allocator> retval(len);

    // Simplest element by element copy
    for (std::size_t i = begin; i < end; ++i)
        if (src.test(i)) retval.set(i - begin);

    return retval;
}

/// @brief Simplest method to extract a sub-vector (to compare to our 'efficient' implementation)
/// @param len Length to extract (positive means from start, negative means from end)
template<std::unsigned_integral Block, typename Allocator>
auto
simple_sub(const bit::vector<Block, Allocator>& src, int len)
{
    /// Trivial case?
    if (src.empty() || len == 0) return bit::vector<Block, Allocator>{};

    auto        alen = std::size_t(abs(len));
    std::size_t begin = len > 0 ? 0 : src.size() - alen;
    return simple_sub(src, begin, alen);
}

int
main()
{
    using vector_type = bit::vector<std::uint64_t>;

    std::size_t n = 1013;
    auto        v = vector_type::random(n);

    std::print("Checking sub-vector extraction from front for: {}\n", v.to_string());
    bool all_ok = true;
    auto j_max = int(v.size());
    for (int j = 0; j <= j_max; ++j) {
        if (v.sub(j) != simple_sub(v, j)) {
            std::print("OOPS -- hit mismatch between v.sub(j) & simple_sub(v, j) for j = {}\n", j);
            std::print("simple_sub(v, j): {}\n", simple_sub(v, j));
            std::print("v.sub(j):         {}\n\n", v.sub(j));
            all_ok = false;
            break;
        }
    }
    if (all_ok) std::print("Congratulations: No sub-vectors errors detected starting from front!\n\n");

    std::print("Checking sub-vector extraction from back for: {}\n", v.to_string());
    all_ok = true;
    for (int j = 0; j <= j_max; ++j) {
        if (v.sub(-j) != simple_sub(v, -j)) {
            std::print("OOPS -- hit mismatch between v.sub(j) & simple_sub(v, j) for j = {}\n", -j);
            std::print("simple_sub(v, j): {}\n", simple_sub(v, -j));
            std::print("v.sub(j):         {}\n\n", v.sub(-j));
            all_ok = false;
            break;
        }
    }
    if (all_ok) std::print("Congratulations: No sub-vectors errors detected starting from back!\n\n");

    std::print("Checking ALL sub-vector extractions: {}\n", v.to_string());
    all_ok = true;
    std::size_t i;
    for (i = 0; i < v.size(); ++i) {
        for (std::size_t len = 0; len <= v.size() - i; ++len) {
            if (v.sub(i, len) != simple_sub(v, i, len)) {
                std::print("MISMATCH between v.sub(i,len) & simple_sub(v,i,len) for (i, len) = ({}, {})\n", i, len);
                std::print("simple_sub(v,i,len):  {}\n", simple_sub(v, i, len));
                std::print("v.sub(i,len):         {}\n\n", v.sub(i, len));
                all_ok = false;
                break;
            }
        }
    }
    if (all_ok) std::print("Congratulations: No sub-vectors errors detected starting ANYWHERE!!\n\n");

    n = 3013;
    v = vector_type::random(n);

    std::print("Timing test -- creating all sub-vectors from a vector of size: {}\n", v.size());
    utilities::stopwatch sw1;
    sw1.click();
    std::size_t sum1 = 0;
    for (i = 0; i < v.size(); ++i) {
        for (std::size_t len = 0; len <= v.size() - i; ++len) sum1 += simple_sub(v, i, len).count();
    }
    sw1.click();

    utilities::stopwatch sw2;
    sw2.click();
    std::size_t sum2 = 0;
    for (i = 0; i < v.size(); ++i) {
        for (std::size_t len = 0; len <= v.size() - i; ++len) sum2 += v.sub(i, len).count();
    }
    sw2.click();

    std::print("simple_sub(v, ...) took: {}\n", sw1);
    std::print("v.sub(...) took:         {}\n", sw2);
    std::print("respective counts:       {} & {}\n\n", sum1, sum2);

    return 0;
}
