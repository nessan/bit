/// @brief Bit riffles
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

template<std::unsigned_integral T>
static constexpr void
riffle(T in, T& lo, T& hi)
{
    constexpr auto T_bits = std::numeric_limits<T>::digits;
    constexpr auto H_bits = T_bits / 2;

    // Split `in` into lo & hi halves.
    constexpr T all_set = std::numeric_limits<T>::max();
    constexpr T lo_mask = all_set >> H_bits;
    lo = in & lo_mask;
    hi = in >> H_bits;

    for (auto i = T_bits / 4; i > 0; i /= 2) {
        T mask = all_set / (1 << i | 1);
        lo = (lo ^ (lo << i)) & mask;
        hi = (hi ^ (hi << i)) & mask;
    }
}

template<typename Iter>
    requires std::is_unsigned_v<typename std::iterator_traits<Iter>::value_type>
static constexpr void
riffle(const Iter src_begin, const Iter src_end, Iter dst)
{
    for (auto src = src_begin; src < src_end; src++) riffle(*src, *dst++, *dst++);
}

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    std::size_t N = 41;

    auto u = vector_type::ones(N);
    auto v = u.riffled();
    std::print("u = {} has size {}\n", u, u.size());
    std::print("v = {} has size {}\n", v, v.size());

    return 0;
}