/// @brief `Try out Jason's polynomial_mod(...) function
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

namespace bit {

/// @brief Maps input bit i to output bit 2*i, with zeros interleaved.
/// @note The high half of the input must be zero!  Not masked here
/// because it's redundant if the argument is extracted by a right shift.
/// (Technically, the high *quarter* does not need to be masked.)
///
/// Parameterizing this sort of SIMD-in-a-word code for arbitrary word
/// sizes makes it hard to read.  Here's the 32-bit version unrolled so
/// you can see how it works.  (_ is a zero bit, * is a garbage bit.)
///     x &= 0x0000ffff;        // ________________abcdefghijklmnop
///     x |= x << 8;            // ________abcdefgh********ijklmnop
///     x &= 0x00ff00ff;        // ________abcdefgh________ijklmnop
///     x |= x << 4;            // ____abcd****efgh____ijkl****mnop
///     x &= 0x0f0f0f0f;        // ____abcd____efgh____ijkl____mnop
///     x |= x << 2;            // __ab**cd__ef**gh__ij**kl__mn**op
///     x &= 0x33333333;        // __ab__cd__ef__gh__ij__kl__mn__op
///     x |= x << 1;            // _a*b_c*d_e*f_g*h_i*j_k*l_m*n_o*p
///     x &= 0x55555555;        // _a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p
///     return x;
///     To create the masks, note that 0xffff == 0xff * 0x101, so
///     0x00ff00ff == 0xffffffff / 0x101.
template<std::unsigned_integral Block>
static constexpr Block
spread_bits(Block x)
{
    constexpr int   bits_per_block = std::numeric_limits<Block>::digits;
    constexpr Block ones = std::numeric_limits<Block>::max();

    static_assert(std::numeric_limits<Block>::radix == 2, "Non-binary computers are evil.");
    static_assert(std::has_single_bit(static_cast<unsigned int>(bits_per_block)),
                  "Only power-of-two sized data types are permitted.");

// #pragma clang loop unroll(full)
#pragma GCC unroll 100
    // #pragma unroll
    for (auto i = bits_per_block / 4; i; i /= 2) {
        x |= x << i;
        x &= ones / (static_cast<Block>(1) << i | static_cast<Block>(1));
    }
    return x;
}

// Spread the bits in src[] into dest[] (non-modular squaring).
// This is done in descending address order, so the source may overlap
// with the start of the destination (but not the end).
template<std::unsigned_integral Block>
static void
spread_blocks(Block* dest, const Block* src, std::size_t size)
{
    constexpr int  bits = std::numeric_limits<Block>::digits;
    constexpr auto low_mask = std::numeric_limits<Block>::max() >> bits / 2;
    while (size-- != 0) {
        Block t = src[size];
        dest[2 * size + 1] = spread_bits(t >> bits / 2);
        dest[2 * size] = spread_bits(t & low_mask);
    }
}

/// @brief  Returns r(x) := x^e mod P(x) where P(x) is a polynomial over GF(2) and e = N or 2^N
/// @param  N We are either interested in reducing x^N or possibly x^(2^N).
/// @param  P The coefficients of P(x) a polynomial over GF(2)
/// @param  N_is_exponent If true e = 2^N which allows for huge powers of x like x^(2^100).
/// @return r The coefficients of the remainder polynomial where e^x = q(x)P(x) + r(x) and degree[r] < degree[P].
/// @note   Implementation uses repeated squaring/multiplication which is much faster than other methods for larger N.
template<std::unsigned_integral Block, typename Allocator>
vector<Block, Allocator>
jason_mod(std::size_t N, const vector<Block, Allocator>& P, bool N_is_exponent = false)
{
    using vector_type = bit::vector<Block, Allocator>;

    // If m is the highest set bit in P then P(x) = p_0 + p_1 x + ... + p_{m-1} x^{m-1}.
    const auto m = P.final_set();

    // If P has no set bits then P(x) is the zero polynomial which isn't allowed.
    if (m == vector_type::npos) throw std::invalid_argument("x^N mod 0 is not defined!");

    // If m == 0 then P(x) = 1 and so x^e % P(x) = 0.
    if (m == 0) return vector_type::zeros(1);

    // P(x) has degree m-1 so h(x) % P(x) has degree at most m - 2.
    vector_type retval(m - 1);

    // We work with the raw arrays for efficiency.
    // Bits m and up are completely ignored, but not forced to zero until immediately before return.
    // This lets us use the input P directly.
    const auto& p = P.blocks();
    auto&       r = retval.blocks();
    const auto  n_blocks = r.size();

    // Some helper functions, implemented as lambdas.

    // r(x) <- r(x)*x mod p(x) step:  This is the innermost loop!
    auto times_x_and_mod = [&](Block* rr) {
        if (raw_shift_up(rr, n_blocks, 1) >> ((m - 1) % P.bits_per_block) & 1) raw_xor(rr, p.data(), n_blocks);
    };

    // Perform the full squaring operation (very hot loop).
    //
    // Implicit arguments are p[], m, n_blocks. low[] and high[] could also be implicit parameters.
    // Making them explicit is just a style choice to make the call sites clearer.
    //
    // Squaring low[] is just spreading the bits, sending bit i to 2*i.
    // The work is all in the modular reduction.
    //
    // The square of low[] can be divided into two m-bit parts:
    // The high half which participates in modular reduction, and the low half which is already of
    // degree < m and can be just XOR'ed into the result.
    // The main design challenge is to get the high half bit-aligned with p[] so that we can XOR as needed.
    //
    // One trick is that, since low[] is of degree at most m-1, its square is of degree at most 2*m-2,
    // and the high half is of degree at most m-1.  We can save a conditional-subtract by unconditionally
    // shifting the high half up by 1 bit to start.
    //
    // When doing this, we must shift in a zero, not the x^(m-1) coefficient from the low half.
    // Fortunately, masking off the low bit is easy.
    //
    // To form these halves, we first divide low[] into two word-aligned parts and spread them separately,
    // the high part into high[], and the low part into low[].  Then we shift high[] as required to get it
    // properly aligned with p[].
    //
    // Each part consists of m bits, which fit into n_blocks blocks, but if n_blocks is odd, we have to divide
    // low[] unevenly, into chunks of size (n_blocks-1) and (n_blocks+1), then move a block over.
    //
    // Second, the most significant part may straddle blocks in such a way that it needs n_blocks+1 to hold it
    // before shifting.
    //
    // Either way, we need to start with n_blocks+1 blocks in the high  half before shifting.
    // The low block is a duplicate of the high block of the low half.
    //
    // The subsequent bit-shift is that required to move bit m-1 to bit 0, which is simply (m-1) & P.bits_per_block.
    auto square_and_mod = [&](Block* low, Block* high) {
        // Square low[], returning the wide result in high[] and low[].
        // low[] contains the low n_blocks, while high[] contains the high
        // n_blocks+1; one block is duplicated.
        if (n_blocks % 2) {
            spread_blocks(high, low + n_blocks / 2, n_blocks / 2 + 1);
            spread_blocks(low, low, n_blocks / 2);
            low[n_blocks - 1] = high[0];
        }
        else {
            spread_blocks(high + 1, low + n_blocks / 2, n_blocks / 2);
            spread_blocks(low, low, n_blocks / 2);
            high[0] = low[n_blocks - 1];
        }
        // Shift high[] down to start at bit m-1.
        int sh = (m - 1) % P.bits_per_block;
        if (sh) raw_shift_down(high, n_blocks, sh, high[n_blocks]);
        high[0] &= static_cast<Block>(-2);

        // Now repeatedly shift high[] and reduce mod p.
        for (std::size_t i = 1; i < m; i++) times_x_and_mod(high);

        // Finally, xor the high half into the low
        raw_xor(low, high, n_blocks);
    };

    // Figure out how many leading bits of the exponent we can
    // fit into retval without any modular reduction.
    // At the end, k < m is the leading exponent, e_bits is the
    // count of additional exponent bits, and (e_mask & N) is
    // the exponent bit we have finished incorporating.
    std::size_t m_bits = static_cast<std::size_t>(std::bit_width(m - 1));
    std::size_t e_mask, e_bits, k;

    if (N_is_exponent) {
        m_bits = std::min(N, m_bits);
        e_bits = N - m_bits;
        k = static_cast<std::size_t>(1) << m_bits;
        e_mask = 0;
    }
    else {
        e_bits = static_cast<std::size_t>(std::bit_width(N));
        m_bits = std::min(e_bits, m_bits);

        e_bits -= m_bits;
        k = N >> e_bits;
        if (k >= m) {
            k >>= 1;
            e_bits++;
        }
        e_mask = static_cast<std::size_t>(1) << e_bits;
    }

    retval.set(k);

    // A temporary used by square_and_mod().
    // The size is equal to n_blocks, but rounded up to an even number of words.
    // Wish we could allocate this on the stack, but variable-length arrays
    // are a G++ extension that's not part of standard C++.
    // Block temp[(n_blocks+1)/2*2];
    auto temp = new Block[(n_blocks + 1) / 2 * 2];

    // The main exponentiation loop
    while (e_bits--) {
        square_and_mod(r.data(), temp);
        e_mask >>= 1;
        if (N & e_mask) times_x_and_mod(r.data());
    }
    delete[] (temp);

    // Clear high bits and return.
    return retval.clean_nonempty();
}

} // namespace bit

int
main()
{
    std::size_t N = 447'124'345;
    auto        p = bit::vector<>::from(1234019u);
    auto        r = bit::polynomial_mod(N, p);
    std::print("r(x) = x^{} mod p(x)\n", N);
    std::print("p(x) = {}\n", p.to_polynomial());
    std::print("r(x) = {}\n", r.to_polynomial());
    return 0;
}