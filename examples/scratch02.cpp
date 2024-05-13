#include "common.h"

template<std::unsigned_integral Block>
static constexpr Block
spread_bits(Block x)
{
    constexpr int   bits_per_block = std::numeric_limits<Block>::digits;
    constexpr Block ones = std::numeric_limits<Block>::max();

    for (auto i = bits_per_block / 4; i; i /= 2) {
        x |= x << i;
        x &= ones / (static_cast<Block>(1) << i | static_cast<Block>(1));
    }
    return x;
}

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

template<std::unsigned_integral Block>
static void
raw_xor(Block* lhs, const Block* rhs, std::size_t size)
{
    for (std::size_t i = 0; i < size; i++) lhs[i] ^= rhs[i];
}

// Shift up 0 < bits < bits_per_block.  Highest-numbered "bits" elements
// are discarded, and lowest-numbered bits are filled with zero.
//
// Returns the ms-word of the input, before shifting.  This is useful for
// the caller, which needs to examine one bit to see if a modular reduction
// is required.
template<std::unsigned_integral Block>
static Block
raw_shift_up(Block* b, std::size_t size, int sh)
{
    constexpr int bits_per_block = std::numeric_limits<Block>::digits;
    Block         y = 0;
    for (std::size_t i = 0; i < size; i++) {
        Block x = b[i];
        b[i] = x << sh | y >> (bits_per_block - sh);
        y = x;
    }
    return y;
}

// Shift down 0 < bits < bits_per_block.  Lowest-numbered "bits" elements
// are discarded.  The ms-bits are filled in from the ms-bits of the "x"
// parameter (if provided).  This enables narrowing shifts.
template<std::unsigned_integral Block>
static void
raw_shift_down(Block* b, std::size_t size, int sh, Block x = 0)
{
    constexpr int bits_per_block = std::numeric_limits<Block>::digits;
    while (size--) {
        // Attempting to encourage the compiler to use SHRD.
        Block y = b[size];
        b[size] = x << (bits_per_block - sh) | y >> sh;
        x = y;
    }
}

template<std::unsigned_integral Block, typename Allocator>
bit::vector<Block, Allocator>
polynomial_mod2(std::size_t N, const bit::vector<Block, Allocator>& P, bool N_is_exponent = false)
{
    const auto m = P.final_set();

    // Cannot do anything with an empty polynomial.
    if (m == P.npos) throw std::invalid_argument("x^N mod P(x) is not defined for an empty P(x).");

    // The residual polynomial has degree less than m -- r0 + r1 x + ... + r_{m-1}x^{m-1}
    bit::vector<Block, Allocator> retval(m);

    // Mod 1 is trivial. Excluding it avoids the need to test for m == 0 or n_blocks == 0 later in the code.
    if (m == 0) return retval;

    // We work with the raw arrays for efficiency.
    // Bits m and up are completely ignored, but not forced to zero until immediately before returning.
    // This lets us use the input P directly.
    const auto& p = P.blocks();
    auto&       r = retval.blocks();
    const auto  n_blocks = r.size();

    // Some helper functions, implemented as lambdas.

    // Perform an r(x) <- x*r(x) mod p(x) step (this is an innermost loop)
    auto times_x_and_mod = [&](Block* rr) {
        if (raw_shift_up(rr, n_blocks, 1) >> ((m - 1) % P.bits_per_block) & 1) raw_xor(rr, p.data(), n_blocks);
    };

    // Perform the full squaring operation (very hot loop).
    //
    // Implicit arguments are p[], m, n_blocks. low[] and high[] could be implicit parameters, too.  Making them
    // explicit is just a style choice to make the call sites clearer.
    //
    // Squaring low[] is just spreading the bits, sending bit i to 2*i.
    // The work is all in the modular reduction.
    //
    // The square of low[] can be divided into two m-bit parts: the high half which participates in modular reduction,
    // and the low half which is already of degree < m and can be just XORed into the result.  The main design challenge
    // is to get the high half bit-aligned with p[] so that we can XOR as needed.
    //
    // One trick is that, since low[] is of degree at most m-1, its square is of degree at most 2*m-2, and the high half
    // is of degree at most m-2.  We can save a conditional-subtract by unconditionally shifting the high half up by 1
    // bit to start.
    //
    // When doing this, we must shift in a zero, not the x^(m-1) coefficient from the low half.  Fortunately, masking
    // off the low bit is easy.
    //
    // To form these halves, we first divide low[] into two word-aligned parts and spread them separately, the high part
    // into high[], and the low part into low[].  Then we shift high[] as required to get it properly aligned with p[].
    //
    // Each part consists of m bits, which fit into n_blocks blocks, but if n_blocks is odd, we have to divide low[]
    // unevenly, into chunks of size (n_blocks-1) and (n_blocks+1), then move a block over.
    //
    // Second, the most significant part may straddle blocks in such a way that it needs n_blocks+1 to hold it before
    // shifting.
    //
    // Either way, we need to start with n_blocks+1 blocks in the high half before shifting.  The low block is a
    // duplicate of the high block of the low half.
    //
    // The subsequent bit-shift is that required to move bit m-1 to bit 0, which is simply (m-1) & P.bits_per_block.
    auto square_and_mod = [&](Block* low, Block* high) {
        // Square low[], returning the wide result in high[] and low[].
        // low[] contains the low n_blocks, while high[] contains the high n_blocks+1; one block is duplicated.
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

    // Figure out how many leading bits of the exponent we can fit into retval without any modular reduction.
    // At the end, k < m is the leading exponent, e_bits is the count of additional exponent bits, and (e_mask & N) is
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
    // Wish we could allocate this on the stack, but variable-length arrays are a G++ extension that's not part of
    // standard C++. Block temp[(n_blocks+1)/2*2];
    auto temp = new Block[(n_blocks + 1) / 2 * 2];

    // The main exponentiation loop
    while (e_bits--) {
        square_and_mod(r.data(), temp);
        e_mask >>= 1;
        if (N & e_mask) times_x_and_mod(r.data());
    }
    delete[] (temp);

    // Clear high bits and return.
    return retval.clean();
}

int
main()
{
    utilities::pretty_print_thousands();

    std::size_t N = 447'124'345;
    auto        p = bit::vector<>::from(1234019u);
    std::print("Computing x^{:L} mod p(x) = {}\n", N, p.to_polynomial());

    utilities::stopwatch sw;

    std::print("Method bit::polynomial_mod  ... ");
    sw.click();
    auto r = bit::polynomial_mod(N, p);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", r.to_polynomial(), sw.lap());

    std::print("Method polynomial_mod2      ... ");
    sw.click();
    r = polynomial_mod2(N, p, false);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", r.to_polynomial(), sw.lap());

    return 0;
}