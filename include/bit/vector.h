/// @brief A bit-vector class bit::vector.
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "bit_assert.h"
#include <array>
#include <bit>
#include <bitset>
#include <chrono>
#include <concepts>
#include <format>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace bit {

/// @brief  A class for vectors over GF(2), the space of two elements {0,1} with arithmetic done mod 2.
/// @tparam Block We pack the elements of the bit-vector into an array of unsigned integers of this type.
/// @tparam Allocator The memory manager that is used to allocate/deallocate space for the blocks as needed.
template<std::unsigned_integral Block = std::uint64_t, typename Allocator = std::allocator<Block>>
class vector {
public:
    /// @brief The individual bit-vector elements/bits are packed into blocks of this unsigned integer type.
    using block_type = Block;

    /// @brief The memory manager used by the bit-vector's store of blocks.
    using allocator_type = Allocator;

    /// @brief A special index value to indicate not-found/no-such-position failure for some methods below.
    static constexpr auto npos = static_cast<std::size_t>(-1);

    /// @brief Create a bit-vector with @c n elements.
    /// @note  The default constructor creates a completely empty bit-vector with size 0.
    explicit constexpr vector(std::size_t n = 0) : m_size{n}, m_blocks(blocks_needed(n))
    {
        // Empty body -- we now have an underlying store vector of blocks all initialized to 0.
        // Note that it's a mistake to use uniform initialization on std::vectors.
    }

    /// @brief Create a bit-vector of size @c n by repeatedly copying the bits of a constant block value.
    /// @param n The size of the bit-vector to create.
    /// @param blk All the blocks in the bit-vector have this value (bar the final one which will be masked for size).
    explicit constexpr vector(std::size_t n, Block blk) : m_size{n}, m_blocks(blocks_needed(n), blk)
    {
        // W e now have a vector of blocks all initialized to blk.
        // The final block may now have excess junk we need to set to zero.
        clean();
    }

    /// @brief Create a bit-vector by copying all the bits from an iteration of unsigned words.
    /// @note  The @c value_type associated with the iterator must be unsigned but not necessarily the same as Block.
    template<typename Iter>
        requires std::is_unsigned_v<typename std::iterator_traits<Iter>::value_type>
    explicit constexpr vector(Iter b, Iter e) : vector{}
    {
        append(b, e);
    }

    /// @brief Create a bit-vector by copying the bits from a @c std::bitset.
    template<std::size_t N>
    explicit constexpr vector(const std::bitset<N>& bs) : vector{}
    {
        append(bs);
    }

    /// @brief Create a bit-vector by calling @c f(i) for @c i=0,...,n-1 where a non-zero return indicates a set bit.
    explicit constexpr vector(std::size_t n, std::invocable<std::size_t> auto f) : vector{n}
    {
        for (std::size_t i = 0; i < n; ++i)
            if (f(i) != 0) set(i);
    }

    /// @brief Create a bit-vector from bits encoded in a string (generally either all 0's & 1's or all hex chars).
    /// @param src The source string which can be prefixed by "0x"/"0X" for hex (and should be to avoid doubt).
    ///        For hex strings there can also be a suffix (one of "_2", "_4", or "_8").
    ///        That suffix give the base for the last character in the string.
    ///        So "0x1" -> 0001, but "0x1_8" -> 001, "0x1_4" -> 01, and "0x1_2" ->  1.
    ///        Without any suffix, hex-strings always parse as bit-vectors with a size divisible by 4.
    /// @param bit_order If true (default is false) the string will have the lowest bit on its right.
    ///        This argument is completely ignored for hex strings.
    /// @throw This method throws a @c std::invalid_argument exception if the string is not recognized
    explicit vector(std::string_view src, bool bit_order = false) : vector{}
    {
        // Defer the work to our factory method that tries to parse the input string.
        auto v = from(src, bit_order);
        if (!v) throw std::invalid_argument("Failed to parse the input string as a valid bit-vector!");
        *this = *v;
    }

    /// @brief Factory method to generate a bit-vector of size @c n where the elements are all 0.
    static constexpr vector zeros(std::size_t n) { return vector{n}; }

    /// @brief Factory method to generate a bit-vector of size @c n where the elements are all 1.
    static constexpr vector ones(std::size_t n)
    {
        // We construct by copying the 111... value into all the needed blocks.
        return vector{n, ones_block()};
    }

    /// @brief Factory method to generate a unit bit-vector of size @c n where only element @c i is set.
    /// @param i Must have @c i<n -- that is only checked on non-release builds.
    static constexpr vector unit(std::size_t n, std::size_t i)
    {
        bit_assert(i < n, "Unit axis i = {} should be less than the bit-vector size n = {}", i, n);
        vector retval{n};
        retval.set(i);
        return retval;
    }

    /// @brief Factory to generate a bit-vector of size @c n where the elements are either 1010101... or 0101010...
    static constexpr vector checker_board(std::size_t n, bool first_element_set = true)
    {
        // We construct by copying the appropriate `checkered_block` into all the needed blocks.
        auto blk = checkered_block(first_element_set);
        return vector{n, blk};
    }

    /// @brief Factory method to generate a bit-vector of size @c n where the elements are from independent coin flips.
    /// @param p The probability of the elements being 1 (defaults to a fair coin, i.e. 50-50).
    static vector random(std::size_t n, double p = 0.5)
    {
        // Need a valid probability ...
        if (p < 0 || p > 1) throw std::invalid_argument("Probability outside valid range [0,1]!");

        // Scale p by 2^64 to remove floating point arithmetic from the main loop below.
        // If we determine p rounds to 1 then we can just set all elements to 1 and return early.
        p = p * 0x1p64 + 0.5;
        if (p >= 0x1p64) return ones(n);

        // p does not round to 1 so we use a 64-bit URNG and check each draw against the 64-bit scaled p.
        auto scaled_p = static_cast<std::uint64_t>(p);

        // The URNG used is a simple congruential uniform 64-bit RNG seeded to a clock-dependent state.
        // The multiplier comes from Steele & Vigna -- see https://arxiv.org/abs/2001.05304
        using lcg = std::linear_congruential_engine<uint64_t, 0xd1342543de82ef95, 1, 0>;
        static lcg rng(static_cast<lcg::result_type>(std::chrono::system_clock::now().time_since_epoch().count()));

        return vector{n, [&](std::size_t) { return rng() < scaled_p; }};
    }

    /// @brief Factory method to create a bit-vector from any unsigned type word.
    /// @note  This isn't a constructor because we don't want @c src to be treated as the number of bit-vector elements!
    static constexpr vector from(std::unsigned_integral auto src)
    {
        vector retval{};
        retval.append(src);
        return retval;
    }

    /// @brief  Factory method that attempts to parse a bit-vector from a string.
    /// @param  src he source string which can be prefixed by "0x"/"0X" for hex (and should be to avoid doubt).
    ///         For hex strings there can also be a suffix (one of "_2", "_4", or "_8").
    ///         That suffix give the base for the last character in the string.
    ///         So "0x1" -> 0b0001, but "0x1_8" -> 0b001, "0x1_4" -> 0b01, and "0x1_2" -> 0b1.
    ///         Without any suffix, hex-strings always parse as bit-vectors with a size divisible by 4.
    /// @param  bit_order If true (default is false) the string will have the lowest bit on its right.
    ///         This argument is completely ignored for hex strings.
    /// @return Returns @c std::nullopt if we fail to parse the input string.
    static std::optional<vector> from(std::string_view src, bool bit_order = false)
    {
        // If the string is all 0's and 1's we assume that the this is a binary string.
        // To force such a string to be intrepreted as hex add the prefix "0x"!
        if (src.find_first_not_of("01") == std::string::npos) {

            // If necessary, reorder the string from the least to the most significant bit.
            std::string s{src};
            if (bit_order) std::reverse(s.begin(), s.end());

            // Create an appropriately sized bit-vector which is all zeros to start with.
            std::size_t n_chars = s.length();
            vector      retval{n_chars};

            // Each '1' in s sets the corresponding bit in retval.
            for (std::size_t i = 0; i < n_chars; ++i)
                if (s[i] == '1') retval.set(i);

            return retval;
        }
        else {
            // It's not a binary string so try our hex decoder.
            return from_hex_string(src);
        }
    }

    /// @brief Returns the number of elements in this bit-vector.
    constexpr std::size_t size() const { return m_size; }

    /// @brief Returns true if the size of this bit-vector is zero.
    constexpr bool empty() const { return size() == 0; }

    /// @brief What is the capacity of this bit-vector (how many bits can it hold without allocating memory)?
    constexpr std::size_t capacity() const { return bits_per_block * m_blocks.capacity(); }

    /// @brief What is the size in bits of any unused capacity in this bit-vector?
    constexpr std::size_t unused() const { return capacity() - size(); }

    /// @brief Increase the bit-vector's capacity so it can hold @c n elements without allocating more memory.
    /// @note  If the capacity increases, the @c size() and values stay the same, but references etc. are invalidated.
    /// @note  This method does nothing if the @c n elements fits inside the current capacity.
    constexpr vector& reserve(std::size_t n)
    {
        m_blocks.reserve(blocks_needed(n));
        return *this;
    }

    /// @brief Try to minimize any unused/excess capacity -- may do nothing.
    constexpr vector& shrink_to_fit()
    {
        m_blocks.resize(blocks_needed(size()));
        m_blocks.shrink_to_fit();
        return *this;
    }

    /// @brief Resize the bit-vector padding out any extra values with 0's.
    ///        If @c n<size() the bit-vector is reduced to the first @c n elements.
    ///        If @c n>size() then the extra appended values are zeros.
    constexpr vector& resize(std::size_t n)
    {
        // Edge case?
        if (size() == n) return *this;

        // Perhaps we need to increase/decrease the size of the underlying data (increases are zeros)
        auto need = blocks_needed(n);
        if (auto have = block_count(); have != need) m_blocks.resize(need);

        // Record the new size and clean up the last occupied word if necessary.
        m_size = n;
        return clean();
    }

    /// @brief Removes all elements from the bit-vector so @c size()==0. Capacity is not changed!
    constexpr vector& clear()
    {
        m_blocks.clear();
        m_size = 0;
        return *this;
    }

    /// @brief Swap the values of elements at locations i & j
    constexpr vector& swap_elements(std::size_t i, std::size_t j)
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < m_size = {}", i, m_size);
        bit_debug_assert(j < m_size, "Index j = {} must be < m_size = {}", j, m_size);
        if (test(i) != test(j)) {
            flip(i);
            flip(j);
        }
        return *this;
    }

    /// @brief Swap the bits of this bit-vector with another
    constexpr vector& swap(vector& other) noexcept
    {
        std::swap(m_size, other.m_size);
        std::swap(m_blocks, other.m_blocks);
        return *this;
    }

    /// @brief Removes the last element from the bit-vector & shrinks it if possible
    constexpr vector& pop()
    {
        if (!empty()) {
            // Zap that last element that we are about to lose to  keep things clean.
            reset(m_size - 1);

            // Shrink the bit-vector
            --m_size;

            // Perhaps we can also shrink the storage data ?
            if (block_count() > blocks_needed(m_size)) m_blocks.pop_back();
        }
        return *this;
    }

    /// @brief Adds a single element to the end of this bit-vector. It will be 0 unless the argument is @c true
    constexpr vector& push(bool one = false)
    {
        // Perhaps we need more storage?
        m_blocks.resize(blocks_needed(m_size + 1));

        // Increase the size of the bit-vector and set the new last bit if appropriate
        ++m_size;
        if (one) set(m_size - 1);
        return *this;
    }

    /// @brief We have several @c append(...) methods so add a synonym for the @c push(bool) method for consistency.
    constexpr vector& append(bool b) { return push(b); }

    /// @brief Append another bit-vector to the end of this one
    constexpr vector& append(const vector& v)
    {
        // Edge case?
        if (v.size() == 0) return *this;

        // Resize the underlying block store so we have enough space to accommodate the extra bits
        m_blocks.resize(blocks_needed(m_size + v.m_size));

        // Where is starting location for where the new bits need to go?
        auto blk = next_block_index();
        auto bit = next_bit_index();

        // Easy case: If we're at the start of a new block we can just copy the blocks in v over to the new space.
        if (bit == 0) {
            for (std::size_t i = 0; i < v.block_count(); ++i) block(blk + i) = v.block(i);
            m_size += v.m_size;
            return *this;
        }

        // Otherwise we may have to put each block from v in two contiguous blocks in our own store.
        for (std::size_t i = 0; i < v.block_count(); ++i) {

            // Split the i'th block of v into lo and hi groups of bits.
            // Note that it may be the case that what's left of v may fit into the lo block alone!
            auto lo = static_cast<Block>(v.block(i) << bit);
            auto hi = static_cast<Block>(v.block(i) >> (bits_per_block - bit));

            // Add the low group to our current block.
            block(blk + i) |= lo;

            // If necessary add the hi group to the next block along in our store
            // Note that if it doesn't fit its not needed!
            if (blk + i + 1 < block_count()) block(blk + i + 1) = hi;
        }

        // Record the new size and go home.
        m_size += v.m_size;
        return *this;
    }

    /// @brief  Append a whole word of bits to the end of this bit-vector.
    /// @tparam Src The type of the word for the source bits (could e.g. be unsigned or unsigned char etc.)
    /// @note   The type of the @c src word need not match the bit-vector's block type.
    template<std::unsigned_integral Src>
    constexpr vector& append(Src src)
    {
        // Resize the underlying data of blocks so we have enough space to accommodate the extra bits
        constexpr std::size_t bits_per_src = std::numeric_limits<Src>::digits;
        m_blocks.resize(blocks_needed(m_size + bits_per_src));

        if constexpr (bits_per_src > bits_per_block) {

            // Handle long source words by splitting them into shorter ones and recursing ...
            // The source needs to be some even multiple of the blocks in bits or something is off!
            static_assert(bits_per_src % bits_per_block == 0, "Cannot pack the source evenly into an array of blocks!");
            constexpr std::size_t blocks_per_src = bits_per_src / bits_per_block;

            // Mask out the appropriate number of bits in the source to create shorter words to push one at a time.
            auto mask = static_cast<Src>(ones_block());
            for (std::size_t i = 0; i < blocks_per_src; ++i, src >>= bits_per_block) {
                auto tmp = static_cast<Block>(src & mask);
                append(tmp);
            }
            return *this;
        }
        else {
            // Short source words so we can safely cast the Src to a Block
            auto tmp = static_cast<Block>(src);

            // Where are we putting things?
            auto blk = next_block_index();
            auto bit = next_bit_index();

            // If we're at the start of a new block we can just copy the src over.
            if (bit == 0) {
                block(blk) = tmp;
                m_size += bits_per_src;
                return *this;
            }

            // Otherwise push as many src bits as possible into the current block.
            std::size_t available = bits_per_block - bit;
            block(blk) |= Block(tmp << bit);

            // If there are bits left in the source word they go at the start of the next block
            if (bits_per_src > available) block(blk + 1) = tmp >> available;
            m_size += bits_per_src;
            return *this;
        }
    }

    /// @brief Add an iterated collection of unsigned words as bits to the end of the bit-vector.
    /// @note  The @c value_type of the iterator should be some unsigned integer type but needn’t be @c Block.
    template<typename Iter>
        requires std::is_unsigned_v<typename std::iterator_traits<Iter>::value_type>
    constexpr vector& append(Iter b, Iter e)
    {
        // Edge case?
        if (b == e) return *this;

        // Number of src words to copy
        auto n_words = static_cast<std::size_t>(std::distance(b, e));

        // Resize the underlying block store so we have enough space to accommodate the extra bits
        using src_type = typename std::iterator_traits<Iter>::value_type;
        std::size_t n_bits = n_words * std::numeric_limits<src_type>::digits;
        m_blocks.resize(blocks_needed(m_size + n_bits));

        // Run through the source words and add each one
        while (b != e) append(*b++);
        return *this;
    }

    /// @brief  Add a whole list of unsigned source words as bits to the end of the bit-vector.
    /// @tparam Src The type of the words for the source bits which need not be @c Block.
    template<std::unsigned_integral Src = Block>
    constexpr vector& append(std::initializer_list<Src> src)
    {
        return append(std::cbegin(src), std::cend(src));
    }

    /// @brief  Add a @c std::vector<Src> of unsigned source words as bits to the end of the bit-vector.
    /// @tparam Src The type of the words for the source bits which need not be @c Block.
    template<std::unsigned_integral Src>
    constexpr vector& append(const std::vector<Src>& src)
    {
        return append(std::cbegin(src), std::cend(src));
    }

    /// @brief  Add a @c std::array<Src,N> of @c N unsigned source words as bits to the end of the bit-vector.
    /// @tparam Src The type of the words for the source bits which need not be @c Block.
    /// @tparam N The fixed array size
    template<std::unsigned_integral Src, std::size_t N>
    constexpr vector& append(const std::array<Src, N>& src)
    {
        return append(std::cbegin(src), std::cend(src));
    }

    /// @brief  Add all the bits from a std::bitset` to the end of this bit-vector.
    /// @tparam N The size of the @c std::bitset which should be deduced in any case.
    template<std::size_t N>
    constexpr vector& append(const std::bitset<N>& src)
    {
        // We do this the dumb way ...
        for (std::size_t i = 0; i < N; ++i) push(src[i]);
        return *this;
    }

    /// @brief  Create a "riffled" version of this bit-vector (i.e. one with our bits interleaved with zeros).
    /// @return If this vector has elements @c abcd then return bit-vector will have elements @c a0b0c0d
    constexpr vector riffled() const
    {
        // We let the riffled(dst) method do the heavy lifting.
        vector retval;
        riffled(retval);
        return retval;
    }

    /// @brief  Create a "riffled" copy of this bit-vector (i.e. one with our bits interleaved with zeros) in @c dst
    /// @return If this vector has elements @c abcd then on return the bit-vector @c dst will have elements @c a0b0c0d
    /// @note   This version is useful in e.g. repeated squaring algorithms where we want to reuse the @c dst storage.
    constexpr void riffled(vector& dst) const
    {
        // Nothing much to do unless there we have at least 2 elements (with two elements `ab` we return `a0b`).
        if (m_size < 2) {
            dst = *this;
            return;
        }

        // Make sure the destination block store is big enough (at this point it may be one block too large).
        auto N = block_count();
        dst.m_blocks.resize(2 * N);

        // Riffle each block into a pair of adjacent blocks in the destination store using a class helper method.
        for (std::size_t i = 0; i < N; ++i) riffle(block(i), dst.block(2 * i), dst.block(2 * i + 1));

        // Our `abcde` example now looks like `a0b0c0d0e0` instead of `a0b0c0d0e`.
        // Chop that last 0 element off by setting the destination size appropriately (no clean() operation needed).
        std::size_t dst_size = 2 * m_size - 1;
        dst.m_blocks.resize(blocks_needed(dst_size));
        dst.m_size = dst_size;
    }

    /// @brief  Create a new bit-vector as a distinct sub-vector of this one.
    /// @param  begin The first element to copy from this bit-vector to index 0 of the returned bit-vector
    /// @param  len The number of elements to copy.
    /// @return A completely new bit-vector of size @c len
    constexpr vector sub(std::size_t begin, std::size_t len) const
    {
        // Edge cases?
        if (empty() || len == 0) return vector{};

        // Another easy one ...
        if (begin == 0 && len == m_size) return vector{*this};

        // DEBUG builds will check the starting position of the sub-vector & the length.
        bit_debug_assert(begin < m_size, "begin = {}, m_size = {}", begin, m_size);
        bit_debug_assert(begin + len < m_size, "begin = {}, len = {}, m_size = {}", begin, len, m_size);

        // Create the right sized rub-vector all initialized to 0's
        vector retval{len};

        // Where does the begin bit of the sub-vector lie in our block store?
        std::size_t begin_blk = block_index_for(begin);
        std::size_t begin_bit = bit_index_for(begin);
        std::size_t complement = bits_per_block - begin_bit;

        // Where does the end bit of the sub-vector lie in our block store?
        std::size_t end_blk = block_index_for(begin + len - 1);

        if (begin_bit == 0) {
            // Can copy over whole blocks and deal with any extra bits copied with a clean()
            for (std::size_t i = begin_blk; i <= end_blk; ++i) retval.block(i - begin_blk) = block(i);
        }
        else {
            // Not working on word boundaries (each block in the sub-vector is a combo of two of ours).
            for (std::size_t i = begin_blk; i < end_blk; ++i) {
                retval.block(i - begin_blk) = static_cast<Block>(block(i) >> begin_bit);
                if (i + 1 < block_count())
                    retval.block(i - begin_blk) |= static_cast<Block>(block(i + 1) << complement);
            }

            // Do we still need to grab the last few bits? Do those the dumb way for now ...
            std::size_t copied = (end_blk - begin_blk) * bits_per_block;
            if (copied < len) {
                for (std::size_t i = copied; i < len; ++i)
                    if (test(begin + i)) retval.set(i);
            }
        }

        // Might have over-copied on the last block so do a cleanup
        return retval.clean();
    }

    /// @brief  Create a new bit-vector as a distinct sub-vector of this one.
    /// @param  len Length to extract (positive means from start, negative means from end).
    /// @return A completely new bit-vector of size @c abs(len)
    constexpr vector sub(int len) const
    {
        // EDge case?
        if (empty() || len == 0) return vector{};

        auto        alen = std::size_t(abs(len));
        std::size_t begin = len > 0 ? 0 : m_size - alen;
        return sub(begin, alen);
    }

    /// @brief Split this vector into two -- one of size @c n_lo and the other with all the rest of the elements.
    /// @param n_lo  The size of @c lo on output -- this should be at most @c size()
    /// @param lo We copy the first @c n_lo elements of this bit-vector into @c lo
    /// @param hi We copy the other @c size()-n elements into @c hi
    /// @note  If necessary, we silently change @c n_lo to be at most @c size()
    /// @note  After we're done a call to @c join(lo,hi) should reproduce this bit-vector.
    constexpr void split(std::size_t n_lo, vector& lo, vector& hi) const
    {
        // Silently correct the n_lo argument if it is too large.
        if (n_lo > m_size) n_lo = m_size;

        // Resize lo to its correct dimension.
        lo.resize(n_lo);

        // Copy over our low blocks to lo -- this can be done a block at time.
        for (std::size_t i = 0; i < lo.block_count(); ++i) lo.block(i) = block(i);

        // From lo's perspective that last block we copied may have some extra "junk" bits so clean those up.
        lo.clean();

        // Move on to hi which we first resize to its correct dimension.
        std::size_t n_hi = m_size - n_lo;
        hi.resize(n_hi);

        // Edge case?
        if (n_hi == 0) return;

        // hi will have a copy of our elements from n_lo on. Need to determine where that element lies on our store.
        std::size_t blk = block_index_for(n_lo);
        std::size_t bit = bit_index_for(n_lo);

        if (bit == 0) {
            // Can just copy over whole blocks from our store to hi's store.
            for (std::size_t i = blk; i < block_count(); ++i) hi.block(i - blk) = block(i);
        }
        else {
            // Not working on word boundaries so most blocks in hi's store are a combination of two of ours.
            std::size_t bit_complement = bits_per_block - bit;
            for (std::size_t i = blk; i < block_count(); ++i) {
                hi.block(i - blk) = static_cast<Block>(block(i) >> bit);
                if (i + 1 < block_count()) hi.block(i - blk) |= static_cast<Block>(block(i + 1) << bit_complement);
            }
        }
        // No hi.clean() call needed as our own store of blocks is clean!
    }

    /// @brief Create a new bit-vector that is a copy of this one without any trailing zeros.
    constexpr vector trimmed_right() const
    {
        auto f = final_set();
        return f != npos ? sub(0, f + 1) : vector{};
    }

    /// @brief Create a new bit-vector that is a copy of this one without any leading zeros.
    constexpr vector trimmed_left() const
    {
        auto f = first_set();
        return f != npos ? sub(f, m_size - f) : vector{};
    }

    /// @brief Create a new bit-vector that is a copy of this one without any leading or trailing zeros.
    constexpr vector trimmed() const
    {
        auto f0 = first_set();
        if (f0 == npos) return vector{};
        auto f1 = final_set();
        return sub(f0, f1 + 1 - f0);
    }

    /// @brief Write access to an individual bit in a bit-vector is through this `vector::reference` proxy class.
    class reference {
    public:
        constexpr reference(vector& v, std::size_t bit) : m_ref{v.block_ref_for(bit)}, m_mask{v.block_mask_for(bit)}
        {
            // Empty body
        }

        // The default compiler generator implementations for most "rule of 5" methods will be fine ...
        constexpr reference(const reference&) = default;
        constexpr reference(reference&&) noexcept = default;
        ~reference() = default;

        // Cannot take a reference to a reference ...
        constexpr void operator&() = delete;

        // Have some custom operator='s
        constexpr reference& operator=(const reference& rhs)
        {
            set_to(rhs.to_bool());
            return *this;
        }
        constexpr reference& operator=(reference&& rhs) noexcept
        {
            set_to(rhs.to_bool());
            return *this;
        }
        constexpr reference& operator=(bool rhs)
        {
            set_to(rhs);
            return *this;
        }
        constexpr reference& set_to(bool rhs)
        {
            rhs ? set() : reset();
            return *this;
        }
        constexpr reference& set()
        {
            m_ref |= m_mask;
            return *this;
        }
        constexpr reference& reset()
        {
            m_ref &= Block(~m_mask);
            return *this;
        }
        constexpr reference& flip()
        {
            m_ref ^= m_mask;
            return *this;
        }

        // Bitwise operations with a right-hand-side bit
        constexpr reference& operator&=(bool rhs)
        {
            if (!rhs) reset();
            return *this;
        }
        constexpr reference& operator|=(bool rhs)
        {
            if (rhs) set();
            return *this;
        }
        constexpr reference& operator^=(bool rhs)
        {
            if (rhs) flip();
            return *this;
        }
        constexpr reference& operator-=(bool rhs)
        {
            if (rhs) reset();
            return *this;
        }
        constexpr reference& operator~() const { return flip(); }

        // Explicitly convert to a bool
        constexpr bool to_bool() const { return (m_ref & m_mask); }

        // Actually you can always use a vector::reference as a bool
        constexpr operator bool() const { return to_bool(); }

    private:
        Block& m_ref;  // This is the block in the bit-vector store that holds the bit-vector element of interest.
        Block  m_mask; // This mask singles out the single bit of interest in that block.
    };

    /// @brief Read-write access to the element at index @c i
    constexpr reference element(std::size_t i)
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < m_size = {}", i, m_size);
        return reference(*this, i);
    }

    /// @brief Read-only access to the element at index @c i
    constexpr bool element(std::size_t i) const
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < m_size = {}", i, m_size);
        return test(i);
    }

    /// @brief Read-write access to element 0
    constexpr reference front() { return element(0); }

    /// @brief Read-only access to element 0
    constexpr bool front() const { return element(0); }

    /// @brief Read-write access to the element at index @c size()-1
    constexpr reference back()
    {
        auto n = size();
        bit_debug_assert(n > 0, "Empty bit-vector!");
        return element(n - 1);
    }

    /// @brief Read-only access to the element at index @c size()-1
    constexpr reference back() const
    {
        auto n = size();
        bit_debug_assert(n > 0, "Empty bit-vector!");
        return element(n - 1);
    }

    /// @brief Read-write access to the bit-vector element at index @c i
    constexpr reference operator[](std::size_t i) { return element(i); }

    /// @brief Read-only access to the bit-vector element at index @c i
    constexpr bool operator[](std::size_t i) const { return element(i); }

    /// @brief Read-write access to the bit-vector element at index @c i
    constexpr reference operator()(std::size_t i) { return element(i); }

    /// @brief Read-only access to the bit-vector element at index @c i
    constexpr bool operator()(std::size_t i) const { return element(i); }

    /// @brief Check whether the bit-vector element at index @c i is set
    constexpr bool test(std::size_t i) const
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < m_size = {}", i, m_size);
        return block_ref_for(i) & block_mask_for(i);
    }

    /// @brief Check whether all the elements in the bit-vector are set.
    constexpr bool all() const
    {
        // Handle empty vectors with an exception if we're in a `BIT_DEBUG` scenario
        bit_debug_assert(!empty(), "Calling this method for an empty vector is likely an error!");

        // The "logical connective" for all() is AND with the identity TRUE, hence the return value for the empty set.
        if (empty()) return true;

        auto final_element = size() - 1;
        auto final_block = block_index_for(final_element);

        // Run through all the fully used blocks and check they are completely set.
        for (std::size_t i = 0; i < final_block; ++i)
            if (block(i) != ones_block()) return false;

        // Check that the used portion of the final block is also fully set.
        return block(final_block) == ones_block() >> bit_index_for(~final_element);
    }

    /// @brief Check whether any of the elements in the bit-vector are set.
    constexpr bool any() const
    {
        // Handle empty vectors with an exception if we're in a `BIT_DEBUG` scenario
        bit_debug_assert(!empty(), "Calling this method for an empty vector is likely an error!");

        // The "logical connective" for any() is OR with the identity FALSE, hence the return value for the empty set.
        for (auto b : m_blocks)
            if (b != 0) return true;

        return false;
    }

    /// @brief Check whether none of of the elements in the bit-vector are set.
    constexpr bool none() const
    {
        // Handle empty vectors with an exception if we're in a `BIT_DEBUG` scenario
        bit_debug_assert(!empty(), "Calling this method for an empty vector is likely an error!");
        return !any();
    }

    /// @brief Returns the number of set elements in the bit-vector.
    /// @note  Tested some other algorithms but @c std::popcount seems pretty good on our platforms!
    constexpr std::size_t count1() const
    {
        // NOTE: We have been careful to keep the excess bits in the last block all at 0.
        std::size_t sum = 0;
        for (auto b : m_blocks) sum += static_cast<std::size_t>(std::popcount(b));
        return sum;
    }

    /// @brief Returns the number of unset elements in the bit-vector.
    constexpr std::size_t count0() const { return m_size - count1(); }

    /// @brief Returns the number of set elements in the bit-vector. This is a synonym for @c count1()
    constexpr std::size_t count() const { return count1(); }

    /// @brief Returns the parity of the bit-vector (number of set bits mod 2).
    constexpr bool parity() const
    {
        // NOTE: We have been careful to keep the excess bits in the last block all at 0.
        Block sum = 0;
        for (auto b : m_blocks) sum ^= b;
        return std::popcount(sum) % 2;
    }

    /// @brief Returns the index of the first set element in the bit-vector or @c npos if there are none set.
    constexpr std::size_t first_set() const
    {
        for (std::size_t i = 0; i < block_count(); ++i)
            if (block(i) != 0) return i * bits_per_block + lsb(block(i));

        // No luck!
        return npos;
    }

    /// @brief Returns the index of the last set element in the bit-vector or @c npos if there are none set.
    constexpr std::size_t final_set() const
    {
        std::size_t i = block_count();
        while (i--)
            if (block(i) != 0) return i * bits_per_block + msb(block(i));

        // No luck!
        return npos;
    }

    /// @brief Returns the index of the next set bit after the argument or @c npos if there are no more set bits.
    constexpr std::size_t next_set(std::size_t pos) const
    {
        // Start our search at element p = (pos + 1)
        std::size_t p = pos + 1;

        // If we're off the end there is nothing to find (this also handles empty vectors where m_size = 0);
        if (p >= m_size) return npos;

        // Iterate through the blocks starting at the one containing element p.
        std::size_t i = block_index_for(p);

        // The first block is masked to exclude any bits below p.
        auto b = static_cast<Block>(ones_block() << bit_index_for(p));
        for (b &= block(i); b == 0; b = block(i))
            if (++i >= block_count()) return npos;
        return i * bits_per_block + lsb(b);
    }

    /// @brief Returns the index of the previous set bit before the argument or @c npos if there are none.
    constexpr std::size_t prev_set(std::size_t pos) const
    {
        // Edge case?
        if (empty() || pos == 0) return npos;

        // Silently fix a very large argument
        if (pos >= m_size) pos = m_size;

        // Start our search at element p = (pos - 1)
        std::size_t p = pos - 1;

        // Iterate backwards through the blocks starting at the one containing element p.
        std::size_t i = block_index_for(p);

        // The first block is masked to exclude bits above p.
        auto b = static_cast<Block>(ones_block() >> (bits_per_block - 1 - bit_index_for(p)));
        for (b &= block(i); b == 0 && i != 0; b = block(--i)) {
            // Empty loop
        }
        return i * bits_per_block + msb(b);
    }

    /// @brief  Returns a unit bit-vector with its 1 at the location of our final set bit.
    /// @param  trimmed If true (default) the returned bit-vector is as small as possible otherwise it is our size.
    /// @return Returns an empty bit-vector if we have no set bits.
    /// @see    See the documentation for std::bit_floor(std::unsigned_integral).
    constexpr vector unit_floor(bool trimmed = true) const
    {
        auto n = final_set();
        if (n == npos) return vector{};
        return unit(trimmed ? n + 1 : size(), n);
    }

    /// @brief  Returns a unit bit-vector with its 1 at the location of one slot past our final set bit.
    /// @param  trimmed If true (default) the returned bit-vector is as small as possible otherwise it is our size + 1.
    /// @return Returns an empty bit-vector if we have no set bits.
    /// @see    See the documentation for std::bit_ceil(std::unsigned_integral).
    constexpr vector unit_ceil(bool trimmed = true) const
    {
        auto n = final_set();
        n = (n == npos) ? 0 : n + 1;
        return unit(trimmed ? n + 1 : size() + 1, n);
    }

    /// @brief Set the element in the bit-vector at index @c i to 1.
    constexpr vector& set(std::size_t i)
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < `m_size` = {}", i, m_size);
        block_ref_for(i) |= block_mask_for(i);
        return *this;
    }

    /// @brief Set @c len elements in the bit-vector starting at @c first to 1.
    constexpr vector& set(std::size_t first, std::size_t len)
    {
        // Check index ranges if appropriate and also handle the edge case where len == 0
        bit_debug_assert(first < m_size, "first = {} but size() = {}", first, m_size);
        if (len == 0) return *this;

        std::size_t final = first + len - 1;
        bit_debug_assert(final < m_size, "len = {} => final = {} but size() = {}", len, final, m_size);

        // Indices of the first and final elements in our block store.
        std::size_t first_block = block_index_for(first);
        std::size_t final_block = block_index_for(final);

        // Useful masks
        auto  first_mask = static_cast<Block>(ones_block() << bit_index_for(first));
        Block final_mask = ones_block() >> (bits_per_block - bit_index_for(final) - 1);

        // Perhaps the range of interest lies in a single block?
        if (first_block == final_block) {
            block(first_block) |= (first_mask & final_mask);
            return *this;
        }

        // The range of interest lies over more than one block.
        block(first_block) |= first_mask;
        for (std::size_t i = first_block + 1; i < final_block; ++i) block(i) = ones_block();
        block(final_block) |= final_mask;

        return *this;
    }

    /// @brief Set all the elements in the bit-vector to 1.
    constexpr vector& set()
    {
        m_blocks.assign(block_count(), ones_block());
        return clean();
    }

    /// @brief Reset the element at index @c i in the bit-vector to 0.
    constexpr vector& reset(std::size_t i)
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < size() = {}", i, m_size);
        block_ref_for(i) &= Block(~block_mask_for(i));
        return *this;
    }

    /// @brief Reset @c len elements starting at @c first in the bit-vector to 0.
    constexpr vector& reset(std::size_t first, std::size_t len)
    {
        // Check index ranges if appropriate and also handle the edge case where len == 0
        bit_debug_assert(first < m_size, "first = {} but size() = {}", first, m_size);
        if (len == 0) return *this;
        std::size_t final = first + len - 1;
        bit_debug_assert(final < m_size, "len = {} => final = {} but size() = {}", len, final, m_size);

        // Indices of the first and final elements in our block store.
        std::size_t first_block = block_index_for(first);
        std::size_t final_block = block_index_for(final);

        // Useful masks
        auto  first_mask = static_cast<Block>(ones_block() << bit_index_for(first));
        Block final_mask = ones_block() >> (bits_per_block - bit_index_for(final) - 1);

        // Perhaps the range of interest lies in a single block?
        if (first_block == final_block) {
            block(first_block) &= Block(~(first_mask & final_mask));
            return *this;
        }

        // The range of interest lies over more than one block.
        block(first_block) &= Block(~first_mask);
        for (std::size_t i = first_block + 1; i < final_block; ++i) block(i) = 0;
        block(final_block) &= Block(~final_mask);

        return *this;
    }

    /// @brief Reset all the elements in the bit-vector to 0.
    constexpr vector& reset()
    {
        for (auto& block : m_blocks) block = 0;
        return *this;
    }

    /// @brief Flip element at @c i in the bit-vector.
    constexpr vector& flip(std::size_t i)
    {
        bit_debug_assert(i < m_size, "Index i = {} must be < m_size = {}", i, m_size);
        block_ref_for(i) ^= block_mask_for(i);
        return *this;
    }

    /// @brief Flip @c len elements starting at @c first in the bit-vector.
    constexpr vector& flip(std::size_t first, std::size_t len)
    {
        // Check index ranges if appropriate and also handle the edge case where len == 0
        bit_debug_assert(first < size(), "first = {} but size() = {}", first, m_size);
        if (len == 0) return *this;
        std::size_t final = first + len - 1;
        bit_debug_assert(final < size(), "len = {} => final = {} but size() = {}", len, final, m_size);

        // Locations of the first and last elements in our block store.
        std::size_t first_block = block_index_for(first);
        std::size_t final_block = block_index_for(final);

        // Useful masks
        auto  first_mask = static_cast<Block>(ones_block() << bit_index_for(first));
        Block final_mask = ones_block() >> (bits_per_block - bit_index_for(final) - 1);

        // Perhaps the range of interest lies in a single block?
        if (first_block == final_block) {
            block(first_block) ^= (first_mask & final_mask);
            return *this;
        }

        // The range of interest lies over more than one block
        block(first_block) ^= first_mask;
        for (std::size_t i = first_block + 1; i < final_block; ++i) block(i) = ~block(i);
        block(final_block) ^= final_mask;

        return *this;
    }

    /// @brief Flip all the elements in the bit-vector.
    constexpr vector& flip()
    {
        for (auto& block : m_blocks) block = ~block;
        return clean();
    }

    /// @brief Set the element @c i in the bit-vector to 1 if `f(i) != 0` otherwise set it to 0.
    /// @param f is function that we will call as `f(i)` for each index in the bit-vector.
    constexpr vector& set_if(std::invocable<std::size_t> auto f)
    {
        reset();
        for (std::size_t i = 0; i < size(); ++i)
            if (f(i) != 0) set(i);
        return *this;
    }

    /// @brief Flip element @c i in the bit-vector if `f(i) != 0` otherwise leave it alone.
    /// @param f is function that we will call as `f(i)` for each index in the bit-vector.
    constexpr vector& flip_if(std::invocable<std::size_t> auto f)
    {
        for (std::size_t i = 0; i < size(); ++i)
            if (f(i) != 0) flip(i);
        return *this;
    }

    /// @brief AND this bit-vector with another of equal size.
    constexpr vector& operator&=(const vector& rhs)
    {
        bit_debug_assert(size() == rhs.size(), "Sizes don't match {} != {}", size(), rhs.size());
        for (std::size_t i = 0; i < block_count(); ++i) block(i) &= rhs.block(i);
        return *this;
    }

    /// @brief OR this bit-vector with another of equal size.
    constexpr vector& operator|=(const vector& rhs)
    {
        bit_debug_assert(size() == rhs.size(), "Sizes don't match {} != {}", size(), rhs.size());
        for (std::size_t i = 0; i < block_count(); ++i) block(i) |= rhs.block(i);
        return *this;
    }

    /// @brief XOR this bit-vector with another of equal size.
    constexpr vector& operator^=(const vector& rhs)
    {
        bit_debug_assert(size() == rhs.size(), "Sizes don't match {} != {}", size(), rhs.size());
        for (std::size_t i = 0; i < block_count(); ++i) block(i) ^= rhs.block(i);
        return *this;
    }

    /// @brief Get back a copy of this bit-vector with all the bits flipped.
    constexpr vector operator~() const
    {
        vector retval{*this};
        retval.flip();
        return retval;
    }

    /// @brief "add" another equal sized bit-vector to this one (in GF(2) addition == XOR)
    constexpr vector& operator+=(const vector& rhs) { return operator^=(rhs); }

    /// @brief "subtract" another equal sized bit-vector from this one (in GF(2) subtraction == XOR)
    constexpr vector& operator-=(const vector& rhs) { return operator^=(rhs); }

    /// @brief Elementwise "multiplication" of another equal sized bit-vector with this one (in GF(2) this is AND).
    constexpr vector& operator*=(const vector& rhs) { return operator&=(rhs); }

    /// @brief Returns the dot-product of this bit-vector with another equal sized one (uses `&` for * and `^` for +).
    constexpr bool dot(const vector& rhs) const
    {
        // DEBUG builds will check that the two bit-vectors are indeed equally sized otherwise we assume it is true.
        bit_debug_assert(size() == rhs.size(), "bit-vector sizes don't match {} != {}", size(), rhs.size());
        Block sum = 0;
        for (std::size_t k = 0; k < block_count(); ++k) sum ^= static_cast<Block>(block(k) & rhs.block(k));
        return std::popcount(sum) % 2;
    }

    /// @brief  Shift this bit-vector @c p elements to the left.
    /// @return This works in vector-order so if v = [v0,v1,v2,v3] then v <<= 1 is [v1,v2,v3,0].
    /// @note   Left shift in vector-order is same as right shift for bit-order!
    constexpr vector& operator<<=(std::size_t p)
    {
        // Edge cases?
        if (p == 0 || empty()) return *this;

        // Perhaps we have just shifted in all zeros?
        if (p >= m_size) {
            reset();
            return *this;
        }

        // For larger values of p we can efficiently start with whole block shifts:
        std::size_t block_shift = p / bits_per_block;
        std::size_t block_end = block_count() - block_shift;

        // Do any whole block shifts first (pushing in whole blocks of zeros to fill the empty slots)
        if (block_shift > 0) {

            // Start by first doing whole block shifts.
            for (std::size_t i = 0; i < block_end; ++i) block(i) = block(i + block_shift);

            // Fill the high order blocks with zeros.
            for (std::size_t i = block_end; i < block_count(); ++i) block(i) = 0;

            // That handles a lot of `p` but there may be some shifts left to do.
            p %= bits_per_block;
        }

        // Perhaps those block shifts didn't do all the work?
        if (p != 0) {

            // Work on the "shift by less than a blocks's worth of bits" piece.
            std::size_t q = bits_per_block - p;
            for (std::size_t i = 0; i < block_end - 1; ++i) {
                auto hi = static_cast<Block>(block(i + 1) << q);
                auto lo = static_cast<Block>(block(i) >> p);
                block(i) = hi | lo;
            }

            // And shift the 'end' block.
            block(block_end - 1) >>= p;
        }

        // Can only have shifted in 0's so no junk to clean up here (contrast to operator>>== below)
        return *this;
    }

    /// @brief  Shift this bit-vector @c p elements to the right.
    /// @return This works in vector-order so if v = [v0,v1,v2,v3] then v >>= 1 is [0,v0,v1,v2].
    /// @note   Right shift in vector-order is same as left shift for bit-order!
    constexpr vector& operator>>=(std::size_t p)
    {
        // Edge cases?
        if (p == 0 || empty()) return *this;

        // Perhaps we have just shifted in all zeros?
        if (p >= m_size) {
            reset();
            return *this;
        }

        // For larger values of `p` we can efficiently start with whole block shifts:
        std::size_t block_shift = p / bits_per_block;
        if (block_shift > 0) {

            // Start by first doing whole block shifts starting from the end.
            for (std::size_t i = block_count() - 1; i >= block_shift; --i) block(i) = block(i - block_shift);

            // Push zeros into the low order blocks.
            for (std::size_t i = 0; i < block_shift; ++i) block(i) = 0;

            // That handles a lot of `p` but there may be some shifts left to do.
            p %= bits_per_block;
        }

        // Perhaps those block shifts didn't do all the work?
        if (p != 0) {

            // Work on the "shift by less than a blocks's worth of bits" piece.
            std::size_t q = bits_per_block - p;
            for (std::size_t i = block_count() - 1; i > block_shift; --i) {
                auto hi = static_cast<Block>(block(i) << p);
                auto lo = static_cast<Block>(block(i - 1) >> q);
                block(i) = hi | lo;
            }

            // And shift that 'first' block.
            block(block_shift) <<= p;
        }

        // We may have added some junk/redundant set bits that need to be zapped.
        return clean();
    }

    /// @brief Get back a bit-vector which is this one left shifted by @c p elements.
    /// @note  This works in vector-order so [v0,v1,v2,v3] << 1 returns [v1,v2,v3,0].
    /// @note  Left shift in vector-order is same as right shift for bit-order!
    constexpr vector operator<<(std::size_t p) const
    {
        vector retval{*this};
        retval <<= p;
        return retval;
    }

    /// @brief Get back a bit-vector which is this one right shifted by @c p elements.
    /// @note  This works in vector-order so [v0,v1,v2,v3] >> 1 returns [0,v0,v1,v2].
    /// @note  Right shift in vector-order is same as left shift for bit-order!
    constexpr vector operator>>(std::size_t p) const
    {
        vector retval{*this};
        retval >>= p;
        return retval;
    }

    /// @brief Starting at bit @c i0 replace our values with those of a passed in replacement bit-vector.
    /// @param i0 is the location where we start putting the new sub-vector in place.
    /// @param with The sub-vector we are putting in place -- it must fit in the existing structure!
    constexpr vector& replace(std::size_t i0, const vector& with)
    {
        // Edge case?
        std::size_t ws = with.size();
        if (ws == 0) return *this;

        // Do a couple of optional sanity checks
        bit_debug_assert(i0 < size(), "i0 = {} but size() = {}", i0, size());
        bit_debug_assert(i0 + ws - 1 < size(), "i0 = {}, with.size() = {}, but size = {}", i0, ws, size());

        // TODO: Replace this loop with something that works on blocks at a time!
        for (std::size_t i = 0; i < ws; ++i) operator[](i0 + i) = with[i];

        return *this;
    }

    /// @brief Starting at bit 0 replace our values with those of a passed in replacement bit-vector
    /// @param with The sub-vector we are putting in place -- it must fit in the existing structure!
    constexpr vector& replace(const vector& with) { return replace(0, with); }

    /// @brief  Get a binary-string representation for this bit-vector using the given characters for set and unset.
    /// @param  pre  A prefix for the string such as "[" -- defaults to the empty string.
    /// @param  post A postfix for the string such as "]" -- defaults to the empty string.
    /// @param  sep  A separator between the elements such as ", " -- defaults to the empty string.
    /// @param  off  The character to use for unset elements -- defaults to '0'.
    /// @param  on   The character to use for set elements -- defaults to '1'.
    /// @return By default something like "10001110010" but it could be "[1,0,0,0,1,1,1,0,0,1,0]"
    std::string to_string(std::string_view pre = "", std::string_view post = "", std::string_view sep = "",
                          char off = '0', char on = '1') const
    {
        // Edge case?
        auto n = size();
        if (n == 0) return std::string{pre} + std::string{post};

        // Initialize the return string to at least roughly the correct size
        std::string retval;
        retval.reserve(pre.size() + n * (sep.size() + 1) + post.size());

        // Construct the string element by element
        retval += pre;
        for (std::size_t i = 0; i < n - 1; ++i) {
            retval.push_back(test(i) ? on : off);
            retval += sep;
        }
        retval.push_back(test(n - 1) ? on : off);
        retval += post;

        return retval;
    }

    /// @brief  Get a binary-string representation for this bit-vector in bit-order.
    /// @param  off The character to use for unset elements--defaults to '0'.
    /// @param  on The character to use for set elements--defaults to '1'.
    /// @return If v = {1,0,0,1,0} then this returns 01001 (i.e. high order bit on the LEFT).
    std::string to_bit_order(char off = '0', char on = '1') const
    {
        auto retval = to_string("", "", "", off, on);
        std::reverse(std::begin(retval), std::end(retval));
        return retval;
    }

    /// @brief  Get a binary-string representation for this bit-vector using the given characters for set and unset.
    /// @param  off  The character to use for unset elements -- defaults to '0'.
    /// @param  on   The character to use for set elements -- defaults to '1'.
    /// @return By default something like "[1 0 0 0 1 1 1 0 0 1 0]"
    std::string to_pretty_string(char off = '0', char on = '1') const { return to_string("[", "]", " ", off, on); }

    /// @brief  Get a hex-string representation for this bit-vector
    /// @note   If the size of the bit-vector is not a multiple of 4 there will be a suffix (one of "_2", "_4", or
    /// "_8").
    ///         The suffix gives the base of the last character in the string.
    ///         Thus the string "A0F1_4" says that the `A`, `0`, and `F` characters are hex (base 16) but that last `1`
    ///         is actually base 4 so should be read as binary '01' as opposed to '0001' which is what it would be if
    ///         we also interpreted the 1 as base 16 (i.e. no suffix).
    /// @return For example, 0x03_8 for v = 0000011 (i.e. 0x for hex, 0000 for hex 0, 011 for 3 base 8).
    /// @return Empty bit-vectors come back as the empty string (i.e. no '0x' prefix).
    std::string to_hex() const
    {
        // Table of hex characters.
        static constexpr std::array hex_char = {'0', '1', '2', '3', '4', '5', '6', '7',
                                                '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
        // Edge case?
        auto n = size();
        if (n == 0) return std::string{};

        // The number of characters in the output string.
        std::size_t digits = (n + 3) / 4;

        // That final character might not be hex -- if not it will need a '_base' suffix.
        auto log2_final_base = n % 4;

        // Return value preallocates the correct number of characters allowing for a "0x" prefix & any '_base'
        std::string retval;
        retval.reserve(2 + digits + (log2_final_base != 0 ? 2 : 0));

        // We always start with a "0x" prefix to eliminate potential mixups between hex and binary formatting.
        retval = "0x";

        Block const* p = m_blocks.data();
        Block        b = 1;

        // Emit digits, reloading buffer word b as necessary (requires that bits_per_block % 4 == 0).
        constexpr Block mask = Block{1} << (bits_per_block - 4);
        while (digits--) {
            Block digit = b % 16;
            b >>= 4;
            if (b == 0) {
                b = *p++;
                digit = b % 16;
                b >>= 4;
                b |= mask;
            }
            retval.push_back(hex_char[digit]);
        }

        // Will need a suffix if the size is not an even multiple of 4
        if (char i = static_cast<char>(n % 4); i != 0) {
            retval.push_back('_');
            retval.push_back('0' + static_cast<char>(1 << i));
        }

        return retval;
    }

    /// @brief Call @c f(pos) over the set bits in the bit-vector in increasing order.
    /// @param f is a function that takes the the position of the set bit as its argument.
    constexpr void if_set_call(std::invocable<std::size_t> auto f) const
    {
        for (std::size_t pos = first_set(); pos != npos; pos = next_set(pos)) f(pos);
    }

    /// @brief Call @c f(pos) over the set bits in the bit-vector in decreasing order.
    /// @param f is a function that takes the the position of the set bit as its argument.
    constexpr void reverse_if_set_call(std::invocable<std::size_t> auto f) const
    {
        for (std::size_t pos = final_set(); pos != npos; pos = prev_set(pos)) f(pos);
    }

    /// @brief Returns a @c std::vector that lists in order all the set indices in this bit-vector.
    std::vector<std::size_t> set_indices() const
    {
        // Create a vector of indices that is the correct size.
        std::vector<std::size_t> retval(count());
        for (std::size_t i = 0, pos = first_set(); pos != npos; i++, pos = next_set(pos)) retval[i] = pos;
        return retval;
    }

    /// @brief Returns a @c std::vector that lists in order all the unset indices in this bit-vector.
    std::vector<std::size_t> unset_indices() const
    {
        // Bit of a cheat for now at least ...
        auto tmp = *this;
        return tmp.flip().set_indices();
    }

    /// @brief Check for equality between two bit-vectors.
    constexpr bool friend operator==(const vector& lhs, const vector& rhs)
    {
        if (&lhs != &rhs) {
            if (lhs.m_size != rhs.m_size) return false;
            for (std::size_t i = 0; i < lhs.block_count(); ++i)
                if (lhs.block(i) != rhs.block(i)) return false;
        }
        return true;
    }

    /// @brief A little debug utility to dump a whole bunch of descriptive data about a bit-vector to a stream.
    /// @note  Don't depend on this format staying constant!
    constexpr void description(std::ostream& s, const std::string& header = "", const std::string& footer = "") const
    {
        if (!header.empty()) s << header << "::\n";
        s << "bit-vector:           " << to_string() << "\n";
        s << "as hex-string:        " << to_hex() << "\n";
        s << "number of bits:       " << size() << "\n";
        s << "number of set bits:   " << count() << "\n";
        s << "bit capacity:         " << capacity() << "\n";
        s << "unused capacity:      " << unused() << "\n";

        if (!empty()) {
            s << "bits-per-block:       " << std::numeric_limits<Block>::digits << "\n";
            s << "blocks used:          " << block_index_for(m_size - 1) + 1 << "\n";
            s << "block store size:     " << block_count() << "\n";
            s << "block store capacity: " << m_blocks.capacity() << "\n";
        }
        s << footer;
    }

    /// @brief A little debug utility to dump a whole bunch of descriptive data about a bit-vector to @c std::cout
    constexpr void description(const std::string& header = "", const std::string& footer = "") const
    {
        description(std::cout, header, footer);
    }

    /// @brief Read-only access to the memory allocator for this bit-vector.
    constexpr Allocator allocator() const { return m_blocks.allocator(); }

    // ----------------------------------------------------------------------------------------------------------------
    // BLOCK METHODS:
    // These are methods that allow access to the raw blocks a bit-vector uses to store its elements.
    // For "advanced" use only -- writing into the block store can leave the bit-vector in an undefined state!
    // ----------------------------------------------------------------------------------------------------------------

    /// @brief The unsigned integer blocks of bit-vector elements are stored in a container of the following type.
    using block_store_type = std::vector<Block, Allocator>;

    /// @brief  A constructor that creates a bit-vector by copying or moving a pre-filled container of blocks.
    /// @tparam Function works with either an r-value or const l-value reference to a @c block_store_type.
    /// @param  n        Size of vector to create. Note that @c blocks_needed(n) must match @c blocks.size().
    /// @param  blocks   The container of blocks we either copy or move over (use @c std::move(blocks) to *move* it).
    /// @param  is_clean If false (the default) we make sure to zap any junk bits in the final block to zeros.
    /// @note   If you set that last parameter to true, be sure that any extra bits are really zero!
    template<typename T>
        requires std::same_as<std::remove_cvref_t<T>, block_store_type>
    explicit constexpr vector(std::size_t n, T&& blocks, bool is_clean = false) : vector()
    {
        // Check that the size of the bit-vector matches with the number of passed in blocks.
        std::size_t needed = blocks_needed(n);
        bit_always_assert(needed == blocks.size(), "needed = {}, blocks.size() = {}", needed, blocks.size());

        // Size is OK & we then copy or move the passed in block store.
        m_size = n;
        m_blocks = std::forward<T>(blocks);

        // Clean out any unused bits in the last block unless otherwise told.
        if (!is_clean) clean();
    }

    /// @brief The number of bit-vector elements/elements that can be stored in each block.
    static constexpr std::size_t bits_per_block = std::numeric_limits<Block>::digits;

    /// @brief Returns the number of blocks needed to hold  a bit-vector with @c n elements.
    /// @note  The block store for an n-element bit-vector will be exactly this size (the store capacity may differ).
    static constexpr std::size_t blocks_needed(std::size_t n) { return (bits_per_block + n - 1) / bits_per_block; }

    /// @brief Returns the index of the block that holds element @c i of the bit-vector.
    static constexpr std::size_t block_index_for(std::size_t i) { return i / bits_per_block; }

    /// @brief Returns the bit position of element @c i of the bit-vector inside the block that holds it.
    static constexpr std::size_t bit_index_for(std::size_t i) { return i % bits_per_block; }

    /// @brief Returns the number of blocks in the block store.
    constexpr std::size_t block_count() const { return m_blocks.size(); }

    /// @brief Read-only access to block @c b in the store (index is not range checked).
    constexpr Block block(std::size_t b) const { return m_blocks[b]; }

    /// @brief   Read-write access to block @c b in the store (index is not range checked).
    /// @warning Writing into the store can leave the vector in a bad state. Use at your own risk.
    constexpr Block& block(std::size_t b) { return m_blocks[b]; }

    /// @brief Read-only access to the full block store.
    constexpr const block_store_type& blocks() const { return m_blocks; }

    /// @brief   Read-write access to the full block store.
    /// @warning Writing into the store can leave the vector in a bad state. Use at your own risk.
    constexpr block_store_type& blocks() { return m_blocks; }

    /// @brief Sets any excess/junk bits in the *last* occupied block to 0.
    /// @note  This is useful if for some reason you write directly into the block store.
    constexpr vector& clean()
    {
        // NOTE: This works fine even if size() == 0
        std::size_t shift = m_size % bits_per_block;
        if (shift != 0) block(block_count() - 1) &= Block(~(ones_block() << shift));
        return *this;
    }

private:
    std::size_t      m_size = 0; // The number of elements in the bit-vector (default is none).
    block_store_type m_blocks;   // The elements are packed into a container of blocks.

    /// @brief A Block with all bits set to 1.
    static constexpr Block ones_block() { return std::numeric_limits<Block>::max(); }

    /// @brief A block that has either the bit-pattern 010101... or 101010...
    static constexpr Block checkered_block(bool first_element_set = true)
    {
        /// NOTE: 0xFFFF.../3 is 0x5555... which has the bit pattern 101010... and shifting that by 1 give 0101010...
        constexpr Block starts_with_1 = ones_block() / 3;
        constexpr Block starts_with_0 = starts_with_1 << 1;
        return first_element_set ? starts_with_1 : starts_with_0;
    }

    /// @brief Returns a writable reference to the block that holds element @c i of the bit-vector.
    constexpr Block& block_ref_for(std::size_t i) { return block(block_index_for(i)); }

    /// @brief Returns a read-only reference to the block that holds element @c i of the bit-vector.
    constexpr Block block_ref_for(std::size_t i) const { return block(block_index_for(i)); }

    /// @brief Returns a bit-mask that isolates a single target bit element in the block that contains it.
    static constexpr Block block_mask_for(std::size_t i) { return Block(Block{1} << bit_index_for(i)); }

    /// @brief Returns the index of the word where the next pushed bit will reside.
    constexpr std::size_t next_block_index() const { return block_index_for(m_size); }

    /// @brief Returns the location where the next pushed bit will be found inside its containing word.
    constexpr std::size_t next_bit_index() const { return bit_index_for(m_size); }

    /// @brief  Returns the index for the least significant set bit in the argument or @c npos if none set.
    /// @tparam Src The (deduced) type of the argument which must be an unsigned integral of some sort.
    template<std::unsigned_integral Src>
    std::size_t static constexpr lsb(Src src)
    {
        if (src == 0) return npos;
        return static_cast<std::size_t>(std::countr_zero(src));
    }

    /// @brief  Returns the index for the most significant set bit in the argument or @c npos if none set.
    /// @tparam Src The (deduced) type of the argument which must be an unsigned integral of some sort.
    /// @note   This version depends on @c npos being std::size_t(-1).
    template<std::unsigned_integral Src>
    std::size_t static constexpr msb(Src src)
    {
        return static_cast<std::size_t>(std::bit_width(src) - 1);
    }

    /// @brief  Factory method to parse a hex string string (e.g. "F27AD3", "0xF2AD3", "0xF2AD3_4") into a bit-vector.
    /// @param  src The source string which optionally can be prefixed by a "0x" or "0X".
    ///         There can also be a suffix "_b" where b is one of 2, 4, or 8.
    ///         If present, the suffix is the alternate base for the last character.
    ///         So "0x1" will get parsed as 0001 (default base is 16), "0x1_8" as 001, "0x1_4" as 01 and "0x1_1" as 1.
    /// @note   Any hex char parses as 4 bits so w/o a suffix the bit-vector size will always be a multiple of 4.
    ///         Using the suffix allows us encode arbitrary sized bit-vectors.
    /// @return Returns a @c std::nullopt if the parsing fails.
    static std::optional<vector> from_hex_string(std::string_view s)
    {
        // String might start with a redundant "0X"/"0x" which we remove.
        if (s.length() >= 2 && s[0] == '0' && tolower(s[1]) == 'x') s.remove_prefix(2);

        // Anything left?
        if (s.empty()) return vector{};

        // The default base for the last character is 16 just like all the others -- no padding bits are used.
        std::size_t padding_bits = 0;

        // Perhaps the string has a suffix (one of "_2", "_4", or "_8") -- the final 4 hex bits are padded out.
        if (s.length() > 2 && s[s.length() - 2] == '_') {
            switch (s.back()) {
                case '2': padding_bits = 3; break;
                case '4': padding_bits = 2; break;
                case '8': padding_bits = 1; break;
                default: return std::nullopt;
            }

            // We've captured the base of the last character so can now remove the suffix
            s.remove_suffix(2);
        }

        // Reserve space for our bit-vector -- if there is a suffix we'll chop it down to size at the end.
        vector retval;
        retval.reserve(4 * s.length());

        // Table of hex digit bit patterns as std::bitset<4>'s
        static constexpr std::bitset<4> hex_bs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        // Convert the hex characters to 0..15 and append.
        // NOTE: By now we know that s.length() > 0, so x will really be initialized inside the loop.
        unsigned char x = 0;
        for (char c : s) {
            x = static_cast<unsigned char>(tolower(c)) - '0';
            if (x > 9) {
                x = x + '0' - 'a';
                if (x > 5) return std::nullopt;
                x += 10;
            }
            retval.append(hex_bs[x]);
        }

        // Finally, handle any padding bits from the last character.
        if (x >= 16 >> padding_bits) return std::nullopt;
        return retval.resize(retval.size() - padding_bits);
    }

public:
    /// @brief  Riffle a block into two output blocks containing the bits from the src interleaved with zeros.
    /// @return With 8-bit blocks and src = `abcdefgh`, on return `lo = a0b0c0d0` and 'hi = e0f0g0h0`.
    static constexpr void riffle(Block src, Block& lo, Block& hi)
    {
        constexpr auto half_block = bits_per_block / 2;
        constexpr auto all_set = ones_block();
        constexpr auto lo_mask = all_set >> half_block;

        // Split the src into lo and hi halves.
        lo = src & lo_mask;
        hi = src >> half_block;

        // Some magic to interleave the respective halves with zeros.
        const Block one = 1;
        for (auto i = bits_per_block / 4; i > 0; i /= 2) {
            Block div = Block(one << i) | one;
            Block mask = all_set / div;
            lo = (lo ^ (lo << i)) & mask;
            hi = (hi ^ (hi << i)) & mask;
        }
    }

    /// @brief All the vector instantiations are friends to each other no matter what the block type is
    template<std::unsigned_integral BlockType, typename AllocType>
    friend class vector;
};

// --------------------------------------------------------------------------------------------------------------------
// NON-MEMBER FUNCTIONS ...
// --------------------------------------------------------------------------------------------------------------------

/// @brief  Overwrite a bit-vector with the bits from an unsigned source word.
/// @tparam Src The type of the word for the source bits (could e.g. be unsigned or unsigned char etc.)
/// @param  dst This is the destination bit-vector which is cleared and then filled from the src.
/// @note   The @c Block type of the bit-vector need not match the @c Src type.
template<std::unsigned_integral Src, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(Src src, vector<Block, Allocator>& dst)
{
    dst.clear();
    dst.append(src);
}

/// @brief  Overwrite a bit-vector with the bits from an iterated collection of unsigned source words.
/// @tparam Iter The @c value_type of this const iterator should be some unsigned integer type.
/// @param  src_b E.g. might be @c std::cbegin(collection).
/// @param  src_e E.g. might be @c std::cend(collection).
/// @param  dst This is the destination bit-vector which is cleared and then filled from the collection.
/// @note   The @c Block type of the bit-vector need not match the @c value_type of the iterator.
template<typename Iter, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(Iter src_b, Iter src_e, vector<Block, Allocator>& dst)
{
    dst.clear();
    dst.append(src_b, src_e);
}

/// @brief  Overwrite a bit-vector with the bits from an initializer list of source words.
/// @tparam Src The type of the words for the source bits (could e.g. be unsigned or unsigned char etc.)
/// @param  dst This is the destination bit-vector which is cleared and then filled from the initializer list.
/// @note   The @c Block type of the bit-vector need not match the @c Src type.
template<std::unsigned_integral Src, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(std::initializer_list<Src> src, vector<Block, Allocator>& dst)
{
    dst.clear();
    dst.append(src);
}

/// @brief  Overwrite a bit-vector with the bits from a @c std::vector of source words.
/// @tparam Src The type of the words for the source bits (could e.g. be unsigned or unsigned char etc.)
/// @param  dst This is the destination bit-vector which is cleared and then filled from the std::vector<Src>.
/// @note   The @c Block type of the bit-vector need not match the @c Src type.
template<std::unsigned_integral Src, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(const std::vector<Src>& src, vector<Block, Allocator>& dst)
{
    dst.clear();
    dst.append(src);
}

/// @brief  Overwrite a bit-vector with the bits from a @c std::array of source words.
/// @tparam Src The type of the words for the source bits (could e.g. be unsigned or unsigned char etc.)
/// @tparam N The fixed size of the source array.
/// @param  dst This is the destination bit-vector which is cleared and then filled from the std::vector<Src>.
/// @note   The @c Block type of the bit-vector need not match the @c Src type.
template<std::unsigned_integral Src, std::size_t N, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(const std::array<Src, N>& src, vector<Block, Allocator>& dst)
{
    dst.clear();
    dst.append(src);
}

/// @brief  Overwrite a bit-vector with the bits from a @c std::bitset.
/// @tparam N The size of the `std::bitset` which should be deduced in any case.
/// @param  dst This is the destination bit-vector which is cleared and then filled from the @c std::bitset<N>.
template<std::size_t N, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(const std::bitset<N>& src, vector<Block, Allocator>& dst)
{
    dst.clear();
    dst.append(src);
}

/// @brief  Overwrite a single unsigned integer word from a bit-vector (at least as much of it as fits in the word)
/// @tparam Dst The (deduced) unsigned destination word-type.
/// @param  src The src bit-vector. Empty src bit-vectors fill the destination with Dst(0).
/// @param  dst E.g. if `dst` is a `uint32_t` it gets filled as far as is possible from the start of the src bit-vector.
/// @note   The @c Dst type need not match the @c Block type.
template<std::unsigned_integral Block, typename Allocator, std::unsigned_integral Dst>
constexpr void
copy(const vector<Block, Allocator>& src, Dst& dst)
{
    // Initialize the destination words (we also return 0 if the src bit-vector is empty).
    dst = 0;
    if (src.empty()) return;

    // How many bits are there per src block and destination types?
    constexpr std::size_t bits_per_block = std::numeric_limits<Block>::digits;
    constexpr std::size_t bits_per_dst = std::numeric_limits<Dst>::digits;

    // Grab a read-only reference to the src blocks
    auto blocks = src.blocks();

    if constexpr (bits_per_dst == bits_per_block) {
        // This is the easiest case -- can just pop the entire first block into the destination and call it day.
        dst = static_cast<Dst>(blocks[0]);
        return;
    }
    else if constexpr (bits_per_dst < bits_per_block) {
        // In this case, destination can't handle all the bits in our first block so just take as many as possible.
        constexpr Block mask = std::numeric_limits<Dst>::max();
        dst = static_cast<Dst>(blocks[0] & mask);
        return;
    }
    else {
        // Destination is a longer word and can take more than one block's worth of bits if they are available.
        // Note the long words should be an even multiple of the short ones or something very odd has happened!
        static_assert(bits_per_dst % bits_per_block == 0, "Cannot pack blocks evenly into the destination word!");

        // How many src blocks will it take to fill dst? Of course that many may not be available
        std::size_t n = std::min(bits_per_dst / bits_per_block, blocks.size());
        std::size_t shift = 0;
        for (std::size_t i = 0; i < n; ++i, shift += bits_per_block) {
            auto src_word = static_cast<Dst>(blocks[i]);
            dst |= static_cast<Dst>(src_word << shift);
        }
        return;
    }
}

/// @brief  Overwrite an iteration of unsigned integer words from a bit-vector (at least as much of it as fits)
/// @tparam Iter The @c value_type associated with the non-const iterator should be some unsigned integer type.
/// @param  src The src bit-vector. Empty src bit-vectors fill the destination with Dst(0)'s.
/// @param  dst_b E.g. might be std::begin(collection).
/// @param  dst_e E.g. might be std::end(collection).
/// @note   The iterator's @c value_type need not match the @c Block type.
template<std::unsigned_integral Block, typename Allocator, typename Iter>
constexpr void
copy(const vector<Block, Allocator>& src, Iter dst_b, Iter dst_e)
{
    // Edge case?
    if (dst_b == dst_e) return;

    // Initialize the entire collection to all zeros
    for (auto b = dst_b; b != dst_e; ++b) *b = 0;

    // How many bits are there per src block and destination types?
    using dst_type = typename std::iterator_traits<Iter>::value_type;
    constexpr std::size_t bits_per_block = std::numeric_limits<Block>::digits;
    constexpr std::size_t bits_per_dst = std::numeric_limits<dst_type>::digits;

    // Grab a read-only reference to the src blocks
    auto blocks = src.blocks();

    if constexpr (bits_per_dst == bits_per_block) {
        // This is the easiest case -- can just pop the blocks into the destination container with assignment
        // We stop if we run out of blocks or run out of space in the container
        for (std::size_t i = 0; i < blocks.size() && dst_b != dst_e; ++i, ++dst_b)
            *dst_b = static_cast<dst_type>(blocks[i]);
        return;
    }
    else if constexpr (bits_per_dst < bits_per_block) {
        // Each block should fit into some integral number of destination words
        static_assert(bits_per_block % bits_per_dst == 0,
                      "Source blocks will not divide evenly into an array of destination words!");

        // How many destination words fit in one block?
        std::size_t dst_per_block = bits_per_block / bits_per_dst;

        // Create a Block word that masks out all but the rightmost bits_per_dst piece0
        constexpr Block mask = std::numeric_limits<dst_type>::max();

        // Copy each block into the appropriate number of words in the container stopping when needed.
        for (std::size_t i = 0; i < blocks.size() && dst_b != dst_e; ++i) {
            auto src_word = blocks[i];
            for (std::size_t j = 0; j < dst_per_block && dst_b != dst_e; ++j, ++dst_b, src_word >>= bits_per_dst)
                *dst_b = static_cast<dst_type>(src_word & mask);
        }
    }
    else {

        // Pushing the smaller blocks into a number of larger destination words (sizes need to be clean multiples)
        static_assert(bits_per_dst % bits_per_block == 0, "Container words are not each an even number of blocks!");

        // How many Blocks fit into one destination word? Will be an clean integer number.
        std::size_t blocks_per_dst = bits_per_dst / bits_per_block;

        // Fill container with our blocks. More than one block will fit in each container word.
        std::size_t i = 0;
        for (; dst_b != dst_e && i < blocks.size(); ++dst_b) {
            for (std::size_t j = 0; j < blocks_per_dst && i < blocks.size(); ++j, ++i) {
                dst_type    src_word = blocks[i];
                std::size_t shift = j * bits_per_block;
                *dst_b |= dst_type(src_word << shift);
            }
        }
    }
}

/// @brief  Overwrite a @c std::array of unsigned integer words from a bit-vector (at least as much of it as fits)
/// @tparam Dst The array holds words of this unsigned type.
/// @tparam N The array is this fixed size
/// @param  src The src bit-vector. Empty src bit-vectors fill the destination with @c Dst(0)'s.
/// @param  dst The destination array is first initialized to all zero then filled as far as possible from src.
/// @note   The @c Dst type need not match the @c Block type.
template<std::unsigned_integral Block, typename Allocator, std::unsigned_integral Dst, std::size_t N>
constexpr void
copy(const vector<Block, Allocator>& src, std::array<Dst, N>& dst)
{
    copy(src, std::begin(dst), std::end(dst));
}

/// @brief  Overwrite a `std::vector<Dst>`using the bits from a bit-vector  (at least as much of it as fits).
/// @tparam Dst The vector holds words of this unsigned type.
/// @param  src The src bit-vector. Empty src bit-vectors fill the destination with @c Dst(0)'s.
/// @param  dst The destination vector is first initialized to all zero then filled as far as possible from src.
/// @note   The @c Dst type need not match the@c Block type.
template<std::unsigned_integral Block, typename Allocator, std::unsigned_integral Dst>
constexpr void
copy(const vector<Block, Allocator>& src, std::vector<Dst>& dst)
{
    copy(src, std::begin(dst), std::end(dst));
}

/// @brief  Overwrite a @c std::bitset with bits from a bit-vector  (at least as much of it as fits).
/// @tparam N The size of the @c std::bitset which should be deduced in any case.
/// @param  dst This is the destination @c std::bitset. first set to all zeros and then filled from the bit-vector.
template<std::size_t N, std::unsigned_integral Block, typename Allocator>
constexpr void
copy(const vector<Block, Allocator>& src, std::bitset<N>& dst)
{
    dst.reset();
    std::size_t n = std::min(src.size(), dst.size());
    for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
}

/// @brief  Resize and then fill a @c std::vector<Dst> using the bits from a bit-vector.
/// @tparam Dst The vector holds words of this unsigned type.
/// @param  src The src bit-vector. Empty src bit-vectors give back an empty @c std::vector<Dst>
/// @param  dst The destination vector is first resized to accommodate all the bit-vector then filled.
/// @note   The @c Dst type need not match the @c Block type.
template<std::unsigned_integral Block, typename Allocator, std::unsigned_integral Dst>
constexpr void
copy_all(const vector<Block, Allocator>& src, std::vector<Dst>& dst)
{
    // Correctly size the output vector (might be to 0) to hold ALL the source bits.
    std::size_t src_bits = src.size();
    std::size_t bits_per_dst = std::numeric_limits<Dst>::digits;
    std::size_t dst_words = (bits_per_dst + src_bits - 1) / bits_per_dst;
    dst.resize(dst_words);
    copy(src, std::begin(dst), std::end(dst));
}

/// @brief Element by element AND of two equal sized bit-vectors.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
operator&(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    bit_debug_assert(lhs.size() == rhs.size(), "sizes don't match {} != {}", lhs.size(), rhs.size());
    vector<Block, Allocator> retval{lhs};
    retval &= rhs;
    return retval;
}

/// @brief Element by element XOR of two equal sized bit-vectors.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
operator^(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    bit_debug_assert(lhs.size() == rhs.size(), "sizes don't match {} != {}", lhs.size(), rhs.size());
    vector<Block, Allocator> retval{lhs};
    retval ^= rhs;
    return retval;
}

/// @brief Element by element OR of two equal sized bit-vectors.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
operator|(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    bit_debug_assert(lhs.size() == rhs.size(), "sizes don't match {} != {}", lhs.size(), rhs.size());
    vector<Block, Allocator> retval{lhs};
    retval |= rhs;
    return retval;
}

/// @brief Element by element "addition" of two equal sized bit-vectors (in GF(2) addition == XOR).
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
operator+(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    return operator^(lhs, rhs);
}

/// @brief Element by element "subtraction" of two equal sized bit-vectors (in GF(2) subtraction == XOR).
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
operator-(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    return operator^(lhs, rhs);
}

/// @brief Element by element "multiplication" of two equal sized bit-vectors (in GF(2) multiplication == AND).
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
operator*(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    return operator&(lhs, rhs);
}

/// @brief   Computes the logical DIFF of two equal sized bit-vectors.
/// @returns A bit-vector @c w where @c w[i] is 1 if @c u[i]!=v[i] and 0 otherwise.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
diff(const vector<Block, Allocator>& u, const vector<Block, Allocator>& v)
{
    bit_debug_assert(u.size() == v.size(), "Sizes don't match {} != {}", u.size(), v.size());

    vector<Block, Allocator> w{u};

    auto& w_blocks = w.blocks();
    auto& v_blocks = v.blocks();
    for (std::size_t i = 0; i < w_blocks.size(); ++i) w_blocks[i] &= ~v_blocks[i];
    return w;
}

/// @brief Joins two bit-vectors (can be any size) to create a new longer one.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
join(const vector<Block, Allocator>& u, const vector<Block, Allocator>& v)
{
    vector<Block, Allocator> w{u};
    w.append(v);
    return w;
}

/// @brief Joins three bit-vectors (can be any size) to create a new longer one.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
join(const vector<Block, Allocator>& u, const vector<Block, Allocator>& v, const vector<Block, Allocator>& w)
{
    vector<Block, Allocator> x{u};
    x.append(v);
    x.append(w);
    return x;
}

/// @brief Compute the dot product two bit-vectors using `&` for multiplication and `^` for addition.
/// @note  The two bit-vectors need to be of the same size (checked in DEBUG builds).
template<std::unsigned_integral Block, typename Allocator>
constexpr bool
dot(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    return lhs.dot(rhs);
}

/// @brief Return the convolution of two bit-vectors.
/// @note  The calculation here is done word-by-word style (thanks to Jason).
///        Faster algorithms exist for very very large bit-vectors.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
convolution(const vector<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    using vector_type = vector<Block, Allocator>;

    // Edge case?
    if (lhs.empty() || rhs.empty()) return vector_type{};

    // Generally the return value will have lhs.size() + rhs.size() - 1 elements (which could be all zeros).
    vector_type retval(lhs.size() + rhs.size() - 1);

    // Edge case? If either bit-vector is all zeros then so too is the convolution.
    if (lhs.none() || rhs.none()) return retval;

    // Only need to consider blocks in rhs up and including the one holding the final set bit.
    // We can ignore any trailing blocks that happen to be all zeros.
    std::size_t num_rhs_blocks = vector_type::block_index_for(rhs.final_set()) + 1;

    // Special first "iteration" where we copy the rhs rather than XOR'ing as we do later.
    for (std::size_t k = 0; k < num_rhs_blocks; ++k) retval.block(k) = rhs.block(k);

    // Work back from the final set element in the lhs using the usual idiom for such reverse iteration.
    for (std::size_t n = lhs.final_set(); n-- > 0;) {
        Block prev = 0;
        for (std::size_t i = 0; i < retval.block_count(); ++i) {
            Block left = prev >> (vector_type::bits_per_block - 1);
            prev = retval.block(i);
            retval.block(i) = static_cast<Block>(prev << 1) | left;
        }

        if (lhs.test(n)) {
            for (std::size_t k = 0; k < num_rhs_blocks; ++k) retval.block(k) ^= rhs.block(k);
        }
    }
    return retval;
}

/// @brief The usual output stream operator for a bit-vector.
template<std::unsigned_integral Block, typename Allocator>
std::ostream&
operator<<(std::ostream& s, const vector<Block, Allocator>& rhs)
{
    return s << rhs.to_pretty_string();
}

/// @brief  Create a bit-vector by reading bits encoded as a binary or hex string from a stream.
/// @param  s The stream to read from (e.g. might read "00111" or "0b00111" or "0x3DEFA3_2").
/// @param  rhs The vector we overwrite with the new bits.
/// @throws @c std::invalid_argument if the parse fails.
/// @note   Implementation uses a string buffer -- could probably do something more direct/efficient.
template<std::unsigned_integral Block, typename Allocator>
std::istream&
operator>>(std::istream& s, vector<Block, Allocator>& rhs)
{
    // Get the input string.
    std::string buffer;
    std::getline(s, buffer);

    // Try to parse it as a bit-vector.
    auto v = vector<Block, Allocator>::from(buffer);

    // Failure?
    if (!v) throw std::invalid_argument(buffer);

    // All good
    rhs = *v;
    return s;
}

} // namespace bit

// --------------------------------------------------------------------------------------------------------------------
// Connect bit-vectors to std::format & friends (not in the `bit` namespace).
// --------------------------------------------------------------------------------------------------------------------

/// @brief Connect bit-vectors to @c std::format and friends by specializing the @c std:formatter struct.
/// @note  We handle {} (default), {:b} (bit-order), {:p} (pretty-string), and {:x} (hex string).
template<std::unsigned_integral Block, typename Allocator>
struct std::formatter<bit::vector<Block, Allocator>> {

    /// @brief Parse a bit-vector format specifier where where we recognize:
    ///        '{}'   returns "v_0v_1v_2..." (elements in vector-order).
    ///        '{:b}' returns "v_{n-1}v_{n-2}..." (elements in bit-order).
    ///        '{:p}' returns "[v_0 v_1 v_2 ...]" (vector order pretty printed).
    ///        '{:x}' returns "0x...." (as a hex value).
    /// @note  Other specifiers will result in an "unrecognized format ..." message.
    ///        C++20 seems a bit weak on rational ways to tell you the specifier is not good at compile time.
    constexpr auto parse(const std::format_parse_context& ctx)
    {
        auto it = ctx.begin();
        while (it != ctx.end() && *it != '}') {
            switch (*it) {
                case 'b': m_bit_order = true; break;
                case 'p': m_pretty = true; break;
                case 'x': m_hex = true; break;
                default: m_error = true;
            }
            ++it;
        }
        return it;
    }

    /// @brief Push out a formatted bit-vector using the various @c to_string(...) methods in the class.
    template<class FormatContext>
    auto format(const bit::vector<Block, Allocator>& rhs, FormatContext& ctx) const
    {
        // Was there a format specification error?
        if (m_error) return std::format_to(ctx.out(), "'UNRECOGNIZED FORMAT SPECIFIER FOR BIT-VECTOR'");

        // Special handling requested?
        if (m_bit_order) return std::format_to(ctx.out(), "{}", rhs.to_bit_order());
        if (m_hex) return std::format_to(ctx.out(), "{}", rhs.to_hex());
        if (m_pretty) return std::format_to(ctx.out(), "{}", rhs.to_pretty_string());

        // Default
        return std::format_to(ctx.out(), "{}", rhs.to_string());
    }

    bool m_bit_order = false;
    bool m_hex = false;
    bool m_pretty = false;
    bool m_error = false;
};
