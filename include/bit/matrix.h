/// @brief A bit-matrix class bit::matrix.
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "bit_assert.h"
#include "vector.h"

#include <optional>
#include <regex>
#include <utility>

namespace bit {

/// @brief  A class for matrices over GF(2), the space of two elements {0, 1} with arithmetic done mod 2.
/// @tparam Block We pack the elements of the bit-matrix by row into arrays of these (some unsigned integer type).
/// @tparam Allocator is the memory manager that is used to allocate/deallocate space for the store as needed.
template<std::unsigned_integral Block = uint64_t, typename Allocator = std::allocator<Block>>
class matrix {
public:
    /// @brief Each row of a @c bit::matrix is a @c bit::vector.
    using vector_type = bit::vector<Block, Allocator>;

    /// @brief Construct an `r x c` bit-matrix.
    /// @note  If either parameter is zero the bit-matrix will be 0 x 0.
    constexpr matrix(std::size_t r, std::size_t c)
    {
        if (r != 0 && c != 0) resize(r, c);
    }

    /// @brief Construct an `n x n` square bit-matrix.
    /// @note  The default constructor for a @c bit::matrix creates an empty/singular 0x0 matrix.
    explicit constexpr matrix(std::size_t n = 0) : matrix(n, n) {}

    /// @brief  Reshape a bit-vector into a bit-matrix with @c r rows (defaults to one row).
    /// @param  v All the elements of @c v are used to create the bit-matrix.
    /// @param  r The bit-matrix will have @c r rows so @c r must divide `v.size()` evenly!
    ///         By default @c r=1 so the returned matrix has a single row.
    ///         If @c r=0 we will return a matrix with a single column instead.
    /// @param  by_rows If true we assume that @c v has the bit-matrix stored by rows (if false, columns).
    /// @throw  @c std::invalid_argument if @c r does not divide @c v.size() evenly.
    constexpr matrix(const vector_type& v, std::size_t r = 1, bool by_rows = true) : matrix()
    {
        // Trivial case?
        std::size_t s = v.size();
        if (s == 0) return;

        // We intepret r = 0 as a command to make a one colum bit-matrix (so v.size() rows)
        if (r == 0) r = s;

        // We only allow reshapes that consume all of v
        if (s % r != 0)
            throw std::invalid_argument("bit::vector size is not compatible with the requested number of matrix rows!");

        // Number of columns (for sure this is an even division)
        std::size_t c = s / r;
        resize(r, c);

        if (by_rows) {
            for (std::size_t i = 0; i < r; ++i) row(i) = v.sub(i * c, c);
        }
        else {
            std::size_t iv = 0;
            for (std::size_t j = 0; j < c; ++j)
                for (std::size_t i = 0; i < r; ++i) m_row[i][j] = v[iv++];
        }
    }

    /// @brief Construct an `n x m` bit-matrix from the outer product or outer sum of two bit-vectors.
    ///        Elements are `mat(i,j) = u[i] & v[j]` or `mat(i,j) = u[i] ^ v[j]` depending on the @c product argument.
    /// @param u has n elements `u[i]` for `i = 0, ..., n-1`
    /// @param v has m elements `v[j]` for `j = 0, ..., m-1`
    /// @param product (defaults to true) means use the outer product. False means use the outer sum instead.
    constexpr matrix(const vector_type& u, const vector_type& v, bool product = true) : matrix(0, v.size())
    {
        std::size_t r = u.size();
        std::size_t c = v.size();
        if (product) {
            for (std::size_t i = 0; i < r; ++i) {
                vector_type row = u[i] ? vector(v) : vector(c);
                m_row.push_back(std::move(row));
            }
        }
        else {
            for (std::size_t i = 0; i < r; ++i) {
                m_row.push_back(v);
                if (u[i]) m_row.back().flip();
            }
        }
    }

    /// @brief Construct an `r x c` bit-matrix by calling @c f(i,j) for each index pair.
    /// @param f If @c f(i,j)!=0 the corresponding element will be set to 1, otherwise it will be set to 0.
    explicit constexpr matrix(std::size_t r, std::size_t c, std::invocable<std::size_t, std::size_t> auto f) :
        matrix(r, c)
    {
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < c; ++j)
                if (f(i, j) != 0) set(i, j);
    }

    /// @brief Construct an `n x n` square bit-matrix by calling @c f(i,j) for each index pair.
    /// @param f If @c f(i,j)!=0 the corresponding element will be set to 1, otherwise it will be set to 0.
    explicit constexpr matrix(std::size_t n, std::invocable<std::size_t, std::size_t> auto f) : matrix(n, n, f)
    {
        // Empty body
    }

    /// @brief Factory method that tries to construct a bit-matrix from a binary or hex string.
    /// @param src The source string which should contain the bit-matrix elements in rows.
    ///        The rows can be separated by newlines, white space, commas, or semi-colons.
    /// @see   The constructor that creates a bit-vector from a string
    /// @param bit_order If true (default is false) the rows will all have their lowest bit on the right.
    ///        This parameter is completely ignored for hex-strings.
    /// @throw This method throws a @c std::invalid_argument exception if the string is not recognized.
    explicit matrix(std::string_view s, bool bit_order = false) : matrix()
    {
        auto m = from(s, bit_order);
        if (!m) throw std::invalid_argument("Failed to parse the input string as a valid bit-matrix!");
        *this = *m;
    }

    /// @brief Factory method to generate a `r x c` bit-matrix where the elements are from independent coin flips.
    /// @param p The probability of the elements being 1.
    static matrix random(std::size_t r, std::size_t c, double p)
    {
        // Need a valid probability ...
        if (p < 0 || p > 1) throw std::invalid_argument("Probability outside valid range [0,1]!");

        // Scale p by 2^64 to remove floating point arithmetic from the main loop below.
        // If we determine p rounds to 1 then we can just set all elements to 1 and return early.
        p = p * 0x1p64 + 0.5;
        if (p >= 0x1p64) return ones(r, c);

        // p does not round to 1 so we use a 64-bit URNG and check each draw against the 64-bit scaled p.
        auto scaled_p = static_cast<std::uint64_t>(p);

        // The URNG used is a simple congruential uniform 64-bit RNG seeded to a clock-dependent state.
        // The multiplier comes from Steele & Vigna -- see https://arxiv.org/abs/2001.05304
        using lcg = std::linear_congruential_engine<uint64_t, 0xd1342543de82ef95, 1, 0>;
        static lcg rng(static_cast<lcg::result_type>(std::chrono::system_clock::now().time_since_epoch().count()));

        return matrix(r, c, [&](std::size_t, std::size_t) { return rng() < scaled_p; });
    }

    /// @brief Factory method to construct an `r x c` bit-matrix where the elements are found by fair coin flips.
    static matrix random(std::size_t r, std::size_t c) { return random(r, c, 0.5); }

    /// @brief Factory method to construct a square bit-matrix where the elements are found by fair coin flips.
    static matrix random(std::size_t n) { return random(n, n, 0.5); }

    /// @brief Factory method to generate an `r x c` bit-matrix with all the elements set to 1.
    static constexpr matrix ones(std::size_t r, std::size_t c)
    {
        matrix retval(r, c);
        retval.set();
        return retval;
    }

    /// @brief Factory method to generate an `n x n` square bit-matrix with all the elements set to 1.
    static constexpr matrix ones(std::size_t n) { return ones(n, n); }

    /// @brief Factory method to generate an `r x c` bit-matrix with all the elements set to 0.
    static constexpr matrix zeros(std::size_t r, std::size_t c)
    {
        matrix retval(r, c);
        return retval;
    }

    /// @brief Factory method to generate an `n x n` square bit-matrix with all the elements set to 0
    static constexpr matrix zeros(std::size_t n) { return zeros(n, n); }

    /// @brief Factory method to generate an `r x c` bit-matrix with a checker-board pattern.
    static constexpr matrix checker_board(std::size_t r, std::size_t c, bool first_element_set = true)
    {
        // return first != 0 ? matrix(r, c, [](size_t i, size_t j) { return (i + j + 1) % 2; })
        //: matrix(r, c, [](size_t i, size_t j) { return (i + j) % 2; });

        matrix retval{0, c};
        retval.m_row.reserve(r);

        for (std::size_t i = 0; i < r; i++, first_element_set = !first_element_set) {
            auto row = vector_type::checker_board(c, first_element_set);
            retval.m_row.push_back(std::move(row));
        }
        return retval;
    }

    /// @brief Factory method to generate an `n x n` bit-matrix with a checker-board pattern.
    static constexpr matrix checker_board(std::size_t n, bool first_element_set = true)
    {
        return checker_board(n, n, first_element_set);
    }

    /// @brief  Factory method to generate the identity bit-matrix.
    /// @param  n The size of the matrix to generate.
    /// @return An `n x n` bit-matrix with 1's on the diagonal and 0's everywhere else.
    static constexpr matrix identity(std::size_t n)
    {
        matrix retval(n);
        retval.set_diagonal();
        return retval;
    }

    /// @brief Factory method to generate the `n x n` shift by @c p-places bit-matrix
    /// @param p Number of places to shift: `p > 0` for right-shift, `p < 0` for left-shift.
    static constexpr matrix shift(std::size_t n, int p = -1)
    {
        matrix retval(n);
        retval.set_diagonal(p);
        return retval;
    }

    /// @brief Factory method to generate the `n x n` rotate by @c p-places bit-matrix
    /// @param p Number of places to rotate: `p > 0` for right-rotate, `p < 0` for left-rotate.
    static constexpr matrix rotate(std::size_t n, int p = -1)
    {
        p %= static_cast<int>(n);
        if (p == 0) return identity(n);

        // Inelegant fix for `implicit conversion changes signedness` compiler warning that is meaningless here.
        auto n_plus_p = std::size_t(int(n) + p);

        matrix retval(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t j = (n_plus_p + i) % n;
            retval(i, j) = 1;
        }
        return retval;
    }

    /// @brief Factory method to create a "companion" matrix (sub-diagonal all ones, arbitrary top row)
    /// @param top_row These are the elements we copy into the top row of our matrix
    static constexpr matrix companion(const vector_type& top_row)
    {
        // Trivial case?
        auto n = top_row.size();
        if (n == 0) return matrix();

        // General case
        matrix retval(n);
        retval.row(0) = top_row;
        retval.set_diagonal(-1);
        return retval;
    }

    /// @brief  Factory method that tries to construct a bit-matrix from a binary or hex string.
    /// @param  src The source string which should contain the bit-matrix elements in rows.
    ///         The rows can be separated by newlines, white space, commas, or semi-colons.
    /// @see    The constructor that creates a bit-vector from a string
    /// @param  bit_order If true (default is false) the rows will all have their lowest bit on the right.
    ///         This parameter is completely ignored for hex-strings.
    /// @return Method returns a @c std::nullopt if the the string is not recognized.
    static std::optional<matrix> from(std::string_view s, bool bit_order = false)
    {
        // Trivial case
        if (s.empty()) return matrix();

        // We split the string into tokens using the standard regex library.
        std::string                src(s);
        auto                       delims = std::regex(R"([\s|,|;]+)");
        std::sregex_token_iterator iter{src.cbegin(), src.cend(), delims, -1};
        std::vector<std::string>   tokens{iter, {}};

        // Zap any empty tokens & check there is something to do
        tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [](std::string_view x) { return x.empty(); }),
                     tokens.end());
        if (tokens.empty()) return matrix();

        // We hope to fill a matrix.
        matrix retval;

        // Iterate through the possible rows
        std::size_t n_rows = tokens.size();
        std::size_t n_cols = 0;
        for (std::size_t i = 0; i < n_rows; ++i) {
            // Attempt to parse the current token as a row for the bit-matrix.
            auto r = vector_type::from(tokens[i], bit_order);

            // Parse failure?
            if (!r) return std::nullopt;

            // We've read a potentially valid matrix row.
            if (i == 0) {
                // First row sets the number of columns
                n_cols = r->size();
                retval.resize(n_rows, n_cols);
            }
            else {
                // Subsequent rows better have the same number of elements!
                if (r->size() != n_cols) return std::nullopt;
            }
            retval.row(i) = *r;
        }

        return retval;
    }

    /// @brief Returns the number of rows in the bit-matrix.
    constexpr std::size_t rows() const { return m_row.size(); }

    /// @brief Returns the number of columns in the bit-matrix.
    constexpr std::size_t cols() const { return m_row.size() > 0 ? m_row[0].size() : 0; }

    /// @brief Returns the number of elements in the bit-matrix.
    constexpr std::size_t size() const { return rows() * cols(); }

    /// @brief Is this a square bit-matrix? Note that empty bit-matrices are NOT considered square.
    constexpr bool is_square() const { return rows() != 0 && rows() == cols(); }

    /// @brief Is this an empty bit-matrix?
    constexpr bool empty() const { return rows() == 0; }

    /// @brief What is the row-capacity of the bit-matrix (i.e. number of rows that be added without a memory alloc)?
    constexpr std::size_t row_capacity() const { return m_row.capacity(); }

    /// @brief What is the column-capacity of the bit-matrix (i.e. number of cols that be added without a memory alloc)?
    /// @note  It is possible that the rows have different capacities but we just report the first one.
    constexpr std::size_t col_capacity() const { return m_row.capacity() > 0 ? m_row.data()->capacity() : 0; }

    /// @brief Resize the bit-matrix, initializing any added elements as zeros.
    /// @param r The new number of rows -- if `r  < rows()` we lose the excess rows.
    /// @param c The new number of columns -- if `c  < cols()` we lose the excess columns.
    constexpr matrix& resize(std::size_t r, std::size_t c)
    {
        // Trivial case ...
        if (rows() == r && cols() == c) return *this;

        // Resizes to zero in either dimension is taken as a zap the lot instruction
        if (r == 0 || c == 0) r = c = 0;

        // Rows, then columns
        m_row.resize(r);
        for (auto& bv : m_row) bv.resize(c);

        return *this;
    }

    /// @brief Resize the bit-matrix to square form, initializing any added values as zeros.
    /// @param n The new number of rows & columns
    constexpr matrix& resize(std::size_t n) { return resize(n, n); }

    /// @brief Add rows to the bit-matrix at the bottom--defaults to adding a single row.
    constexpr matrix& add_row(std::size_t n = 1) { return resize(rows() + n, cols()); }

    /// @brief Add columns to the bit-matrix at the right--defaults to adding a single column.
    constexpr matrix& add_col(std::size_t n = 1) { return resize(rows(), cols() + n); }

    /// @brief Removes the last row of the bit-matrix
    constexpr matrix& pop_row()
    {
        if (rows() > 0) m_row.pop_back();
        return *this;
    }

    /// @brief Removes the last column of the bit-matrix.
    constexpr matrix& pop_col()
    {
        if (cols() > 0) {
            for (auto& bv : m_row) bv.pop();
        }
        return *this;
    }

    /// @brief Removes all elements from the bit-matrix (does not change the capacity).
    constexpr matrix& clear()
    {
        resize(0, 0);
        return *this;
    }

    /// @brief Request to minimize any unused capacity in the bit-matrix.
    constexpr matrix& shrink_to_fit()
    {
        for (auto& bv : m_row) bv.shrink_to_fit();
        m_row.shrink_to_fit();
        return *this;
    }

    /// @brief Read-only element access for the bit-matrix.
    constexpr bool element(std::size_t i, std::size_t j) const
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        return m_row[i][j];
    }

    /// @brief  Read-write element access for the bit-matrix.
    /// @return A @c bit::vector::reference
    constexpr auto element(std::size_t i, std::size_t j)
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        return m_row[i][j];
    }

    /// @brief Read-only element access for the bit-matrix by index pair.
    constexpr bool operator()(std::size_t i, std::size_t j) const { return element(i, j); }

    /// @brief  Read-write element access for the bit-matrix.
    /// @return A @c bit::vector::reference
    constexpr auto operator()(std::size_t i, std::size_t j) { return element(i, j); }

    /// @brief Read-only access to row @c i in the bit-matrix.
    constexpr const vector_type& row(std::size_t i) const
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        return m_row[i];
    }

    /// @brief Read-write access to row @c i in the bit-matrix.
    constexpr vector_type& row(std::size_t i)
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        return m_row[i];
    }

    /// @brief Read-only access to row @c i.  Synonym for @c row(i)
    constexpr const vector_type& operator[](std::size_t i) const { return row(i); }

    /// @brief Read-write access to row @c i. Synonym for @c row(i)
    constexpr vector_type& operator[](std::size_t i) { return row(i); }

    /// @brief Read-only access to columns @c j
    constexpr vector_type col(std::size_t j) const
    {
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        vector_type retval(rows());
        for (std::size_t i = 0; i < rows(); ++i) retval[i] = m_row[i][j];
        return retval;
    }

    /// @brief  Extract the `r x c` sub-matrix starting at @c (i0,j0).
    /// @return A completely independent new bit-matrix of size `r x c`
    constexpr matrix sub(std::size_t i0, std::size_t j0, std::size_t r, std::size_t c) const
    {
        // Start point OK?
        bit_debug_assert(i0 < rows(), "i0 = {},  rows() = {}", i0, rows());
        bit_debug_assert(j0 < cols(), "i0 = {},  rows() = {}", j0, cols());

        // Trivial case?
        if (r == 0 || c == 0) return matrix{};

        // End point OK?
        bit_debug_assert(i0 + r - 1 < rows(), "i0 + r - 1 = {}, rows() = {}", i0 + r - 1, rows());
        bit_debug_assert(j0 + c - 1 < cols(), "j0 + c - 1 = {}, cols() = {}", j0 + c - 1, cols());

        // Set up the return value with the correct number of columns for each row.
        // Then reserve the space for the correct number of rows.
        matrix retval(0, c);
        retval.m_row.reserve(r);

        // Push back a copy of the appropriate piece from each row.
        for (std::size_t i = i0; i < i0 + r; ++i) retval.m_row.push_back(m_row[i].sub(j0, c));

        return retval;
    }

    /// @brief  Extract the `r x c` sub-matrix starting at `(0, 0)`.
    /// @return A completely independent new bit-matrix of size `r x c`
    constexpr matrix sub(std::size_t r, std::size_t c) const { return sub(0, 0, r, c); }

    /// @brief  Extract the `r x r` sub-matrix starting at `(0, 0)`.
    /// @return A completely independent new bit-matrix of size `r x r`
    constexpr matrix sub(std::size_t r) const { return sub(0, 0, r, r); }

    /// @brief  Extract the lower triangular part of this matrix.
    /// @param  strict If true we exclude the diagonal -- default is to include it.
    /// @return A completely independent new bit-matrix.
    constexpr matrix lower(bool strict = false) const
    {
        // Trivial case?
        if (empty()) return matrix();

        // Make a copy of the full matrix
        matrix retval = *this;

        // Zero out an appropriate portion of each row in that copy
        std::size_t nr = retval.rows();
        std::size_t nc = retval.cols();
        std::size_t di = (strict ? 0 : 1);
        for (std::size_t i = 0; i < nr; ++i) {

            // First element to zero out (may be done already if matrix is rectangular)
            auto first = i + di;
            if (first >= nc) break;
            retval.row(i).reset(first, nc - first);
        }
        return retval;
    }

    /// @brief  Extract the strictly lower triangular part of this matrix (excludes the diagonal).
    /// @return A completely independent new bit-matrix.
    constexpr matrix strictly_lower() const { return lower(true); }

    /// @brief Extract the lower triangular part of this matrix (with ones on the diagonal)
    /// @return A completely independent new bit-matrix.
    constexpr matrix unit_lower() const
    {
        auto retval = lower(true);
        retval.set_diagonal();
        return retval;
    }

    /// @brief  Extract the upper triangular part of this matrix.
    /// @param  strict If true we exclude the diagonal -- default is to include it.
    /// @return A completely independent new bit-matrix.
    constexpr matrix upper(bool strict = false) const
    {
        // Trivial case?
        if (empty()) return matrix();

        // Make a copy of the full matrix
        matrix retval = *this;

        // Zero out an appropriate portion of each row in that copy
        std::size_t nr = retval.rows();
        std::size_t nc = retval.cols();
        std::size_t di = (strict ? 1 : 0);
        for (std::size_t i = 0; i < nr; ++i) {

            // Number of elements to zero out
            auto len = std::min(i + di, nc);
            retval.row(i).reset(0, len);
        }
        return retval;
    }

    /// @brief  Extract the strictly upper triangular part of this matrix (excludes the diagonal).
    /// @return A completely independent new bit-matrix.
    constexpr matrix strictly_upper() const { return upper(true); }

    /// @brief  Extract the strictly upper triangular part of this matrix (with ones on the diagonal).
    /// @return A completely independent new bit-matrix.
    constexpr matrix unit_upper() const
    {
        auto retval = upper(true);
        retval.set_diagonal();
        return retval;
    }

    /// @brief Transpose a square matrix in-place.
    /// @note  There is a non-member function @c transpose(bit::matrix) for arbitrary rectangular bit-matrices.
    constexpr matrix& to_transpose()
    {
        bit_assert(is_square(), "Matrix is {} x {} -- needs to be square!", rows(), cols());
        for (std::size_t i = 1; i < rows(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                bool tmp = m_row[i][j];
                m_row[i][j] = m_row[j][i];
                m_row[j][i] = tmp;
            }
        }
        return *this;
    }

    /// @brief Swap rows @c i0 and @c i1
    constexpr matrix& swap_rows(std::size_t i0, std::size_t i1)
    {
        bit_debug_assert(i0 < rows(), "i0 = {},  rows() = {}", i0, rows());
        bit_debug_assert(i1 < rows(), "i1 = {},  rows() = {}", i1, rows());
        std::swap(m_row[i0], m_row[i1]);
        return *this;
    }

    /// @brief Swap columns @c j0 and @c j1
    constexpr matrix& swap_cols(std::size_t j0, std::size_t j1)
    {
        bit_debug_assert(j0 < cols(), "j0 = {},  cols() = {}", j0, cols());
        bit_debug_assert(j1 < cols(), "j1 = {},  cols() = {}", j1, cols());
        if (j0 == j1) return *this;
        for (std::size_t i = 0; i < rows(); ++i) {
            bool tmp = m_row[i][j0];
            m_row[i][j0] = m_row[i][j1];
            m_row[i][j1] = tmp;
        }
        return *this;
    }

    /// @brief Test whether a specific element is set.
    constexpr bool test(std::size_t i, std::size_t j) const
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        return m_row[i].test(j);
    }

    /// @brief Test whether all the elements are set.
    constexpr bool all() const
    {
        // Handle empty matrices with an exception if we're in a `BIT_DEBUG` scenario
        bit_debug_assert(!empty(), "Calling this method for an empty matrix is likely an error!");

        for (const auto& row : m_row)
            if (!row.all()) return false;
        return true;
    }

    /// @brief Test whether any of the elements are set.
    constexpr bool any() const
    {
        // Handle empty matrices with an exception if we're in a `BIT_DEBUG` scenario
        bit_debug_assert(!empty(), "Calling this method for an empty matrix is likely an error!");

        for (const auto& row : m_row)
            if (row.any()) return true;
        return false;
    }

    /// @brief Test whether none of the elements are set.
    constexpr bool none() const
    {
        // Handle empty matrices with an exception if we're in a `BIT_DEBUG` scenario
        bit_debug_assert(!empty(), "Calling this method for an empty matrix is likely an error!");

        for (const auto& row : m_row)
            if (!row.none()) return false;
        return true;
    }

    /// @brief Returns the number of set elements in the bit-matrix.
    constexpr std::size_t count() const
    {
        std::size_t retval = 0;
        for (const auto& row : m_row) retval += row.count();
        return retval;
    }

    /// @brief Compute the number of set elements on the bit-matrix diagonal.
    constexpr std::size_t count_diagonal() const
    {
        bit_debug_assert(is_square(), "Matrix is {} x {} but it should be square!", rows(), cols());
        std::size_t retval = 0;
        for (std::size_t i = 0; i < rows(); ++i)
            if (m_row[i][i]) ++retval;
        return retval;
    }

    /// @brief Compute the trace (the "sum" of the diagonal elements) of this bit-matrix.
    constexpr bool trace() const { return count_diagonal() % 2 == 1; }

    /// @brief Is this the  zero matrix?
    constexpr bool is_zero() const { return !any(); }

    /// @brief Is this the all ones matrix?
    constexpr bool is_ones() const { return all(); }

    /// @brief Is this the identity matrix?
    constexpr bool is_identity() const
    {
        if (!is_square()) return false;
        for (std::size_t i = 0; i < rows(); ++i) {
            auto r = row(i);
            r.flip(i);
            if (r.any()) return false;
        }
        return true;
    }

    /// @brief Is this bit-matrix symmetric?
    constexpr bool is_symmetric() const
    {
        if (!is_square()) return false;
        std::size_t N = rows();
        for (std::size_t i = 1; i < N; ++i) {
            for (std::size_t j = 0; j < i; ++j)
                if (m_row[i][j] != m_row[j][i]) return false;
        }
        return true;
    }

    /// @brief Set the element at a specific bit-matrix location.
    constexpr matrix& set(std::size_t i, std::size_t j)
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        m_row[i].set(j);
        return *this;
    }

    /// @brief Set all the elements in the bit-matrix.
    constexpr matrix& set()
    {
        for (auto& row : m_row) row.set();
        return *this;
    }

    /// @brief Set the elements on some diagonal of the bit-matrix.
    /// @param d `d > 0` for a super-diagonal, `d < 0` for a sub-diagonal, `d == 0` for the main diagonal (default).
    constexpr matrix& set_diagonal(int d = 0)
    {
        bit_assert(is_square(), "Matrix is {} x {} -- needs to be square!", rows(), cols());

        // Method (silently) does nothing if the diagonal is out of range
        if (const auto ad = static_cast<std::size_t>(abs(d)); ad < rows()) {
            std::size_t iLo = d >= 0 ? 0 : ad;
            std::size_t iHi = d >= 0 ? rows() - ad : rows();
            std::size_t j = d >= 0 ? iLo + ad : iLo - ad;
            for (std::size_t i = iLo; i < iHi; ++i) m_row[i][j++] = 1;
        }
        return *this;
    }

    /// @brief Reset the element in a specific bit-matrix location.
    constexpr matrix& reset(std::size_t i, std::size_t j)
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        m_row[i].reset(j);
        return *this;
    }

    /// @brief Reset all the elements in the bit-matrix.
    constexpr matrix& reset()
    {
        for (auto& row : m_row) row.reset();
        return *this;
    }

    /// @brief Reset the elements on some diagonal of the bit-matrix.
    /// @param d `d > 0` for a super-diagonal, `d < 0` for a sub-diagonal, `d == 0` for the main diagonal (default).
    constexpr matrix& reset_diagonal(int d = 0)
    {
        bit_assert(is_square(), "Matrix is {} x {} -- needs to be square!", rows(), cols());

        // Method (silently) does nothing if the diagonal is out of range
        if (const auto ad = static_cast<std::size_t>(abs(d)); ad < rows()) {
            std::size_t iLo = d >= 0 ? 0 : ad;
            std::size_t iHi = d >= 0 ? rows() - ad : rows();
            std::size_t j = d >= 0 ? iLo + ad : iLo - ad;
            for (std::size_t i = iLo; i < iHi; ++i) m_row[i][j++] = 0;
        }
        return *this;
    }

    /// @brief Flip a the element in a specific bit-matrix location.
    constexpr matrix& flip(std::size_t i, std::size_t j)
    {
        bit_debug_assert(i < rows(), "i = {}, rows() = {}", i, rows());
        bit_debug_assert(j < cols(), "j = {}, cols() = {}", j, cols());
        m_row[i].flip(j);
        return *this;
    }

    /// @brief Flip all the elements in the bit-matrix.
    constexpr matrix& flip()
    {
        for (auto& row : m_row) row.flip();
        return *this;
    }

    /// @brief Flips the elements on some diagonal of the bit-matrix.
    /// @param d `d > 0` for a super-diagonal, `d < 0` for a sub-diagonal, `d == 0` for the main diagonal (default).
    constexpr matrix& flip_diagonal(int d = 0)
    {
        bit_assert(is_square(), "Matrix is {} x {} -- needs to be square!", rows(), cols());

        // Method (silently) does nothing if the diagonal is out of range
        if (const auto ad = static_cast<std::size_t>(abs(d)); ad < rows()) {
            std::size_t iLo = d >= 0 ? 0 : ad;
            std::size_t iHi = d >= 0 ? rows() - ad : rows();
            std::size_t j = d >= 0 ? iLo + ad : iLo - ad;
            for (std::size_t i = iLo; i < iHi; ++i) m_row[i][j++].flip();
        }
        return *this;
    }

    /// @brief Set element `(i, j)` to 1 if `f(i,j) != 0` otherwise set it to 0.
    /// @param f is function that we expect to call as `f(i, j)` for each index pair in the bit-matrix.
    constexpr matrix& set_if(std::invocable<std::size_t, std::size_t> auto f)
    {
        reset();
        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t j = 0; j < cols(); ++j)
                if (f(i, j) != 0) set(i, j);
        return *this;
    }

    /// @brief Flips the value of element `(i, j)` if `f(i,j) != 0` otherwise leaves it alone.
    /// @param f is function that we expect to call as `f(i, j)` for each index pair in the bit-matrix.
    constexpr matrix& flip_if(std::invocable<std::size_t, std::size_t> auto f)
    {
        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t j = 0; j < cols(); ++j)
                if (f(i, j) != 0) flip(i, j);
        return *this;
    }

    /// @brief Element by element AND'ing.
    constexpr matrix& operator&=(const matrix& rhs)
    {
        bit_debug_assert(rhs.rows() == rows(), "rhs.rows() = {}, rows() = {}", rhs.rows(), rows());
        bit_debug_assert(rhs.cols() == cols(), "rhs.cols() = {}, cols() = {}", rhs.cols(), cols());
        for (std::size_t i = 0; i < rows(); ++i) row(i) &= rhs.row(i);
        return *this;
    }

    /// @brief Element by element OR'ing.
    constexpr matrix& operator|=(const matrix& rhs)
    {
        bit_debug_assert(rhs.rows() == rows(), "rhs.rows() = {}, rows() = {}", rhs.rows(), rows());
        bit_debug_assert(rhs.cols() == cols(), "rhs.cols() = {}, cols() = {}", rhs.cols(), cols());
        for (std::size_t i = 0; i < rows(); ++i) row(i) |= rhs.row(i);
        return *this;
    }

    /// @brief Element by element XOR'ing.
    constexpr matrix& operator^=(const matrix& rhs)
    {
        bit_debug_assert(rhs.rows() == rows(), "rhs.rows() = {}, rows() = {}", rhs.rows(), rows());
        bit_debug_assert(rhs.cols() == cols(), "rhs.cols() = {}, cols() = {}", rhs.cols(), cols());
        for (std::size_t i = 0; i < rows(); ++i) row(i) ^= rhs.row(i);
        return *this;
    }

    /// @brief Element by element addition (in GF(2) addition is just XOR).
    constexpr matrix& operator+=(const matrix& rhs) { return operator^=(rhs); }

    /// @brief Element by element subtraction (in GF(2) subtraction is just XOR).
    constexpr matrix& operator-=(const matrix& rhs) { return operator^=(rhs); }

    /// @brief Element by element multiplication (in GF(2) multiplication is just AND).
    constexpr matrix& operator*=(const matrix& rhs) { return operator&=(rhs); }

    /// @brief Get a copy of this bit-matrix with all the elements flipped.
    constexpr matrix operator~() const
    {
        matrix retval{*this};
        retval.flip();
        return retval;
    }

    /// @brief Left shift all the rows by @c p places.
    constexpr matrix& operator<<=(std::size_t p)
    {
        for (auto& row : m_row) row <<= p;
        return *this;
    }

    /// @brief Right shift all the rows by @c p places.
    constexpr matrix& operator>>=(std::size_t p)
    {
        for (auto& row : m_row) row >>= p;
        return *this;
    }

    /// @brief Get a copy of this bit-matrix where all the rows are left shifted by @c p places.
    constexpr matrix operator<<(std::size_t p) const
    {
        matrix retval{*this};
        retval <<= p;
        return retval;
    }

    /// @brief Get a copy of this bit-matrix where all the rows are right shifted by @c p places.
    constexpr matrix operator>>(std::size_t p) const
    {
        matrix retval{*this};
        retval >>= p;
        return retval;
    }

    /// @brief Augment this matrix in-place by appending a vector as a new column to the right so M -> M|v.
    /// @param v The vector must have the same size as there are rows in the matrix.
    constexpr matrix& append(const vector<Block, Allocator>& v)
    {
        if (v.size() != rows()) throw std::invalid_argument("Incompatible input vector -- different number of rows!");
        for (std::size_t i = 0; i < rows(); ++i) m_row[i].append(v[i]);
        return *this;
    }

    /// @brief Augment this matrix in-place by appending another matrix as new columns to the right so M -> M|V.
    /// @param V The matrix must have the same number of rows as there are rows in the matrix.
    constexpr matrix& append(const matrix<Block, Allocator>& V)
    {
        if (V.rows() != rows()) throw std::invalid_argument("Incompatible input matrix -- different number of rows!");
        for (std::size_t i = 0; i < rows(); ++i) m_row[i].append(V.m_row[i]);
        return *this;
    }

    /// @brief Starting at @c (i0,j0) replace our values with those from a passed in replacement bit-matrix @c with.
    constexpr matrix& replace(std::size_t i0, std::size_t j0, const matrix& with)
    {
        // Start point OK?
        bit_debug_assert(i0 < rows(), "i0 = {},  rows() = {}", i0, rows());
        bit_debug_assert(j0 < cols(), "i0 = {},  rows() = {}", j0, cols());

        // Size of replacement OK?
        bit_debug_assert(i0 + with.rows() - 1 < rows(), "i1 = {}, rows() = {}", i0 + with.rows() - 1, rows());
        bit_debug_assert(j0 + with.cols() - 1 < cols(), "j1 = {}, cols() = {}", j0 + with.cols() - 1, cols());

        // Just use the bit-vector replacement mechanism for each effected row ...
        for (std::size_t i = 0; i < with.rows(); ++i) row(i0 + i).replace(j0, with.row(i));

        return *this;
    }

    /// @brief Starting on the diagonal at @c (i0,i0) replace our values with those from a passed in bit-matrix @c with
    constexpr matrix& replace(std::size_t i0, const matrix& with) { return replace(i0, i0, with); }

    /// @brief Starting on the diagonal at @c (0,0) replace our values with those from a passed in bit-matrix @c with
    constexpr matrix& replace(const matrix& with) { return replace(0, 0, with); }

    /// @brief Turn this bit-matrix into a bit-vector.
    /// @param by_rows If true (the default) then we merge by rows, otherwise we merge by columns
    constexpr auto to_vector(bool by_rows = true) const
    {
        vector<Block, Allocator> retval;
        retval.reserve(size());
        if (by_rows) {
            for (std::size_t i = 0; i < rows(); ++i) retval.append(row(i));
        }
        else {
            for (std::size_t j = 0; j < cols(); ++j) retval.append(col(j));
        }
        return retval;
    }

    /// @brief Get a bit-string representation for this bit-matrix using the given characters for set and unset.
    std::string to_string(std::string_view delim = "\n", char off = '0', char on = '1') const
    {
        // Handle a trivial case ...
        std::size_t r = rows();
        if (r == 0) return std::string{};

        std::size_t c = cols();
        std::string retval;
        retval.reserve(r * (c + delim.length()));

        for (std::size_t i = 0; i < r; ++i) {
            retval += m_row[i].to_string("", "", "", off, on);
            if (i + 1 < r) retval += delim;
        }
        return retval;
    }

    /// @brief Get a bit-string representation for this bit-matrix using the given characters for set and unset.
    std::string to_pretty_string(char off = '0', char on = '1') const
    {
        // Handle completely empty matrices
        std::size_t r = rows();
        if (r == 0) return "[]";

        // Handle matrices with just one row
        if (r == 1) return m_row[0].to_string("[", "]", " ", off, on);

        // Light vertical bar in Unicode
        std::string bar = "\u2502";

        // Space for the return string
        std::size_t c = cols();
        std::string retval;
        retval.reserve(r * (2 * c + 2 * bar.size() + 1));

        for (std::size_t i = 0; i < r; ++i) {
            retval += m_row[i].to_string(bar, bar, " ", off, on);
            if (i + 1 < r) retval += '\n';
        }

        return retval;
    }

    /// @brief Get a hex-string representation for this bit-matrix.
    std::string to_hex(std::string_view delim = "\n") const
    {
        // Handle a trivial case ...
        std::size_t r = rows();
        if (r == 0) return std::string{};

        // Otherwise we set up the return string with the correct amount of space
        std::size_t c = cols();
        std::string retval;
        retval.reserve(r * (c / 4 + 4));

        for (std::size_t i = 0; i < r; ++i) {
            retval += m_row[i].to_hex();
            if (i + 1 < r) retval += delim;
        }
        return retval;
    }

    /// @brief Probability that an `n x n` bit matrix with elements picked by flipping fair coins, is invertible.
    /// @note  For large @c n the value is about 29% and that holds even for n as low as 10.
    static double probability_invertible(std::size_t n)
    {
        // A 0x0 matrix is not well defined throw an error!
        if (n == 0) throw std::invalid_argument("Matrix should not be 0x0--likely an error somewhere upstream!");

        // Formula is p(n) = \prod_{k = 1}^{n} (1 - 2^{-k}) which runs out of juice once n hits any size at all!
        std::size_t n_prod = std::min(n, static_cast<size_t>(std::numeric_limits<double>::digits));

        // Compute the product over the range that matters
        double product = 1;
        double pow2 = 1;
        while (n_prod--) {
            pow2 *= 0.5;
            product *= 1 - pow2;
        }
        return product;
    }

    /// @brief Probability that an `n x n` bit matrix with elements picked by flipping fair coins, is singular.
    /// @note  For large @c n the value is about 71% and that holds even for n as low as 10.
    static double probability_singular(std::size_t n) { return 1.0 - probability_invertible(n); }

    /// @brief Check for equality between two bit-matrices
    constexpr bool friend operator==(const matrix& lhs, const matrix& rhs)
    {
        if (&lhs != &rhs) {
            if (lhs.rows() != rhs.rows()) return false;
            for (std::size_t i = 0; i < lhs.rows(); ++i)
                if (lhs[i] != rhs[i]) return false;
        }
        return true;
    }

    /// @brief Converts this matrix in-place to row-echelon form.
    /// @param pivot_col If present, on return this will have a set bit for every column with a pivot in this matrix.
    matrix& to_echelon_form(vector_type* pivot_col = 0)
    {
        bit_always_assert(!empty(), "Empty matrices are not supported by this method!");

        // Were we asked to track which columns contain pivots?
        if (pivot_col) {
            pivot_col->resize(cols());
            pivot_col->reset();
        }

        // Working on row number ...
        std::size_t nr = rows();
        std::size_t r = 0;

        // Iterate through the columns
        for (std::size_t j = 0; j < cols(); ++j) {

            // Find a non-zero element on or below the diagonal in this column -- a pivot.
            std::size_t p = r;
            while (p < nr && m_row[p][j] == 0) ++p;

            // Perhaps no such element exists in this column? If so move on to the next one.
            if (p == nr) continue;

            // Getting to here means that column j does have a pivot
            if (pivot_col) pivot_col->element(j) = true;

            // If necessary, swap the pivot row into place
            if (p != r) swap_rows(p, r);

            // Below the working row make sure column j is all zeros by elimination as necessary
            for (std::size_t i = r + 1; i < nr; ++i) {
                if (m_row[i][j] == 1) row(i) ^= row(r);
            }

            // Move on to the next row if there is one
            if (++r == nr) break;
        }
        return *this;
    }

    /// @brief Converts this matrix in place to reduced-row-echelon-form
    /// @param pivot_col If present, on return this will have a set bit for every column with a pivot in this matrix.
    matrix& to_reduced_echelon_form(vector_type* pivot_col = nullptr)
    {
        // Start by going to echelon form
        to_echelon_form(pivot_col);

        // Then iterate from the bottom row up, keeping at most one non-zero entry per row.
        for (std::size_t i = rows(); i--;) {

            // Find the pivot column -- i.e. the first set bit in row i & skip the row if it is all zeros
            auto p = row(i).first_set();
            if (p == vector<>::npos) continue;

            // Zap everything in column p above row i
            for (std::size_t e = 0; e < i; ++e) {
                if (m_row[e][p] == 1) row(e) ^= row(i);
            }
        }
        return *this;
    }

    /// @brief A little debug utility to dump a whole bunch of descriptive data about a bit-matrix to a stream.
    /// @note  Don't depend on this format remaining constant!
    constexpr void description(std::ostream& s, const std::string& header = "", const std::string& footer = "\n") const
    {
        if (!header.empty()) s << header << ":\n";
        s << "bit-matrix dimension:   " << rows() << " x " << cols() << "\n";
        s << "bit-matrix capacity:    " << row_capacity() << " x " << col_capacity() << "\n";
        s << "number of set elements: " << count() << "\n";
        std::size_t r = rows();
        for (std::size_t i = 0; i < r; ++i) {
            s << "    " << m_row[i].to_string() << "  =  0x" << m_row[i].to_hex() << "\n";
        }
        s << footer;
    }

    /// @brief A little debug utility to dump a whole bunch of descriptive data about a bit-matrix to `std::cout`
    constexpr void description(const std::string& header = "", const std::string& footer = "\n") const
    {
        description(std::cout, header, footer);
    }

private:
    /// @brief The matrix data
    std::vector<vector_type> m_row;
};

// --------------------------------------------------------------------------------------------------------------------
// NON-MEMBER FUNCTIONS ...
// --------------------------------------------------------------------------------------------------------------------

/// @brief bit-matrix, bit-vector multiplication (dimensions must be compatible).
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
dot(const matrix<Block, Allocator>& lhs, const vector<Block, Allocator>& rhs)
{
    bit_debug_assert(lhs.cols() == rhs.size(), "Matrix cols = {}, vector size = {}", lhs.cols(), rhs.size());
    std::size_t              r = lhs.rows();
    vector<Block, Allocator> retval(r);
    for (std::size_t i = 0; i < r; ++i) retval[i] = bit::dot(lhs.row(i), rhs);
    return retval;
}

/// @brief bit-matrix, bit-vector multiplication (dimensions must be compatible).
/// @note We store bit-matrices by rows so @c dot(matrix,vector) will generally be faster than this.
template<std::unsigned_integral Block, typename Allocator>
constexpr vector<Block, Allocator>
dot(const vector<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    bit_debug_assert(lhs.size() == rhs.rows(), "Matrix rows = {}, vector size = {}", rhs.rows(), lhs.size());
    std::size_t              c = rhs.cols();
    vector<Block, Allocator> retval(c);
    for (std::size_t j = 0; j < c; ++j) retval[j] = bit::dot(lhs, rhs.col(j));
    return retval;
}

/// @brief bit-matrix bit-matrix multiplication (bit-matrices must have compatible dimensions).
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
dot(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    bit_debug_assert(lhs.cols() == rhs.rows(), "lhs.cols() = {}, rhs.rows() = {}", lhs.cols(), rhs.rows());
    std::size_t              r = lhs.rows();
    std::size_t              c = rhs.cols();
    matrix<Block, Allocator> retval(r, c);
    for (std::size_t j = 0; j < c; ++j) {
        auto rhsCol = rhs.col(j);
        for (std::size_t i = 0; i < r; ++i) retval(i, j) = bit::dot(lhs.row(i), rhsCol);
    }
    return retval;
}

/// @brief Given a bit-matrix M and a compatible bit-vector v we return the augmented bit-matrix M|v.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
join(const matrix<Block, Allocator>& M, const vector<Block, Allocator>& v)
{
    auto retval = M;
    return retval.append(v);
}

/// @brief Given a bit-matrix M and a compatible bit-matrix V we return the augmented bit-matrix M|V.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
join(const matrix<Block, Allocator>& M, const matrix<Block, Allocator>& V)
{
    auto retval = M;
    return retval.append(V);
}

/// @brief Returns the transpose of an arbitrary bit-matrix. Does not alter the input bit-matrix.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
transpose(const matrix<Block, Allocator>& M)
{
    std::size_t              r = M.rows();
    std::size_t              c = M.cols();
    matrix<Block, Allocator> retval(c, r);
    for (std::size_t i = 0; i < r; ++i) {
        for (std::size_t j = 0; j < c; ++j) retval(j, i) = M(i, j);
    }
    return retval;
}

/// @brief Raise a square bit-matrix to any power @c n.
/// @note  This version is a straight left-to-right bits in n square & multiply.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
pow(const matrix<Block, Allocator>& M, std::size_t n)
{
    bit_assert(M.is_square(), "Matrix is {} x {} but it should be square!", M.rows(), M.cols());

    // Trivial case M^0 = I?
    if (n == 0) return matrix<Block, Allocator>::identity(M.rows());

    // n != 0: Note that if e.g. n = 0b00010111 then std::bit_floor(n) = 0b00010000.
    std::size_t n2 = std::bit_floor(n);

    // Start with our product being M
    matrix<Block, Allocator> retval{M};

    // That takes care of the most significant binary digit in n.
    n2 >>= 1;

    // More to go?
    while (n2) {
        // Need a squaring step ...
        retval = dot(retval, retval);

        // May also need a straight multiply step.
        if (n & n2) retval = dot(retval, M);

        // Finished with another binary digit in n.
        n2 >>= 1;
    }
    return retval;
}

/// @brief Raise a square bit-matrix to the power @c 2^n e.g. 2^128 which is not representable as an @c uint64_t
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
pow2(const matrix<Block, Allocator>& M, std::size_t n)
{
    bit_assert(M.is_square(), "Matrix has dimensions {} x {} but it should be square!", M.rows(), M.cols());

    // Note the 2^0 = 1 so we can start with a straight copy of M
    matrix<Block, Allocator> retval{M};

    // Square as often as requested ...
    for (uint64_t i = 0; i < n; ++i) retval = dot(retval, retval);

    return retval;
}

/// @brief Element by element bit-matrix, bit-matrix &'ing
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
operator&(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    matrix<Block, Allocator> retval{lhs};
    retval &= rhs;
    return retval;
}

/// @brief Element by element bit-matrix, bit-matrix |'ing.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
operator|(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    matrix<Block, Allocator> retval{lhs};
    retval |= rhs;
    return retval;
}

/// @brief Element by element bit-matrix, bit-matrix ^'ing.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
operator^(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    matrix<Block, Allocator> retval{lhs};
    retval ^= rhs;
    return retval;
}

/// @brief Element by element bit-matrix, bit-matrix addition (in GF(2) addition is just XOR).
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
operator+(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    matrix<Block, Allocator> retval{lhs};
    retval ^= rhs;
    return retval;
}

/// @brief Element by element bit-matrix, bit-matrix subtraction (in GF(2) subtraction is just XOR).
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
operator-(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    matrix<Block, Allocator> retval{lhs};
    retval ^= rhs;
    return retval;
}

/// @brief Element by element bit-matrix, bit-matrix multiplication (in GF(2) multiplication is just AND).
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
operator*(const matrix<Block, Allocator>& lhs, const matrix<Block, Allocator>& rhs)
{
    matrix<Block, Allocator> retval{lhs};
    retval &= rhs;
    return retval;
}

/// @brief Computes the result of using a square bit-matrix as the argument in a polynomial.
/// @param p The polynomial coefficients as a bit-vector where the polynomial is p0 + p1 x + p2 x^2 + ...
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
polynomial_sum(const vector<Block, Allocator>& p, const matrix<Block, Allocator>& M)
{
    // The bit-matrix must be square.
    bit_assert(M.is_square(), "Matrix has dimensions {} x {} but it should be square!", M.rows(), M.cols());

    // The returned bit-matrix will be N x N.
    auto N = M.rows();

    // Handle a singular/empty polynomial with an exception if we're in a `BIT_DEBUG` scenario
    bit_debug_assert(!p.empty(), "Calling this method for an empty bit-vector is likely an error!");

    // Otherwise handling a singular/empty polynomial is a bit arbitrary but needs must ...
    if (p.empty()) return matrix<Block, Allocator>(N);

    // Of course if the polynomial is all zeros then so too is the polynomial sum.
    if (p.none()) return matrix<Block, Allocator>(N);

    // Highest power in the polynomial--now know there is one.
    auto n = p.final_set();

    // Start with the polynomial sum being the identity matrix.
    auto retval = matrix<Block, Allocator>::identity(N);

    // Work backwards a la Horner
    while (n > 0) {

        retval = dot(M, retval);

        // Add the identity to the sum if the corresponding polynomial coefficient is 1.
        if (p[n - 1])
            for (std::size_t i = 0; i < N; ++i) retval(i, i) ^= 1;

        // And count down ...
        n--;
    }

    return retval;
}

/// @brief Returns the row-echelon form of a bit-matrix. Does not alter the input matrix.
/// @param pivot_col If present, on return this will have a set bit for every column with a pivot in this matrix.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
echelon_form(const matrix<Block, Allocator>& M, vector<Block, Allocator>* pivot_col = nullptr)
{
    auto A = M;
    return A.to_echelon_form(pivot_col);
}

/// @brief Returns the reduced-row-echelon form of a bit-matrix. Does not alter the input matrix.
/// @param pivot_col If present, on return this will have a set bit for every column with a pivot in this matrix.
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
reduced_echelon_form(const matrix<Block, Allocator>& M, vector<Block, Allocator>* pivot_col = 0)
{
    auto A = M;
    return A.to_reduced_echelon_form(pivot_col);
}

/// @brief Returns the inverse of a bit-matrix or @c std::nullopt if the matrix is singular.
template<std::unsigned_integral Block, typename Allocator>
std::optional<matrix<Block, Allocator>>
invert(const matrix<Block, Allocator>& M)
{
    bit_assert(M.is_square(), "Matrix has dimensions {} x {} but it should be square!", M.rows(), M.cols());

    // Create the augmented matrix M|I
    auto n = M.rows();
    auto A = join(M, matrix<Block, Allocator>::identity(n));

    // Compute the reduced row echelon form for A
    A.to_reduced_echelon_form();

    // If all went well A is now I|M^(-1)
    if (A.sub(n).is_identity()) return A.sub(0, n, n, n);

    // Apparently things did not go well -- A is not invertible
    return std::nullopt;
}

/// @brief Prints a bit-matrix A and a bit-vector b side-by-side.
/// @param strm  The bit-matrix and bit-vector appear on this stream.
/// @param delim The bit-matrix and bit-vector are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(std::ostream& strm, const bit::matrix<Block, Alloc>& A, const bit::vector<Block, Alloc>& b,
      std::string_view delim = "\t")
{
    // If either is empty there was likely a bug somewhere!
    bit_debug_assert(!A.empty(), "Matrix A is empty which is likely an error!");
    bit_debug_assert(!b.empty(), "Vector b is empty which is likely an error!");

    // Most rows we ever print
    auto nr = std::max(A.rows(), b.size());

    // Space filler might be needed if the matrix has fewer rows than the vector
    std::string A_fill(A.cols(), ' ');

    // Little lambda that turns a bool into a string
    auto b2s = [](bool x) { return x ? "1" : "0"; };

    // Print away ...
    // clang-format off
    for (std::size_t r = 0; r < nr; ++r) {
        strm << (r < A.rows() ? A.row(r).to_string() : A_fill) << delim
             << (r < b.size() ? b2s(b[r]) : " ") << "\n";
    }
    // clang-format on
    return;
}

/// @brief Prints a bit-matrix A and a bit-vector b side-by-side to @c std::cout
/// @param delim The bit-matrix and bit-vector are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(const bit::matrix<Block, Alloc>& A, const bit::vector<Block, Alloc>& b, std::string_view delim = "\t")
{
    print(std::cout, A, b, delim);
}

/// @brief Prints a bit-matrix A, and bit-vectors b & x side-by-side.
/// @param strm  The bit-matrix and bit-vectors appear on this stream
/// @param delim The bit-matrix and bit-vectors are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(std::ostream& strm, const bit::matrix<Block, Alloc>& A, const bit::vector<Block, Alloc>& b,
      const bit::vector<Block, Alloc>& x, std::string_view delim = "\t")
{
    // If any are empty there was likely a bug somewhere!
    bit_debug_assert(!A.empty(), "Matrix A is empty which is likely an error!");
    bit_debug_assert(!b.empty(), "Vector b is empty which is likely an error!");
    bit_debug_assert(!x.empty(), "Vector x is empty which is likely an error!");

    // Most rows we ever print
    auto nr = std::max({A.rows(), b.size(), x.size()});

    // Space filler might be needed if the matrix has fewer rows than the vector
    std::string A_fill(A.cols(), ' ');

    // Little lambda that turns a bool into a string
    auto b2s = [](bool z) { return z ? "1" : "0"; };

    // Print away ...
    // clang-format off
    for (std::size_t r = 0; r < nr; ++r) {
        strm << (r < A.rows() ? A.row(r).to_string() : A_fill) << delim
             << (r < b.size() ? b2s(b[r]) : " ") << delim
             << (r < x.size() ? b2s(x[r]) : " ") << "\n";
    }
    // clang-format on

    return;
}

/// @brief Prints a bit-matrix A and bit-vectors b & x side-by-side to @c std::cout
/// @param delim The bit-matrix and bit-vectors are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(const bit::matrix<Block, Alloc>& A, const bit::vector<Block, Alloc>& b, const bit::vector<Block, Alloc>& x,
      std::string_view delim = "\t")
{
    print(std::cout, A, b, x, delim);
}

/// @brief Prints a bit-matrix A, and three bit-vectors b, x, & y side-by-side.
/// @param strm  The bit-matrix and bit-vectors appear on this stream
/// @param delim The bit-matrix and bit-vectors are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(std::ostream& strm, const bit::matrix<Block, Alloc>& A, const bit::vector<Block, Alloc>& b,
      const bit::vector<Block, Alloc>& x, const bit::vector<Block, Alloc>& y, std::string_view delim = "\t")
{
    // If any are empty there was likely a bug somewhere!
    bit_debug_assert(!A.empty(), "Matrix A is empty which is likely an error!");
    bit_debug_assert(!b.empty(), "Vector b is empty which is likely an error!");
    bit_debug_assert(!x.empty(), "Vector x is empty which is likely an error!");
    bit_debug_assert(!y.empty(), "Vector y is empty which is likely an error!");

    // Most rows we ever print
    auto nr = std::max({A.rows(), b.size(), x.size(), y.size()});

    // Space filler might be needed if the matrix has fewer rows than the vector
    std::string A_fill(A.cols(), ' ');

    // Little lambda that turns a bool into a string
    auto b2s = [](bool z) { return z ? "1" : "0"; };

    // Print away ...
    // clang-format off
    for (std::size_t r = 0; r < nr; ++r) {
        strm << (r < A.rows() ? A.row(r).to_string() : A_fill) << delim
             << (r < b.size() ? b2s(b[r]) : " ") << delim
             << (r < x.size() ? b2s(x[r]) : " ") << delim
             << (r < y.size() ? b2s(y[r]) : " ") << "\n";
    }
    // clang-format on

    return;
}

/// @brief Prints a bit-matrix A and three bit-vectors b, x, & y side-by-side to @c std::cout
/// @param delim The bit-matrix and bit-vectors are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(const bit::matrix<Block, Alloc>& A, const bit::vector<Block, Alloc>& b, const bit::vector<Block, Alloc>& x,
      const bit::vector<Block, Alloc>& y, std::string_view delim = "\t")
{
    print(std::cout, A, b, x, y, delim);
}

/// @brief Prints two bit-matrices A, B side-by-side.
/// @param strm  The bit-matrices appear on this stream.
/// @param delim The bit-matrices are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(std::ostream& strm, const bit::matrix<Block, Alloc>& A, const bit::matrix<Block, Alloc>& B,
      std::string_view delim = "\t")
{
    // If any matrix is empty there was likely a bug somewhere!
    bit_debug_assert(!A.empty(), "Matrix A is empty which is likely an error!");
    bit_debug_assert(!B.empty(), "Matrix B is empty which is likely an error!");

    // Most rows we ever print
    auto nr = std::max(A.rows(), B.rows());

    // Space fillers might be needed if the matrices have different row dimensions
    std::string A_fill(A.cols(), ' ');
    std::string B_fill(B.cols(), ' ');

    // Print away ...
    for (std::size_t r = 0; r < nr; ++r) {
        strm << (r < A.rows() ? A.row(r).to_string() : A_fill) << delim
             << (r < B.rows() ? B.row(r).to_string() : B_fill) << "\n";
    }
    return;
}

/// @brief Prints two bit-matrices A, B side-by-side to @c std::cout.
/// @param delim The bit-matrices are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(const bit::matrix<Block, Alloc>& A, const bit::matrix<Block, Alloc>& B, std::string_view delim = "\t")
{
    print(std::cout, A, B, delim);
}

/// @brief Prints three bit-matrices A, B, C side-by-side to @c std::cout.
/// @param delim The bit-matrices are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(const bit::matrix<Block, Alloc>& A, const bit::matrix<Block, Alloc>& B, const bit::matrix<Block, Alloc>& C,
      std::string_view delim = "\t")
{
    print(std::cout, A, B, C, delim);
}

/// @brief Prints three bit-matrices A, B, C side-by-side.
/// @param strm  The bit-matrices appear on this stream
/// @param delim The bit-matrices are separated by this which defaults to a tab.
template<std::unsigned_integral Block, typename Alloc>
void
print(std::ostream& strm, const bit::matrix<Block, Alloc>& A, const bit::matrix<Block, Alloc>& B,
      const bit::matrix<Block, Alloc>& C, std::string_view delim = "\t")
{
    // If any matrix is empty there was likely a bug somewhere!
    bit_debug_assert(!A.empty(), "Matrix A is empty which is likely an error!");
    bit_debug_assert(!B.empty(), "Matrix B is empty which is likely an error!");
    bit_debug_assert(!C.empty(), "Matrix C is empty which is likely an error!");

    // Most rows we ever print
    auto nr = std::max({A.rows(), B.rows(), C.rows()});

    // Space fillers might be needed if the matrices have different row dimensions
    std::string A_fill(A.cols(), ' ');
    std::string B_fill(B.cols(), ' ');
    std::string C_fill(C.cols(), ' ');

    // Print away ...
    for (std::size_t r = 0; r < nr; ++r) {
        strm << (r < A.rows() ? A.row(r).to_string() : A_fill) << delim
             << (r < B.rows() ? B.row(r).to_string() : B_fill) << delim
             << (r < C.rows() ? C.row(r).to_string() : C_fill) << "\n";
    }
    return;
}

/// @brief Usual output stream operator for a bit-matrix.
template<std::unsigned_integral Block, typename Allocator>
std::ostream&
operator<<(std::ostream& s, const matrix<Block, Allocator>& rhs)
{
    return s << rhs.to_pretty_string();
}

/// @brief  Create a bit-matrix by reading bits encoded as a binary or hex string from a stream.
/// @param  s The stream to read from where the bit-matrix is in rows separated by white space, commas, semi-colons.
/// @param  rhs The bit-matrix we overwrite with the new bits.
/// @throws @c std::invalid_argument if the parse fails.
/// @note   Implementation uses a string buffer -- could probably do something more direct/efficient.
template<std::unsigned_integral Block, typename Allocator>
std::istream&
operator>>(std::istream& s, bit::matrix<Block, Allocator>& rhs)
{
    // Get the input string.
    std::string buffer;
    std::getline(s, buffer);

    // Try to parse it as a bit-matrix.
    auto m = bit::matrix<Block, Allocator>::from(buffer);

    // Failure?
    if (!m) throw std::invalid_argument(buffer);

    // All good
    rhs = *m;
    return s;
}

} // namespace bit
