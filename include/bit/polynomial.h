/// @brief Polynomials over GF(2).
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "vector.h"
#include "matrix.h"

namespace bit {

/// @brief  A class for polynomials over GF(2), the space of two elements {0,1} with arithmetic done mod 2.
/// @tparam Block We store the polynomial coefficients as a bit-vector that uses this block type.
/// @tparam Allocator The memory manager that is used to allocate/deallocate space for the blocks as needed.
template<std::unsigned_integral Block = std::uint64_t, typename Allocator = std::allocator<Block>>
class polynomial {
public:
    /// @brief The polynomial coefficients are stored in a bit-vector of this type.
    using vector_type = bit::vector<Block, Allocator>;

    /// @brief A bit::polynomial is a function of either a simple bool or a square bit-matrix argument of this type.
    using matrix_type = bit::matrix<Block, Allocator>;

    /// @brief Special value used to indicate "no degree" for the zero polynomial.
    static constexpr std::size_t ndeg = vector_type::npos;

    /// @brief Construct a zero polynomial with n coefficients: p_0, ..., p_{n-1}
    ///        The polynomial is p(x) := p_0 + p_1*x + ... + p_{n-1} x^{n-1}  with all coefficients set to 0.
    /// @note  The default constructor creates the empty polynomial (also treated as p(x) := 0).
    constexpr explicit polynomial(std::size_t n = 0) : m_coeffs{n} {}

    /// @brief Construct a polynomial by *copying* a bit-vector of coefficients.
    constexpr explicit polynomial(const vector_type& coeffs) : m_coeffs{coeffs} { update(); }

    /// @brief  Construct a polynomial by *moving* a bit-vector of coefficients in place.
    /// @note   Use @c std::move(coeffs) in the constructor argument to get this version.
    /// @return After construction the argument bit-vector is no longer valid.
    constexpr polynomial(vector_type&& coeffs) : m_coeffs{std::move(coeffs)}
    {
        // "Invalidate" the arg & update our own cache.
        coeffs.clear();
        update();
    }

    /// @brief Construct a polynomial by calling @c f(i) for @c i=0,...,n-1 to set the coefficients.
    explicit constexpr polynomial(std::size_t n, std::invocable<std::size_t> auto f) : m_coeffs{n, f} { update(); }

    /// @brief  Factory method to generate the polynomial p(x) = x^n.
    static polynomial power(std::size_t n)
    {
        // Set up a polynomial with n + 1 zero coefficients and set the last one to 1.
        polynomial retval{n + 1};
        retval.m_coeffs[n] = 1;
        retval.m_degree = n;
        return retval;
    }

    /// @brief  Factory method to generate a random bit-polynomial of @b degree n.
    /// @param  n The desired polynomial degree.
    /// @param  p The probability that any randomly chosen coefficient is 1 (defaults to 50-50).
    /// @return If n > 0, the coefficient of x^n will be 1, others are determined based on independent coin flips.
    static polynomial random(std::size_t n, double p = 0.5)
    {
        // Polynomial of degree n has n+1 coefficients which we generate at random based on probability p.
        auto coeffs = vector_type::random(n + 1, p);

        // If n > 0 we want the coefficient of x^n to be 1 for sure. If n == 0 then our random 0/1 is fine.
        if (n > 0) coeffs[n] = 1;

        // Use the move-the-coefficients polynomial constructor which will also update the cached data members.
        return polynomial{std::move(coeffs)};
    }

    /// @brief How many coefficients are there in the polynomial?
    constexpr std::size_t size() const { return m_coeffs.size(); }

    /// @brief Returns true if the polynomial has no coefficients (another form of the zero polynomial).
    constexpr bool empty() const { return m_coeffs.empty(); }

    /// @brief How many coefficients might the polynomial have with causing a memory allocation?
    constexpr std::size_t capacity() const { return m_coeffs.capacity(); }

    /// @brief Resize the polynomial so it has n coefficients: p_0, p_1, ..., p_{n-1}.
    /// @note  If n > size() then the added coefficients are all zeros so do not change the polynomial degree.
    ///        On the other hand if n < size() we drop terms in the polynomial which may lower its degree.
    constexpr polynomial& resize(std::size_t n)
    {
        // Edge case.
        if (size() == n) return *this;

        // Clearing back to the zero polynomial?
        if (n == 0) return clear();

        // Note that any added coefficients are automatically zeros & do not change the polynomial's degree.
        m_coeffs.resize(n);

        // Polynomial now has degree at most n - 1 so perhaps we need to update the cached polynomial degree value.
        if (m_degree > n - 1) update();
        return *this;
    }

    /// @brief Turn this polynomial into the zero polynomial.
    /// @note  This does not release any used memory.
    constexpr polynomial& clear()
    {
        m_coeffs.clear();
        m_degree = ndeg;
        return *this;
    }

    /// @brief Returns true if this is the polynomial P(x) := 0.
    constexpr bool zero() const { return m_degree == ndeg; }

    /// @brief Returns true if this polynomial is not the zero polynomial.
    constexpr bool nonzero() const { return m_degree != ndeg; }

    /// @brief Returns true if this is the polynomial P(x) := 1.
    constexpr bool one() const { return m_degree == 0; }

    /// @brief Returns true if the polynomial is either P(x) := 0 or 1.
    constexpr bool constant() const { return zero() || one(); }

    /// @brief Returns the @b degree of the polynomial.
    constexpr std::size_t degree() const { return m_degree; }

    /// @brief Returns true if the polynomial is *monic* (i.e. no trailing zero coefficients).
    /// @note  Zero polynomials are not monic.
    constexpr bool monic() const { return nonzero() && m_coeffs.size() == m_degree + 1; }

    /// @brief Kills any trailing zero coefficients, so e.g., 1 + x^4 + 0*x^5 + 0*x^6 -> 1 + x^4.
    /// @note  This does nothing to any form of the zero polynomial.
    constexpr polynomial& make_monic()
    {
        if (nonzero()) m_coeffs.resize(m_degree + 1);
        return *this;
    }

    /// @brief Try to minimize the storage used by this polynomial -- may do nothing.
    constexpr polynomial& shrink_to_fit()
    {
        make_monic();
        m_coeffs.shrink_to_fit();
        return *this;
    }

    /// @brief Returns the number of 1-coefficients in the polynomial.
    constexpr std::size_t count1() const
    {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < monic_blocks(); ++i)
            sum += static_cast<std::size_t>(std::popcount(m_coeffs.block(i)));
        return sum;
    }

    /// @brief Returns the number of 0-coefficients in the polynomial.
    constexpr std::size_t count0() const { return size() - count1(); }

    /// @brief Write access to an individual polynomial coefficient is through this `polynomial::reference` proxy class.
    class reference {
    public:
        constexpr reference(polynomial& p, std::size_t i) : m_poly{p}, m_index{i} {}

        // The default compiler generator implementations for most "rule of 5" methods will be fine ...
        constexpr reference(const reference&) = default;
        constexpr reference(reference&&) noexcept = default;
        ~reference() = default;

        // Cannot take a reference to a reference ...
        constexpr void operator&() = delete;

        // Explicitly convert the reference to a bool.
        constexpr bool to_bool() const { return m_poly.get(m_index); }

        // You can always use the reference as bool.
        constexpr operator bool() const { return to_bool(); }

        // Set the underlying polynomial coefficient to a specific boolean value.
        constexpr reference& set_to(bool rhs)
        {
            m_poly.set(m_index, rhs);
            return *this;
        }

        // Have some custom operator='s
        constexpr reference& operator=(const reference& rhs) { return set_to(rhs.to_bool()); }
        constexpr reference& operator=(reference&& rhs) { return set_to(rhs.to_bool()); }
        constexpr reference& operator=(bool rhs) { return set_to(rhs); }

        // A few bit operations such as p[i].set() etc.
        constexpr reference& set() { return set_to(true); }
        constexpr reference& reset() { return set_to(false); }
        constexpr reference& flip() { return set_to(!to_bool()); }

    private:
        polynomial& m_poly;  // The polynomial of interest.
        std::size_t m_index; // The index of the polynomial coefficient of interest
    };

    /// @brief Read-only access to the i'th polynomial coefficient (synonym for the @c get method).
    constexpr bool operator[](std::size_t i) const { return get(i); }

    /// @brief Read-write access to the i'th polynomial coefficient using a @c polynomial::reference
    constexpr auto operator[](std::size_t i) { return reference{*this, i}; }

    /// @brief Read-only access to the i'th polynomial coefficient.
    constexpr bool get(std::size_t i) const
    {
        bit_debug_assert(i < size(), "Trying to access p_{} but coefficients stop at p_{}.", i, size() - 1);
        return m_coeffs.element(i);
    }

    /// @brief Set the i'th polynomial coefficient to 1 to the given value and return a reference to this.
    /// @note  The default value is 1 -- see also the @c reset(i) method.
    /// @note  You can a "natural" version of this method using the @c operator[](i) method.
    constexpr polynomial& set(std::size_t i, bool val = true)
    {
        bit_debug_assert(i < size(), "Trying to set p_{} to {} but coefficients stop at p_{}.", i, val, size() - 1);
        if (val) {
            m_coeffs.set(i);
            if (i > m_degree || m_degree == ndeg) m_degree = i;
        }
        else {
            m_coeffs.reset(i);
            if (i == m_degree) update();
        }
        return *this;
    }

    /// @brief Reset the i'th polynomial coefficient to 0 and return a reference to this.
    constexpr polynomial& reset(std::size_t i) { return set(i, false); }

    /// @brief Set all the polynomial coefficients to 1 and return a reference to this.
    constexpr polynomial& set()
    {
        m_coeffs.set();
        m_degree = m_coeffs.size() - 1;
        return *this;
    }

    /// @brief Reset all the polynomial coefficients to 0 and return a reference to this.
    /// @note  This does not change the polynomial's @c size()
    constexpr polynomial& reset()
    {
        m_coeffs.reset();
        m_degree = 0;
        return *this;
    }

    /// @brief Read-only access to all the polynomial coefficients.
    constexpr const vector_type& coefficients() const { return m_coeffs; }

    /// @brief Set the polynomial coefficients by copying them from a pre-filled bit-vector.
    constexpr polynomial& set_coefficients(const vector_type& coeffs)
    {
        m_coeffs = coeffs;
        update();
        return *this;
    }

    /// @brief  Set the polynomial coefficients by *moving* a bit-vector of coefficients into place.
    /// @note   Use @c std::move(coeffs) in the argument to get this version of @c set_coefficients.
    /// @return After the call the argument bit-vector is no longer valid.
    constexpr polynomial& set_coefficients(vector_type&& coeffs)
    {
        // Move over the argument vector and "invalidate" it. Then update our own cache.
        m_coeffs = std::move(coeffs);
        coeffs.m_size = 0;
        update();
        return *this;
    }

    /// @brief Returns the value of the polynomial evaluated for the boolean value @c x
    constexpr bool operator()(bool x) const
    {
        // Edge case
        if (zero()) return 0;

        // p(0):
        if (x == 0) return m_coeffs[0];

        // p(1):
        Block sum = 0;
        for (std::size_t b = 0; b < monic_blocks(); ++b) sum ^= m_coeffs.block(b);
        return std::popcount(sum) % 2;
    }

    /// @brief Returns the value of the polynomial evaluated for the input square bit-matrix @c M
    constexpr matrix_type operator()(const matrix_type& M) const
    {
        // The bit-matrix argument must be square.
        bit_assert(M.is_square(), "Matrix must be square -- not {} x {}!", M.rows(), M.cols());

        // The returned bit-matrix will be N x N.
        auto N = M.rows();

        // Edge case: If the polynomial is zero then the return value is the N x N zero matrix.
        if (zero()) return matrix_type{N, N};

        // Otherwise we start with the polynomial sum being the N x N identity matrix.
        auto retval = matrix_type::identity(N);

        // Work backwards a la Horner ...
        auto n = degree();
        while (n > 0) {

            // Always multiply ...
            retval = dot(M, retval);

            // Add the identity to the sum if the corresponding polynomial coefficient is 1.
            if (m_coeffs[n - 1]) retval.add_identity();

            // And count down ...
            n--;
        }

        return retval;
    }

    /// @brief Adds another polynomial to this one and returns a reference to the result.
    constexpr polynomial& operator+=(const polynomial& rhs)
    {
        // Edge case.
        if (rhs.zero()) return *this;

        // Another edge case.
        if (zero()) {
            *this = rhs;
            return *this;
        }

        // Perhaps we need to get bigger to accommodate the rhs? Note that added coefficients are zeros,
        if (m_coeffs.size() < rhs.m_degree + 1) m_coeffs.resize(rhs.m_degree + 1);

        // Add in the active rhs blocks.
        for (std::size_t b = 0; b < rhs.monic_blocks(); ++b) m_coeffs.block(b) ^= rhs.m_coeffs.block(b);

        // Perhaps we need to update our cached degree data value? Not necessary if rhs was of lower degree.
        if (rhs.m_degree > m_degree)
            m_degree = rhs.m_degree;
        else if (rhs.m_degree == m_degree)
            update();

        return *this;
    }

    /// @brief Subtracts another polynomial from this one and returns a reference to the result.
    constexpr polynomial& operator-=(const polynomial& rhs)
    {
        // In GF(2) subtraction is the same as addition.
        return operator+=(rhs);
    }

    /// @brief Multiplies this by another polynomial and returns a reference to the result.
    constexpr polynomial& operator*=(const polynomial& rhs)
    {
        // Edge cases
        if (zero()) return *this;
        if (rhs.zero()) return reset();
        if (rhs.one()) return *this;
        if (one()) {
            *this = rhs;
            return *this;
        }

        // Generally we pass the work to the convolution method for bit-vectors.
        *this = polynomial{std::move(convolution(m_coeffs, rhs.m_coeffs))};
        return *this;
    }

    /// @brief Multiplies this polynomial by x^n where n defaults to 1 and returns a reference to this.
    /// @note  This will be faster than using the multiplication operator.
    constexpr polynomial& times_x(std::size_t n = 1)
    {
        // Edge cases.
        if (n == 0 || zero()) return *this;

        // If necessary: add space for higher order polynomial coefficients.
        auto new_degree = m_degree + n;
        auto new_size = new_degree + 1;
        if (m_coeffs.size() < new_size) m_coeffs.resize(new_size);
        m_coeffs >>= n;
        m_degree = new_degree;
        return *this;
    }

    /// @brief Returns a new polynomial that is the square of this one.
    /// @note  This will be faster than using the multiplication operator.
    constexpr polynomial squared() const
    {
        // Pass the heavy lifting to the next method below.
        polynomial retval;
        squared(retval);
        return retval;
    }

    /// @brief Fills a destination polynomial with coefficients that make it the square of this one.
    /// @note  This version can be used for repeated squaring where we want to reuse the @c dst storage.
    constexpr void squared(polynomial& dst) const
    {
        // Edge case: The square of p(x) := 0 or 1 is just the same thing.
        if (constant()) {
            dst = *this;
            return;
        }

        // In GF(2) if p(x) = a + b*x + c*x^2 + d*x^3 + ... then p(x)^2 = a^2 + b^2*x^2 + c^2*x^4 + d*x^6 + ...
        // bit::vector has a fast riffled() method that interleaves the extra zero coefficients in the right spots.
        m_coeffs.riffled(dst.m_coeffs);
        dst.m_degree = 2 * m_degree;
    }

    /// @brief  Create a new polynomial as a distinct sub-polynomial of this one.
    /// @param  n The number of coefficients to copy.
    constexpr polynomial sub(std::size_t n) const { return polynomial{m_coeffs.sub(0, n)}; }

    /// @brief Split this polynomial p(x) into two pieces lo(x) and hi(x) so that p(x) = lo(x) + x^n * hi(x).
    /// @param n On return the lo(x) polynomial will have degree that is less than n.
    /// @param lo, hi On return these two polynomials will be such that p(x) = lo(x) + x^n * hi(x).
    constexpr void split(std::size_t n, polynomial& lo, polynomial& hi)
    {
        // Use the bit-vector's split(...) method to copy the coefficients to lo & hi.
        // This automatically puts hi's coefficients n slots "down" so lo(x) + x^n*hi(x) = this polynomial.
        m_coeffs.split(n, lo.m_coeffs, hi.m_coeffs);
        lo.update();
        hi.update();
    }

    /// @brief Check for equality between two bit-polynomials.
    constexpr bool friend operator==(const polynomial& lhs, const polynomial& rhs)
    {
        // Edge case.
        if (&lhs == &rhs) return true;

        // Check the active blocks for equality.
        auto m_blocks = lhs.monic_blocks();
        if (rhs.monic_blocks() != m_blocks) return false;
        for (std::size_t b = 0; b < m_blocks; ++b)
            if (lhs.m_coeffs.block(b) != rhs.m_coeffs.block(b)) return false;

        // Made it
        return true;
    }

    /// @brief   Returns a string representation of the polynomial.
    /// @param x By default we print the polynomial in terms of "x" -- you can override that by setting @c x
    /// @returns Default representation for coefficient vector [1,0,1,0,0] is "1 + x^2"
    std::string to_string(std::string_view x = "x") const
    {
        // Edge case.
        if (zero()) return "0";

        // Otherwise we construct the string ...
        std::ostringstream ss;

        // All terms other than the first one are preceded by a "+"
        bool first_term = true;
        for (std::size_t i = 0; i <= degree(); ++i) {
            if (m_coeffs[i]) {
                if (i == 0) { ss << "1"; }
                else {
                    if (!first_term) ss << " + ";
                    ss << x << "^" << i;
                }
                first_term = false;
            }
        }
        return ss.str();
    }

    /// @brief If we are P(x) this method returns x^e mod P(x) where e = N or 2^N for some unsigned integer N.
    /// @param N We are either interested in reducing x^N or possibly x^(2^N).
    /// @param N_is_exponent If true e = 2^N which allows for huge powers of x like x^(2^100).
    /// @note  We use repeated squaring/multiplication which is much faster than other methods for larger N.
    polynomial reduce(std::size_t N, bool N_is_exponent = false) const
    {
        // Error check: ... mod 0 is not defined.
        if (zero()) throw std::invalid_argument("... mod P(x) is not defined for P(x) := 0.");

        // Edge case: Anything mod 1 = 0.
        if (one()) return polynomial{};

        // Polynomial is non-zero and can be written as P(x) = x^n + P_{n-1} x^{n-1} + ... P_0 where:
        auto n = degree();

        // P(x) can be written as x^n + p(x) where n > 0 and degree[p] < n: p(x) = p_0 + ... + p_{n-1}*x^{n-1}.
        // The algorithm works on the coefficient bit-vectors which we conflate in the comments with their polynomial.
        auto p = m_coeffs.sub(0, n);

        // Return value r(x) := x^e mod P(x) has degree < n: r(x) = r_0 + r_1 x + ... + r_{n-1} x^{n-1}
        vector_type r{n};

        // lambda: If degree[q] < n, this performs: q(x) <- x*q(x) mod P(x).
        auto times_x_step = [&](auto& q) {
            bool add_p = q[n - 1];
            q >>= 1;
            if (add_p) q ^= p;
        };

        // We precompute x^{n + i} mod P(x) for i = 0, ..., n-1 starting from x^n mod P(x) = p.
        std::vector<vector_type> power_mod(n, vector_type{n});
        power_mod[0] = p;
        for (std::size_t i = 1; i < n; ++i) {
            power_mod[i] = power_mod[i - 1];
            times_x_step(power_mod[i]);
        }

        // Some workspace we use/reuse below in order to minimize allocations/deallocations.
        vector_type s{2 * n}, h{n};

        // lambda: If degree[q] < n, this performs: q(x) <- q(x)^2 mod P(x).
        auto square_step = [&](auto& q) {
            // Square q(x) storing the result in workspace `s`.
            q.riffled(s);

            // Split s(x) = l(x) + x^n * h(x) where l(x) & h(x) are both of degree less than n.
            // We reuse q(x) for l(x).
            s.split(n, q, h);

            // s(x) = q(x) + h(x) so s(x) mod P(x) = q(x) + h(x) mod P(x) which we handle term by term.
            // If h(x) != 0 then at most every second term in h(x) is 1 (nature of polynomial squares in GF(2)).
            auto h_first = h.first_set();
            if (h_first != vector_type::npos) {
                auto h_final = h.final_set();
                for (std::size_t i = h_first; i <= h_final; i += 2)
                    if (h[i]) q ^= power_mod[i];
            }
        };

        // Case e = 2^N: Do N square_step calls to to compute x^(2^N) mod P(x) == (x^2)^N mod P(x).
        if (N_is_exponent) {

            // Edge case: N = 0 => x^(2^N) = x: If n == 1 then P(x) = x + p_0 and x^(2^N) mod P(x) = p_0
            if (N == 0 && n == 1) {
                r[0] = p[0];
                return polynomial{r};
            }

            // General case: Start with r(x) = x mod P(x) -> x^2 mod P(x) -> x^4 mod P(x) ...
            r[1] = 1;
            for (std::size_t i = 0; i < N; ++i) square_step(r);
            return polynomial{r};
        }

        // Case e = N < n: Then x^N mod P(x) = x^N
        if (N < n) {
            r[N] = 1;
            return polynomial{r};
        }

        // Case e = N = n: Then x^N mod P(x) = p(x).
        if (N == n) return polynomial{p};

        // Case e = N > n: We use a square & multiply algorithm:

        // Note that if e.g. N = 0b00010111 then std::bit_floor(N) = 0b00010000.
        std::size_t N_bit = std::bit_floor(N);

        // Start with r(x) = x mod P(x) which takes care of the most significant binary digit in n.
        // TODO: We could start a bit further along with a higher power r(x) := x^? mod P(x) where ? < n but > 1.
        r[1] = 1;
        N_bit >>= 1;

        // And off we go ...
        while (N_bit) {

            // Always do a square step and then a times_x step if necessary (i.e. if current bit in N is set).
            square_step(r);
            if (N & N_bit) times_x_step(r);

            // On to the next bit position in n.
            N_bit >>= 1;
        }

        return polynomial{r};
    }

private:
    // The bit-polynomial coefficients are stored in a bit-vector which might be all zeros.
    // Profiling suggests it is advantageous to precompute and cache the degree of the polynomial.
    // The downside of caching is that makes it slightly more complicated to set/alter the polynomial coefficients.
    vector_type m_coeffs;
    std::size_t m_degree = ndeg;

    /// @brief Forces an update of our cached data values.
    constexpr void update() { m_degree = m_coeffs.final_set(); }

    /// @brief Returns the number of "active" blocks in the underlying data store.
    constexpr std::size_t monic_blocks() const
    {
        return m_degree != ndeg ? vector_type::block_index_for(m_degree) + 1 : 0;
    }
};

// --------------------------------------------------------------------------------------------------------------------
// NON-MEMBER FUNCTIONS ...
// --------------------------------------------------------------------------------------------------------------------

/// @brief Returns a polynomial that is p(x) times x^n where n defaults to 1
template<std::unsigned_integral Block, typename Allocator>
constexpr polynomial<Block, Allocator>
times_x(const polynomial<Block, Allocator>& p, std::size_t n = 1)
{
    polynomial<Block, Allocator> retval{p};
    retval.times_x(n);
    return retval;
}

/// @brief Add two bit-polynomials to get a new one.
template<std::unsigned_integral Block, typename Allocator>
constexpr polynomial<Block, Allocator>
operator+(const polynomial<Block, Allocator>& lhs, const polynomial<Block, Allocator>& rhs)
{
    // Avoid unnecessary resizing by adding the smaller degree polynomial to the larger one ...
    if (lhs.degree() >= rhs.degree()) {
        polynomial<Block, Allocator> retval{lhs};
        retval += rhs;
        return retval;
    }
    else {
        polynomial<Block, Allocator> retval{rhs};
        retval += lhs;
        return retval;
    }
}

/// @brief Subtract two bit-polynomials to get a new one.
template<std::unsigned_integral Block, typename Allocator>
constexpr polynomial<Block, Allocator>
operator-(const polynomial<Block, Allocator>& lhs, const polynomial<Block, Allocator>& rhs)
{
    // Subtraction is identical to addition in GF(2).
    return operator+(lhs, rhs);
}

/// @brief Multiply two bit-polynomials to get a new one.
template<std::unsigned_integral Block, typename Allocator>
constexpr polynomial<Block, Allocator>
operator*(const polynomial<Block, Allocator>& lhs, const polynomial<Block, Allocator>& rhs)
{
    // Relay the problem to the bit-vector convolution function.
    return polynomial<Block, Allocator>{bit::convolution(lhs.coefficients(), rhs.coefficients())};
}

/// @brief The usual output stream operator for a bit::polynomial
template<std::unsigned_integral Block, typename Allocator>
std::ostream&
operator<<(std::ostream& s, const polynomial<Block, Allocator>& rhs)
{
    return s << rhs.to_string();
}

} // namespace bit

// --------------------------------------------------------------------------------------------------------------------
// Connect bit-polynomials to std::format & friends (not in the `bit` namespace).
// --------------------------------------------------------------------------------------------------------------------

/// @brief Connect bit-polynomials to std::format & friends by specializing the @c std:formatter struct.
/// @note  The default variable is 'x' but use specifier {:y} e.g. to get 'y' as the variable.
template<std::unsigned_integral Block, typename Allocator>
struct std::formatter<bit::polynomial<Block, Allocator>> {

    /// @brief Parse a bit-polynomial format specifier for a variable name (default is "x").
    /// @note  So @c std::format("{:mat}",p) will result in p0* + p1*mat + p2*mat^2 etc.
    constexpr auto parse(const std::format_parse_context& ctx)
    {
        auto it = ctx.begin();
        if (*it != '}') m_var.clear();
        while (it != ctx.end() && *it != '}') {
            m_var += *it;
            ++it;
        }
        return it;
    }

    /// @brief Push out a formatted bit-polynomial using the @c to_string(...) method in the class.
    template<class FormatContext>
    auto format(const bit::polynomial<Block, Allocator>& rhs, FormatContext& ctx) const
    {
        // Default
        return std::format_to(ctx.out(), "{}", rhs.to_string(m_var));
    }

    std::string m_var = "x";
};
