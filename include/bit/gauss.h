/// @brief Gauss-Jordan elimination class to solve A.x = b in GF(2)
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "bit_assert.h"
#include "matrix.h"

#include <optional>
namespace bit {

/// @brief  Gaussian elimination solver for systems A.x = b where A is a bit::matrix and b is a bit::vector.
/// @tparam Template parameters are those used by bit::vector and bit::matrix.
template<std::unsigned_integral Block = uint64_t, typename Allocator = std::allocator<Block>>
class gauss {
public:
    using vector_type = vector<Block, Allocator>;
    using matrix_type = matrix<Block, Allocator>;
    using location_type = std::vector<std::size_t>; // Index locations of the free variables

    /// @brief Create a gauss_solver object for the system A.x = b
    /// @param A A The lhs square bit-matrix.
    /// @param b The rhs bit-vector whose size must match the number of rows in A.
    gauss(const matrix_type& A, const vector_type& b)
    {
        // Only handle square matrices.
        bit_always_assert(A.is_square(), "Matrix is {} x {} but it should be square!", A.rows(), A.cols());

        // The number of rows in the lhs matrix and the rhs vector must agree.
        bit_always_assert(A.rows() == b.size(), "A.rows() = {} but b.size() = {}.", A.rows(), b.size());

        // Create the augmented matrix A|b and convert that to reduced row echelon form
        vector_type pivots;
        auto        tmp = join(A, b).to_reduced_echelon_form(&pivots);

        // Final column in the pivots is not needed (corresponds to the "extra" b)
        pivots.pop();

        // Pull out the left hand and right hand sides of the reduced row echelon version of the system A.x = b
        m_lhs = tmp.sub(A.rows());
        m_rhs = tmp.col(A.cols());

        // Rank of A is just the number of pivots.
        m_rank = pivots.count();

        // Any unset index in pivots indicates a free variable (may be none of those of course)
        m_free = pivots.unset_indices();

        // Capture the index location of the first set bit in each row
        m_first.resize(m_lhs.rows());
        for (std::size_t i = 0; i < m_lhs.rows(); ++i) m_first[i] = m_lhs.row(i).first_set();

        // If the system is consistent then there is at least one solution otherwise none.
        if (consistent()) {

            // The number of possible solutions in a consistent system is 2^f where f is the number of free variables.
            std::size_t f = m_free.size();

            // However, the number of "addressable" solutions has to fit into a std::size_t
            std::size_t b_max = std::numeric_limits<std::size_t>::digits - 1;
            std::size_t f_max = std::min(f, b_max);
            m_count = (1ULL << f_max);
        }
        else {
            m_count = 0;
        }
    }

    /// @brief Returns the number of equations in the system
    constexpr std::size_t equation_count() const { return m_lhs.rows(); }

    /// @brief Returns the rank of the underlying left hand side bit-matrix.
    constexpr std::size_t rank() const { return m_rank; }

    /// @brief Is the system consistent? Does it have any solutions?
    constexpr bool is_consistent() const { return m_count > 0; }

    /// @brief Returns the number of "free" variables in the system.
    constexpr std::size_t free_count() const { return m_free.size(); }

    /// @brief Read only access to the index locations we will use for free variables
    location_type free_indices() const { return m_free; }

    /// @brief Read-only access to the reduced echelon form of A
    const matrix_type& lhs() const { return m_lhs; }

    /// @brief Read-only access to the altered b
    const vector_type& rhs() const { return m_rhs; }

    /// @brief Returns the number of solutions we can find for the system A.x = b.
    /// @note If there are f free variables the true number of solutions is 2^f. However, the most solutions we
    /// can address must fit into a `std::size_t` so may be less than that (typically at most 2^63 addressable)
    std::size_t solution_count() const { return m_count; }

    /// @brief Returns a solution for the system of equations A.x = b.
    /// If there are free variables this will be a random choice over any of the possible 2^f variants.
    vector_type operator()() const
    {
        // If we are asked for a solution and there are none then we throw an exception
        bit_always_assert(is_consistent(), "System is inconsistent so has NO solutions");

        // Default all the elements of x to random values.
        auto x = vector_type::random(equation_count());

        // Back-substitution step will over-write any non-free variables with their true values.
        back_substitute(x);
        return x;
    }

    /// @brief Returns a solution for the system of equations A.x = b.
    /// @param ns The number of the solution we want to access.
    /// @note In GF(2) there can be 0 or 2^f solutions where f is the number of free variables. We can address a lot
    /// of those solutions (typically up to min(2^f, 2^63)
    /// @note This function will throw an error is you ask for a solution that doesn't exist.
    /// @note The numbering of the solutions is certainly not unique
    vector_type operator()(std::size_t ns) const
    {
        // If we are asked for a solution and there are none then we throw an exception
        bit_always_assert(ns < m_count, "Argument ns = {} is not less than solution count = {}!", ns, m_count);

        // Our solution will have the free variables set bases on the bit-pattern in ns.
        vector_type x(equation_count());
        for (std::size_t i = 0; i < m_free.size(); ++i) {
            x[m_free[i]] = (ns & 1);
            ns >>= 1;
        }

        // Back-substitution step will over-write any non-free variables with their true values.
        back_substitute(x);
        return x;
    }

private:
    matrix_type   m_lhs;   // This is the reduced echelon form of A (system is A.x = b)
    vector_type   m_rhs;   // This is the equivalent form for b
    location_type m_free;  // Index locations for the non-pivot/free variables
    location_type m_first; // Index locations of the first set bit in each row
    std::size_t   m_rank;  // The rank of the underlying A matrix
    std::size_t   m_count; // The number of solutions we can index/address for A.x = b

    /// @brief Is the echelon system consistent where zero rows on the left are matched with zeros on the right?
    constexpr bool consistent() const
    {
        // Iterate from the bottom up and check the zero rows on the left are matched by zeros on the right
        for (auto i = m_lhs.rows(); i--;) {

            // Once we hit a non-zero row we are all done as in echelon form zero rows can only be at the bottom
            if (m_lhs.row(i).any()) return true;

            // Get here iff the lhs row is all zeros so for consistency it better be matched by a zero in the rhs!
            if (m_rhs[i]) return false;
        }

        // Get to here in the degenerate case that both the lhs and rhs are just all zeros. So x can be anything!
        return true;
    }

    /// @brief Back-substitution step on x. This over-writes any non-free variables with their true values.
    constexpr void back_substitute(vector_type& x) const
    {
        // We start at the first non-zero row of the matrix
        auto n = equation_count();
        for (auto i = m_rank; i--;) {
            // Find the first set bit in this non-zero reduced echelon row
            auto j = m_first[i];

            // Solve for the non-free variable x[j]
            x[j] = m_rhs[i];
            for (auto k = j + 1; k < n; ++k)
                if (m_lhs(i, k)) x[j] ^= x[k];
        }
    }
};

// --------------------------------------------------------------------------------------------------------------------
// NON-MEMBER FUNCTIONS ...
// --------------------------------------------------------------------------------------------------------------------

/// @brief Sometimes all you want to do is to find one solution to A.x = b. This function will do that.
/// Returns `std::nullopt` if the system is not well formed or inconsistent, otherwise an x wrapped in an optional.
template<std::unsigned_integral Block, typename Allocator>
constexpr std::optional<vector<Block, Allocator>>
solve(const matrix<Block, Allocator>& A, const vector<Block, Allocator>& b)
{
    // Only handle square matrices
    if (!A.is_square()) return std::nullopt;

    // The number of rows in the matrix and vector must agree
    if (A.rows() != b.size()) return std::nullopt;

    // With those pre-conditions met we can safely construct our solver without exceptions getting thrown.
    gauss solver(A, b);

    // Perhaps there are no solutions?
    if (solver.solution_count() == 0) return std::nullopt;

    // OK there is at least one solution to return
    return solver();
}

} // namespace bit