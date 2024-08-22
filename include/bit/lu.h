/// @brief An LU decomposition object for a square bit-matrix A
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "verify.h"
#include "matrix.h"

#include <optional>

namespace bit {

template<std::unsigned_integral Block, typename Allocator>
class lu {
public:
    using vector_type = vector<Block, Allocator>;
    using matrix_type = matrix<Block, Allocator>;
    using permutation_type = std::vector<std::size_t>;

    explicit lu(const matrix_type& A) : m_lu{A}, m_swap{A.rows()}, m_rank{A.rows()}
    {
        // Only handle square matrices
        bit_verify(A.is_square(), "Matrix is {} x {} but it should be square!", A.rows(), A.cols());

        // Iterate through the bit-matrix.
        std::size_t N = m_lu.rows();
        for (std::size_t j = 0; j < N; ++j) {

            // Initialize this element of the  row swap instruction vector.
            m_swap[j] = j;

            // Find a non-zero element on or below the diagonal in column -- a pivot.
            std::size_t p = j;
            while (p < N && m_lu(p, j) == 0) ++p;

            // Perhaps no such element exists in this column? If so the matrix is singular!
            if (p == N) {
                m_rank--;
                continue;
            }

            // If necessary, swap the pivot row into place & record the rows swap instruction
            if (p != j) {
                m_lu.swap_rows(p, j);
                m_swap[j] = p;
            }

            // At this point m_lu(j,j) == 1
            for (std::size_t i = j + 1; i < N; ++i) {
                if (m_lu(i, j))
                    for (std::size_t k = j + 1; k < N; ++k) m_lu(i, k) ^= m_lu(j, k);
            }
        }
    }

    /// @brief Read-only access to the rank of the underlying bit-matrix
    constexpr std::size_t rank() const { return m_rank; }

    /// @brief Is the system non-singular? (i.e. was the underlying bit-matrix of full rank?)
    constexpr bool non_singular() const { return m_rank == m_lu.rows(); }

    /// @brief Is the system singular? (i.e. was the underlying bit-matrix no of full rank?)
    constexpr bool singular() const { return !non_singular(); }

    /// @brief Returns the determinant of the underlying matrix
    constexpr bool determinant() const { return non_singular(); }

    /// @brief Read-only access to the LU form of the bit-matrix where A -> [L\U]
    constexpr matrix_type& LU() const { return m_lu; }

    /// @brief Copy of the L (in P.A = L.U) as a square bit-matrix
    constexpr matrix_type L() const { return m_lu.unit_lower(); }

    /// @brief Copy of the U (in P.A = L.U) as a square bit-matrix
    constexpr matrix_type U() const { return m_lu.upper(); }

    /// @brief Read-only access to the row swap instruction vector (the P in P.A = L.U)
    permutation_type row_swaps() const { return m_swap; }

    /// @brief Read-only access to the row swap instruction vector as a permutation vector (the P in P.A = L.U).
    permutation_type permutation_vector() const
    {
        std::size_t      N = m_swap.size();
        permutation_type retval{N};
        for (std::size_t i = 0; i < N; ++i) retval[i] = i;
        for (std::size_t i = 0; i < N; ++i) std::swap(retval[m_swap[i]], retval[i]);
        return retval;
    }

    /// @brief Permute the rows of the input matrix `B` in-place using our row-swap instruction vector
    constexpr void permute(matrix_type& B) const
    {
        std::size_t N = m_swap.size();
        bit_verify(B.rows() == N, "Matrix has {} rows but row-swap instruction vector has {}.", B.rows(), N);
        for (std::size_t i = 0; i < N; ++i) B.swap_rows(i, m_swap[i]);
    }

    /// @brief Permute the rows of the input vector `b` in-place using our row-swap instruction vector
    constexpr void permute(vector_type& b) const
    {
        std::size_t N = m_swap.size();
        bit_verify(b.size() == N, "Vector has {} elements but row-swap instruction vector has {}.", b.size(), N);
        for (std::size_t i = 0; i < N; ++i)
            if (m_swap[i] != i) b.swap_elements(i, m_swap[i]);
    }

    /// @brief Attempts to solve the system A.x = b. Returns `std::nullopt` if the system is singular
    std::optional<vector_type> operator()(const vector_type& b) const
    {
        auto N = m_lu.rows();
        bit_verify(b.size() == N, "RHS b has {} elements but elements but LHS matrix has {} rows.", b.size(), N);

        // Can we solve this at all?
        if (singular()) return std::nullopt;

        // Make a permuted copy of the the right hand side
        auto x = b;
        permute(x);

        // Forward substitution
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < i; ++j)
                if (m_lu(i, j)) x[i] ^= x[j];
        }

        // Backward substitution
        for (std::size_t i = N; i--;) {
            for (std::size_t j = i + 1; j < N; ++j)
                if (m_lu(i, j)) x[i] ^= x[j];
        }

        return x;
    }

    /// @brief Attempts to solve the systems A.x = B. Returns `std::nullopt` if the system is singular.
    /// @param B Each column b of B is an independent right hand side for the system A.x = b
    std::optional<matrix_type> operator()(const matrix_type& B) const
    {
        auto N = m_lu.rows();
        bit_verify(B.rows() == N, "RHS B has {} rows but elements but LHS matrix has {} rows.", B.rows(), N);

        // Can we solve this at all?
        if (singular()) return std::nullopt;

        // Make a permuted copy of the the right hand side
        auto X = B;
        permute(X);

        // Solve for each column
        for (std::size_t j = 0; j < N; ++j) {

            // Forward substitution
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t k = 0; k < i; ++k)
                    if (m_lu(i, k)) X(i, j) ^= X(k, j);
            }

            // Backward substitution
            for (std::size_t i = N; i--;) {
                for (std::size_t k = i + 1; k < N; ++k)
                    if (m_lu(i, k)) X(i, j) ^= X(k, j);
            }
        }

        return X;
    }

    /// @brief Attempts to invert the bit-matrix A. Returns `std::nullopt` if A is singular
    std::optional<matrix_type> invert() const
    {
        // We just solve the system A.A_inv = I for A_inv
        return operator()(matrix_type::identity(m_lu.rows()));
    }

private:
    matrix_type      m_lu;   // The LU decomposition stored in one bit-matrix.
    permutation_type m_swap; // The list of row swap instructions (a permutation in LAPACK format)
    std::size_t      m_rank; // The rank of the underlying matrix
};

} // namespace bit
