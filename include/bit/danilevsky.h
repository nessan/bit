/// @brief Danilevsky's algorithm transforms a square bit-matrix to Frobenius form.
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "verify.h"
#include "matrix.h"
#include "polynomial.h"

namespace bit {

/// @brief  Returns the characteristic polynomial for a bit-matrix @c A.
/// @param  A The input bit-matrix which must be square N x N. Left unchanged on output
template<std::unsigned_integral Block, typename Allocator>
polynomial<Block, Allocator>
characteristic_polynomial(const matrix<Block, Allocator>& A)
{
    // Matrix needs to be non-empty and square.
    bit_verify(A.is_square(), "Matrix is {} x {} but it needs to be square!", A.rows(), A.cols());

    // Make working copy of A.
    matrix<Block, Allocator> A_copy{A};

    // Get the Frobenius form of A as a vector of companion matrices (matrix gets destroyed doing that so use a copy).
    auto companion_matrices = compact_frobenius_form(A_copy);
    bit_verify(companion_matrices.size() > 0, "Something went wrong--the Frobenius form of A is empty!");

    auto retval = companion_matrix_characteristic_polynomial(companion_matrices[0]);
    for (std::size_t i = 1; i < companion_matrices.size(); ++i)
        retval *= companion_matrix_characteristic_polynomial(companion_matrices[i]);

    return retval;
}

/// @brief Returns the characteristic polynomial for a companion bit-matrix.
/// @param top_row The companion matrix is passed in compact top-row only form (i.e. as a bit-vector)
template<std::unsigned_integral Block, typename Allocator>
polynomial<Block, Allocator>
companion_matrix_characteristic_polynomial(const vector<Block, Allocator>& top_row)
{
    using vector_type = vector<Block, Allocator>;
    using polynomial_type = polynomial<Block, Allocator>;

    // The companion matrix is N x N so the characteristic polynomial will be order N with N+1 coefficients.
    std::size_t N = top_row.size();
    vector_type c{N + 1};

    // The characteristic polynomial is x^N + A[0,0] x^(N-1) + A[0,1] x^(N-1) + ... + A[0,N-1].
    for (std::size_t j = 0; j < N; ++j) c[N - 1 - j] = top_row(j);
    c[N] = 1;

    // Move those coefficients into a bit-polynomial.
    return polynomial_type{std::move(c)};
}

/// @brief   Returns the companion bit-matrix blocks that make up the Frobenius form of an input bit-matrix @c A.
/// @param A The input bit-matrix which must be square -- it is unchanged on output.
/// @return  Companion matrices are in top-row form so each one is a bit-vector as we return a @c std::vector of those.
template<std::unsigned_integral Block, typename Allocator>
std::vector<vector<Block, Allocator>>
compact_frobenius_form(const matrix<Block, Allocator>& A)
{
    // The matrix needs to be non-empty and square
    bit_verify(A.is_square(), "Matrix is {} x {} but it should be square!", A.rows(), A.cols());

    // Our return value is a vector of top rows for one or more companion matrices
    std::vector<vector<Block, Allocator>> retval;

    // Need a working copy of A
    matrix<Block, Allocator> A_copy{A};

    // Work through all the companion matrix blocks from the bottom right up in the Frobenius form af A
    auto n = A_copy.rows();
    while (n > 0) {
        auto c = danilevsky(A_copy, n);
        retval.push_back(c);
        n -= c.size();
    }
    return retval;
}

/// @brief   Uses Danilevsky's algorithm to compute the the bottom right @e companion matrix for a square matrix @c A
/// @param A The input bit-matrix which must be square N x N -- the argument is @b altered by this function!
/// @param n Rather than all of @c A we only consider @c A.sub(n,n) where by default @c n=N -- mostly for internal use.
///          Setting @c n<N let's us use function iteratively on successively smaller top left sub-matrices as we fill
///          in more and more of the bottom right companion matrix blocks.
/// @return  We return the bottom right companion bit-matrix block of `A.sub(n)` in @e top-row only format (a vector).
template<std::unsigned_integral Block, typename Allocator>
vector<Block, Allocator>
danilevsky(matrix<Block, Allocator>& A, std::size_t n = 0)
{
    // The bit-matrix needs to be non-empty and square
    bit_verify(A.is_square(), "Matrix is {} x {} but it should be square!", A.rows(), A.cols());

    // Default case is to work on all of A
    std::size_t N = A.rows();
    if (n == 0) n = N;

    // If we were asked to look at a specific sub-matrix it better fit.
    bit_verify(n <= N, "Asked to look at {} rows but matrix has only {} of them!", n, N);

    // Handle an edge case.
    if (n == 1) {
        vector<Block, Allocator> retval{1};
        retval[0] = A(0, 0);
        return retval;
    }

    // In step k of the algorithm we attempt to reduce row k to companion-matrix form via a similarity transform. By
    // construction the rows from k+1 to n-1 will already be there (all zero except for sub-diagonal).
    std::size_t k = n - 1;
    while (k > 0) {

        // If row k's sub-diagonal is 0 try to find an earlier column with a non-zero element & swap that column
        // here. If found we also swap the rows with the same indices to maintain the similarity balance.
        if (A(k, k - 1) == 0) {
            for (std::size_t j = 0; j < k - 1; ++j) {
                if (A(k, j)) {
                    A.swap_cols(j, k - 1);
                    A.swap_rows(j, k - 1);
                    break;
                }
            }
        }

        // No joy? Then we have a companion-matrix in the lowest left block and can return that.
        if (A(k, k - 1) == 0) break;

        // Otherwise the sub-diagonal is not zero proceed with the transformation step Inverse[M] x A x M
        // M is identity except for M.row(k-1) = A.row(k) (that one non-trivial row is captured here as m).
        auto m = A.row(k);

        // Inverse[M] == M which is sparse so we can compute A <- Inverse[M] x A by altering a few elements of A.
        for (std::size_t j = 0; j < n; ++j) A(k - 1, j) = dot(m, A.col(j));

        // Similarly we efficiently compute A <- A x M again using the sparsity structure of M
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                bool tmp = A(i, k - 1) && m[j];
                A(i, j) = j != k - 1 ? A(i, j) ^ tmp : tmp;
            }
        }

        // We can now put row k into companion-matrix form (rows below k are already in that form by now).
        A.row(k).reset();
        A(k, k - 1) = 1;

        // Done with row k
        k--;
    }

    // Either k = 0 or the bit-matrix has a non-removable zero on the sub-diagonal of row k. Either way the bottom
    // right (n-k) x (n-k) sub-bit-matrix block A(k,k) through A(n-1,n-1) is in companion-matrix form.
    // We return that in top-row only format.
    vector<Block, Allocator> top_row{n - k};
    for (std::size_t j = 0; j < n - k; ++j) top_row[j] = A(k, k + j);
    return top_row;
}

} // namespace bit