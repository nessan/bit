/// @brief Danilevsky's algorithm transforms a square bit-matrix to Frobenius form.
/// @link  https://nessan.github.io/bit
///
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#pragma once

#include "bit_assert.h"
#include "matrix.h"

namespace bit {

/// @brief  Returns the coefficients of characteristic polynomial for a bit-matrix A.
/// @param  A The input bit-matrix which must be square N x N. Left unchanged on output
/// @return A Vector `c` where the characteristic polynomial is `c[N] x^N + c[N-1] x^(N-1) + ... + c[1] x + c[0]`
template<std::unsigned_integral Block, typename Allocator>
vector<Block, Allocator>
characteristic_polynomial(const matrix<Block, Allocator>& A)
{
    // The matrix needs to be non-empty and square
    bit_always_assert(A.is_square(), "Matrix is {} x {} but it needs to be square!", A.rows(), A.cols());

    // Need a working copy of A
    matrix<Block, Allocator> A_copy{A};

    // Get all the compact versions of the companion matrix blocks in the Frobenius form of A
    auto top_rows = compact_frobenius_form(A_copy);
    bit_always_assert(top_rows.size() > 0, "Something went wrong--Frobenius form of A has NO blocks!");

    auto retval = companion_matrix_characteristic_polynomial(top_rows[0]);
    for (std::size_t i = 1; i < top_rows.size(); ++i) {
        auto p = companion_matrix_characteristic_polynomial(top_rows[i]);
        auto q = convolution(retval, p);
        retval = q;
    }
    return retval;
}

/// @brief Returns the characteristic polynomial for a companion bit-matrix where that is passed in top row only form.
/// @return A Vector `c` where the characteristic polynomial is `c[N] x^N + c[N-1] x^(N-1) + ... + c[1] x + c[0]`
template<std::unsigned_integral Block, typename Allocator>
vector<Block, Allocator>
companion_matrix_characteristic_polynomial(const vector<Block, Allocator>& top_row)
{
    // The companion matrix is N x N
    std::size_t N = top_row.size();

    // It has an O(N) characteristic polynomial so N+1 polynomial coefficients
    vector<Block, Allocator> c(N + 1);

    // The coefficient for x^N in the characteristic polynomial is 1 and the top row now has all the other
    // coefficients ordered as A[0,0] x^(N-1) + A[0,1] x^(N-1) + ... + A[0,N-1] so we reverse their order per
    // this function's description.
    for (std::size_t j = 0; j < N; ++j) c[N - 1 - j] = top_row(j);
    c[N] = 1;

    return c;
}

/// @brief Returns the companion bit-matrix blocks that make up the Frobenius form of an input bit-matrix A.
/// @note We return the top rows only. The matrix A is untouched by this function.
template<std::unsigned_integral Block, typename Allocator>
std::vector<vector<Block, Allocator>>
compact_frobenius_form(const matrix<Block, Allocator>& A)
{
    // The matrix needs to be non-empty and square
    bit_always_assert(A.is_square(), "Matrix is {} x {} but it should be square!", A.rows(), A.cols());

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

/// @brief Returns the bottom right companion bit-matrix block using Danilevsky's algorithm.
/// @param A The input bit-matrix which must be square N x N. A is altered by this function!
/// @param n We actually only consider `A.sub(n)` where by default n = N. Idea is to be able to call this method on
/// successively smaller top left sub-matrices as we fill in more and more of the bottom right companion matrix blocks.
/// @return  We return the bottom right companion bit-matrix block of `A.sub(n)` in top-row only format.
template<std::unsigned_integral Block, typename Allocator>
vector<Block, Allocator>
danilevsky(matrix<Block, Allocator>& A, std::size_t n = 0)
{
    // The bit-matrix needs to be non-empty and square
    bit_always_assert(A.is_square(), "Matrix is {} x {} but it should be square!", A.rows(), A.cols());

    // Default case is to work on all of A
    std::size_t N = A.rows();
    if (n == 0) n = N;

    // If we were asked to look at a specific sub-matrix it better fit.
    bit_always_assert(n <= N, "Asked to look at {} rows but matrix has only {} of them!", n, N);

    // Handle a trivial case.
    if (n == 1) {
        vector<Block, Allocator> retval(1);
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
    vector<Block, Allocator> top_row(n - k);
    for (std::size_t j = 0; j < n - k; ++j) top_row[j] = A(k, k + j);
    return top_row;
}

} // namespace bit