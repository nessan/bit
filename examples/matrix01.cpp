/// @brief Exercise some of the basic functionality for the @c bit::matrix class.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT

// No need for speed here -- always do need bounds checking!
#ifndef BIT_VERIFY
    #define BIT_VERIFY
#endif

#include "common.h"

int
main()
{
    using matrix_type = bit::matrix<uint64_t>;

    std::size_t n = 9;
    std::size_t m = 9;

    matrix_type M(n, m);
    M.description("Default construction");

    M.flip();
    M.description("After M.flip()");

    M.resize(6);
    M.description("After M.resize(6,6)");

    M.resize(10);
    M.description("After M.resize(10,10)");

    M.set_diagonal();
    M.description("After M.set_diagonal()");

    M.set_diagonal(1);
    M.description("After M.set_diagonal(1)");

    M.set_diagonal(-1);
    M.description("After M.set_diagonal(-1)");

    M.clear();
    M.description("After M.clear()");

    M.shrink_to_fit();
    M.description("After M.shrink_to_fit()");

    M.resize(5, 7);
    M.description("After M.resize(5, 7)");

    M = matrix_type::random(7uL, 5uL);
    M.description("A random 7 x 5 bit-matrix");

    M.add_row();
    M.description("After M.add_row()");

    M.add_col();
    M.description("After M.add_col()");

    M.set_if([&M](size_t i, size_t) { return i == M.rows() - 1; });
    M.description("After setting last row with a lambda");

    M.set_if([&M](size_t, size_t j) { return j == M.cols() - 1; });
    M.description("After setting last col with a lambda");

    M.pop_row();
    M.description("After M.pop_row()");

    M.pop_col();
    M.description("After M.pop_col()");

    M.resize(12).reset();
    M.description("After M.resize(12).reset()");

    M.replace(matrix_type::identity(6));
    M.description("After M.replace(bit::matrix<>::identity(6))");

    M.replace(6, matrix_type::identity(6));
    M.description("After M.replace(6, bit::matrix<>::identity(6))");

    auto ur = M.to_vector();
    auto uc = M.to_vector(false);
    ur.description("M as a vector by row");
    uc.description("M as a vector by column");

    matrix_type(ur, 12).description("M = bit::matrix(row-vector, 12)");
    matrix_type(ur, 24).description("M = bit::matrix(row-vector, 24)");
    matrix_type(uc, 12, false).description("M = bit::matrix(column-vector, 12, false)");
    matrix_type(uc, 6, false).description("M = bit::matrix(column-vector, 12, false)");
    matrix_type(ur).description("M = bit::matrix(row-vector) -- a one row bit-matrix");
    matrix_type(ur, false).description("M = bit::matrix(column-vector, false) -- a one column bit-matrix");

    return 0;
}
