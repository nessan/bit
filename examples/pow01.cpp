/// @brief Check that left-to-right bit-matrix power algorithm matches older version.
/// SPDX-FileCopyrightText:  2024 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

namespace bit {

/// @brief Previous version of code to raise a square bit-matrix to any power @c n
template<std::unsigned_integral Block, typename Allocator>
constexpr matrix<Block, Allocator>
old(const matrix<Block, Allocator>& M, std::size_t n)
{
    bit_assert(M.is_square(), "Matrix is {} x {} but it should be square!", M.rows(), M.cols());

    // Note the M^0 is the identity matrix so we start with that
    matrix<Block, Allocator> retval = matrix<Block, Allocator>::identity(M.rows());

    // Make a copy of the input matrix which we will square as needed
    matrix<Block, Allocator> A{M};

    // And we are off ...
    while (n > 0) {

        // At the odd values of n we do a multiply:
        if ((n & 1) == 1) retval = dot(retval, A);

        // Are we done?
        n >>= 1;
        if (n == 0) break;

        // Square the input matrix once again
        A = dot(A, A);
    }
    return retval;
}

} // namespace bit

int
main()
{
    using matrix_type = bit::matrix<std::uint64_t>;

    // A random NxN matrix.
    std::size_t N = 30;
    auto        M0 = matrix_type::random(N, N);

    // Power to raise the matrix to.
    std::size_t p = 623;

    // Number of trials & a tick size
    std::size_t n_trials = 1000;
    std::size_t n_tick = n_trials / 20;

    // And a timer
    utilities::stopwatch sw;

    // Do things the old way:
    matrix_type M1(N, N);
    std::print("Running {} calls of bit::old(M[{}x{}], {}]): ", n_trials, M0.rows(), M0.rows(), p);
    sw.click();
    for (std::size_t n = 0; n < n_trials; ++n) {
        if (n % n_tick == 0) std::cout << '.' << std::flush;
        M1 = bit::old(M0, p);
    }
    sw.click();
    auto lap1 = sw.lap();
    std::print(" done.\n");

    // Do things the new way:
    matrix_type M2(N, N);
    std::print("Running {} calls of bit::pow(M[{}x{}], {}]): ", n_trials, M0.rows(), M0.rows(), p);
    sw.click();
    for (std::size_t n = 0; n < n_trials; ++n) {
        if (n % n_tick == 0) std::cout << '.' << std::flush;
        M2 = bit::pow(M0, p);
    }
    sw.click();
    auto lap2 = sw.lap();
    std::print(" done.\n");

    // Print the timing information ...
    std::print("Old time: {:.2f}s.\n", lap1);
    std::print("New time: {:.2f}s.\n", lap2);
    std::print("Ratio:    {:.2f}.\n", lap2 / lap1);

    // Also check that the final result from the two methods matched.
    if (M1 != M2) {
        std::print("METHOD MISMATCH, METHOD MISMATCH, METHOD MISMATCH, METHOD MISMATCH!\n");
        exit(1);
    }

    // All OK
    std::print("Both methods gave the same results!\n");
    return 0;
}
