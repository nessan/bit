/// @brief Checks on Gaussian Elimination for bit-matrices.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using block_type = std::uint64_t;
    using vector_type = bit::vector<block_type>;
    using matrix_type = bit::matrix<block_type>;

    std::size_t m = 16;

    auto A = matrix_type::random(m);
    auto b = vector_type::random(m);
    std::print("Solving the system A.x = b for the following A & b:\n");
    print(A, b);

    // Create a solver object for the system
    auto solver = bit::gauss(A, b);
    std::print("The echelon forms of A & b are:\n");
    print(solver.lhs(), solver.rhs());

    // Maybe there were no solutions?
    auto num_solutions = solver.solution_count();
    if (num_solutions == 0) {
        std::print("This system is inconsistent and has NO solutions!\n");
        return 0;
    }

    // Print some general information
    std::print("Rank of the bit-matrix A:        {}\n", solver.rank());
    std::print("Number of free variables:        {}\n", solver.free_count());
    std::print("Free variables are at indices:   {}\n", solver.free_indices());
    std::print("Number of solutions to A.x = b:  {}\n", num_solutions);

    // Iterate through all the solutions we can address & check each one is an actual solution.
    for (std::size_t ns = 0; ns < num_solutions; ++ns) {
        auto        x = solver(ns);
        auto        Ax = bit::dot(A, x);
        std::string msg = Ax == b ? "which matches our rhs b." : "which DOES NOT match our rhs b!!!";
        std::print("Solution {} has A.x = {} {}\n", x, Ax, msg);
    }

    return 0;
}
