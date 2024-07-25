/// @brief Timing check on the @c reduce method in the bit-polynomial class vs. the slower iterative function.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"
#include "reduce.h"

int
main()
{
    utilities::pretty_print_thousands();

    // Set up a polynomial ...
    std::size_t degree = 17;
    auto        p = bit::polynomial<>::random(degree);
    std::print("Polynomial p(x): {}\n", p);

    // Going to compute x^N mod p(x) where N = 2^n.
    // NOTE: If n is say about 28 then the iterative method takes 16s on a recent Mac!
    std::size_t n = 27;
    std::size_t N = 1ULL << n;
    std::print("Computing x^2^{:L} mod p(x) == x^{:L} mod p(x).\n", n, N);

    utilities::stopwatch sw;

    // Use the reduce method with repeated squaring on n.
    std::print("Method `p.reduce({:L}, true)` returns ...\n", n);
    sw.click();
    auto rr = p.reduce(n, true);
    sw.click();
    std::print("\t{} in {:.6f} seconds.\n", rr, sw.lap());

    // Use the general reduce method on N.
    std::print("Method `p.reduce({:L}, false)` returns ...\n", N);
    sw.click();
    rr = p.reduce(N);
    sw.click();
    std::print("\t{} in {:.6f} seconds.\n", rr, sw.lap());

    // Use the iterative method which might be dog slow!
    std::print("Method `iterative_mod({:L})` returns ...\n", N);
    sw.click();
    auto ri = bit::iterative_mod(N, p.coefficients());
    sw.click();
    std::print("\t{} in {:.6f} seconds.\n", bit::polynomial{ri}, sw.lap());

    return 0;
}