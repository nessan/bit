/// @brief Check on the polynomial_mod function by using a simple/slow alternative.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

/// @brief  Returns the coefficients of r(x) := x^N mod P(x) where P(x) is a polynomial over GF(2).
/// @param  N Power of x we are interested in.
/// @param  P The coefficients of P(x).
/// @return r Coefficients of the remainder polynomial where x^N = q(x)P(x) + r(x).
template<std::unsigned_integral Block, typename Allocator>
bit::vector<Block, Allocator>
iterative_mod(std::size_t N, const bit::vector<Block, Allocator>& P)
{
    // Only the *monic* part of P(x) matters--we can drop any trailing zero coefficients in P.
    auto monic = P.trimmed_right();

    // If all you are left with is the empty bit-vector then P(x) is the zero polynomial which is a problem.
    if (monic.empty()) throw std::invalid_argument("x^N mod 0 is not defined!");

    // The monic version of P(x) is x^m + p(x) -- we only need the lower order bit p(x).
    auto p = monic.sub(0, monic.size() - 1);
    auto m = p.size();

    // Looking for poly r(x) where deg[r(x)] < m and x^N = q(x)P(x) + r(x) for some (unknown) quotient poly q(x).
    // The remainder r(x) will be returned as a bit-vector of at most m coefficients.
    // Case N < m: Remainder is the unit vector with bit N set.
    if (N < m) {
        bit::vector<Block, Allocator> r(N + 1);
        r.set(N);
        return r;
    }

    // Case N == m: Remainder is p itself.
    auto r = p;
    if (N == m) return r;

    // Case N > m: Use an iteration that depends on a simple formula for x*r(x) mod p(x) in terms of r(x) mod p(x).
    for (std::size_t i = m; i < N; ++i) r = r[m - 1] ? p ^ (r >> 1) : (r >> 1);

    // Eliminate the trailing zero coefficients in r
    return r.trimmed_right();
}

int
main()
{
    bit::vector p(13);
    p.set();
    std::print("Polynomial p(x): {}\n", p.to_polynomial());

    // NOTE: If n is say about 28 then the iterative method above takes 16s on a recent Mac
    std::size_t n = 27;
    std::size_t N = 1ULL << n;

    std::print("Computing x^2^{} mod p(x) == x^{} mod p(x)\n", n, N);

    utilities::stopwatch sw;

    std::print("Method bit::polynomial_mod with argument n   ... ");
    sw.click();
    auto r = bit::polynomial_mod(N, p);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", r.to_polynomial(), sw.lap());

    std::print("Method bit::polynomial_mod with argument 2^n ... ");
    sw.click();
    r = bit::polynomial_mod(n, p, true);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", r.to_polynomial(), sw.lap());

    std::print("Method iterative_mod                         ... ");
    sw.click();
    r = iterative_mod(N, p);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", r.to_polynomial(), sw.lap());

    return 0;
}