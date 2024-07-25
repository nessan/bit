#include "common.h"
#include "reduce.h"

int
main()
{
    utilities::pretty_print_thousands();

    std::size_t N = 447'124'345;
    auto        p = bit::vector<>::from(1234019u);
    std::print("Computing x^{:L} mod p(x) = {}\n", N, bit::polynomial(p));

    utilities::stopwatch sw;

    std::print("Method bit::polynomial_mod  ... ");
    sw.click();
    auto r = bit::polynomial_mod(N, p);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", bit::polynomial(r), sw.lap());

    std::print("Method polynomial_mod2      ... ");
    sw.click();
    r = jsn::polynomial_mod(N, p, false);
    sw.click();
    std::print("returned {} in {:.6f} seconds.\n", bit::polynomial(r), sw.lap());

    return 0;
}