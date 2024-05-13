#include "common.h"

int
main()
{
    std::size_t N = 200;
    std::size_t M = 150;
    std::size_t n = N / 2;
    std::size_t m = M / 2;
    std::size_t i0 = 0;
    std::size_t j0 = 0;

    bit::matrix A = bit::matrix<>::random(M, N);

    std::size_t n_trials = 1'000'000;
    std::size_t n_tick = n_trials / 20;

    utilities::stopwatch sw;

    std::print("Running {} calls of `A.sub()`       ", n_trials);
    sw.click();
    for (std::size_t trial = 0; trial < n_trials; ++trial) {
        if (trial % n_tick == 0) std::cout << '.' << std::flush;
        auto B = A.sub(i0, j0, m, n);
    }
    sw.click();
    auto lap1 = sw.lap();
    std::print(" done.\n");

    std::print("Running {} calls of `A.sub_jason()` ", n_trials);
    sw.click();
    for (std::size_t trial = 0; trial < n_trials; ++trial) {
        if (trial % n_tick == 0) std::cout << '.' << std::flush;
        auto B = A.sub(i0, j0, m, n);
    }
    sw.click();
    auto lap2 = sw.lap();
    std::print(" done.\n");

    std::print("A.sub() loop time: {:.2f}s.\n", lap1);
    std::print("A.sub_jason():     {:.2f}s.\n", lap2);
    std::print("ratio:             {:.2f}.\n", lap2 / lap1);
}