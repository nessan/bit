/// @brief Timing of word-by-word convolution vs element-by-element version.
/// @note  You need to compile this with optimization turned on.
/// @copyright Copyright (retval) 2024 Nessan Fitzmaurice
#include "common.h"
#include "convolution.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    // Convolving bit-vectors u and v with the following sizes.
    std::size_t nu = 5000;
    std::size_t nv = 5000;

    // Create the two input bit-vectors.
    auto u = vector_type::random(nu);
    auto v = vector_type::random(nv);

    // Space for the convolution results.
    vector_type w1, w2;

    // Number of trials & a tick size
    std::size_t n_trials = 1000;
    std::size_t n_tick = n_trials / 20;

    // And a timer
    utilities::stopwatch sw;

    // Do things the fast way
    std::print("Running {} calls of `bit::convolution(u[{}], v[{}])`   ", n_trials, u.size(), v.size());
    sw.click();
    for (std::size_t n = 0; n < n_trials; ++n) {
        if (n % n_tick == 0) std::cout << '.' << std::flush;
        bit::convolution(u, v);
    }
    sw.click();
    auto lap1 = sw.lap();
    std::print(" done.\n");

    // Do things the slower way
    std::print("Running {} calls of `simple_convolution(u[{}], v[{}])` ", n_trials, u.size(), v.size());
    sw.click();
    for (std::size_t n = 0; n < n_trials; ++n) {
        if (n % n_tick == 0) std::cout << '.' << std::flush;
        simple_convolution(u, v);
    }
    sw.click();
    auto lap2 = sw.lap();
    std::print(" done.\n");

    std::print("bit::convolution loop time:   {:.2f}s.\n", lap1);
    std::print("simple_convolution loop time: {:.2f}s.\n", lap2);
    std::print("ratio:                        {:.2f}.\n", lap2 / lap1);

    // Also check that the final result from the two methods matched.
    if (w1 != w2) {
        // They didn't so print a message and exit with an error-code ...
        std::print("METHOD MISMATCH, METHOD MISMATCH, METHOD MISMATCH, METHOD MISMATCH!\n");
        std::print("Input bit-vector u: {}\n", u);
        std::print("Input bit-vector v: {}\n", v);
        std::print("Simple convolution: {}\n", w1);
        std::print("Faster convolution: {}\n", w2);
        exit(1);
    }
}
