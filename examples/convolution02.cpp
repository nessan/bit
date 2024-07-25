/// @brief Checks on word-by-word convolution vs element-by-element version over a wide range of inputs.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"
#include "convolution.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    // A simple Knuth style linear congruential generator seeded to a clock dependent state.
    using lcg = std::linear_congruential_engine<uint64_t, 6364136223846793005U, 1442695040888963407U, 0U>;
    static lcg rng(static_cast<lcg::result_type>(std::chrono::system_clock::now().time_since_epoch().count()));

    // We pull the two bit-vector sizes from a uniform distribution over this range.
    std::uniform_int_distribution<> uniform(1, 1000);

    std::size_t n_trials = 1000;
    for (std::size_t n = 0; n < n_trials; ++n) {

        // Size of the two bit-vectors on this trial run.
        auto nu = static_cast<std::size_t>(uniform(rng));
        auto nv = static_cast<std::size_t>(uniform(rng));

        // Create the two bit-vectors for this trial run.
        auto u = vector_type::random(nu);
        auto v = vector_type::random(nv);

        // Do the two different style convolutions.
        auto w1 = simple_convolution(u, v);
        auto w2 = bit::convolution(u, v);

        // Check that the results match.
        if (w1 != w2) {
            // They didn't so print an error message and exit early ...
            std::print("METHOD MISMATCH, METHOD MISMATCH, METHOD MISMATCH, METHOD MISMATCH!\n");
            std::print("Input bit-vector u: {}\n", u);
            std::print("Input bit-vector v: {}\n", v);
            std::print("Simple convolution: {}\n", w1);
            std::print("Faster convolution: {}\n", w2);
            exit(1);
        }
        else {
            // Results matched to print a "progress meter" ..
            std::print(".");
            std::cout << std::flush;
        }
    }
    std::print("\nHurray! All {} trials matched!\n", n_trials);
}
