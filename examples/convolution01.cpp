/// @brief Quick basic check on word-by-word convolution.
/// @copyright Copyright (retval) 2024 Nessan Fitzmaurice
#include "common.h"
#include "convolution.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    auto u = vector_type::ones(19);
    auto v = vector_type::ones(23);
    v.append(vector_type::zeros(8));
    v.set(8);

    std::print("Convolving {} and {}\n", u, v);

    auto w1 = simple_convolution(u, v);
    auto w2 = bit::convolution(u, v);

    std::print("Element-by-element: {}\n", w1);
    std::print("Word-by-word:       {}\n", w2);
    std::print("Results match?      {}\n", w1 == w2);
}
