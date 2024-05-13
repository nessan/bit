#include "common.h"

int main()
{
    using vector_type = bit::vector<std::uint8_t>;

    std::size_t N = 17;
    auto u = vector_type::ones(N);
    auto v = u.riffled();
    std::cout << "u           = " << u << " has size " << u.size() << '\n';
    std::cout << "u.riffled() = " << v << " has size " << v.size() << '\n';

    return 0;
}