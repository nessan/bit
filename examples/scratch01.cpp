#include <bit/bit.h>
int main()
{
    // lambda: Turns the degree of a polynomial into a string.
    auto deg = [](auto& p) { return p.degree() == bit::polynomial<>::ndeg ? "NONE" : std::format("{}", p.degree()); };

    auto p0 = bit::polynomial<>::random(0);
    std::cout << std::format("p0(x) = {} has degree: {}.\n", p0, deg(p0));

    auto p1 = bit::polynomial<>::random(7);
    std::cout << std::format("p0(x) = {} has degree: {}.\n", p1, deg(p1));

    auto p2 = bit::polynomial<>::random(7, 0.9);
    std::cout << std::format("p0(x) = {} has degree: {}.\n", p2, deg(p2));
}