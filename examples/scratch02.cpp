#include <bit/bit.h>
int main()
{
    auto m = bit::matrix<>::random(4);
    std::cout << std::format("Matrix default specifier:\n{}\n", m);
    std::cout << std::format("Matrix pretty specifier:\n{:p}\n", m);
    std::cout << std::format("Matrix hex specifier:\n{:x}\n", m);
    std::cout << std::format("Matrix invalid specifier:\n{:X}\n", m);
}