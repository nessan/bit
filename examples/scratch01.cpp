

#ifndef BIT_DEBUG
    #define BIT_DEBUG
#endif
#include <bit/bit.h>
int main()
{
    std::size_t n = 12;         // <1>
    bit::vector<> v(n);
    v.set(n);                   // <2>
    std::cout << v << "\n";
}