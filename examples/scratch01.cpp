#define BIT_VERIFY              // <1>
#include <bit/bit.h>
int main()
{
    std::size_t   n = 12;       // <2>
    bit::vector<> v(n);
    v.set(n);                   // <3>
    std::cout << v << "\n";
}