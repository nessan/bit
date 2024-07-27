#undef  NDEBUG
#define BIT_DEBUG
#include <bit/bit.h>
int main()
{
    std::size_t n = 12;         // <1>
    bit::vector<> v(n);
    v.set(n);                   // <2>
    std::cout << v << "\n";
}