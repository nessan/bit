#include <bit/bit.h>
int main()
{
    bit::vector           v1;                                           // <1>
    bit::vector           v2{32};                                       // <2>
    std::vector<uint16_t> vec{65535, 0};
    bit::vector           v3{vec};                                      // <3>
    bit::vector           v4{32, [](size_t k) { return (k + 1) % 2; }}; // <4>
    std::bitset<32>       bs{65535};
    bit::vector           v5{bs};                                       // <5>
    std::cout << "v1 = " << v1.to_string()    << '\n';
    std::cout << "v2 = " << v2.to_string()    << '\n';
    std::cout << "v3 = " << v3.to_string()    << '\n';
    std::cout << "v4 = " << v4.to_string()    << '\n';
    std::cout << "bs = " << bs                << '\n';
    std::cout << "v5 = " << v5.to_string()    << '\n';
    std::cout << "v5 = " << v5.to_bit_order() << " in bit-order!\n";
}