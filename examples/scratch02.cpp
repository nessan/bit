#include <bit/bit.h>
int main()
{
    bit::vector v;                                          // <1>
    std::cout << "v: " << v << '\n';
    v.import_bits(std::uint8_t(0));                         // <2>
    std::cout << "v: " << v << '\n';
    v.import_bits({std::uint8_t(255), std::uint8_t(0)});    // <3>
    std::cout << "v: " << v << '\n';
    std::vector<std::uint8_t> vec{255, 0};                  // <4>
    v.import_bits(vec);
    std::cout << "v: " << v << '\n';
    v.import_bits(vec.cbegin(), vec.cend());                // <5>
    std::cout << "v: " << v << '\n';
    std::bitset<8> bs(255);                                 // <6>
    v.import_bits(bs);
    std::cout << "v: " << v << '\n';
}