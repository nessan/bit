#include "common.h"
int
main()
{
    auto v = bit::vector<std::uint8_t>::ones(77);

    std::cout << std::format("bit::vector: {:b}\n", v);
    std::cout << "Exporting that bit-vector to words of various types:\n";
    uint16_t word16;
    v.export_bits(word16);
    std::cout << std::format("uint16_t:    {}\n", word16);
    uint32_t word32;
    v.export_bits(word32);
    std::cout << std::format("uint32_t:    {}\n", word32);
    uint64_t word64;
    v.export_bits(word64);
    std::cout << std::format("uint64_t:    {}\n", word64);
    std::cout << std::endl;

    std::cout << std::format("bit::vector: {:b}\n", v);
    std::cout << "Exporting that bit-vector to a std::array of words of various types:\n";
    constexpr std::size_t   N = 4;
    std::array<uint16_t, N> arr16;
    v.export_bits(arr16);
    std::cout << std::format("std::array<uint16_t,4>: {}\n", arr16);
    std::array<uint32_t, N> arr32;
    v.export_bits(arr32);
    std::cout << std::format("std::array<uint32_t,4>: {}\n", arr32);
    std::array<uint64_t, N> arr64;
    v.export_bits(arr64);
    std::cout << std::format("std::array<uint64_t,4>: {}\n", arr64);
    std::cout << std::endl;

    std::cout << std::format("bit::vector: {:b}\n", v);
    std::cout << "Exporting that bit-vector to a std::vector of words of various types:\n";
    std::vector<uint16_t> vec16(N);
    v.export_bits(vec16);
    std::cout << std::format("std::vector<uint16_t>: {}\n", vec16);
    std::vector<uint32_t> vec32(N);
    v.export_bits(vec32);
    std::cout << std::format("std::vector<uint32_t>: {}\n", vec32);
    std::vector<uint64_t> vec64(N);
    v.export_bits(vec64);
    std::cout << std::format("std::vector<uint64_t>: {}\n", vec64);
    std::cout << std::endl;

    std::cout << std::format("bit::vector: {:b}\n", v);
    std::cout << "Exporting ALL of that bit-vector to a std::vector of words of various types:\n";
    v.export_all_bits(vec16);
    std::cout << std::format("std::vector<uint16_t>: {}\n", vec16);
    v.export_all_bits(vec32);
    std::cout << std::format("std::vector<uint32_t>: {}\n", vec32);
    v.export_all_bits(vec64);
    std::cout << std::format("std::vector<uint64_t>: {}\n", vec64);
    std::cout << std::endl;
}
