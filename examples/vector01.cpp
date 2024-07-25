/// @brief Exercise some of the basic functionality for the @c bit::vector class.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT

// NOTE: Speed is not important here and we always want bounds checking.
#ifndef BIT_DEBUG
    #define BIT_DEBUG
#endif

#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    std::size_t n = 17;
    std::size_t i = 0;

    vector_type v(n);
    v.description("construction");

    std::print("Setting successive bits:\n");
    for (i = 0; i < n; ++i) {
        v[i] = true;
        std::print("{}\n", v.to_string());
    }
    std::print("\n");

    std::print("Resetting successive bits:\n");
    for (i = 0; i < n; ++i) {
        v[i] = false;
        std::print("{}\n", v.to_string());
    }
    std::print("\n");

    std::print("Pushing bits:\n");
    for (i = 0; i < n; ++i) {
        v.push(true);
        std::print("{}\n", v.to_string());
    }
    v.description("after pushing lots of bits");

    std::print("Popping bits:\n");
    for (i = 0; i < n; ++i) {
        v.pop();
        std::print("{}\n", v.to_string());
    }
    v.description("after popping lots of bits");

    v.clear();
    v.description("after clear()");

    v.shrink_to_fit();
    v.description("after shrink_to_fit()");

    uint8_t w08 = 1;
    v.append(w08);
    v.description("appended 1 as 8 bits");

    uint16_t w16 = 3;
    v.append(w16);
    v.description("appended 3 as 16 bits");

    uint32_t w32 = 7;
    v.append(w32);
    v.description("appended 7 as 32 bits");

    uint64_t w64 = 15;
    v.append(w64);
    v.description("appended 15 as 64 bits");

    v.clear();
    v.append({1u, 3u, 7u, 15u, 31u});
    v.description("cleared and pushed an initializer list of unsigned's");

    v.set();
    v.description("set all bits");

    v.reset();
    v.description("reset all bits");

    v.flip();
    v.description("flipped all bits");

    std::size_t n_random = 143;
    auto        v00 = vector_type::random(n_random, 0.0);
    v00.description("random fill p=0.00");

    auto v25 = vector_type::random(n_random, 0.25);
    v25.description("random fill p=0.25");

    auto v50 = vector_type::random(n_random);
    v50.description("random fill p=0.50");

    auto v75 = vector_type::random(n_random, 0.75);
    v75.description("random fill p=0.75");

    auto v100 = vector_type::random(n_random, 1.0);
    v100.description("random fill p=1.00");

    std::print("Our 50-50 randomly filled & its flip\n");
    auto v50c = ~v50;
    std::print("{}\n", v50.to_string());
    std::print("{}\n\n", v50c.to_string());

    std::print("Our 50-50 randomly filled & its flip as hex-strings\n");
    std::print("{}\n", v50.to_hex());
    std::print("{}\n\n", v50c.to_hex());

    std::print("AND'ed:\n");
    std::print("{}\n\n", (v50 & v50c).to_string());

    std::print("OR'ed:\n");
    std::print("{}\n\n", (v50 | v50c).to_string());

    std::print("XOR'ed:\n");
    std::print("{}\n\n", (v50 ^ v50c).to_string());

    std::print("AND'ed:\n");
    std::print("{}\n\n", (v50 & v50c).to_string());

    std::print("v50 & v25:\n");
    std::print("v50:       {}\n", v50.to_string());
    std::print("v25:       {}\n", v25.to_string());
    std::print("v50 & v25: {}\n\n", (v50 & v25).to_string());

    std::print("v50 | v25:\n");
    std::print("v50:       {}\n", v50.to_string());
    std::print("v25:       {}\n", v25.to_string());
    std::print("v50 | v25: {}\n\n", (v50 | v25).to_string());

    std::print("v50 ^ v25:\n");
    std::print("v50:       {}\n", v50.to_string());
    std::print("v25:       {}\n", v25.to_string());
    std::print("v50 ^ v25: {}\n\n", (v50 ^ v25).to_string());

    std::print("v50 - v25:\n");
    std::print("v50:       {}\n", v50.to_string());
    std::print("v25:       {}\n", v25.to_string());
    std::print("v50 - v25: {}\n\n", (v50 - v25).to_string());

    std::print("Left shift:\n");
    std::print("v50:        {}\n", v50.to_string());
    std::print("v50 << 1:   {}\n", (v50 << 1).to_string());
    std::print("v50 << 9:   {}\n", (v50 << 9).to_string());
    std::print("v50 << 19:  {}\n", (v50 << 19).to_string());
    std::print("v50 << 21:  {}\n", (v50 << 21).to_string());
    std::print("v50 << 23:  {}\n", (v50 << 23).to_string());
    std::print("v50 << 259: {}\n\n", (v50 << 259).to_string());

    std::print("Left shift:\n");
    std::print("v50:        {}\n", v50.to_string());
    std::print("v50 >> 1:   {}\n", (v50 >> 1).to_string());
    std::print("v50 >> 9:   {}\n", (v50 >> 9).to_string());
    std::print("v50 >> 19:  {}\n", (v50 >> 19).to_string());
    std::print("v50 >> 21:  {}\n", (v50 >> 21).to_string());
    std::print("v50 >> 23:  {}\n", (v50 >> 23).to_string());
    std::print("v50 >> 259: {}\n\n", (v50 >> 259).to_string());

    n = 411;
    vector_type vPattern(n, [](size_t k) { return (k + 1) % 2; });
    std::print("Listing all the set indices in:\n{}\n\n", vPattern.to_string());
    bool none = true;
    auto npos = vector_type::npos;
    auto pos = vPattern.first_set();
    while (pos != npos) {
        none = false;
        std::print("{} ", pos);
        pos = vPattern.next_set(pos);
    }
    std::print("\n");
    if (none) std::print("NO SET BITS!");
    std::print("\n");

    std::print("Listing all the set indices using a function call instead:\n");
    vPattern.if_set_call([](std::size_t k) { std::cout << k << ' '; });
    std::print("\n\n");

    std::print("Listing all the set indices in reverse order:\n");
    pos = vPattern.final_set();
    while (pos != npos) {
        std::print("{} ", pos);
        pos = vPattern.prev_set(pos);
    }
    std::print("\n\n");

    std::print("Listing all the set indices using the built-in bit-vector set_indices() method:\n");
    std::print("{}\n\n", vPattern.set_indices());

    std::print("Listing all the UNset indices using the built-in bit-vector unset_indices() method:\n");
    std::print("{}\n\n", vPattern.unset_indices());

    std::print("Listing all the set indices in reverse order using a function call instead:\n");
    vPattern.reverse_if_set_call([](std::size_t k) { std::cout << k << ' '; });
    std::print("\n\n");

    std::print("Bits to hex and back again (from hex the size may change)...\n");
    auto s50 = v50.to_hex();
    auto v50h = vector_type::from(s50);
    if (!v50h) {
        std::print("Failed to parse '{}' as a bit-vector!\n", s50);
        exit(1);
    }
    auto v50hs = v50h->sub(0, v50.size());

    std::string msg = v50hs == v50 ? "YES" : "*** ERROR ERROR ERROR !!!no match!!! ERROR ERROR ERROR ***";
    std::print("v50:              {}\n", v50.to_string());
    std::print("s50:              {}\n", s50);
    std::print("From hex:         {}\n", v50h->to_string());
    std::print("Sub-vector match: {}\n\n", msg);

    std::print("Bits to binary string and back again ...\n");
    auto b50 = v50.to_string();
    auto v50b = vector_type::from(b50);
    if (!v50b) {
        std::print("Failed to parse '{}' as a bit-vector!n", b50);
        exit(1);
    }
    msg = (v50 == v50b) ? "YES" : "*** NO VECTORS DO NOT MATCH! ***";
    std::print("v50:       {}\n", v50.to_string());
    std::print("b50:       {}\n", b50);
    std::print("From bits: {}\n", v50b->to_string());
    std::print("Match:     {}\n\n", msg);

    // clang-format off
    std::print("Copying that bit-vector into words of various types:\n");
    uint8_t  wd8;   bit::copy(v50, wd8);
    uint16_t wd16;  bit::copy(v50, wd16);
    uint32_t wd32;  bit::copy(v50, wd32);
    uint64_t wd64;  bit::copy(v50, wd64);
    std::print("uint8_t:  {}\n", static_cast<int>(wd8));
    std::print("uint16_t: {}\n", wd16);
    std::print("uint32_t: {}\n", wd32);
    std::print("uint64_t: {}\n\n", wd64);

    std::print("Copying that bit-vector into arrays of words of various types:\n");
    constexpr std::size_t N = 4;
    std::array<uint8_t,  N> a08;  bit::copy(v50, a08);
    std::array<uint16_t, N> a16;  bit::copy(v50, a16);
    std::array<uint32_t, N> a32;  bit::copy(v50, a32);
    std::array<uint64_t, N> a64;  bit::copy(v50, a64);
    std::print("uint8_t:  {}\n", a08);
    std::print("uint16_t: {}\n", a16);
    std::print("uint32_t: {}\n", a32);
    std::print("uint64_t: {}\n\n", a64);

    std::print("Copying ALL of that bit-vector into vectors of words of various types:\n");
    std::vector<uint8_t>  v08;  bit::copy_all(v50, v08);
    std::vector<uint16_t> v16;  bit::copy_all(v50, v16);
    std::vector<uint32_t> v32;  bit::copy_all(v50, v32);
    std::vector<uint64_t> v64;  bit::copy_all(v50, v64);
    std::print("uint8_t:  {}\n", v08);
    std::print("uint16_t: {}\n", v16);
    std::print("uint32_t: {}\n", v32);
    std::print("uint64_t: {}\n\n", v64);

    std::print("Construction/reconstruction from std::vector<T>:\n");
    vector_type r08(v08.cbegin(), v08.cend());
    vector_type r16(v16.cbegin(), v16.cend());
    vector_type r32(v32.cbegin(), v32.cend());
    vector_type r64(v64.cbegin(), v64.cend());
    std::print("Original vector:             {}\n", v50.to_string());
    std::print("Reconstructed from uint8_t:  {}\n", r08.to_string());
    std::print("Reconstructed from uint16_t: {}\n", r16.to_string());
    std::print("Reconstructed from uint32_t: {}\n", r32.to_string());
    std::print("Reconstructed from uint64_t: {}\n\n", r64.to_string());

    v.resize(4); v.set();
    vector_type u1(3);
    vector_type u2(17);  u2.set();
    vector_type u3(6);
    vector_type u4(13);  u4.set();

    std::print("Starting with {}\n", v);
    v.append(u1); std::print("Appended {} to get {}\n", u1, v);
    v.append(u2); std::print("Appended {} to get {}\n", u2, v);
    v.append(u3); std::print("Appended {} to get {}\n", u3, v);
    v.append(u4); std::print("Appended {} to get {}\n", u4, v);
    std::print("\n");
    // clang-format on

    return 0;
}
