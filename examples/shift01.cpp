/// @brief Simple check on right shifting versus same thing using the string representation.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT

#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    while (true) {

        // Get a bit-vector size from the user.
        std::string input;
        std::cout << "Bit-vector size (x to exit ...): ";
        std::cin >> input;
        if (input == "x" || input == "X") exit(0);
        auto n = utilities::possible<std::size_t>(input);
        if (!n) {
            std::print("Failed to parse '{}' as a number ... try again\n", input);
            continue;
        }

        // Create a random bit-vector of that size and get its compact string representation.
        auto v0 = vector_type::random(*n);
        auto v0_str = v0.to_string();

        // Get a shift amount from the user.
        std::cout << "Shift size (x to exit ...):      ";
        std::cin >> input;
        if (input == "x" || input == "X") exit(0);
        auto p = utilities::possible<std::size_t>(input);
        if (!p) {
            std::print("Failed to parse '{}' as a number ... try again\n", input);
            continue;
        }

        // Shift our vector & get its compact string representation.
        auto v1 = v0 >> *p;
        auto v1_str = v1.to_string();

        // Create a 'shifted' string version of the original bit-vector v.
        auto vs = v0_str;
        if (*p > 0) {
            auto nz = std::min(*p, vs.length());
            vs.erase(vs.length() - nz, nz);
            vs.insert(0, nz, '0');
        }

        // Print everything
        std::cout << "bit-vector v:     " << v0_str << '\n';
        std::cout << "v >>= p:          " << v1_str << '\n';
        std::cout << "v-string-shifted: " << vs << '\n';
        std::cout << "Match?            " << (v1_str == vs ? "yes" : "DO NOT MATCH!") << '\n';
    }
}