/// @brief Look at setting/resetting/flipping blocks of bits for the `bit::vector` class.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    using vector_type = bit::vector<std::uint8_t>;

    std::size_t n = 133;
    std::size_t i = 0;

    vector_type v(n);

    std::print("Setting ranges of elements to 1:\n");
    v.reset();
    std::print("Starting with bit-vector of size {}: {}\n", v.size(), v);
    for (i = 0; i < v.size(); ++i) {
        std::size_t maxLen = v.size() - i + 1;
        for (std::size_t len = 1; len < maxLen; ++len) {
            v.reset();
            std::print("Setting {} element(s) starting at position {}: {}\n", len, i, v.set(i, len).to_string());
        }
    }
    std::print("\n");

    std::print("Resetting ranges of elements to 0:\n");
    v.set();
    std::print("Starting with bit-vector of size {}: {}\n", v.size(), v);
    for (i = 0; i < v.size(); ++i) {
        std::size_t maxLen = v.size() - i + 1;
        for (std::size_t len = 1; len < maxLen; ++len) {
            v.set();
            std::print("Resetting {} element(s) starting at position {}: {}\n", len, i, v.reset(i, len).to_string());
        }
    }
    std::print("\n");

    std::print("Flipping ranges of elements from 1 to 0:\n");
    v.set();
    std::print("Starting with bit-vector of size {}: {}\n", v.size(), v);
    for (i = 0; i < v.size(); ++i) {
        std::size_t maxLen = v.size() - i + 1;
        for (std::size_t len = 1; len < maxLen; ++len) {
            v.set();
            std::print("Flipping {} element(s) starting at position {}: {}\n", len, i, v.flip(i, len).to_string());
        }
    }
    std::print("\n");

    return 0;
}
