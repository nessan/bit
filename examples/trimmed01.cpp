/// @brief Exercise the various bit-vector @c trim methods.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"

int
main()
{
    auto v0 = bit::vector<>::zeros(6);
    auto v1 = bit::vector<>::ones(12);
    auto v3 = bit::join(v0, v1, v0);
    auto vr = v3.trimmed_right();
    auto vl = v3.trimmed_left();
    auto vt = v3.trimmed();

    std::print("bit-vector:    size {} {}\n", v3.size(), v3);
    std::print("trimmed right: size {} {}\n", vr.size(), vr);
    std::print("trimmed left:  size {} {}\n", vl.size(), vl);
    std::print("trimmed:       size {} {}\n", vt.size(), vt);
}