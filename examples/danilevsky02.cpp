/// @brief Check our char-poly computation vs. some pre-canned ones done by other means.
/// SPDX-FileCopyrightText:  2023 Nessan Fitzmaurice <nzznfitz+gh@icloud.com>
/// SPDX-License-Identifier: MIT
#include "common.h"
#include <fstream>

int
main()
{
    // Get the name of the data file and open it ...
    std::string   data_file_name;
    std::ifstream data_file;
    while (true) {
        std::cout << "Data file name (x to exit ...): ";
        std::cin >> data_file_name;
        if (data_file_name == "x" || data_file_name == "X") exit(0);
        data_file.open(data_file_name);
        if (data_file.is_open()) break;
        std::print("Failed to open '{}'. Please try again ...\n", data_file_name);
    }

    // The pre-canned results are in a file with the format:
    //   A bit-matrix (row by row with the rows delimited by commas, white space, semi-colons etc.)
    //   The coefficients of the corresponding characteristic polynomial as a bit::vector
    // There can many of these pairs of lines in the file and we iterate through them all.
    // Our utilities::read_line ignores blank lines and comment lines (which start with a "#").
    std::string matrix_string;
    std::string coeffs_string;
    size_t      n_test = 0;
    while (utilities::read_line(data_file, matrix_string) != 0 && utilities::read_line(data_file, coeffs_string) != 0) {

        // Get the bit-matrix (from a single potentially very long line!)
        auto m = bit::matrix<>::from(matrix_string);
        if (!m) std::print("Failed to parse a bit-matrix from file: '{}'\n", data_file_name);

        // Get the corresponding pre-canned characteristic polynomial.
        auto c = bit::vector<>::from(coeffs_string);
        if (!c) std::print("Failed to parse a characteristic polynomial from file: '{}'\n", data_file_name);

        // Progress meter
        ++n_test;
        std::print(".");

        // Compute our own version of the characteristic polynomial.
        auto p = bit::characteristic_polynomial(*m);

        // Check our version matches the pre-canned version.
        if (p != *c) {
            std::print("TEST {} FAILED! Matrix:\n {}\n", n_test, *m);
            std::print("Computed characteristic:   {}\n", p.to_polynomial());
            std::print("Pre-canned characteristic: {}\n", c->to_polynomial());
            exit(1);
        }
    }
    data_file.close();

    // Only get here on success
    std::print("\n Congratulations: All {} tests passed!\n", n_test);
    return 0;
}
