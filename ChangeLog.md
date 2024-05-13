# Changelog

This file documents notable changes to the project.

## 30-Apr-2024

- Moved the repo to github
- Moved the docs from Antora/AsciiDoc to Quarto/Markdown.

## 16-Jan-2024

- Bug fix for the `bit::polynomial_sum(...)` function applied to bit-matrices.
- Much faster block-at-a-time version of `bit::convolution(...)`.

Thanks @jason!

- Added `from(string)` factory functions to parse bit-vectors and bit-matrices from strings.
- Added a factory function to create unit bit-vectors.
- Added `unit_floor()` and `unit_ceil()` methods for bit-vectors.
- Added technical documentation covering the design of the library.
- Documented the algorithm behind the `polynomial_mod(...)` function.
- A large number of smaller fixes for both the code and the documentation.

## Late 2023

- Various small fixes for the code and documentation along with formatting updates.

## 28-Aug-2023

- Better version of the `polynomial_mod` function.

## 01-Aug-2023

- Initial releases for the `bit` library which is a replacement for an earlier `GF2` library.
